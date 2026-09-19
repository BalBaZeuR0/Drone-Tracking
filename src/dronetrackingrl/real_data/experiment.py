"""Taşınabilir gerçek-veri deneyi: önbellek (cache) oluştur + PPO eğit + raporla.

İki aşamalı çalışır:

1. ``build-cache``  (kareleri olan makinede, bir kez): gerçek karelerden
   maskeleme çıktılarını + ground-truth görünürlüğü küçük bir ``.npz``
   dosyasına yazar. Yavaş kısım (kare okuma + arka plan çıkarma) burasıdır.
2. ``train``        (herhangi bir makinede): sadece ``.npz`` önbelleğini
   kullanır -- kare/veri seti gerekmez. Encoder'ı ön-eğitir, PPO'yu eğitir,
   baseline'larla karşılaştırır ve tüm çıktıları tek bir klasöre + zip'e yazar.

Önbellek, ``RealCameraSignalProvider``'ın ürettiği sinyalin birebir kaydıdır
(aynı sınıf, kimlik encoder'ı ile çağrılıp ham kırpımlar saklanır) -- yeni bir
görünürlük/maskeleme mantığı yoktur. Reward hâlâ SADECE ground-truth'tan gelir.
"""

import argparse
import csv
import json
import logging
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dronetrackingrl.baselines.policies import FixedCameraPolicy, OraclePolicy, RandomPolicy
from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.real_data.real_signal_provider import RealCameraSignalProvider
from dronetrackingrl.real_data.sync import is_recording, mapped_frame
from dronetrackingrl.real_data.training import load_detections
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv
from dronetrackingrl.rl_env.signal_provider import CameraStepSignal

CACHE_VERSION = 2  # v2: maskeleme SmallTargetDetector (kırpım = blackhat tepkisi, blob merkezi ham piksel)
logger = logging.getLogger("dronetrackingrl.experiment")


# --------------------------------------------------------------------------
# Önbellek
# --------------------------------------------------------------------------
@dataclass
class SignalCache:
    cameras: List[int]
    reference_camera: int
    ref_start: int
    ref_end: int
    crop_size: int
    crops: np.ndarray       # uint8   (n_ref, n_cam, crop, crop)  maskelenmiş kırpım * 255
    recording: np.ndarray   # bool    (n_ref, n_cam)  kamera o an senkronize kayıtta mı
    visible: np.ndarray     # bool    (n_ref, n_cam)  ground-truth görünürlük (REWARD kaynağı)
    mask_pixel: np.ndarray  # float64 (n_ref, n_cam, 2)  maskeleme blob merkezi, yoksa NaN
    gt_pixel: np.ndarray    # float64 (n_ref, n_cam, 2)  ground-truth (x,y), görünmüyorsa NaN

    @property
    def num_frames(self) -> int:
        return self.ref_end - self.ref_start + 1

    def slice(self, ref_start: int, ref_end: int) -> "SignalCache":
        """[ref_start, ref_end] alt aralığı (önbelleğin içinde olmalı)."""
        if not (self.ref_start <= ref_start < ref_end <= self.ref_end):
            raise ValueError(
                f"Aralık [{ref_start}, {ref_end}] önbellek aralığı [{self.ref_start}, {self.ref_end}] içinde değil"
            )
        a, b = ref_start - self.ref_start, ref_end - self.ref_start + 1
        return SignalCache(
            cameras=list(self.cameras), reference_camera=self.reference_camera,
            ref_start=ref_start, ref_end=ref_end, crop_size=self.crop_size,
            crops=self.crops[a:b], recording=self.recording[a:b], visible=self.visible[a:b],
            mask_pixel=self.mask_pixel[a:b], gt_pixel=self.gt_pixel[a:b],
        )

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            np.savez_compressed(
                handle,
                cameras=np.array(self.cameras, dtype=np.int64),
                meta=np.array(
                    [self.reference_camera, self.ref_start, self.ref_end, self.crop_size, CACHE_VERSION],
                    dtype=np.int64,
                ),
                crops=self.crops,
                recording=self.recording,
                visible=self.visible,
                mask_pixel=self.mask_pixel,
                gt_pixel=self.gt_pixel,
            )

    @classmethod
    def load(cls, path) -> "SignalCache":
        data = np.load(Path(path))
        reference_camera, ref_start, ref_end, crop_size, version = (int(v) for v in data["meta"])
        if version != CACHE_VERSION:
            raise ValueError(f"Önbellek sürümü uyumsuz: dosya v{version}, kod v{CACHE_VERSION}")
        return cls(
            cameras=[int(c) for c in data["cameras"]],
            reference_camera=reference_camera,
            ref_start=ref_start,
            ref_end=ref_end,
            crop_size=crop_size,
            crops=data["crops"],
            recording=data["recording"],
            visible=data["visible"],
            mask_pixel=data["mask_pixel"],
            gt_pixel=data["gt_pixel"],
        )


class _FlattenEncoder:
    """RealCameraSignalProvider'a 'encoder' olarak verilir; kırpımı olduğu gibi
    (düzleştirilmiş) geri döndürür, böylece ham kırpımı provider'ın kendi
    mantığından elde ederiz (mantığı kopyalamadan)."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


def build_signal_cache(
    frames_root,
    detections_dir,
    cameras: Sequence[int],
    ref_start: int,
    ref_end: int,
    reference_camera: int = 0,
    crop_size: int = 64,
    progress_every: int = 500,
    log=print,
) -> SignalCache:
    cameras = list(cameras)
    detections = load_detections(Path(detections_dir), cameras)
    provider = RealCameraSignalProvider(
        frames_root=Path(frames_root),
        detections=detections,
        cameras=cameras,
        encoder=_FlattenEncoder(),
        latent_dim=crop_size * crop_size,
        start_ref_frame=ref_start,
        end_ref_frame=ref_end,
        reference_camera=reference_camera,
        crop_size=crop_size,
    )
    n_ref = ref_end - ref_start + 1
    n_cam = len(cameras)
    crops = np.zeros((n_ref, n_cam, crop_size, crop_size), dtype=np.uint8)
    recording = np.zeros((n_ref, n_cam), dtype=bool)
    visible = np.zeros((n_ref, n_cam), dtype=bool)
    mask_pixel = np.full((n_ref, n_cam, 2), np.nan, dtype=np.float64)
    gt_pixel = np.full((n_ref, n_cam, 2), np.nan, dtype=np.float64)

    def store(index: int, signals: List[CameraStepSignal]) -> None:
        ref_frame = ref_start + index
        for j, (camera, signal) in enumerate(zip(cameras, signals)):
            crops[index, j] = np.rint(signal.latent.reshape(crop_size, crop_size) * 255.0).astype(np.uint8)
            recording[index, j] = is_recording(ref_frame, camera, reference_camera)
            visible[index, j] = signal.visible
            if signal.pixel is not None:
                mask_pixel[index, j] = signal.pixel
            if signal.visible:
                frame_id = round(mapped_frame(ref_frame, reference_camera, camera))
                gt_pixel[index, j] = detections[camera][frame_id]

    started = time.time()
    store(0, provider.reset())
    index = 1
    while ref_end > ref_start:
        signals, done = provider.step()
        store(index, signals)
        index += 1
        if progress_every and index % progress_every == 0:
            elapsed = time.time() - started
            eta = elapsed / index * (n_ref - index)
            log(f"  önbellek: {index}/{n_ref} kare  ({elapsed:.0f}s geçti, ~{eta:.0f}s kaldı)")
        if done:
            break

    return SignalCache(
        cameras=cameras,
        reference_camera=reference_camera,
        ref_start=ref_start,
        ref_end=ref_end,
        crop_size=crop_size,
        crops=crops,
        recording=recording,
        visible=visible,
        mask_pixel=mask_pixel,
        gt_pixel=gt_pixel,
    )


class CachedSignalProvider:
    """CameraSignalProvider: RealCameraSignalProvider ile aynı sinyal, ama
    kare okumadan, önbellekten. ``encoder=None`` ise gözlem sıfır vektörüdür
    (Random/Fixed/Oracle baseline'ları gözlemi kullanmaz)."""

    def __init__(
        self,
        cache: SignalCache,
        encoder: Optional[MaskedCropAutoencoder],
        latent_dim: int,
        latent_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.cache = cache
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.latent_stats = latent_stats  # (ortalama, std): gözlemi standartlaştırır
        self.num_cameras = len(cache.cameras)
        self._ref_frame = cache.ref_start

    def reset(self) -> List[CameraStepSignal]:
        self._ref_frame = self.cache.ref_start
        return self.signals_for_ref_frame(self._ref_frame)

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        self._ref_frame += 1
        signals = self.signals_for_ref_frame(self._ref_frame)
        return signals, self._ref_frame >= self.cache.ref_end

    def signals_for_ref_frame(self, ref_frame: int) -> List[CameraStepSignal]:
        i = ref_frame - self.cache.ref_start
        if self.encoder is None:
            latents = np.zeros((self.num_cameras, self.latent_dim), dtype=np.float32)
        else:
            batch = torch.from_numpy(self.cache.crops[i].astype(np.float32) / 255.0).unsqueeze(1)
            with torch.no_grad():
                latents = self.encoder.encode(batch).numpy()
            if self.latent_stats is not None:
                mean, std = self.latent_stats
                latents = ((latents - mean) / std).astype(np.float32)
        signals = []
        for j in range(self.num_cameras):
            px = self.cache.mask_pixel[i, j]
            pixel = None if np.isnan(px[0]) else (float(px[0]), float(px[1]))
            signals.append(
                CameraStepSignal(latent=latents[j], visible=bool(self.cache.visible[i, j]), pixel=pixel)
            )
        return signals


class RotatingProvider:
    """Birden fazla sağlayıcıyı (farklı zaman aralıkları) her reset'te sırayla
    kullanır: ajan hem 'cam0 hep görünür' hem 'cam0 kaybolur' bölümlerini görür."""

    def __init__(self, providers: Sequence[CachedSignalProvider]):
        self.providers = list(providers)
        self.latent_dim = self.providers[0].latent_dim
        self.num_cameras = self.providers[0].num_cameras
        self._index = -1
        self._current = self.providers[0]

    def reset(self) -> List[CameraStepSignal]:
        self._index = (self._index + 1) % len(self.providers)
        self._current = self.providers[self._index]
        return self._current.reset()

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        return self._current.step()


# --------------------------------------------------------------------------
# Encoder ön-eğitimi (önbellekten, mini-batch)
# --------------------------------------------------------------------------
def pretrain_encoder(
    encoder: MaskedCropAutoencoder,
    cache,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> List[float]:
    """Sadece maskelemenin drone bulduğu kırpımlarla (boş kırpımlar hariç)
    autoencoder'ı eğitir. Epoch başına ortalama kayıp listesini döndürür."""
    caches = cache if isinstance(cache, (list, tuple)) else [cache]
    crops = np.concatenate([c.crops[~np.isnan(c.mask_pixel[..., 0])] for c in caches])
    if len(crops) == 0:
        return []
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    epoch_losses: List[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(crops))
        total, count = 0.0, 0
        for start in range(0, len(crops), batch_size):
            batch_u8 = crops[order[start : start + batch_size]]
            batch = torch.from_numpy(batch_u8.astype(np.float32) / 255.0).unsqueeze(1)
            optimizer.zero_grad()
            loss = loss_fn(encoder(batch), batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_u8)
            count += len(batch_u8)
        epoch_losses.append(total / count)
    return epoch_losses


def compute_latent_stats(encoder: MaskedCropAutoencoder, cache, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Eğitim önbelleğindeki TÜM kırpımların (boşlar dahil) latent ortalama /
    std'si. PPO'nun MLP'si küçük ölçekli, neredeyse sabit girdilere tepki
    veremiyor; standartlaştırma bunu düzeltmek için."""
    caches = cache if isinstance(cache, (list, tuple)) else [cache]
    flat = np.concatenate([c.crops.reshape(-1, c.crop_size, c.crop_size) for c in caches])
    chunks = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            batch = torch.from_numpy(flat[start : start + batch_size].astype(np.float32) / 255.0).unsqueeze(1)
            chunks.append(encoder.encode(batch).numpy())
    latents = np.concatenate(chunks)
    return latents.mean(axis=0), np.maximum(latents.std(axis=0), 1e-3)


# --------------------------------------------------------------------------
# Değerlendirme
# --------------------------------------------------------------------------
def run_episode(cache: SignalCache, encoder, latent_dim: int, policy=None, model=None, latent_stats=None) -> List[dict]:
    """Bir bölüm (tüm önbellek aralığı) çalıştırır. Her adım için: hangi kare
    (ref_frame) için hangi kamera seçildi, seçilen görünür müydü, en az bir
    kamera görünür müydü, ve her kameranın görünürlüğü."""
    env = CameraSwitchEnv(CachedSignalProvider(cache, encoder, latent_dim, latent_stats))
    obs, info = env.reset()
    ref_frame = cache.ref_start
    rows: List[dict] = []
    while True:
        if model is not None:
            action = int(model.predict(obs, deterministic=True)[0])
        else:
            action = int(policy.select_action(info))
        visible = list(info["visible"])
        obs, _, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "ref_frame": ref_frame,
                "action": action,
                "camera": cache.cameras[action],
                "chosen_visible": bool(visible[action]),
                "any_visible": bool(any(visible)),
                **{f"visible_cam{c}": bool(v) for c, v in zip(cache.cameras, visible)},
            }
        )
        ref_frame += 1
        if terminated or truncated:
            return rows


def summarize_rows(rows: List[dict], cameras: List[int]) -> dict:
    chosen = np.array([r["chosen_visible"] for r in rows], dtype=bool)
    any_visible = np.array([r["any_visible"] for r in rows], dtype=bool)
    actions = np.array([r["action"] for r in rows])
    return {
        "steps": len(rows),
        "avg_reward": float(np.mean(2.0 * chosen - 1.0)),
        "visible_rate": float(chosen.mean()),
        "best_possible_rate": float(any_visible.mean()),
        "switches": int((actions[1:] != actions[:-1]).sum()),
        "camera_share": {str(c): float((actions == k).mean()) for k, c in enumerate(cameras)},
    }


def observation_diagnostics(cache: SignalCache) -> dict:
    """RL gözleminin (maskeleme çıktısı) ground-truth ile ne kadar örtüştüğü —
    kamera başına. Düşük precision/recall = gözlem bilgi taşımıyor demektir."""
    result = {}
    for j, camera in enumerate(cache.cameras):
        rec = cache.recording[:, j]
        gt = cache.visible[:, j]
        det = ~np.isnan(cache.mask_pixel[:, j, 0])
        tp = int((gt & det).sum())
        fp = int((rec & ~gt & det).sum())
        fn = int((gt & ~det).sum())
        tn = int((rec & ~gt & ~det).sum())
        both = gt & det
        if both.any():
            err = np.linalg.norm(cache.mask_pixel[both, j] - cache.gt_pixel[both, j], axis=1)
            err_median, err_mean = float(np.median(err)), float(err.mean())
        else:
            err_median = err_mean = None
        result[str(camera)] = {
            "recording_rate": float(rec.mean()),
            "gt_visible_rate": float(gt.mean()),
            "mask_detect_rate": float(det[rec].mean()) if rec.any() else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "centroid_error_px_median": err_median,
            "centroid_error_px_mean": err_mean,
        }
    return result


def baseline_metrics(cache: SignalCache, latent_dim: int, random_seeds: int) -> dict:
    cameras = cache.cameras
    n = len(cameras)
    out = {}
    randoms = [
        summarize_rows(run_episode(cache, None, latent_dim, policy=RandomPolicy(n, seed=s)), cameras)
        for s in range(random_seeds)
    ]
    rates = [r["visible_rate"] for r in randoms]
    out["random"] = {"visible_rate_mean": float(np.mean(rates)), "visible_rate_std": float(np.std(rates)), "runs": random_seeds}
    fixed = {
        str(c): summarize_rows(run_episode(cache, None, latent_dim, policy=FixedCameraPolicy(k)), cameras)
        for k, c in enumerate(cameras)
    }
    out["fixed"] = fixed
    best_cam = max(fixed, key=lambda c: fixed[c]["visible_rate"])
    out["best_fixed"] = {"camera": int(best_cam), **fixed[best_cam]}
    out["oracle"] = summarize_rows(run_episode(cache, None, latent_dim, policy=OraclePolicy()), cameras)
    return out


# --------------------------------------------------------------------------
# Deney
# --------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    train_cache: str
    out_dir: str
    eval_cache: Optional[str] = None
    timesteps: int = 100_000
    seeds: List[int] = field(default_factory=lambda: [0])
    latent_dim: int = 16
    encoder_epochs: int = 20
    encoder_batch_size: int = 256
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.0
    ppo_lr: float = 3e-4
    ppo_ent_coef: float = 0.0
    normalize_latents: bool = True
    policy: str = "shared"  # "shared" (kameradan bağımsız skorlayıcı) veya "mlp" (düz MLP)
    random_seeds: int = 5
    pack: bool = True
    train_range: Optional[Tuple[int, int]] = None  # önbelleğin alt aralığı
    extra_train: List[Tuple[str, int, int]] = field(default_factory=list)  # (önbellek, başlangıç, bitiş)
    eval_range: Optional[Tuple[int, int]] = None


def _write_trace(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_summary(results: dict) -> str:
    lines = [f"Deney: {results['config']['out_dir']}", f"Toplam süre: {results['elapsed_seconds']:.0f}s", ""]
    for split, info in results["splits"].items():
        lines.append(f"=== {split.upper()} (ref_frame {info['ref_start']}..{info['ref_end']}, {info['steps']} adım) ===")
        base = results["baselines"][split]
        lines.append(f"  En iyi mümkün (oracle) görünür oranı : {base['oracle']['visible_rate']:.4f}")
        lines.append(f"  Rastgele                             : {base['random']['visible_rate_mean']:.4f} ± {base['random']['visible_rate_std']:.4f}")
        lines.append(f"  En iyi sabit kamera (cam{base['best_fixed']['camera']})      : {base['best_fixed']['visible_rate']:.4f}")
        ppo_rates = [run["splits"][split]["visible_rate"] for run in results["ppo"].values()]
        lines.append(f"  PPO ({len(ppo_rates)} seed)                        : {np.mean(ppo_rates):.4f} ± {np.std(ppo_rates):.4f}   (seed başına: {', '.join(f'{r:.4f}' for r in ppo_rates)})")
        for seed, run in results["ppo"].items():
            lines.append(f"    seed {seed}: kamera payı {run['splits'][split]['camera_share']}, geçiş sayısı {run['splits'][split]['switches']}")
        lines.append("")
    lines.append("Yorum ipucu: PPO ~ 'en iyi sabit kamera' ise ajan kamera değiştirmeyi öğrenmemiş, tek kameraya yapışmış demektir.")
    lines.append("Gözlem kalitesi için results.json -> observation_diagnostics (precision/recall) bölümüne bak.")
    return "\n".join(lines) + "\n"


def run_experiment(config: ExperimentConfig) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "run.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)

    started = time.time()
    caches = {"train": SignalCache.load(config.train_cache)}
    if config.train_range:
        caches["train"] = caches["train"].slice(*config.train_range)
    if config.eval_cache or config.eval_range:
        eval_cache = SignalCache.load(config.eval_cache or config.train_cache)
        caches["eval"] = eval_cache.slice(*config.eval_range) if config.eval_range else eval_cache
    for k, (path, a, b) in enumerate(config.extra_train, start=1):
        caches[f"train_extra{k}"] = SignalCache.load(path).slice(a, b)
    cameras = caches["train"].cameras
    for name, cache in caches.items():
        if cache.cameras != cameras:
            raise ValueError(f"'{name}' önbelleğinin kameraları farklı: {cache.cameras} != {cameras}")
        logger.info("%s önbelleği: ref_frame %d..%d, kameralar %s", name, cache.ref_start, cache.ref_end, cache.cameras)

    results: dict = {
        "config": asdict(config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "splits": {},
        "observation_diagnostics": {},
        "baselines": {},
        "ppo": {},
    }
    for name, cache in caches.items():
        results["observation_diagnostics"][name] = observation_diagnostics(cache)
        results["baselines"][name] = baseline_metrics(cache, config.latent_dim, config.random_seeds)
        results["splits"][name] = {
            "ref_start": cache.ref_start,
            "ref_end": cache.ref_end,
            "steps": cache.num_frames - 1,
        }
        logger.info(
            "%s baseline'ları: oracle %.4f | rastgele %.4f | en iyi sabit cam%s %.4f",
            name,
            results["baselines"][name]["oracle"]["visible_rate"],
            results["baselines"][name]["random"]["visible_rate_mean"],
            results["baselines"][name]["best_fixed"]["camera"],
            results["baselines"][name]["best_fixed"]["visible_rate"],
        )

    train_caches = [c for n, c in caches.items() if n.startswith("train")]
    train_cache = train_caches[0]
    for seed in config.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        logger.info("--- seed %d: encoder ön-eğitimi ---", seed)
        torch.manual_seed(seed)
        encoder = MaskedCropAutoencoder(crop_size=train_cache.crop_size, latent_dim=config.latent_dim)
        losses = pretrain_encoder(
            encoder, train_caches, config.encoder_epochs, config.encoder_batch_size, seed=seed
        )
        if losses:
            logger.info("encoder kaybı: %.5f -> %.5f", losses[0], losses[-1])

        latent_stats = compute_latent_stats(encoder, train_caches) if config.normalize_latents else None
        logger.info("--- seed %d: PPO eğitimi (%d adım) ---", seed, config.timesteps)
        vec_env = make_vec_env(
            lambda: CameraSwitchEnv(
                RotatingProvider([CachedSignalProvider(c, encoder, config.latent_dim, latent_stats) for c in train_caches])
            ),
            n_envs=1,
        )
        if config.policy == "shared":
            from dronetrackingrl.real_data.policy import SharedCameraPolicy

            policy, policy_kwargs = SharedCameraPolicy, dict(
                num_cameras=len(cameras), latent_dim=config.latent_dim
            )
        elif config.policy == "mlp":
            policy, policy_kwargs = "MlpPolicy", {}
        else:
            raise ValueError(f"Bilinmeyen politika: {config.policy}")
        model = PPO(
            policy, vec_env, verbose=0, seed=seed, device="cpu", policy_kwargs=policy_kwargs,
            n_steps=config.ppo_n_steps, batch_size=config.ppo_batch_size, gamma=config.ppo_gamma,
            learning_rate=config.ppo_lr, ent_coef=config.ppo_ent_coef,
        )
        model.set_logger(configure(str(seed_dir), ["stdout", "csv"]))
        train_started = time.time()
        model.learn(total_timesteps=config.timesteps)
        model.save(str(seed_dir / "ppo_model"))
        torch.save(encoder.state_dict(), seed_dir / "encoder.pt")
        if latent_stats is not None:
            np.savez(seed_dir / "latent_stats.npz", mean=latent_stats[0], std=latent_stats[1])

        run = {"encoder_loss_first": losses[0] if losses else None,
               "encoder_loss_last": losses[-1] if losses else None,
               "train_seconds": time.time() - train_started, "splits": {}}
        for name, cache in caches.items():
            rows = run_episode(cache, encoder, config.latent_dim, model=model, latent_stats=latent_stats)
            _write_trace(seed_dir / f"trace_{name}.csv", rows)
            run["splits"][name] = summarize_rows(rows, cache.cameras)
            logger.info("seed %d %s: görünür oranı %.4f, kamera payı %s",
                        seed, name, run["splits"][name]["visible_rate"], run["splits"][name]["camera_share"])
        results["ppo"][str(seed)] = run

    results["elapsed_seconds"] = time.time() - started
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    (out_dir / "summary.txt").write_text(_format_summary(results), encoding="utf-8")
    logger.info("Yazıldı: %s", out_dir)
    if config.pack:
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=str(out_dir.parent), base_dir=out_dir.name)
        logger.info("Tek dosya paketi (geri götürülecek): %s", archive)
    logging.shutdown()
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dronetrackingrl.real_data.experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-cache", help="Karelerden önbellek üret (kareleri olan makinede)")
    build.add_argument("--frames-root", required=True)
    build.add_argument("--detections-dir", required=True)
    build.add_argument("--ref-start", type=int, required=True)
    build.add_argument("--ref-end", type=int, required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    build.add_argument("--reference-camera", type=int, default=0)
    build.add_argument("--crop-size", type=int, default=64)

    inspect = sub.add_parser("inspect-cache", help="Önbelleğin gözlem-kalitesi tanılarını yazdır")
    inspect.add_argument("cache")

    train = sub.add_parser("train", help="Önbellekten PPO eğit + raporla (kare gerekmez)")
    train.add_argument("--train-cache", required=True)
    train.add_argument("--eval-cache")
    train.add_argument("--extra-train", nargs="+", metavar="ÖNBELLEK:BAŞLANGIÇ:BİTİŞ", default=[],
                       help="ek eğitim kaynakları (ör. caches/eval.npz:28534:31500); her bölümde dönüşümlü kullanılır")
    train.add_argument("--train-range", type=int, nargs=2, metavar=("START", "END"),
                       help="eğitim önbelleğinin alt aralığı (ref_frame)")
    train.add_argument("--eval-range", type=int, nargs=2, metavar=("START", "END"),
                       help="test için alt aralık (--eval-cache verilmezse eğitim önbelleğinden alınır)")
    train.add_argument("--out-dir", required=True)
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--seeds", type=int, nargs="+", default=[0])
    train.add_argument("--latent-dim", type=int, default=16)
    train.add_argument("--encoder-epochs", type=int, default=20)
    train.add_argument("--encoder-batch-size", type=int, default=256)
    train.add_argument("--ppo-n-steps", type=int, default=2048)
    train.add_argument("--ppo-batch-size", type=int, default=64)
    train.add_argument("--policy", choices=["shared", "mlp"], default="shared",
                       help="shared: kameradan bağımsız skorlayıcı (varsayılan); mlp: düz MLP")
    train.add_argument("--ppo-lr", type=float, default=3e-4)
    train.add_argument("--ppo-ent-coef", type=float, default=0.0)
    train.add_argument("--no-normalize-latents", action="store_true", help="Gözlem standartlaştırmasını kapat")
    train.add_argument("--ppo-gamma", type=float, default=0.0,
                       help="İndirim faktörü. Kamera seçimi sonraki kareyi etkilemediği için (switch_penalty=0) 0 varsayılan.")
    train.add_argument("--random-seeds", type=int, default=5)
    train.add_argument("--no-pack", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build-cache":
        cache = build_signal_cache(
            args.frames_root, args.detections_dir, args.cameras, args.ref_start, args.ref_end,
            reference_camera=args.reference_camera, crop_size=args.crop_size,
            log=lambda msg: print(msg, flush=True),
        )
        cache.save(args.out)
        print(f"Kaydedildi: {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)", flush=True)
    elif args.command == "inspect-cache":
        print(json.dumps(observation_diagnostics(SignalCache.load(args.cache)), indent=2))
    else:
        run_experiment(
            ExperimentConfig(
                train_cache=args.train_cache, eval_cache=args.eval_cache, out_dir=args.out_dir,
                timesteps=args.timesteps, seeds=args.seeds, latent_dim=args.latent_dim,
                encoder_epochs=args.encoder_epochs, encoder_batch_size=args.encoder_batch_size,
                ppo_n_steps=args.ppo_n_steps, ppo_batch_size=args.ppo_batch_size, ppo_gamma=args.ppo_gamma, ppo_lr=args.ppo_lr, ppo_ent_coef=args.ppo_ent_coef,
                normalize_latents=not args.no_normalize_latents, policy=args.policy,
                random_seeds=args.random_seeds, pack=not args.no_pack,
                train_range=tuple(args.train_range) if args.train_range else None,
                extra_train=[(p, int(a), int(b)) for p, a, b in (spec.rsplit(':', 2) for spec in args.extra_train)],
                eval_range=tuple(args.eval_range) if args.eval_range else None,
            )
        )


if __name__ == "__main__":
    main()
