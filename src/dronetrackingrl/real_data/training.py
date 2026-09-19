from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.masking.small_target_detector import SmallTargetDetector
from dronetrackingrl.real_data.detections import parse_detections_file
from dronetrackingrl.real_data.real_signal_provider import RealCameraSignalProvider, default_frame_reader
from dronetrackingrl.real_data.sync import is_recording, mapped_frame
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv


def load_detections(
    detections_dir: Path, cameras: List[int]
) -> Dict[int, Dict[int, Optional[Tuple[float, float]]]]:
    return {
        camera: parse_detections_file(str(Path(detections_dir) / f"cam{camera}.txt"))
        for camera in cameras
    }


def collect_crops(
    frames_root: Path,
    cameras: List[int],
    start_ref_frame: int,
    end_ref_frame: int,
    reference_camera: int = 0,
    crop_size: int = 64,
) -> List[np.ndarray]:
    """Encoder ön-eğitimi için kameraların kayıtta olduğu ve drone'un
    gerçekten tespit edildiği karelerde maskelenmiş kırpım toplar
    (RealCameraSignalProvider'dan bağımsız, kendi SmallTargetDetector'larıyla
    — encode edilmemiş ham kırpımlara ihtiyaç var). Tespit edilmeyen (boş)
    kareler bilerek atlanıyor: dataset3'ün gerçek görünürlük oranları
    (%42-94) göz önüne alındığında, bunları dahil etmek eğitim setinin
    büyük bir kısmını tekdüze sıfır kırpımla doldurup encoder'ı bozardı."""
    crops: List[np.ndarray] = []
    detectors = {camera: SmallTargetDetector(crop_size=crop_size) for camera in cameras}
    for ref_frame in range(start_ref_frame, end_ref_frame + 1):
        for camera in cameras:
            if not is_recording(ref_frame, camera, reference_camera):
                continue
            camera_frame_id = round(mapped_frame(ref_frame, reference_camera, camera))
            frame = default_frame_reader(frames_root, camera, camera_frame_id)
            if frame is None:
                continue
            result = detectors[camera].update(frame)
            if result.visible:
                crops.append(result.mask_crop)
    return crops


def _train_encoder_in_batches(
    encoder: MaskedCropAutoencoder,
    crops: List[np.ndarray],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> None:
    """train_autoencoder (Faz 1, encoder/autoencoder.py) her epoch'ta tüm
    kırpımları tek seferde (full-batch) işliyor -- dataset3 ölçeğinde
    (yüz binlerce kırpım olabilir) bu bellek taşırır. Faz 1'in paylaşılan
    modülüne dokunmadan, burada mini-batch'li bir eğitim döngüsü
    kullanıyoruz."""
    if not crops:
        return
    crops_array = np.stack(crops)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    num_samples = len(crops_array)
    for _ in range(epochs):
        indices = rng.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = torch.tensor(crops_array[batch_indices], dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            reconstruction = encoder(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()


def train_and_evaluate(
    frames_root: Path,
    detections_dir: Path,
    cameras: List[int],
    start_ref_frame: int,
    end_ref_frame: int,
    reference_camera: int = 0,
    crop_size: int = 64,
    latent_dim: int = 16,
    total_timesteps: int = 2000,
    encoder_batch_size: int = 256,
    seed: int = 0,
) -> float:
    detections = load_detections(detections_dir, cameras)
    encoder = MaskedCropAutoencoder(crop_size=crop_size, latent_dim=latent_dim)

    crops = collect_crops(frames_root, cameras, start_ref_frame, end_ref_frame, reference_camera, crop_size)
    _train_encoder_in_batches(encoder, crops, epochs=20, batch_size=encoder_batch_size, seed=seed)

    def make_env():
        provider = RealCameraSignalProvider(
            frames_root=frames_root,
            detections=detections,
            cameras=cameras,
            encoder=encoder,
            latent_dim=latent_dim,
            start_ref_frame=start_ref_frame,
            end_ref_frame=end_ref_frame,
            reference_camera=reference_camera,
            crop_size=crop_size,
        )
        return CameraSwitchEnv(provider)

    train_env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=total_timesteps)

    eval_env = make_env()
    obs, _ = eval_env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(int(action))
        total_reward += reward
        steps += 1
    return total_reward / steps if steps else 0.0
