import numpy as np
import torch
from stable_baselines3 import PPO

from dronetrackingrl.real_data.experiment import CachedSignalProvider, SignalCache
from dronetrackingrl.real_data.policy import SharedCameraPolicy
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv

NUM_CAMERAS, LATENT_DIM = 4, 5


def _cache() -> SignalCache:
    n = 40
    rng = np.random.default_rng(0)
    return SignalCache(
        cameras=list(range(NUM_CAMERAS)), reference_camera=0, ref_start=0, ref_end=n - 1, crop_size=4,
        crops=np.zeros((n, NUM_CAMERAS, 4, 4), dtype=np.uint8),
        recording=np.ones((n, NUM_CAMERAS), dtype=bool),
        visible=rng.random((n, NUM_CAMERAS)) > 0.5,
        mask_pixel=np.full((n, NUM_CAMERAS, 2), np.nan), gt_pixel=np.full((n, NUM_CAMERAS, 2), np.nan),
    )


def _make_model() -> PPO:
    class _Provider(CachedSignalProvider):
        """Gözlem = kameranın görünürlüğü (öğrenilebilir, kameradan bağımsız bir sinyal)."""

        def signals_for_ref_frame(self, ref_frame):
            signals = super().signals_for_ref_frame(ref_frame)
            return [s._replace(latent=np.full(LATENT_DIM, float(s.visible), dtype=np.float32)) for s in signals]

    env = CameraSwitchEnv(_Provider(_cache(), None, LATENT_DIM))
    return PPO(
        SharedCameraPolicy, env, device="cpu", seed=0, n_steps=64, batch_size=64,
        policy_kwargs=dict(num_cameras=NUM_CAMERAS, latent_dim=LATENT_DIM),
    )


def test_logits_permute_with_cameras():
    model = _make_model()
    obs = torch.randn(3, NUM_CAMERAS * LATENT_DIM)
    perm = [2, 0, 3, 1]
    permuted = obs.reshape(3, NUM_CAMERAS, LATENT_DIM)[:, perm].reshape(3, -1)

    with torch.no_grad():
        logits = model.policy.get_distribution(obs).distribution.logits
        logits_permuted = model.policy.get_distribution(permuted).distribution.logits

    torch.testing.assert_close(logits_permuted, logits[:, perm])


def test_action_layer_is_frozen_identity():
    model = _make_model()
    assert torch.equal(model.policy.action_net.weight, torch.eye(NUM_CAMERAS))
    assert not model.policy.action_net.weight.requires_grad


def test_learns_to_pick_a_visible_camera_and_survives_save_load(tmp_path):
    model = _make_model()
    model.learn(total_timesteps=4000)

    path = tmp_path / "model.zip"
    model.save(str(path))
    loaded = PPO.load(str(path), device="cpu")

    cache = _cache()
    hits = total = 0
    for i in range(cache.num_frames - 1):
        obs = np.concatenate(
            [np.full(LATENT_DIM, float(v), dtype=np.float32) for v in cache.visible[i]]
        )
        action = int(loaded.predict(obs, deterministic=True)[0])
        hits += bool(cache.visible[i, action]) or not cache.visible[i].any()
        total += 1
    assert hits / total > 0.9
