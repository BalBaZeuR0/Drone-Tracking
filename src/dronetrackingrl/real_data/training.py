from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder, train_autoencoder
from dronetrackingrl.masking.background_subtraction import BackgroundSubtractor
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
    """Encoder ön-eğitimi için kameraların kayıtta olduğu her karede
    maskelenmiş kırpım toplar (RealCameraSignalProvider'dan bağımsız,
    kendi BackgroundSubtractor'larıyla — encode edilmemiş ham kırpımlara
    ihtiyaç var)."""
    crops: List[np.ndarray] = []
    subtractors = {camera: BackgroundSubtractor(crop_size=crop_size) for camera in cameras}
    for ref_frame in range(start_ref_frame, end_ref_frame + 1):
        for camera in cameras:
            if not is_recording(ref_frame, camera, reference_camera):
                continue
            camera_frame_id = round(mapped_frame(ref_frame, reference_camera, camera))
            frame = default_frame_reader(frames_root, camera, camera_frame_id)
            if frame is None:
                continue
            result = subtractors[camera].update(frame)
            crops.append(result.mask_crop)
    return crops


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
    seed: int = 0,
) -> float:
    detections = load_detections(detections_dir, cameras)
    encoder = MaskedCropAutoencoder(crop_size=crop_size, latent_dim=latent_dim)

    crops = collect_crops(frames_root, cameras, start_ref_frame, end_ref_frame, reference_camera, crop_size)
    if crops:
        crops_tensor = torch.tensor(np.stack(crops), dtype=torch.float32).unsqueeze(1)
        train_autoencoder(encoder, crops_tensor, epochs=20)

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
