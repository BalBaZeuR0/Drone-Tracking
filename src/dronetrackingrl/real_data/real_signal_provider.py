import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.masking.small_target_detector import SmallTargetDetector
from dronetrackingrl.real_data.sync import is_recording, mapped_frame
from dronetrackingrl.rl_env.signal_provider import CameraStepSignal

logger = logging.getLogger(__name__)


def default_frame_reader(frames_root: Path, camera: int, frame_id: int) -> Optional[np.ndarray]:
    path = Path(frames_root) / f"cam{camera}" / f"{frame_id:06d}.jpg"
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


class RealCameraSignalProvider:
    """Gerçek çıkarılmış karelerden ve detections/camN.txt ground-truth'undan
    beslenen CameraSignalProvider implementasyonu.

    `visible` (reward için) SADECE ground-truth detections'tan ve senkronize
    kayıt durumundan gelir -- maskeleme modülü asla reward'a karışmaz, sadece
    latent gözlemi üretir (spec'in "iki ayrı kullanım" ayrımı).

    Maskeleme için varsayılan `SmallTargetDetector` (gerçek çekimlerde
    ground-truth'a karşı doğrulandı); ilk kare boş olmak zorunda değil, sadece
    ilk birkaç kare (warm-up) tespit üretmez. `detector_factory(crop_size)`
    ile başka bir dedektör (örn. Faz 1'in BackgroundSubtractor'ı) verilebilir.
    """

    def __init__(
        self,
        frames_root: Path,
        detections: Dict[int, Dict[int, Optional[Tuple[float, float]]]],
        cameras: List[int],
        encoder: MaskedCropAutoencoder,
        latent_dim: int,
        start_ref_frame: int,
        end_ref_frame: int,
        reference_camera: int = 0,
        crop_size: int = 64,
        frame_reader: Callable[[Path, int, int], Optional[np.ndarray]] = default_frame_reader,
        detector_factory: Callable[[int], object] = SmallTargetDetector,
    ):
        self.frames_root = frames_root
        self.detections = detections
        self.cameras = cameras
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.num_cameras = len(cameras)
        self.start_ref_frame = start_ref_frame
        self.end_ref_frame = end_ref_frame
        self.reference_camera = reference_camera
        self.crop_size = crop_size
        self.frame_reader = frame_reader
        self._detectors = {camera: detector_factory(crop_size) for camera in cameras}
        self._ref_frame = start_ref_frame

    def reset(self) -> List[CameraStepSignal]:
        for detector in self._detectors.values():
            detector.reset()
        self._ref_frame = self.start_ref_frame
        return self.signals_for_ref_frame(self._ref_frame)

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        self._ref_frame += 1
        signals = self.signals_for_ref_frame(self._ref_frame)
        done = self._ref_frame >= self.end_ref_frame
        return signals, done

    def signals_for_ref_frame(self, ref_frame: int) -> List[CameraStepSignal]:
        signals = []
        for camera in self.cameras:
            camera_frame_id = round(mapped_frame(ref_frame, self.reference_camera, camera))
            recording = is_recording(ref_frame, camera, self.reference_camera)
            detector = self._detectors[camera]

            frame = None
            if recording:
                frame = self.frame_reader(self.frames_root, camera, camera_frame_id)
                if frame is None:
                    logger.warning(
                        "Missing/unreadable frame for camera %d, frame_id %d "
                        "(ref_frame %d) -- treating as not visible for this step.",
                        camera, camera_frame_id, ref_frame,
                    )

            if recording and frame is not None:
                mask_result = detector.update(frame)
                # `self.detections[camera]` (not `.get(camera, {})`): a camera
                # entirely missing from `detections` is a caller wiring
                # mistake and must raise loudly, not silently degrade to
                # permanently invisible. A missing `camera_frame_id` *within*
                # a present camera's dict is the normal "no detection for
                # this exact frame" case and still resolves to None.
                ground_truth = self.detections[camera].get(camera_frame_id)
                visible = ground_truth is not None
                pixel = mask_result.centroid
                crop = mask_result.mask_crop
            else:
                visible = False
                pixel = None
                crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)

            crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                latent = self.encoder.encode(crop_tensor).squeeze(0).numpy()

            signals.append(CameraStepSignal(latent=latent, visible=visible, pixel=pixel))
        return signals
