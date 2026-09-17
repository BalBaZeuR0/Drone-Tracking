from typing import List, Tuple

import numpy as np

from dronetrackingrl.rl_env.signal_provider import CameraStepSignal


class FakeCameraSignalProvider:
    """Deterministic three-camera, three-frame script for tests.

    Frame 0: camera 0 visible. Frames 1-2: camera 1 visible.
    """

    latent_dim = 2
    num_cameras = 3

    def __init__(self):
        self._visible_camera_per_frame = [0, 1, 1]
        self._frame_index = -1

    def reset(self) -> List[CameraStepSignal]:
        self._frame_index = 0
        return self._signals_for_frame(self._frame_index)

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        self._frame_index += 1
        signals = self._signals_for_frame(self._frame_index)
        done = self._frame_index == len(self._visible_camera_per_frame) - 1
        return signals, done

    def _signals_for_frame(self, frame_index: int) -> List[CameraStepSignal]:
        visible_camera = self._visible_camera_per_frame[frame_index]
        return [
            CameraStepSignal(
                latent=np.array([float(camera), float(frame_index)]),
                visible=(camera == visible_camera),
                pixel=(float(camera), float(frame_index)),
            )
            for camera in range(self.num_cameras)
        ]
