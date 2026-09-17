import random
from typing import Dict, List


class FixedCameraPolicy:
    def __init__(self, camera_index: int):
        self.camera_index = camera_index

    def select_action(self, info: Dict) -> int:
        return self.camera_index


class RandomPolicy:
    def __init__(self, num_cameras: int, seed: int = 0):
        self.num_cameras = num_cameras
        self._rng = random.Random(seed)

    def select_action(self, info: Dict) -> int:
        return self._rng.randrange(self.num_cameras)


class OraclePolicy:
    def select_action(self, info: Dict) -> int:
        visible: List[bool] = info["visible"]
        for camera_index, is_visible in enumerate(visible):
            if is_visible:
                return camera_index
        return 0
