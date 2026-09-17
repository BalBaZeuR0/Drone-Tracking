from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Camera:
    k: np.ndarray  # 3x3 intrinsic matrix
    r: np.ndarray  # 3x3 rotation, world -> camera
    t: np.ndarray  # 3, translation, world -> camera
    width: int
    height: int


@dataclass
class ProjectionResult:
    pixel: Optional[Tuple[float, float]]
    in_frame: bool


def project_point(camera: Camera, point_world: np.ndarray) -> ProjectionResult:
    p_cam = camera.r @ point_world + camera.t
    if p_cam[2] <= 0:
        return ProjectionResult(pixel=None, in_frame=False)

    p_img_h = camera.k @ p_cam
    u = p_img_h[0] / p_img_h[2]
    v = p_img_h[1] / p_img_h[2]
    in_frame = bool(0 <= u < camera.width and 0 <= v < camera.height)
    return ProjectionResult(pixel=(u, v), in_frame=in_frame)
