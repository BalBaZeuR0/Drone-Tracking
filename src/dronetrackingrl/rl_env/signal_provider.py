from typing import List, NamedTuple, Optional, Protocol, Tuple

import numpy as np


class CameraStepSignal(NamedTuple):
    latent: np.ndarray
    visible: bool
    pixel: Optional[Tuple[float, float]]


class CameraSignalProvider(Protocol):
    latent_dim: int
    num_cameras: int

    def reset(self) -> List[CameraStepSignal]:
        ...

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        ...
