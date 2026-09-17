from typing import List, NamedTuple, Optional, Protocol, Tuple

import numpy as np


class CameraStepSignal(NamedTuple):
    latent: np.ndarray
    # Should be GROUND-TRUTH visibility — in the full system, derived from a
    # future XYZ+calibration projection (per spec), not from a detector's own
    # belief about what it sees (e.g. MaskResult.visible from the masking
    # module). CameraSwitchEnv.step() uses this field directly as the reward
    # signal; wiring it from a detector's own output instead of ground truth
    # makes the reward partly self-referential (the agent gets rewarded for
    # agreeing with the detector, not for tracking the drone).
    visible: bool
    pixel: Optional[Tuple[float, float]]


class CameraSignalProvider(Protocol):
    latent_dim: int
    num_cameras: int

    def reset(self) -> List[CameraStepSignal]:
        ...

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        ...
