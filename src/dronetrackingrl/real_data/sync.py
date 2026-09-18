from typing import Tuple

# dataset3 için gerçek, ölçülmüş senkronizasyon parametreleri — kaynak:
# drone-tracking-datasets/dataset3/README.md, "Ground truth synchronization
# parameters". Satır = referans kamera, sütun = hedef kamera:
# frame_j = alpha[ref][target] * frame_i + beta[ref][target].
DATASET3_ALPHA: Tuple[Tuple[float, ...], ...] = (
    (1.0000, 0.5005, 0.4960, 0.4171, 0.5000, 0.8341),
    (1.9982, 1.0000, 0.9910, 0.8333, 0.9990, 1.6667),
    (2.0163, 1.0091, 1.0000, 0.8409, 1.0081, 1.6819),
    (2.3978, 1.2000, 1.1892, 1.0000, 1.1988, 2.0000),
    (2.0001, 1.0010, 0.9919, 0.8342, 1.0000, 1.6683),
    (1.1989, 0.6000, 0.5946, 0.5000, 0.5994, 1.0000),
)
DATASET3_BETA: Tuple[Tuple[float, ...], ...] = (
    (0.00, 1013.95, 546.98, 251.16, 961.02, 137.51),
    (-2026.04, 0.00, -457.83, -593.82, -51.96, -1552.47),
    (-1102.90, 461.99, 0.00, -208.81, 409.59, -782.45),
    (-602.21, 712.57, 248.32, 0.00, 659.93, -364.81),
    (-1922.12, 52.01, -406.29, -551.00, 0.00, -1465.78),
    (-164.85, 931.45, 465.22, 182.40, 878.60, 0.00),
)
# Her kameranın detections/camN.txt'indeki max frame_id'si (dataset3).
DATASET3_MAX_FRAME_ID: Tuple[int, ...] = (33875, 19960, 17166, 14196, 18900, 28080)


def mapped_frame(ref_frame: int, from_cam: int, to_cam: int) -> float:
    return DATASET3_ALPHA[from_cam][to_cam] * ref_frame + DATASET3_BETA[from_cam][to_cam]


def is_recording(ref_frame: int, camera: int, reference_camera: int = 0) -> bool:
    target_frame = mapped_frame(ref_frame, reference_camera, camera)
    return 0 <= target_frame <= DATASET3_MAX_FRAME_ID[camera]
