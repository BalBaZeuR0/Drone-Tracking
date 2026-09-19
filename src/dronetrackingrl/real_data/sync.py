from dataclasses import dataclass
from typing import Dict, Tuple

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


DATASET4_ALPHA: Tuple[Tuple[float, ...], ...] = (
    (1.0000, 0.4983, 0.5005, 0.4999, 0.5000, 0.8342, 0.4171),
    (2.0069, 1.0000, 1.0044, 1.0033, 1.0034, 1.6741, 0.8370),
    (1.9981, 0.9956, 1.0000, 0.9989, 0.9989, 1.6667, 0.8334),
    (2.0003, 0.9967, 1.0011, 1.0000, 1.0001, 1.6686, 0.8349),
    (2.0002, 0.9966, 1.0011, 0.9999, 1.0000, 1.6685, 0.8343),
    (1.1988, 0.5973, 0.6000, 0.5993, 0.5993, 1.0000, 0.5000),
    (2.3975, 1.1947, 1.2000, 1.1978, 1.1986, 2.0000, 1.0000),
)
DATASET4_BETA: Tuple[Tuple[float, ...], ...] = (
    (0.00, 1156.29, 1128.78, 1219.88, 889.37, 3018.11, -1562.26),
    (-2320.60, 0.00, -32.64, 59.77, -270.82, 1082.34, -2529.59),
    (-2255.38, 32.50, 0.00, 92.38, -238.21, 1136.75, -2502.62),
    (-2440.13, -59.56, -92.46, 0.00, -330.57, 982.64, -2587.35),
    (-1778.91, 269.91, 238.46, 330.57, 0.00, 1534.20, -2304.50),
    (-3618.11, -646.52, -682.02, -588.87, -919.51, 0.00, -3071.00),
    (3745.56, 3022.07, 3003.04, 3099.07, 2762.22, 6142.00, 0.00),
)
# dataset4 README ("Ground truth synchronization parameters") + detections max frame_id'leri.
DATASET4_MAX_FRAME_ID: Tuple[int, ...] = (31075, 15409, 15678, 10933, 17640, 32016, 11292)


@dataclass(frozen=True)
class SyncTable:
    alpha: Tuple[Tuple[float, ...], ...]
    beta: Tuple[Tuple[float, ...], ...]
    max_frame_id: Tuple[int, ...]

    @property
    def num_cameras(self) -> int:
        return len(self.max_frame_id)


DATASET3 = SyncTable(DATASET3_ALPHA, DATASET3_BETA, DATASET3_MAX_FRAME_ID)
DATASET4 = SyncTable(DATASET4_ALPHA, DATASET4_BETA, DATASET4_MAX_FRAME_ID)
SYNC_TABLES: Dict[str, SyncTable] = {"dataset3": DATASET3, "dataset4": DATASET4}


def mapped_frame(ref_frame: int, from_cam: int, to_cam: int, table: SyncTable = DATASET3) -> float:
    return table.alpha[from_cam][to_cam] * ref_frame + table.beta[from_cam][to_cam]


def is_recording(ref_frame: int, camera: int, reference_camera: int = 0, table: SyncTable = DATASET3) -> bool:
    target_frame = mapped_frame(ref_frame, reference_camera, camera, table)
    return 0 <= target_frame <= table.max_frame_id[camera]
