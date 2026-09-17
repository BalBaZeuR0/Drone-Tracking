from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class MaskResult:
    mask_crop: np.ndarray
    centroid: Optional[Tuple[float, float]]
    visible: bool


class BackgroundSubtractor:
    def __init__(self, crop_size: int = 64, diff_threshold: int = 25, min_blob_area: int = 5):
        self.crop_size = crop_size
        self.diff_threshold = diff_threshold
        self.min_blob_area = min_blob_area
        self._background: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._background = None

    def update(self, frame: np.ndarray) -> MaskResult:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        if self._background is None:
            # First frame of a sequence is assumed drone-free and becomes the
            # static reference frame for differencing.
            self._background = gray
            return self._no_detection()

        diff = np.abs(gray - self._background)
        binary = (diff > self.diff_threshold).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        best_label = None
        best_area = self.min_blob_area - 1
        for label in range(1, num_labels):  # label 0 is background
            area = stats[label, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_label = label

        if best_label is None:
            return self._no_detection()

        cx, cy = centroids[best_label]
        masked = np.where(labels == best_label, gray, 0.0)
        crop = self._crop_around(masked, cx, cy)
        return MaskResult(mask_crop=crop / 255.0, centroid=(float(cx), float(cy)), visible=True)

    def _no_detection(self) -> MaskResult:
        return MaskResult(
            mask_crop=np.zeros((self.crop_size, self.crop_size), dtype=np.float32),
            centroid=None,
            visible=False,
        )

    def _crop_around(self, image: np.ndarray, cx: float, cy: float) -> np.ndarray:
        half = self.crop_size // 2
        height, width = image.shape
        x0, y0 = int(cx) - half, int(cy) - half
        x1, y1 = x0 + self.crop_size, y0 + self.crop_size

        src_x0, src_y0 = max(x0, 0), max(y0, 0)
        src_x1, src_y1 = min(x1, width), min(y1, height)

        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)

        crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        if src_x1 > src_x0 and src_y1 > src_y0:
            crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
        return crop
