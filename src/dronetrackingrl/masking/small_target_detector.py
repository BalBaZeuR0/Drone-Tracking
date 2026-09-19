from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from dronetrackingrl.masking.background_subtraction import MaskResult


@dataclass
class _Candidate:
    peak: float
    position: Tuple[float, float]  # çalışma çözünürlüğünde (x, y)


class SmallTargetDetector:
    """Gökyüzündeki küçük, koyu, hareketli hedefi (drone) bulur.

    Ham kareyi ilk kareye göre farklamak (BackgroundSubtractor) gerçek
    çekimlerde çalışmıyor: bulutlar/ışık değişir ve en büyük "fark" bloğu
    drone değil gökyüzü olur (dataset3'te gerçek etiketlere göre isabet
    ~%43, yanlış tespit ~%43). Bu sınıf onun yerine:

    1. karenin genişliği ``work_width``'ten büyükse küçültür (drone hâlâ
       birkaç piksel; 4K kameralarda hızı ciddi artırır),
    2. ``blackhat`` ile "çevresinden koyu olan küçük noktaları" öne çıkarır
       (yavaş değişen bulut/ışık bunda görünmez),
    3. bu tepkiden hızlı güncellenen zamansal bir arka planı çıkarır (durağan
       koyu yapılar -- ağaç, bina -- silinir),
    4. eşik üstü bileşenlerden, bir önceki karede de yakınında aday olanları
       (kalıcılık) tutar ve en güçlüsünü seçer.

    Dönen ``centroid`` HAM (tam çözünürlük) piksel koordinatındadır, yani
    ``detections/camN.txt`` etiketleriyle doğrudan karşılaştırılabilir.
    ``mask_crop``, ``crop_size`` x ``crop_size`` boyutunda, tespit çevresinden
    alınmış blackhat tepkisidir (0..1); tespit yoksa sıfırdır.

    Parametreler dataset3'ün gerçek etiketlerine karşı ayarlanıp, ayarda
    kullanılmayan zaman aralıklarında doğrulandı (7 pencere x 150 kare x 6
    kamera: isabet %85.6, yanlış yerde tespit %0.6, kaçırma %13.8, etiketsiz
    karede tespit %12; eski yöntem: %43 / %43 / %14 / %71). Bilinen sınır:
    drone bir koyu zeminin (ağaç hattı) önündeyse blackhat onu göremez.
    """

    def __init__(
        self,
        crop_size: int = 64,
        work_width: int = 960,
        kernel_size: int = 15,
        background_rate: float = 0.15,
        min_response: float = 14.0,
        noise_sigmas: float = 6.0,
        min_area: int = 2,
        max_area: int = 400,
        warmup_frames: int = 5,
        persist_radius: float = 20.0,
        response_scale: float = 100.0,
    ):
        self.crop_size = crop_size
        self.work_width = work_width
        self.background_rate = background_rate
        self.min_response = min_response
        self.noise_sigmas = noise_sigmas
        self.min_area = min_area
        self.max_area = max_area
        self.warmup_frames = warmup_frames
        self.persist_radius = persist_radius
        self.response_scale = response_scale
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.reset()

    def reset(self) -> None:
        self._background: Optional[np.ndarray] = None
        self._frames_seen = 0
        self._previous_positions: List[Tuple[float, float]] = []

    def update(self, frame: np.ndarray) -> MaskResult:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        if gray.shape[1] > self.work_width:
            scale = self.work_width / gray.shape[1]
            size = (self.work_width, int(round(gray.shape[0] * scale)))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        dark_spots = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, self._kernel).astype(np.float32)

        if self._background is None:
            self._background = dark_spots.copy()
            self._frames_seen = 1
            return self._no_detection()

        response = dark_spots - self._background
        self._background = (1.0 - self.background_rate) * self._background + self.background_rate * dark_spots
        self._frames_seen += 1
        if self._frames_seen <= self.warmup_frames:
            return self._no_detection()

        candidates = self._find_candidates(response)
        previous, self._previous_positions = self._previous_positions, [c.position for c in candidates]
        persistent = [
            c
            for c in candidates
            if any(np.hypot(c.position[0] - px, c.position[1] - py) <= self.persist_radius for px, py in previous)
        ]
        if not persistent:
            return self._no_detection()

        x, y = persistent[0].position
        crop = _crop_window(response, x, y, self.crop_size)
        return MaskResult(
            mask_crop=np.clip(crop / self.response_scale, 0.0, 1.0).astype(np.float32),
            centroid=(x / scale, y / scale),
            visible=True,
        )

    def _find_candidates(self, response: np.ndarray) -> List[_Candidate]:
        median = float(np.median(response))
        mad = float(np.median(np.abs(response - median))) * 1.4826 + 1e-3
        threshold = max(self.min_response, median + self.noise_sigmas * mad)
        binary = (response > threshold).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        candidates = []
        for k in range(1, count):
            area = stats[k, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            region = labels[y : y + h, x : x + w] == k
            peak = float(response[y : y + h, x : x + w][region].max())
            candidates.append(_Candidate(peak=peak, position=(float(centroids[k][0]), float(centroids[k][1]))))
        candidates.sort(key=lambda c: -c.peak)
        return candidates

    def _no_detection(self) -> MaskResult:
        return MaskResult(
            mask_crop=np.zeros((self.crop_size, self.crop_size), dtype=np.float32),
            centroid=None,
            visible=False,
        )


def _crop_window(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    half = size // 2
    height, width = image.shape
    x0, y0 = int(cx) - half, int(cy) - half
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x0 + size, width), min(y0 + size, height)
    crop = np.zeros((size, size), dtype=np.float32)
    if src_x1 > src_x0 and src_y1 > src_y0:
        crop[src_y0 - y0 : src_y1 - y0, src_x0 - x0 : src_x1 - x0] = image[src_y0:src_y1, src_x0:src_x1]
    return crop
