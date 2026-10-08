from typing import Optional, Tuple

import numpy as np

from dronetrackingrl.masking.background_subtraction import MaskResult
from dronetrackingrl.masking.small_target_detector import SmallTargetDetector, _crop_window


class HybridReloDetector:
    """SmallTargetDetector drone'u bulur, RELO takip eder (bkz.
    docs/superpowers/specs/2026-10-08-relo-hybrid-detector-design.md).

    Arayüzü SmallTargetDetector ile aynıdır (``reset()``, ``update(frame) ->
    MaskResult``), yani RealCameraSignalProvider'a ``detector_factory`` olarak
    verilebilir. İki durum:

    - **ARAMA:** çıktı klasik dedektörünkü. Klasik dedektör ``init_streak``
      ardışık karede, her biri bir öncekine ``persist_radius`` px içinde tespit
      verirse RELO o merkezde ``init_box_px`` kenarlı bir kutuyla başlatılır.
    - **TAKİP:** çıktı merkezi RELO kutusunun merkezi. Güven ``lost_patience``
      kare üst üste ``lost_threshold`` altında kalırsa ARAMA'ya dönülür (o karenin
      çıktısı klasik dedektörünkü olur).

    Klasik dedektör her karede çalışır (zamansal arka plan modeli kesintisiz
    güncellensin). Kırpım her zaman klasik dedektörün blackhat tepkisinden, çıktı
    merkezi etrafında alınır: encoder'ın gördüğü sinyal türü değişmez.
    Görünürlük (ödül) ile hiçbir ilgisi yoktur -- o yalnızca ground truth'tan gelir.
    """

    def __init__(
        self,
        crop_size: int = 64,
        classic=None,
        tracker=None,
        tracker_factory=None,
        init_streak: int = 3,
        init_box_px: float = 20.0,
        lost_threshold: float = 0.2,
        lost_patience: int = 5,
        persist_radius: float = 20.0,
        response_scale: float = 100.0,
        **classic_kwargs,
    ):
        self.crop_size = crop_size
        self.classic = classic if classic is not None else SmallTargetDetector(crop_size=crop_size, **classic_kwargs)
        if tracker is None and tracker_factory is None:
            raise ValueError("tracker veya tracker_factory verilmeli")
        self._tracker = tracker
        self._tracker_factory = tracker_factory  # GPU modeli ilk ihtiyaçta, işçi sürecinde yüklensin
        self.init_streak = init_streak
        self.init_box_px = init_box_px
        self.lost_threshold = lost_threshold
        self.lost_patience = lost_patience
        self.persist_radius = persist_radius
        self.response_scale = response_scale
        self.reset()

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = self._tracker_factory()
        return self._tracker

    def reset(self) -> None:
        self.classic.reset()
        self.tracking = False
        self._streak = 0
        self._last_classic: Optional[Tuple[float, float]] = None
        self._low_count = 0

    def update(self, frame: np.ndarray) -> MaskResult:
        classic = self.classic.update(frame)

        if self.tracking:
            box, confidence = self.tracker.track(frame)
            self._low_count = self._low_count + 1 if confidence < self.lost_threshold else 0
            if self._low_count < self.lost_patience:
                x, y, w, h = box
                center = (x + 0.5 * w, y + 0.5 * h)
                return MaskResult(
                    mask_crop=self._crop(center), centroid=center, visible=True,
                    relo_score=float(confidence), relo_box_wh=(float(w), float(h)), relo_tracking=True,
                )
            self.tracking = False
            self._streak = 0
            self._last_classic = None

        self._update_streak(classic.centroid)
        if self._streak >= self.init_streak:
            cx, cy = classic.centroid
            half = 0.5 * self.init_box_px
            self.tracker.init(frame, (cx - half, cy - half, self.init_box_px, self.init_box_px))
            self.tracking = True
            self._low_count = 0
            return MaskResult(
                mask_crop=classic.mask_crop, centroid=classic.centroid, visible=classic.visible,
                relo_tracking=True,
            )
        return classic

    def _update_streak(self, centroid: Optional[Tuple[float, float]]) -> None:
        if centroid is None:
            self._streak = 0
        elif self._last_classic is not None and np.hypot(
            centroid[0] - self._last_classic[0], centroid[1] - self._last_classic[1]
        ) <= self.persist_radius:
            self._streak += 1
        else:
            self._streak = 1
        self._last_classic = centroid

    def _crop(self, center: Tuple[float, float]) -> np.ndarray:
        response = self.classic.last_response
        if response is None:
            return np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        scale = self.classic.last_scale
        crop = _crop_window(response, center[0] * scale, center[1] * scale, self.crop_size)
        return np.clip(crop / self.response_scale, 0.0, 1.0).astype(np.float32)
