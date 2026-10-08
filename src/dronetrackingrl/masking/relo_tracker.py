import copy
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class ReloTracker:
    """RELO (Chen vd., ICML 2026) takipçisine ince adaptör.

    RELO deposu bu repoya kopyalanmaz (lisans dosyası yok): ``relo_root``
    klonlanmış depodur, ağırlıklar ``tools/download_checkpoints.py`` ile onun
    altına indirilir. İçe aktarma tembeldir, yani RELO kurulu olmayan
    makinelerde bu modülü içe aktarmak bir şey bozmaz.

    ``confidence``: politika logit haritasının (16x16) softmax maksimumu, [0,1] --
    RELO'nun kendi ``best_score``'u Hann penceresi uygulanmış ham logit (sınırsız),
    gözlem özelliği olarak kullanışsız. Cihazı RELO kendisi seçer (CUDA varsa GPU).
    Kutu, kenarı ``min_box_px``'ten küçük olmayacak şekilde tutulur: birkaç
    piksellik drone'da kutu çökerse arama bölgesi (4 x kutu) anlamsızlaşır.
    """

    def __init__(self, relo_root, variant: str = "t256", min_box_px: float = 8.0):
        root = Path(relo_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import torch  # noqa: F401  (RELO'dan önce yüklensin)
        from lib.config.relo.config import cfg, update_config_from_file
        from lib.test.tracker.relo import RELO
        from lib.test.utils import TrackerParams

        update_config_from_file(str(root / "experiments" / "relo" / f"relo_{variant}.yaml"))
        config = copy.deepcopy(cfg)
        config.MODEL.ENCODER.PRETRAIN_TYPE = str(root / config.MODEL.ENCODER.PRETRAIN_TYPE)

        params = TrackerParams()
        params.cfg = config
        params.yaml_name = f"relo_{variant}"
        params.template_factor = config.TEST.TEMPLATE_FACTOR
        params.template_size = config.TEST.TEMPLATE_SIZE
        params.search_factor = config.TEST.SEARCH_FACTOR
        params.search_size = config.TEST.SEARCH_SIZE
        params.checkpoint = str(
            root / "checkpoints" / "train" / "relo" / f"relo_{variant}" / f"RELO_ep{config.TEST.EPOCH:04d}.pth.tar"
        )
        params.save_all_boxes = False

        class _CapturingRELO(RELO):
            def _select_location_map(self, out_dict):
                self.last_policy_logits = out_dict["policy_logits"]
                return super()._select_location_map(out_dict)

        self._tracker = _CapturingRELO(params, "DEFAULT")
        self.min_box_px = min_box_px

    def init(self, frame: np.ndarray, box_xywh: Tuple[float, float, float, float]) -> None:
        self._tracker.initialize(_to_rgb(frame), {"init_bbox": [float(v) for v in self._clamp(box_xywh)]})

    def track(self, frame: np.ndarray) -> Tuple[Tuple[float, float, float, float], float]:
        import torch

        out = self._tracker.track(_to_rgb(frame))
        box = self._clamp(out["target_bbox"])
        self._tracker.state = list(box)  # bir sonraki arama bölgesi kırpılmış kutudan
        logits = self._tracker.last_policy_logits.detach().flatten()
        confidence = float(torch.softmax(logits.float(), dim=0).max().item())
        return box, confidence

    def _clamp(self, box_xywh):
        x, y, w, h = (float(v) for v in box_xywh)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        w, h = max(w, self.min_box_px), max(h, self.min_box_px)
        return (cx - 0.5 * w, cy - 0.5 * h, w, h)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
