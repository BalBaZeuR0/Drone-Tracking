import os

import numpy as np
import pytest

RELO_ROOT = os.environ.get("RELO_ROOT")


@pytest.mark.skipif(not RELO_ROOT, reason="RELO kurulu değil (RELO_ROOT ayarlı değil)")
def test_real_relo_follows_a_moving_dark_square():
    from dronetrackingrl.masking.relo_tracker import ReloTracker

    tracker = ReloTracker(RELO_ROOT, variant="t256")

    def frame(cx):
        image = np.full((240, 320, 3), 200, dtype=np.uint8)
        image[110:130, cx - 10 : cx + 10] = 30
        return image

    tracker.init(frame(100), (90.0, 110.0, 20.0, 20.0))
    for cx in range(104, 140, 4):
        (x, y, w, h), confidence = tracker.track(frame(cx))
    assert abs((x + w / 2) - 136) < 8 and abs((y + h / 2) - 120) < 8
    assert 0.0 <= confidence <= 1.0


def test_clamp_keeps_center_and_enforces_minimum_box():
    from dronetrackingrl.masking.relo_tracker import ReloTracker

    tracker = ReloTracker.__new__(ReloTracker)  # model yüklemeden yalnızca yardımcıyı test et
    tracker.min_box_px = 8.0
    assert tracker._clamp((10.0, 20.0, 2.0, 4.0)) == (7.0, 18.0, 8.0, 8.0)
    assert tracker._clamp((0.0, 0.0, 30.0, 10.0)) == (0.0, 0.0, 30.0, 10.0)
