from pathlib import Path

import cv2
import numpy as np

from dronetrackingrl.masking.small_target_detector import SmallTargetDetector
from dronetrackingrl.real_data.detections import parse_detections_file

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dataset3_sample"


def _sky(height: int, width: int, brightness: float) -> np.ndarray:
    gradient = np.linspace(150, 190, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    return np.clip(gradient + brightness, 0, 255).astype(np.uint8)


def _draw_dot(frame: np.ndarray, cx: int, cy: int, half: int, value: int) -> None:
    frame[cy - half : cy + half + 1, cx - half : cx + half + 1] = value


def test_finds_moving_dark_dot_after_warmup():
    detector = SmallTargetDetector(crop_size=32)
    results = []
    for t in range(14):
        frame = _sky(200, 300, brightness=0.5 * t)
        _draw_dot(frame, 50 + 4 * t, 100, 2, 60)
        results.append(detector.update(frame))

    assert not any(r.visible for r in results[:5])  # warm-up: no detections
    for t in range(9, 14):
        assert results[t].visible
        x, y = results[t].centroid
        assert abs(x - (50 + 4 * t)) < 3 and abs(y - 100) < 3
        assert results[t].mask_crop.shape == (32, 32)
        assert results[t].mask_crop.max() > 0.5


def test_ignores_slow_brightness_drift_and_static_dark_specks():
    detector = SmallTargetDetector(crop_size=32)
    for t in range(20):
        frame = _sky(200, 300, brightness=1.0 * t)  # slow global drift, like clouds
        for cx, cy in ((60, 60), (200, 150), (250, 40)):
            _draw_dot(frame, cx, cy, 2, 60)  # static small dark features (trees, poles)
        result = detector.update(frame)
        assert not result.visible
        assert not result.mask_crop.any()


def test_centroid_is_reported_in_native_resolution_for_wide_frames():
    detector = SmallTargetDetector(crop_size=32, work_width=960)
    result = None
    for t in range(14):
        frame = _sky(1080, 1920, brightness=0.5 * t)
        _draw_dot(frame, 400 + 8 * t, 500, 5, 60)
        result = detector.update(frame)

    assert result.visible
    x, y = result.centroid
    assert abs(x - (400 + 8 * 13)) < 6 and abs(y - 500) < 6  # native pixels, not 960-wide pixels


def test_reset_restarts_warmup():
    detector = SmallTargetDetector(crop_size=32)
    for t in range(14):
        frame = _sky(200, 300, brightness=0.5 * t)
        _draw_dot(frame, 50 + 4 * t, 100, 2, 60)
        detector.update(frame)

    detector.reset()
    frame = _sky(200, 300, brightness=0.0)
    _draw_dot(frame, 50, 100, 2, 60)
    assert not detector.update(frame).visible


def test_tracks_real_drone_within_a_few_pixels_on_consecutive_real_frames():
    frames_dir = FIXTURE / "frames_motion" / "cam0"
    truth = parse_detections_file(str(FIXTURE / "detections_motion" / "cam0.txt"))
    detector = SmallTargetDetector()

    detected = {}
    for frame_id in range(22000, 22013):
        gray = cv2.imread(str(frames_dir / f"{frame_id:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        detected[frame_id] = detector.update(gray).centroid

    assert all(detected[f] is None for f in range(22000, 22005))  # warm-up
    for frame_id in range(22008, 22013):
        assert detected[frame_id] is not None
        gt_x, gt_y = truth[frame_id]
        assert np.hypot(detected[frame_id][0] - gt_x, detected[frame_id][1] - gt_y) < 8.0
