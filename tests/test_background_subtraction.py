import numpy as np
from dronetrackingrl.masking.background_subtraction import BackgroundSubtractor


def test_first_frame_has_no_blob_and_sets_background():
    subtractor = BackgroundSubtractor(crop_size=16)
    background_frame = np.full((40, 40), 50, dtype=np.uint8)

    result = subtractor.update(background_frame)

    assert result.visible is False
    assert result.centroid is None


def test_second_frame_detects_bright_blob():
    subtractor = BackgroundSubtractor(crop_size=16, diff_threshold=25, min_blob_area=5)
    background_frame = np.full((40, 40), 50, dtype=np.uint8)
    subtractor.update(background_frame)

    blob_frame = background_frame.copy()
    blob_frame[10:14, 20:24] = 255  # 4x4 bright blob centered at (21.5, 11.5)

    result = subtractor.update(blob_frame)

    assert result.visible is True
    assert result.centroid is not None
    cx, cy = result.centroid
    assert abs(cx - 21.5) < 1.0
    assert abs(cy - 11.5) < 1.0
    assert result.mask_crop.shape == (16, 16)
    assert result.mask_crop.max() > 0


def test_reset_allows_reinitializing_background():
    subtractor = BackgroundSubtractor(crop_size=16)
    first_frame = np.full((40, 40), 50, dtype=np.uint8)
    subtractor.update(first_frame)

    subtractor.reset()
    result = subtractor.update(first_frame)

    assert result.visible is False
    assert result.centroid is None
