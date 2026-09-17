from dronetrackingrl.evaluation.metrics import (
    frame_in_view_rate,
    pixel_trajectory_error,
    switch_count,
)


def test_frame_in_view_rate_computes_fraction_visible():
    assert frame_in_view_rate([True, True, False, True]) == 0.75


def test_frame_in_view_rate_empty_sequence_is_zero():
    assert frame_in_view_rate([]) == 0.0


def test_pixel_trajectory_error_averages_euclidean_distance():
    predicted = [(0.0, 0.0), (3.0, 4.0)]
    ground_truth = [(0.0, 0.0), (0.0, 0.0)]
    assert pixel_trajectory_error(predicted, ground_truth) == 2.5


def test_pixel_trajectory_error_skips_missing_predictions():
    predicted = [None, (3.0, 4.0)]
    ground_truth = [(0.0, 0.0), (0.0, 0.0)]
    assert pixel_trajectory_error(predicted, ground_truth) == 5.0


def test_switch_count_counts_camera_changes():
    assert switch_count([0, 0, 1, 1, 2, 0]) == 3
