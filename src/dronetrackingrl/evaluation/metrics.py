from typing import Optional, Sequence, Tuple


def frame_in_view_rate(visible_flags: Sequence[bool]) -> float:
    if not visible_flags:
        return 0.0
    return sum(1 for flag in visible_flags if flag) / len(visible_flags)


def pixel_trajectory_error(
    predicted: Sequence[Optional[Tuple[float, float]]],
    ground_truth: Sequence[Tuple[float, float]],
) -> float:
    errors = []
    for pred, truth in zip(predicted, ground_truth):
        if pred is None:
            continue
        dx = pred[0] - truth[0]
        dy = pred[1] - truth[1]
        errors.append((dx ** 2 + dy ** 2) ** 0.5)
    if not errors:
        return float("inf")
    return sum(errors) / len(errors)


def switch_count(camera_indices: Sequence[int]) -> int:
    return sum(1 for prev, curr in zip(camera_indices, camera_indices[1:]) if prev != curr)
