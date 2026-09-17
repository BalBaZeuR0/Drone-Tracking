import numpy as np
from dronetrackingrl.geometry.projection import Camera, project_point


def _make_camera():
    k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    r = np.eye(3)
    t = np.zeros(3)
    return Camera(k=k, r=r, t=t, width=100, height=100)


def test_point_on_axis_projects_to_principal_point():
    camera = _make_camera()
    result = project_point(camera, np.array([0.0, 0.0, 10.0]))
    assert result.in_frame is True
    assert result.pixel == (50.0, 50.0)


def test_point_outside_image_bounds_is_not_in_frame():
    camera = _make_camera()
    result = project_point(camera, np.array([5.0, 0.0, 10.0]))
    assert result.pixel == (100.0, 50.0)
    assert result.in_frame is False


def test_point_behind_camera_has_no_pixel():
    camera = _make_camera()
    result = project_point(camera, np.array([0.0, 0.0, -5.0]))
    assert result.pixel is None
    assert result.in_frame is False
