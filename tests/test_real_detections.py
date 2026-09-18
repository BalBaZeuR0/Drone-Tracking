from dronetrackingrl.real_data.detections import parse_detections_file

CAM0_FIXTURE = "tests/fixtures/dataset3_sample/detections/cam0.txt"
CAM3_FIXTURE = "tests/fixtures/dataset3_sample/detections/cam3.txt"


def test_parses_visible_frames_from_real_cam0_fixture():
    signals = parse_detections_file(CAM0_FIXTURE)
    assert len(signals) == 22
    assert signals[1] == (742.82211823, 897.10093596)
    assert signals[19] == (743.07363879, 897.60862411)
    assert signals[50] == (743.47244720, 897.75087019)


def test_zero_zero_maps_to_none_on_real_cam0_fixture():
    signals = parse_detections_file(CAM0_FIXTURE)
    assert signals[13442] is None


def test_zero_zero_maps_to_none_on_real_cam3_fixture():
    signals = parse_detections_file(CAM3_FIXTURE)
    assert signals[1] is None
    assert signals[50] is None
    assert signals[100] is None
    assert signals[14194] is None


def test_real_visible_detection_preserved_on_real_cam3_fixture():
    signals = parse_detections_file(CAM3_FIXTURE)
    assert signals[14192] == (350.64351480, 461.23966033)
    assert signals[14193] == (343.53522167, 460.19926108)


def test_unknown_frame_id_is_absent_not_none():
    signals = parse_detections_file(CAM0_FIXTURE)
    assert 999999 not in signals
