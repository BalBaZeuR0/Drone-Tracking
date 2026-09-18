from dronetrackingrl.real_data.sync import is_recording, mapped_frame


def test_mapped_frame_at_reference_start_matches_real_beta():
    assert mapped_frame(0, 0, 3) == 251.16
    assert mapped_frame(0, 0, 1) == 1013.95


def test_cam3_is_recording_near_start_of_cam0_timeline():
    assert is_recording(0, camera=3) is True


def test_cam3_stops_recording_near_end_of_cam0_timeline():
    # El ile doğrulandı: mapped_frame(33875, 0, 3) ~= 14380.4, bu da cam3'ün
    # detections/cam3.txt'teki max frame_id'sini (14196) aşıyor.
    assert is_recording(33875, camera=3) is False
    assert is_recording(33000, camera=3) is True


def test_cam1_covers_the_entire_cam0_timeline():
    # El ile doğrulandı: mapped_frame(33875, 0, 1) ~= 17968.4, cam1'in max
    # frame_id'sinin (19960) hâlâ içinde.
    assert is_recording(33875, camera=1) is True


def test_reference_camera_is_recording_within_its_own_range():
    assert is_recording(0, camera=0) is True
    assert is_recording(33875, camera=0) is True
    assert is_recording(33876, camera=0) is False
