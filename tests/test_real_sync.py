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


def test_sync_tables_are_internally_consistent_inverses():
    # frame_j = a_ij * frame_i + b_ij  =>  a_ji ~ 1/a_ij and b_ji ~ -b_ij / a_ij.
    # Catches transcription slips in the hand-copied README tables.
    from dronetrackingrl.real_data.sync import SYNC_TABLES

    for name, table in SYNC_TABLES.items():
        n = table.num_cameras
        assert len(table.alpha) == n and all(len(row) == n for row in table.alpha), name
        assert len(table.beta) == n and all(len(row) == n for row in table.beta), name
        for i in range(n):
            assert table.alpha[i][i] == 1.0 and table.beta[i][i] == 0.0, name
            for j in range(n):
                assert abs(table.alpha[i][j] * table.alpha[j][i] - 1.0) < 0.01, (name, i, j)
                assert abs(table.beta[j][i] + table.beta[i][j] / table.alpha[i][j]) < 12.0, (name, i, j)


def test_dataset4_uses_its_own_table():
    from dronetrackingrl.real_data.sync import DATASET4

    assert mapped_frame(0, 0, 6, DATASET4) == -1562.26
    assert is_recording(0, 6, table=DATASET4) is False  # cam6 has not started yet
    assert is_recording(4000, 6, table=DATASET4) is True
    assert is_recording(31075, 0, table=DATASET4) is True
    assert is_recording(31076, 0, table=DATASET4) is False
