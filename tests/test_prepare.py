import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronetrackingrl.real_data.prepare import discover_cameras, extract_dataset, reassemble

NUM_FRAMES = 12


def _make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    rng = np.random.default_rng(0)
    for _ in range(NUM_FRAMES):
        writer.write(rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8))
    writer.release()


def _make_split_dataset(root: Path, with_spanning_marker: bool) -> Path:
    """cam0 gibi bir kamera klasörü: videoyu zip'leyip bayt düzeyinde 3 parçaya böler
    (.z01, .z02 ve merkezi dizini içeren son .zip) -- gerçek dataset'teki biçim."""
    dataset = root / "dataset"
    (dataset / "cam0").mkdir(parents=True)
    (dataset / "detections").mkdir()
    video = root / "cam0.avi"
    _make_video(video)

    archive_path = root / "whole.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(video, "cam0.avi")
    blob = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        start_dir = archive.start_dir
    cut1, cut2 = start_dir // 3, 2 * start_dir // 3

    first = blob[:cut1]
    if with_spanning_marker:
        first = b"PK\x07\x08" + first
    (dataset / "cam0" / "cam0.z01").write_bytes(first)
    (dataset / "cam0" / "cam0.z02").write_bytes(blob[cut1:cut2])
    (dataset / "cam0" / "cam0.zip").write_bytes(blob[cut2:])

    lines = [" frame no.            x            y"] + [f"{i}.000000 1.0 2.0" for i in range(1, NUM_FRAMES + 1)]
    (dataset / "detections" / "cam0.txt").write_text("\n".join(lines) + "\n")
    return dataset


@pytest.mark.parametrize("with_spanning_marker", [False, True])
def test_reassemble_split_archive_verifies_crc(tmp_path, with_spanning_marker):
    dataset = _make_split_dataset(tmp_path, with_spanning_marker)
    video = reassemble(dataset / "cam0", "cam0", out_dir=tmp_path / "work")

    assert video.read_bytes() == (tmp_path / "cam0.avi").read_bytes()


def test_reassemble_rejects_corrupted_part(tmp_path):
    dataset = _make_split_dataset(tmp_path, with_spanning_marker=False)
    middle = dataset / "cam0" / "cam0.z02"
    data = bytearray(middle.read_bytes())
    data[len(data) // 2] ^= 0xFF
    middle.write_bytes(bytes(data))

    with pytest.raises((ValueError, zlib.error)):
        reassemble(dataset / "cam0", "cam0", out_dir=tmp_path / "work")


def test_extract_dataset_writes_numbered_frames_and_resumes(tmp_path):
    dataset = _make_split_dataset(tmp_path, with_spanning_marker=True)
    frames = tmp_path / "frames"

    assert discover_cameras(dataset) == ["cam0"]
    first = extract_dataset(dataset, frames, workers=1, work_dir=tmp_path / "work")
    assert first[0]["extracted"] == NUM_FRAMES and first[0]["match"] and not first[0]["skipped"]
    assert sorted(p.name for p in (frames / "cam0").glob("*.jpg")) == [f"{i:06d}.jpg" for i in range(1, NUM_FRAMES + 1)]
    assert not list((tmp_path / "work").glob("*.avi"))  # geçici video silinmiş olmalı

    again = extract_dataset(dataset, frames, workers=1, work_dir=tmp_path / "work")
    assert again[0]["skipped"] and again[0]["match"]
