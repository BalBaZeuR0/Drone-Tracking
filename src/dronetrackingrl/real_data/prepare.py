"""Ham dataset'ten (çok parçalı zip'ler) kare (JPEG) çıkarma.

`drone-tracking-datasets` içindeki videolar ``camN.z01 .. camN.zip`` şeklinde
bölünmüş. Standart araçlar (unzip, zip -FF, 7z) bunları açamıyor; parçalar PKZIP
"spanned" değil, tek bir zip'in bayt düzeyinde bölünmüş halidir. Yöntem: ilk
parçadaki yerel dosya başlığından DEFLATE akışının başlangıcını bulup, ham
sıkıştırılmış baytları parçalar boyunca (.z01 -> .z02 -> ... -> .zip; .zip HER
ZAMAN son parça ve merkezi dizini içerir) birleştirip doğrudan zlib ile açmak.
Sonuç, merkezi dizindeki CRC32 ve boyutla doğrulanır.

Kullanım:
    python -m dronetrackingrl.real_data.prepare extract \\
        --dataset-dir <.../drone-tracking-datasets/dataset3> --frames-dir <çıktı/frames> --workers 3

Çıktı: ``<frames-dir>/camN/000001.jpg`` (1'den başlayan, 6 haneli) -- her dataset'in
``detections/camN.txt`` frame_id'leriyle birebir aynı numaralandırma. Tamamlanmış
kameralar (kare sayısı = detections'taki son frame_id) atlanır; yarıda kesilirse
aynı komut kaldığı yerden devam eder.
"""

import argparse
import re
import struct
import sys
import time
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

LOCAL_HEADER_SIG = b"PK\x03\x04"
SPANNING_MARKER = b"PK\x07\x08"


def find_parts(cam_dir: Path, cam_name: str) -> List[Path]:
    numbered = sorted(
        cam_dir.glob(f"{cam_name}.z[0-9][0-9]"),
        key=lambda p: int(re.search(r"\.z(\d+)$", p.name).group(1)),
    )
    final = cam_dir / f"{cam_name}.zip"
    if not final.exists():
        raise FileNotFoundError(f"son parça yok: {final}")
    return numbered + [final]


def _data_offset(first_part: Path) -> int:
    with open(first_part, "rb") as handle:
        head = handle.read(4)
        start = 4 if head == SPANNING_MARKER else 0  # ilk parça 4 baytlık işaretle başlayabilir
        handle.seek(start)
        header = handle.read(30)
        if header[0:4] != LOCAL_HEADER_SIG:
            raise ValueError(f"{first_part} yerel dosya başlığıyla başlamıyor")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        return start + 30 + name_len + extra_len


def reassemble(cam_dir: Path, cam_name: str, out_dir: Optional[Path] = None) -> Path:
    """Parçaları tek bir videoya birleştirir; CRC32/boyut uyuşmazsa hata verir."""
    parts = find_parts(cam_dir, cam_name)
    data_offset = _data_offset(parts[0])
    with zipfile.ZipFile(parts[-1]) as archive:
        infos = archive.infolist()
        if len(infos) != 1:
            raise ValueError(f"tek girdi bekleniyordu, {len(infos)} bulundu")
        entry_name, expected_crc, expected_size = infos[0].filename, infos[0].CRC, infos[0].file_size

    out_dir = Path(out_dir) if out_dir else cam_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(entry_name).name or f"{cam_name}.mp4")
    if not out_path.suffix:
        out_path = out_dir / f"{cam_name}.mp4"

    decompressor = zlib.decompressobj(-15)
    crc = total = 0
    with open(out_path, "wb") as out:
        for index, part in enumerate(parts):
            with open(part, "rb") as handle:
                if index == 0:
                    handle.seek(data_offset)
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    data = decompressor.decompress(chunk)
                    if data:
                        out.write(data)
                        crc = zlib.crc32(data, crc)
                        total += len(data)
        tail = decompressor.flush()
        if tail:
            out.write(tail)
            crc = zlib.crc32(tail, crc)
            total += len(tail)

    if crc != expected_crc or total != expected_size:
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"{cam_name}: CRC/boyut uyuşmuyor (bulunan crc={crc:#010x} boyut={total}, "
            f"beklenen crc={expected_crc:#010x} boyut={expected_size})"
        )
    return out_path


def expected_frame_count(dataset_dir: Path, cam_name: str) -> int:
    """detections/camN.txt'teki son frame_id (= videodaki toplam kare sayısı)."""
    last = None
    with open(Path(dataset_dir) / "detections" / f"{cam_name}.txt") as handle:
        for line in handle:
            if line.strip():
                last = line
    return int(float(last.split()[0]))


def extract_camera(dataset_dir, cam_name: str, frames_dir, work_dir=None, jpeg_quality: int = 90) -> dict:
    dataset_dir, frames_dir = Path(dataset_dir), Path(frames_dir)
    expected = expected_frame_count(dataset_dir, cam_name)
    out_dir = frames_dir / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in out_dir.glob("*.jpg"))
    if existing == expected:
        return {"cam": cam_name, "extracted": existing, "expected": expected, "skipped": True, "match": True}

    started = time.time()
    video = reassemble(dataset_dir / cam_name, cam_name, out_dir=work_dir)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cv2 videoyu açamadı: {video}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        cv2.imwrite(str(out_dir / f"{index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    capture.release()
    video.unlink()
    return {
        "cam": cam_name, "extracted": index, "expected": expected, "skipped": False,
        "match": index == expected, "seconds": round(time.time() - started),
    }


def _extract_job(args) -> dict:
    return extract_camera(*args)


def discover_cameras(dataset_dir: Path) -> List[str]:
    return sorted(
        p.name for p in Path(dataset_dir).iterdir()
        if p.is_dir() and re.fullmatch(r"cam\d+", p.name) and (p / f"{p.name}.zip").exists()
    )


def extract_dataset(dataset_dir, frames_dir, cameras: Optional[Sequence[str]] = None, workers: int = 1,
                    work_dir=None) -> List[dict]:
    dataset_dir = Path(dataset_dir)
    cameras = list(cameras) if cameras else discover_cameras(dataset_dir)
    jobs = [(str(dataset_dir), cam, str(frames_dir), None if work_dir is None else str(work_dir)) for cam in cameras]
    if workers <= 1:
        results = [_extract_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_extract_job, jobs))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dronetrackingrl.real_data.prepare")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Bir dataset'in videolarını kareye böl (JPEG)")
    extract.add_argument("--dataset-dir", required=True, help="ör. .../drone-tracking-datasets/dataset3")
    extract.add_argument("--frames-dir", required=True, help="karelerin yazılacağı klasör (camN/ alt klasörleri)")
    extract.add_argument("--workers", type=int, default=1, help="paralel kamera sayısı (süreç başına ~0.5GB RAM)")
    extract.add_argument("--cameras", nargs="+", help="sadece bu kameralar (ör. cam0 cam3); varsayılan: hepsi")
    extract.add_argument("--work-dir", help="geçici video (birkaç GB) için klasör; varsayılan: kamera klasörü")
    args = parser.parse_args(argv)

    results = extract_dataset(args.dataset_dir, args.frames_dir, args.cameras, args.workers, args.work_dir)
    print("\n=== ÖZET ===")
    for r in results:
        status = "ATLANDI (zaten tam)" if r["skipped"] else ("OK" if r["match"] else "UYUŞMAZLIK")
        print(f"{r['cam']}: {r['extracted']} / {r['expected']} kare [{status}]")
    return 0 if all(r["match"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
