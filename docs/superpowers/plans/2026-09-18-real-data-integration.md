# Gerçek Veri Entegrasyonu (Faz 1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faz 1'in sentetik-veriyle doğrulanmış çekirdek bileşenlerini (maskeleme, encoder, `CameraSwitchEnv`) gerçek `drone-tracking-datasets/dataset3` verisine bağlamak — gerçek `detections/camN.txt` ground-truth'undan ve gerçek kameralar-arası senkronizasyon parametrelerinden beslenen bir `RealCameraSignalProvider` yazmak, ve gerçek verilerin küçük bir diliminde uçtan uca PPO eğitiminin çalıştığını kanıtlamak.

**Architecture:** Faz 1'in `CameraSignalProvider` protokolü (değişmedi) yeni bir implementasyonla dolduruluyor: `RealCameraSignalProvider` gerçek `.jpg` karelerini diskten okur (mevcut `BackgroundSubtractor` + paylaşılan `MaskedCropAutoencoder` ile RL gözlemi/latent üretir — bu ikisi hiç değişmiyor), ve ödül için kullanılan `görünür_mü` bilgisini **sadece** `detections/camN.txt`'teki gerçek elle-etiketlenmiş konumlardan (`(0,0)` = görünmüyor) + gerçek alpha/beta senkronizasyon parametreleriyle hesaplanan "kamera o an kayıtta mı" durumundan üretir. Maskeleme modülü asla ödüle karışmıyor (spec'teki "iki ayrı kullanım" ayrımı korunuyor).

**Tech Stack:** Faz 1 ile aynı (Python, numpy, opencv-python, PyTorch, Gymnasium, Stable-Baselines3 PPO, pytest). Yeni bağımlılık yok.

**Spec:** `docs/superpowers/specs/2026-09-16-camera-switching-rl-design.md` (§ Gerçek Veri Entegrasyonu (Faz 1b))

## Global Constraints

- `görünür_mü` (reward kaynağı) **sadece** `detections/camN.txt`'ten gelir
  (`(x,y) != (0,0)`) **VE** kamera o an senkronize zaman diliminde kayıtta
  olmalı — maskeleme modülünün kendi `visible`/`centroid` çıktısı asla
  reward'a karışmaz, sadece RL gözlemi (latent) için kullanılır.
- `CameraSwitchEnv`'in reward mantığı (+1/-1, `terminated`/`truncated`
  semantiği) **değişmiyor** — sadece yeni bir `CameraSignalProvider`
  implementasyonu ekleniyor.
- Senkronizasyon parametreleri (`alpha`, `beta`, kamera başına
  `max_frame_id`) dataset3'ün kendi `README.md`'sinden **gerçek, ölçülmüş
  değerler** olarak koda gömülü — hesaplanmıyor/tahmin edilmiyor.
- Referans kamera: **cam0** (en yüksek fps, en uzun kayıt).
- Test fixture'ları (`tests/fixtures/dataset3_sample/`) gerçek dataset3
  dosyalarından birebir kopyalanmış küçük dilimlerdir (fabrikasyon yok) —
  bkz. her task'ın Files bölümü.
- Bu plan kapsamında `cameras.txt`/kalibrasyon `<model>.json` parser'ı
  **yazılmıyor** (spec'te YAGNI olarak işaretlendi — sadece terk edilen
  merkezleme skoru için gerekiyordu).

---

### Task 1: Detections Parser

**Files:**
- Create: `src/dronetrackingrl/real_data/__init__.py`
- Create: `src/dronetrackingrl/real_data/detections.py`
- Create: `tests/fixtures/dataset3_sample/detections/cam0.txt` (gerçek
  `dataset3/detections/cam0.txt`'ten birebir kopyalanmış satırlar: kare
  1-19, 50, 100, 13442 — hepsi gerçek, elle etiketlenmiş değerler)
- Create: `tests/fixtures/dataset3_sample/detections/cam3.txt` (gerçek
  `dataset3/detections/cam3.txt`'ten birebir kopyalanmış satırlar: kare
  1-19, 50, 100, 14192-14196)
- Test: `tests/test_real_detections.py`

**Interfaces:**
- Consumes: yok.
- Produces: `parse_detections_file(path: str) -> Dict[int, Optional[Tuple[float, float]]]`. Task 3 ve Task 4 bunu kullanacak.

- [ ] **Step 1: Fixture dosyalarını oluştur**

`tests/fixtures/dataset3_sample/detections/cam0.txt`:
```
 frame no.            x            y
1.000000 742.82211823 897.10093596
2.000000 742.88421712 897.37083636
3.000000 742.90043057 897.36550299
4.000000 742.91565473 897.36364778
5.000000 742.83458128 897.81536946
6.000000 742.94341950 897.36952144
7.000000 742.95610228 897.37682507
8.000000 742.96808010 897.38675643
9.000000 743.21650246 897.51995074
10.000000 742.99020520 897.41365188
11.000000 743.00049465 897.43019074
12.000000 743.01036347 897.44850687
13.000000 743.31615764 897.32059113
14.000000 743.02912358 897.48962051
15.000000 743.03815704 897.51199280
16.000000 743.04705420 897.53529192
17.000000 742.63344828 897.28862069
18.000000 743.06472399 897.58382018
19.000000 743.07363879 897.60862411
50.000000 743.47244720 897.75087019
100.000000 742.83185607 897.71276601
13442.000000   0.00000000   0.00000000
```

`tests/fixtures/dataset3_sample/detections/cam3.txt`:
```
 frame no.            x            y
1.000000   0.00000000   0.00000000
2.000000   0.00000000   0.00000000
3.000000   0.00000000   0.00000000
4.000000   0.00000000   0.00000000
5.000000   0.00000000   0.00000000
6.000000   0.00000000   0.00000000
7.000000   0.00000000   0.00000000
8.000000   0.00000000   0.00000000
9.000000   0.00000000   0.00000000
10.000000   0.00000000   0.00000000
11.000000   0.00000000   0.00000000
12.000000   0.00000000   0.00000000
13.000000   0.00000000   0.00000000
14.000000   0.00000000   0.00000000
15.000000   0.00000000   0.00000000
16.000000   0.00000000   0.00000000
17.000000   0.00000000   0.00000000
18.000000   0.00000000   0.00000000
19.000000   0.00000000   0.00000000
14192.000000 350.64351480 461.23966033
14193.000000 343.53522167 460.19926108
14194.000000   0.00000000   0.00000000
14195.000000   0.00000000   0.00000000
14196.000000   0.00000000   0.00000000
50.000000   0.00000000   0.00000000
100.000000   0.00000000   0.00000000
```

(Bu iki dosya zaten `tests/fixtures/dataset3_sample/detections/` altında
mevcutsa — 2026-09-18'de gerçek dataset3'ten kopyalandı — bu adım sadece
doğrulama; içerik farklıysa yukarıdaki gerçek değerlerle değiştir.)

- [ ] **Step 2: Boş `__init__.py` oluştur**

`src/dronetrackingrl/real_data/__init__.py` — boş dosya.

- [ ] **Step 3: Başarısız testi yaz**

`tests/test_real_detections.py`:
```python
from dronetrackingrl.real_data.detections import parse_detections_file

CAM0_FIXTURE = "tests/fixtures/dataset3_sample/detections/cam0.txt"
CAM3_FIXTURE = "tests/fixtures/dataset3_sample/detections/cam3.txt"


def test_parses_visible_frames_from_real_cam0_fixture():
    signals = parse_detections_file(CAM0_FIXTURE)
    assert len(signals) == 21
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
```

- [ ] **Step 4: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_real_detections.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 5: Parser'ı implemente et**

`src/dronetrackingrl/real_data/detections.py`:
```python
from typing import Dict, Optional, Tuple


def parse_detections_file(path: str) -> Dict[int, Optional[Tuple[float, float]]]:
    signals: Dict[int, Optional[Tuple[float, float]]] = {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 3:
                continue  # header or blank line
            try:
                frame_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue  # header
            signals[frame_id] = None if (x == 0.0 and y == 0.0) else (x, y)
    return signals
```

- [ ] **Step 6: Testin geçtiğini doğrula**

Run: `pytest tests/test_real_detections.py -v`
Expected: PASS (5/5)

- [ ] **Step 7: Commit**

```bash
git add src/dronetrackingrl/real_data/__init__.py src/dronetrackingrl/real_data/detections.py tests/fixtures/dataset3_sample/detections tests/test_real_detections.py
git commit -m "feat: add real detections.txt parser with (0,0)-as-invisible handling"
```

---

### Task 2: Senkronize Zaman Ekseni (Sync)

**Files:**
- Create: `src/dronetrackingrl/real_data/sync.py`
- Test: `tests/test_real_sync.py`

**Interfaces:**
- Consumes: yok (gerçek dataset3 alpha/beta/max-frame-id değerleri koda gömülü).
- Produces: `mapped_frame(ref_frame: int, from_cam: int, to_cam: int) -> float`, `is_recording(ref_frame: int, camera: int, reference_camera: int = 0) -> bool`. Task 3 bunları kullanacak.

- [ ] **Step 1: Başarısız testi yaz**

`tests/test_real_sync.py`:
```python
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
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_real_sync.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Sync modülünü implemente et**

`src/dronetrackingrl/real_data/sync.py`:
```python
from typing import Tuple

# dataset3 için gerçek, ölçülmüş senkronizasyon parametreleri — kaynak:
# drone-tracking-datasets/dataset3/README.md, "Ground truth synchronization
# parameters". Satır = referans kamera, sütun = hedef kamera:
# frame_j = alpha[ref][target] * frame_i + beta[ref][target].
DATASET3_ALPHA: Tuple[Tuple[float, ...], ...] = (
    (1.0000, 0.5005, 0.4960, 0.4171, 0.5000, 0.8341),
    (1.9982, 1.0000, 0.9910, 0.8333, 0.9990, 1.6667),
    (2.0163, 1.0091, 1.0000, 0.8409, 1.0081, 1.6819),
    (2.3978, 1.2000, 1.1892, 1.0000, 1.1988, 2.0000),
    (2.0001, 1.0010, 0.9919, 0.8342, 1.0000, 1.6683),
    (1.1989, 0.6000, 0.5946, 0.5000, 0.5994, 1.0000),
)
DATASET3_BETA: Tuple[Tuple[float, ...], ...] = (
    (0.00, 1013.95, 546.98, 251.16, 961.02, 137.51),
    (-2026.04, 0.00, -457.83, -593.82, -51.96, -1552.47),
    (-1102.90, 461.99, 0.00, -208.81, 409.59, -782.45),
    (-602.21, 712.57, 248.32, 0.00, 659.93, -364.81),
    (-1922.12, 52.01, -406.29, -551.00, 0.00, -1465.78),
    (-164.85, 931.45, 465.22, 182.40, 878.60, 0.00),
)
# Her kameranın detections/camN.txt'indeki max frame_id'si (dataset3).
DATASET3_MAX_FRAME_ID: Tuple[int, ...] = (33875, 19960, 17166, 14196, 18900, 28080)


def mapped_frame(ref_frame: int, from_cam: int, to_cam: int) -> float:
    return DATASET3_ALPHA[from_cam][to_cam] * ref_frame + DATASET3_BETA[from_cam][to_cam]


def is_recording(ref_frame: int, camera: int, reference_camera: int = 0) -> bool:
    target_frame = mapped_frame(ref_frame, reference_camera, camera)
    return 0 <= target_frame <= DATASET3_MAX_FRAME_ID[camera]
```

- [ ] **Step 4: Testin geçtiğini doğrula**

Run: `pytest tests/test_real_sync.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add src/dronetrackingrl/real_data/sync.py tests/test_real_sync.py
git commit -m "feat: add dataset3 cross-camera sync timeline with real alpha/beta params"
```

---

### Task 3: RealCameraSignalProvider

**Files:**
- Create: `src/dronetrackingrl/real_data/real_signal_provider.py`
- Create: `tests/fixtures/dataset3_sample/frames/cam0/000001.jpg` (gerçek
  dataset3 cam0 kare 1)
- Create: `tests/fixtures/dataset3_sample/frames/cam0/000002.jpg` (gerçek kare 2)
- Create: `tests/fixtures/dataset3_sample/frames/cam0/000003.jpg` (gerçek kare 3)
- Create: `tests/fixtures/dataset3_sample/frames/cam0/000050.jpg` (gerçek kare 50)
- Create: `tests/fixtures/dataset3_sample/frames/cam0/000100.jpg` (gerçek kare 100)
- Create: `tests/fixtures/dataset3_sample/frames/cam0/013442.jpg` (gerçek,
  ground-truth'a göre drone'un görünmediği kare — background referansı testi için)
- Test: `tests/test_real_signal_provider.py`

**Interfaces:**
- Consumes: `parse_detections_file` (Task 1), `mapped_frame`/`is_recording` (Task 2), `BackgroundSubtractor`/`MaskResult` (Faz 1, değişmedi), `MaskedCropAutoencoder` (Faz 1, değişmedi), `CameraStepSignal`/`CameraSignalProvider` protokolü (Faz 1, değişmedi).
- Produces: `RealCameraSignalProvider` sınıfı — `CameraSignalProvider` protokolünü karşılar (`reset() -> List[CameraStepSignal]`, `step() -> Tuple[List[CameraStepSignal], bool]`, `latent_dim`, `num_cameras`), artı `signals_for_ref_frame(ref_frame: int) -> List[CameraStepSignal]` (test edilebilirlik için public). Task 4 bunu kullanacak.

**Not:** Bu task'ın fixture kare dosyaları (`.jpg`) zaten
`tests/fixtures/dataset3_sample/frames/` altında mevcutsa (2026-09-18'de
gerçek dataset3'ten kopyalandı — cam0'ın kendi videosundan çıkarılan gerçek
1920×1080 kareler), bu adım sadece doğrulama; eksikse dataset3'ün tam
kareye bölünmüş halinden (`DatasetDroneTracker/drone-tracking-datasets/dataset3/frames/cam0/`)
aynı 6 dosyayı (`000001.jpg`, `000002.jpg`, `000003.jpg`, `000050.jpg`,
`000100.jpg`, `013442.jpg`) kopyala.

- [ ] **Step 1: Başarısız testi yaz**

`tests/test_real_signal_provider.py`:
```python
from pathlib import Path

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.real_data.detections import parse_detections_file
from dronetrackingrl.real_data.real_signal_provider import RealCameraSignalProvider

FRAMES_ROOT = Path("tests/fixtures/dataset3_sample/frames")
CAM0_DETECTIONS_PATH = "tests/fixtures/dataset3_sample/detections/cam0.txt"
LATENT_DIM = 8


def _make_provider(start_ref_frame: int, end_ref_frame: int) -> RealCameraSignalProvider:
    detections = {0: parse_detections_file(CAM0_DETECTIONS_PATH)}
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    return RealCameraSignalProvider(
        frames_root=FRAMES_ROOT,
        detections=detections,
        cameras=[0],
        encoder=encoder,
        latent_dim=LATENT_DIM,
        start_ref_frame=start_ref_frame,
        end_ref_frame=end_ref_frame,
        reference_camera=0,
    )


def test_reset_on_real_confirmed_empty_frame_reports_not_visible():
    provider = _make_provider(start_ref_frame=13442, end_ref_frame=13443)
    signals = provider.reset()
    assert signals[0].visible is False
    assert signals[0].latent.shape == (LATENT_DIM,)


def test_signals_for_ref_frame_reports_ground_truth_visibility_on_real_visible_frame():
    provider = _make_provider(start_ref_frame=13442, end_ref_frame=13443)
    provider.reset()
    signals = provider.signals_for_ref_frame(50)
    assert signals[0].visible is True
    assert signals[0].latent.shape == (LATENT_DIM,)


def test_step_advances_through_consecutive_real_frames_and_signals_done():
    provider = _make_provider(start_ref_frame=1, end_ref_frame=3)
    reset_signals = provider.reset()
    assert reset_signals[0].visible is True  # gerçek kare 1, ground truth'a göre görünür

    signals_2, done_2 = provider.step()
    assert signals_2[0].visible is True  # gerçek kare 2, görünür
    assert done_2 is False

    signals_3, done_3 = provider.step()
    assert signals_3[0].visible is True  # gerçek kare 3, görünür
    assert done_3 is True
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_real_signal_provider.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Provider'ı implemente et**

`src/dronetrackingrl/real_data/real_signal_provider.py`:
```python
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.masking.background_subtraction import BackgroundSubtractor
from dronetrackingrl.real_data.sync import is_recording, mapped_frame
from dronetrackingrl.rl_env.signal_provider import CameraStepSignal


def default_frame_reader(frames_root: Path, camera: int, frame_id: int) -> Optional[np.ndarray]:
    path = Path(frames_root) / f"cam{camera}" / f"{frame_id:06d}.jpg"
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


class RealCameraSignalProvider:
    """Gerçek çıkarılmış karelerden ve detections/camN.txt ground-truth'undan
    beslenen CameraSignalProvider implementasyonu.

    `visible` (reward için) SADECE ground-truth detections'tan ve senkronize
    kayıt durumundan gelir -- maskeleme modülü asla reward'a karışmaz, sadece
    latent gözlemi üretir (spec'in "iki ayrı kullanım" ayrımı).

    Ön koşul: `start_ref_frame`, çağıranın detections[reference_camera]'da
    drone'un görünmediğini (None) doğruladığı bir referans-kamera karesi
    olmalı -- BackgroundSubtractor'ın ilk update() çağrısı drone-free bir
    kare varsayıyor.
    """

    def __init__(
        self,
        frames_root: Path,
        detections: Dict[int, Dict[int, Optional[Tuple[float, float]]]],
        cameras: List[int],
        encoder: MaskedCropAutoencoder,
        latent_dim: int,
        start_ref_frame: int,
        end_ref_frame: int,
        reference_camera: int = 0,
        crop_size: int = 64,
        frame_reader: Callable[[Path, int, int], Optional[np.ndarray]] = default_frame_reader,
    ):
        self.frames_root = frames_root
        self.detections = detections
        self.cameras = cameras
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.num_cameras = len(cameras)
        self.start_ref_frame = start_ref_frame
        self.end_ref_frame = end_ref_frame
        self.reference_camera = reference_camera
        self.crop_size = crop_size
        self.frame_reader = frame_reader
        self._subtractors = {camera: BackgroundSubtractor(crop_size=crop_size) for camera in cameras}
        self._ref_frame = start_ref_frame

    def reset(self) -> List[CameraStepSignal]:
        for subtractor in self._subtractors.values():
            subtractor.reset()
        self._ref_frame = self.start_ref_frame
        return self.signals_for_ref_frame(self._ref_frame)

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        self._ref_frame += 1
        signals = self.signals_for_ref_frame(self._ref_frame)
        done = self._ref_frame >= self.end_ref_frame
        return signals, done

    def signals_for_ref_frame(self, ref_frame: int) -> List[CameraStepSignal]:
        signals = []
        for camera in self.cameras:
            camera_frame_id = round(mapped_frame(ref_frame, self.reference_camera, camera))
            recording = is_recording(ref_frame, camera, self.reference_camera)
            subtractor = self._subtractors[camera]

            if recording:
                frame = self.frame_reader(self.frames_root, camera, camera_frame_id)
                mask_result = subtractor.update(frame)
                ground_truth = self.detections.get(camera, {}).get(camera_frame_id)
                visible = ground_truth is not None
                pixel = mask_result.centroid
                crop = mask_result.mask_crop
            else:
                visible = False
                pixel = None
                crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)

            crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                latent = self.encoder.encode(crop_tensor).squeeze(0).numpy()

            signals.append(CameraStepSignal(latent=latent, visible=visible, pixel=pixel))
        return signals
```

- [ ] **Step 4: Testin geçtiğini doğrula**

Run: `pytest tests/test_real_signal_provider.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add src/dronetrackingrl/real_data/real_signal_provider.py tests/fixtures/dataset3_sample/frames tests/test_real_signal_provider.py
git commit -m "feat: add RealCameraSignalProvider backed by real frames + detections"
```

---

### Task 4: Gerçek Veri Eğitim/Değerlendirme Fonksiyonu

**Files:**
- Create: `src/dronetrackingrl/real_data/training.py`
- Test: `tests/test_real_training.py`

**Interfaces:**
- Consumes: `parse_detections_file` (Task 1), `RealCameraSignalProvider` (Task 3), `MaskedCropAutoencoder` (Faz 1), `CameraSwitchEnv` (Faz 1).
- Produces: `load_detections(detections_dir, cameras) -> Dict[...]`, `collect_crops(...) -> List[np.ndarray]` (encoder ön-eğitimi için), `train_and_evaluate(...) -> float` (ortalama değerlendirme ödülü). `train_and_evaluate`, dataset3'ün tam çıkarılmış hali üzerinde gerçek eğitim koşusu için de (bu plan dışında, elle) kullanılacak.

- [ ] **Step 1: Başarısız testi yaz**

`tests/test_real_training.py`:
```python
from pathlib import Path

import torch

from dronetrackingrl.real_data.training import train_and_evaluate

FRAMES_ROOT = Path("tests/fixtures/dataset3_sample/frames")
DETECTIONS_DIR = Path("tests/fixtures/dataset3_sample/detections")


def test_train_and_evaluate_runs_on_real_fixture_and_reports_full_visibility_reward():
    torch.manual_seed(0)
    avg_reward = train_and_evaluate(
        frames_root=FRAMES_ROOT,
        detections_dir=DETECTIONS_DIR,
        cameras=[0],
        start_ref_frame=1,
        end_ref_frame=3,
        reference_camera=0,
        total_timesteps=200,
        seed=0,
    )
    # Tek kamera, 3 gerçek fixture karesinin (1,2,3) hepsi ground truth'a
    # göre görünür -- ödül politikadan bağımsız her zaman +1, yani bu bir
    # kablolama (wiring) duman testi, öğrenme testi değil (PPO'nun gerçekten
    # rastgeleyi yendiğinin kanıtı sentetik Task 7'de zaten var).
    assert avg_reward == 1.0
```

- [ ] **Step 2: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_real_training.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Eğitim fonksiyonunu implemente et**

`src/dronetrackingrl/real_data/training.py`:
```python
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder, train_autoencoder
from dronetrackingrl.masking.background_subtraction import BackgroundSubtractor
from dronetrackingrl.real_data.detections import parse_detections_file
from dronetrackingrl.real_data.real_signal_provider import RealCameraSignalProvider, default_frame_reader
from dronetrackingrl.real_data.sync import is_recording, mapped_frame
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv


def load_detections(
    detections_dir: Path, cameras: List[int]
) -> Dict[int, Dict[int, Optional[Tuple[float, float]]]]:
    return {
        camera: parse_detections_file(str(Path(detections_dir) / f"cam{camera}.txt"))
        for camera in cameras
    }


def collect_crops(
    frames_root: Path,
    cameras: List[int],
    start_ref_frame: int,
    end_ref_frame: int,
    reference_camera: int = 0,
    crop_size: int = 64,
) -> List[np.ndarray]:
    """Encoder ön-eğitimi için kameraların kayıtta olduğu her karede
    maskelenmiş kırpım toplar (RealCameraSignalProvider'dan bağımsız,
    kendi BackgroundSubtractor'larıyla — encode edilmemiş ham kırpımlara
    ihtiyaç var)."""
    crops: List[np.ndarray] = []
    subtractors = {camera: BackgroundSubtractor(crop_size=crop_size) for camera in cameras}
    for ref_frame in range(start_ref_frame, end_ref_frame + 1):
        for camera in cameras:
            if not is_recording(ref_frame, camera, reference_camera):
                continue
            camera_frame_id = round(mapped_frame(ref_frame, reference_camera, camera))
            frame = default_frame_reader(frames_root, camera, camera_frame_id)
            if frame is None:
                continue
            result = subtractors[camera].update(frame)
            crops.append(result.mask_crop)
    return crops


def train_and_evaluate(
    frames_root: Path,
    detections_dir: Path,
    cameras: List[int],
    start_ref_frame: int,
    end_ref_frame: int,
    reference_camera: int = 0,
    crop_size: int = 64,
    latent_dim: int = 16,
    total_timesteps: int = 2000,
    seed: int = 0,
) -> float:
    detections = load_detections(detections_dir, cameras)
    encoder = MaskedCropAutoencoder(crop_size=crop_size, latent_dim=latent_dim)

    crops = collect_crops(frames_root, cameras, start_ref_frame, end_ref_frame, reference_camera, crop_size)
    if crops:
        crops_tensor = torch.tensor(np.stack(crops), dtype=torch.float32).unsqueeze(1)
        train_autoencoder(encoder, crops_tensor, epochs=20)

    def make_env():
        provider = RealCameraSignalProvider(
            frames_root=frames_root,
            detections=detections,
            cameras=cameras,
            encoder=encoder,
            latent_dim=latent_dim,
            start_ref_frame=start_ref_frame,
            end_ref_frame=end_ref_frame,
            reference_camera=reference_camera,
            crop_size=crop_size,
        )
        return CameraSwitchEnv(provider)

    train_env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=total_timesteps)

    eval_env = make_env()
    obs, _ = eval_env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(int(action))
        total_reward += reward
        steps += 1
    return total_reward / steps if steps else 0.0
```

- [ ] **Step 4: Testin geçtiğini doğrula**

Run: `pytest tests/test_real_training.py -v`
Expected: PASS (1/1)

- [ ] **Step 5: Tüm test paketini çalıştır**

Run: `pytest -v`
Expected: Faz 1'in 21 testi + bu planın yeni testleri (5+5+3+1=14) hepsi PASS (toplam 35).

- [ ] **Step 6: Commit**

```bash
git add src/dronetrackingrl/real_data/training.py tests/test_real_training.py
git commit -m "feat: add real-data train_and_evaluate wiring PPO to RealCameraSignalProvider"
```

---

## Bu Plandan Sonra (kapsam dışı, elle çalıştırılacak)

Bu planın 4 task'ı, gerçek veri boru hattının **doğru kablolandığını**
küçük gerçek fixture'larla kanıtlıyor. dataset3'ün **tam** çıkarılmış
hali üzerinde (6 kamera, ~132.000 kare) gerçek bir eğitim koşusu bu planın
kapsamı dışında — `train_and_evaluate(...)`'ı gerçek yollarla (harici
`DatasetDroneTracker/drone-tracking-datasets/dataset3/frames/` ve
`detections/`), tüm 6 kamerayla, daha büyük `total_timesteps` ile, ve her
kameranın kendi `detections/camN.txt`'inden bulunacak gerçek boş bir
başlangıç karesiyle (cam0 için zaten bulundu: kare 13442) çağırarak elle
yapılacak. Sonuç, Faz 1'deki gibi Oracle/Sabit-kamera/Rastgele
baseline'larıyla karşılaştırılacak (spec § Baseline'lar ve Değerlendirme).
