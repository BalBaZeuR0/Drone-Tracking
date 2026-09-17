# Çekirdek RL Bileşenleri Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Veri seti indirmesi tamamlanmadan, RL kamera-seçim sisteminin veri-setinden-bağımsız çekirdek bileşenlerini (ground-truth projeksiyon, maskeleme, latent encoder, RL ortamı, baseline politikalar, metrikler) TDD ile inşa etmek ve sentetik veriyle uçtan uca bir PPO eğitim akışının çalıştığını kanıtlamak.

**Architecture:** Her bileşen bağımsız, dar arayüzlerle (Protocol/dataclass) test edilebilir bir Python modülü olarak yazılır. Gerçek görüntü/kalibrasyon dosya formatları henüz bilinmediği için, bileşenler arası bağlantı `CameraSignalProvider` protokolü üzerinden soyutlanır — testlerde sahte (fake) sağlayıcılar, son entegrasyon testinde ise prosedürel olarak üretilmiş sentetik kareler kullanılır. Gerçek veri seti entegrasyonu (dosya formatı parse etme) bu planın kapsamı dışındadır, veri seti indirilince ayrı bir plan olarak yazılacaktır.

**Tech Stack:** Python 3.10+, numpy, opencv-python (klasik CV / arka plan çıkarma), PyTorch (autoencoder), Gymnasium (özel RL ortamı), Stable-Baselines3 PPO, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-camera-switching-rl-design.md`

## Global Constraints

- Aksiyon uzayı `Discrete(N)` (N = kamera sayısı), algoritma PPO (Stable-Baselines3).
- Latent encoder tüm kameralar arasında **paylaşılan tek bir model** (kamera başına ayrı model yok).
- Ödül **sadece** "kadrajda olma/görünürlük" kriterine dayanır (merkeze yakınlık, boyut gibi ek kriterler bu aşamada yok).
- RL gözlemine **sadece latent vektör** girer; elle çıkarılan (x/y/confidence) sinyali gözleme dahil edilmez — sadece trajectory kaydı için ayrıca saklanır.
- Maskeleme modülü XYZ/kalibrasyona hiç bakmadan, saf görüntü tabanlı (kör) çalışır. XYZ+kalibrasyon sadece otomatik doğrulama/ground-truth referansı için kullanılır.
- Bu plan veri-setinden-bağımsız çekirdek bileşenleri kapsar; gerçek dosya formatı parse/ingestion işi kapsam dışı, ayrı bir sonraki plana bırakılmıştır.

---

### Task 0: Proje İskeleti

**Files:**
- Create: `pyproject.toml`
- Create: `src/dronetrackingrl/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: yok (ilk task).
- Produces: `dronetrackingrl` adında pip-editable-installable bir paket; `pytest` çalıştırılabilir hale gelir.

- [ ] **Step 1: pyproject.toml oluştur**

```toml
[project]
name = "dronetrackingrl"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "opencv-python>=4.9",
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
    "torch>=2.2",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Paket giriş dosyasını oluştur**

`src/dronetrackingrl/__init__.py` — boş dosya.

- [ ] **Step 3: .gitignore oluştur**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Sanal ortam kur ve paketi kur**

Run:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
```
Expected: kurulum hatasız tamamlanır.

- [ ] **Step 5: pytest'in paketi bulduğunu doğrula**

Run: `pytest --collect-only`
Expected: hata yok, "no tests ran" / 0 test toplandığını bildirir (henüz test dosyası yok).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/dronetrackingrl/__init__.py .gitignore
git commit -m "chore: scaffold dronetrackingrl package"
```

---

### Task 1: Ground-Truth Projeksiyon Modülü

**Files:**
- Create: `src/dronetrackingrl/geometry/__init__.py`
- Create: `src/dronetrackingrl/geometry/projection.py`
- Test: `tests/test_projection.py`

**Interfaces:**
- Consumes: yok (saf numpy matematiği).
- Produces: `Camera` dataclass (`k`, `r`, `t`, `width`, `height`), `ProjectionResult` dataclass (`pixel: Optional[Tuple[float,float]]`, `in_frame: bool`), `project_point(camera: Camera, point_world: np.ndarray) -> ProjectionResult`. İleride (sonraki plan) gerçek kalibrasyon dosyalarından `Camera` üretecek bir adaptör bu fonksiyonu kullanacak.

- [ ] **Step 1: Boş `__init__.py` oluştur**

`src/dronetrackingrl/geometry/__init__.py` — boş dosya.

- [ ] **Step 2: Başarısız testi yaz**

`tests/test_projection.py`:
```python
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
```

- [ ] **Step 3: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_projection.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'dronetrackingrl.geometry.projection'`)

- [ ] **Step 4: Modülü implemente et**

`src/dronetrackingrl/geometry/projection.py`:
```python
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Camera:
    k: np.ndarray  # 3x3 intrinsic matrix
    r: np.ndarray  # 3x3 rotation, world -> camera
    t: np.ndarray  # 3, translation, world -> camera
    width: int
    height: int


@dataclass
class ProjectionResult:
    pixel: Optional[Tuple[float, float]]
    in_frame: bool


def project_point(camera: Camera, point_world: np.ndarray) -> ProjectionResult:
    p_cam = camera.r @ point_world + camera.t
    if p_cam[2] <= 0:
        return ProjectionResult(pixel=None, in_frame=False)

    p_img_h = camera.k @ p_cam
    u = p_img_h[0] / p_img_h[2]
    v = p_img_h[1] / p_img_h[2]
    in_frame = 0 <= u < camera.width and 0 <= v < camera.height
    return ProjectionResult(pixel=(u, v), in_frame=in_frame)
```

- [ ] **Step 5: Testin geçtiğini doğrula**

Run: `pytest tests/test_projection.py -v`
Expected: PASS (3/3)

- [ ] **Step 6: Commit**

```bash
git add src/dronetrackingrl/geometry tests/test_projection.py
git commit -m "feat: add pinhole camera projection for ground-truth visibility"
```

---

### Task 2: Maskeleme (Arka Plan Çıkarma) Modülü

**Files:**
- Create: `src/dronetrackingrl/masking/__init__.py`
- Create: `src/dronetrackingrl/masking/background_subtraction.py`
- Test: `tests/test_background_subtraction.py`

**Interfaces:**
- Consumes: yok (opencv + numpy).
- Produces: `MaskResult` dataclass (`mask_crop: np.ndarray` şekli `(crop_size, crop_size)` float32 [0,1], `centroid: Optional[Tuple[float,float]]`, `visible: bool`), `BackgroundSubtractor` sınıfı: `update(frame: np.ndarray) -> MaskResult`, `reset() -> None`. Task 6 ve Task 7 bu sınıfı kullanacak.

- [ ] **Step 1: Boş `__init__.py` oluştur**

`src/dronetrackingrl/masking/__init__.py` — boş dosya.

- [ ] **Step 2: Başarısız testi yaz**

`tests/test_background_subtraction.py`:
```python
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
```

- [ ] **Step 3: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_background_subtraction.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Modülü implemente et**

`src/dronetrackingrl/masking/background_subtraction.py`:
```python
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class MaskResult:
    mask_crop: np.ndarray
    centroid: Optional[Tuple[float, float]]
    visible: bool


class BackgroundSubtractor:
    def __init__(self, crop_size: int = 64, diff_threshold: int = 25, min_blob_area: int = 5):
        self.crop_size = crop_size
        self.diff_threshold = diff_threshold
        self.min_blob_area = min_blob_area
        self._background: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._background = None

    def update(self, frame: np.ndarray) -> MaskResult:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        if self._background is None:
            # First frame of a sequence is assumed drone-free and becomes the
            # static reference frame for differencing.
            self._background = gray
            return self._no_detection()

        diff = np.abs(gray - self._background)
        binary = (diff > self.diff_threshold).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        best_label = None
        best_area = self.min_blob_area - 1
        for label in range(1, num_labels):  # label 0 is background
            area = stats[label, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_label = label

        if best_label is None:
            return self._no_detection()

        cx, cy = centroids[best_label]
        masked = np.where(labels == best_label, gray, 0.0)
        crop = self._crop_around(masked, cx, cy)
        return MaskResult(mask_crop=crop / 255.0, centroid=(float(cx), float(cy)), visible=True)

    def _no_detection(self) -> MaskResult:
        return MaskResult(
            mask_crop=np.zeros((self.crop_size, self.crop_size), dtype=np.float32),
            centroid=None,
            visible=False,
        )

    def _crop_around(self, image: np.ndarray, cx: float, cy: float) -> np.ndarray:
        half = self.crop_size // 2
        height, width = image.shape
        x0, y0 = int(cx) - half, int(cy) - half
        x1, y1 = x0 + self.crop_size, y0 + self.crop_size

        src_x0, src_y0 = max(x0, 0), max(y0, 0)
        src_x1, src_y1 = min(x1, width), min(y1, height)

        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)

        crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        if src_x1 > src_x0 and src_y1 > src_y0:
            crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
        return crop
```

- [ ] **Step 5: Testin geçtiğini doğrula**

Run: `pytest tests/test_background_subtraction.py -v`
Expected: PASS (3/3)

- [ ] **Step 6: Commit**

```bash
git add src/dronetrackingrl/masking tests/test_background_subtraction.py
git commit -m "feat: add static-reference background subtraction masking"
```

---

### Task 3: Değerlendirme Metrikleri

**Files:**
- Create: `src/dronetrackingrl/evaluation/__init__.py`
- Create: `src/dronetrackingrl/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: yok (saf fonksiyonlar).
- Produces: `frame_in_view_rate(visible_flags) -> float`, `pixel_trajectory_error(predicted, ground_truth) -> float`, `switch_count(camera_indices) -> int`.

- [ ] **Step 1: Boş `__init__.py` oluştur**

`src/dronetrackingrl/evaluation/__init__.py` — boş dosya.

- [ ] **Step 2: Başarısız testi yaz**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 3: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Modülü implemente et**

`src/dronetrackingrl/evaluation/metrics.py`:
```python
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
```

- [ ] **Step 5: Testin geçtiğini doğrula**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (5/5)

- [ ] **Step 6: Commit**

```bash
git add src/dronetrackingrl/evaluation tests/test_metrics.py
git commit -m "feat: add evaluation metrics (in-view rate, trajectory error, switch count)"
```

---

### Task 4: RL Ortamı ve Sinyal Sağlayıcı Arayüzü

**Files:**
- Create: `src/dronetrackingrl/rl_env/__init__.py`
- Create: `src/dronetrackingrl/rl_env/signal_provider.py`
- Create: `src/dronetrackingrl/rl_env/camera_switch_env.py`
- Create: `tests/fakes.py`
- Test: `tests/test_camera_switch_env.py`

**Interfaces:**
- Consumes: yok (Gymnasium API üzerine kurulu, dış modüllere bağımlı değil).
- Produces: `CameraStepSignal` (NamedTuple: `latent: np.ndarray`, `visible: bool`, `pixel: Optional[Tuple[float,float]]`), `CameraSignalProvider` Protocol (`latent_dim: int`, `num_cameras: int`, `reset() -> List[CameraStepSignal]`, `step() -> Tuple[List[CameraStepSignal], bool]`), `CameraSwitchEnv(gym.Env)` — Task 5 (baselines) ve Task 7 (entegrasyon) bu arayüzü kullanacak. `tests/fakes.py` içindeki `FakeCameraSignalProvider` Task 5'in testlerinde de yeniden kullanılacak.

**Önemli tasarım notu (zamanlama):** `step(action)` önce **o anda gözlemlenen** karenin (son `reset()`/`step()` çağrısının döndürdüğü sinyaller) görünürlüğüne göre ödülü hesaplar, **sonra** sağlayıcıyı bir sonraki kareye ilerletip yeni gözlemi döndürür. Böylece ajan, seçtiği aksiyonun değerlendirildiği kareyi gerçekten görmüş olur (bir sonraki kareyi önceden bilemez).

- [ ] **Step 1: Boş `__init__.py` oluştur**

`src/dronetrackingrl/rl_env/__init__.py` — boş dosya.

- [ ] **Step 2: Sinyal sağlayıcı arayüzünü yaz**

`src/dronetrackingrl/rl_env/signal_provider.py`:
```python
from typing import List, NamedTuple, Optional, Protocol, Tuple

import numpy as np


class CameraStepSignal(NamedTuple):
    latent: np.ndarray
    visible: bool
    pixel: Optional[Tuple[float, float]]


class CameraSignalProvider(Protocol):
    latent_dim: int
    num_cameras: int

    def reset(self) -> List[CameraStepSignal]:
        ...

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        ...
```

- [ ] **Step 3: Test çifti (fake) ve başarısız testi yaz**

`tests/fakes.py`:
```python
from typing import List, Tuple

import numpy as np

from dronetrackingrl.rl_env.signal_provider import CameraStepSignal


class FakeCameraSignalProvider:
    """Deterministic three-camera, three-frame script for tests.

    Frame 0: camera 0 visible. Frames 1-2: camera 1 visible.
    """

    latent_dim = 2
    num_cameras = 3

    def __init__(self):
        self._visible_camera_per_frame = [0, 1, 1]
        self._frame_index = -1

    def reset(self) -> List[CameraStepSignal]:
        self._frame_index = 0
        return self._signals_for_frame(self._frame_index)

    def step(self) -> Tuple[List[CameraStepSignal], bool]:
        self._frame_index += 1
        signals = self._signals_for_frame(self._frame_index)
        done = self._frame_index == len(self._visible_camera_per_frame) - 1
        return signals, done

    def _signals_for_frame(self, frame_index: int) -> List[CameraStepSignal]:
        visible_camera = self._visible_camera_per_frame[frame_index]
        return [
            CameraStepSignal(
                latent=np.array([float(camera), float(frame_index)]),
                visible=(camera == visible_camera),
                pixel=(float(camera), float(frame_index)),
            )
            for camera in range(self.num_cameras)
        ]
```

`tests/test_camera_switch_env.py`:
```python
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv
from fakes import FakeCameraSignalProvider


def test_reset_returns_observation_with_expected_shape():
    env = CameraSwitchEnv(FakeCameraSignalProvider())
    obs, info = env.reset()
    assert obs.shape == (6,)
    assert info["visible"] == [True, False, False]


def test_step_rewards_choosing_the_visible_camera():
    env = CameraSwitchEnv(FakeCameraSignalProvider())
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 1.0  # camera 0 was visible in the frame just observed
    assert info["visible"] == [False, True, False]  # info now reflects the next frame


def test_step_applies_switch_penalty_on_camera_change():
    env = CameraSwitchEnv(FakeCameraSignalProvider(), switch_penalty=0.5)
    env.reset()
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(1)
    assert reward == 0.5  # base reward 1.0 (camera 1 visible) minus switch penalty 0.5
    assert terminated is True
```

- [ ] **Step 4: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_camera_switch_env.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'dronetrackingrl.rl_env.camera_switch_env'`)

- [ ] **Step 5: Ortamı implemente et**

`src/dronetrackingrl/rl_env/camera_switch_env.py`:
```python
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dronetrackingrl.rl_env.signal_provider import CameraSignalProvider


class CameraSwitchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, signal_provider: CameraSignalProvider, switch_penalty: float = 0.0):
        super().__init__()
        self.provider = signal_provider
        self.switch_penalty = switch_penalty
        self.num_cameras = signal_provider.num_cameras
        self.latent_dim = signal_provider.latent_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_cameras * self.latent_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_cameras)
        self._current_signals = None
        self._last_action: Optional[int] = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        signals = self.provider.reset()
        self._current_signals = signals
        self._last_action = None
        return self._signals_to_obs(signals), self._signals_to_info(signals)

    def step(self, action: int):
        reward = 1.0 if self._current_signals[action].visible else -1.0
        if self._last_action is not None and action != self._last_action:
            reward -= self.switch_penalty
        self._last_action = action

        signals, done = self.provider.step()
        self._current_signals = signals
        obs = self._signals_to_obs(signals)
        info = self._signals_to_info(signals)
        return obs, reward, done, False, info

    def _signals_to_obs(self, signals):
        return np.concatenate([signal.latent for signal in signals]).astype(np.float32)

    def _signals_to_info(self, signals):
        return {
            "visible": [signal.visible for signal in signals],
            "pixel": [signal.pixel for signal in signals],
        }
```

- [ ] **Step 6: Testin geçtiğini doğrula**

Run: `pytest tests/test_camera_switch_env.py -v`
Expected: PASS (3/3)

- [ ] **Step 7: Commit**

```bash
git add src/dronetrackingrl/rl_env tests/fakes.py tests/test_camera_switch_env.py
git commit -m "feat: add CameraSwitchEnv gymnasium environment"
```

---

### Task 5: Baseline Politikalar

**Files:**
- Create: `src/dronetrackingrl/baselines/__init__.py`
- Create: `src/dronetrackingrl/baselines/policies.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: `CameraSwitchEnv` (Task 4), `FakeCameraSignalProvider` (`tests/fakes.py`, Task 4).
- Produces: `FixedCameraPolicy`, `RandomPolicy`, `OraclePolicy` — hepsi `select_action(info: dict) -> int`. Task 7 (entegrasyon) `RandomPolicy`'yi karşılaştırma için kullanacak.

- [ ] **Step 1: Boş `__init__.py` oluştur**

`src/dronetrackingrl/baselines/__init__.py` — boş dosya.

- [ ] **Step 2: Başarısız testi yaz**

`tests/test_policies.py`:
```python
from dronetrackingrl.baselines.policies import FixedCameraPolicy, OraclePolicy, RandomPolicy
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv
from fakes import FakeCameraSignalProvider


def _run_episode(policy) -> float:
    env = CameraSwitchEnv(FakeCameraSignalProvider())
    _, info = env.reset()
    total_reward = 0.0
    terminated = False
    while not terminated:
        action = policy.select_action(info)
        _, reward, terminated, _, info = env.step(action)
        total_reward += reward
    return total_reward


def test_oracle_policy_always_picks_the_visible_camera():
    total_reward = _run_episode(OraclePolicy())
    assert total_reward == 2.0  # both scored frames have a visible camera and are picked correctly


def test_fixed_camera_policy_uses_configured_camera():
    policy = FixedCameraPolicy(camera_index=0)
    total_reward = _run_episode(policy)
    assert total_reward == 0.0  # camera 0 is visible at frame 0 but not afterwards


def test_random_policy_stays_within_action_space():
    policy = RandomPolicy(num_cameras=3, seed=1)
    env = CameraSwitchEnv(FakeCameraSignalProvider())
    _, info = env.reset()
    action = policy.select_action(info)
    assert 0 <= action < 3
```

- [ ] **Step 3: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_policies.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Politikaları implemente et**

`src/dronetrackingrl/baselines/policies.py`:
```python
import random
from typing import Dict, List


class FixedCameraPolicy:
    def __init__(self, camera_index: int):
        self.camera_index = camera_index

    def select_action(self, info: Dict) -> int:
        return self.camera_index


class RandomPolicy:
    def __init__(self, num_cameras: int, seed: int = 0):
        self.num_cameras = num_cameras
        self._rng = random.Random(seed)

    def select_action(self, info: Dict) -> int:
        return self._rng.randrange(self.num_cameras)


class OraclePolicy:
    def select_action(self, info: Dict) -> int:
        visible: List[bool] = info["visible"]
        for camera_index, is_visible in enumerate(visible):
            if is_visible:
                return camera_index
        return 0
```

- [ ] **Step 5: Testin geçtiğini doğrula**

Run: `pytest tests/test_policies.py -v`
Expected: PASS (3/3)

- [ ] **Step 6: Commit**

```bash
git add src/dronetrackingrl/baselines tests/test_policies.py
git commit -m "feat: add oracle/fixed/random baseline policies"
```

---

### Task 6: Latent Autoencoder

**Files:**
- Create: `src/dronetrackingrl/encoder/__init__.py`
- Create: `src/dronetrackingrl/encoder/autoencoder.py`
- Test: `tests/test_autoencoder.py`

**Interfaces:**
- Consumes: `BackgroundSubtractor.mask_crop` çıktısıyla aynı şekle sahip tensörler (Task 2), ama doğrudan bağımlı değil — girdi olarak herhangi bir `(N, 1, crop_size, crop_size)` float tensörü kabul eder.
- Produces: `MaskedCropAutoencoder(nn.Module)` (`encode(x) -> Tensor`, `forward(x) -> Tensor`), `train_autoencoder(model, crops, epochs, lr) -> List[float]`. Task 7 bu ikisini kullanacak.

- [ ] **Step 1: Boş `__init__.py` oluştur**

`src/dronetrackingrl/encoder/__init__.py` — boş dosya.

- [ ] **Step 2: Başarısız testi yaz**

`tests/test_autoencoder.py`:
```python
import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder, train_autoencoder


def test_encode_produces_expected_latent_shape():
    model = MaskedCropAutoencoder(crop_size=32, latent_dim=8)
    crops = torch.rand(4, 1, 32, 32)

    latent = model.encode(crops)

    assert latent.shape == (4, 8)


def test_training_reduces_reconstruction_loss():
    torch.manual_seed(0)
    model = MaskedCropAutoencoder(crop_size=32, latent_dim=8)
    crops = torch.rand(8, 1, 32, 32)

    losses = train_autoencoder(model, crops, epochs=20, lr=1e-2)

    assert losses[-1] < losses[0]
```

- [ ] **Step 3: Testin başarısız olduğunu doğrula**

Run: `pytest tests/test_autoencoder.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 4: Autoencoder'ı implemente et**

`src/dronetrackingrl/encoder/autoencoder.py`:
```python
from typing import List

import torch
from torch import nn


class MaskedCropAutoencoder(nn.Module):
    def __init__(self, crop_size: int = 64, latent_dim: int = 16):
        super().__init__()
        reduced = crop_size // 4
        self.crop_size = crop_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * reduced * reduced, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * reduced * reduced),
            nn.ReLU(),
            nn.Unflatten(1, (16, reduced, reduced)),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def train_autoencoder(
    model: MaskedCropAutoencoder, crops: torch.Tensor, epochs: int = 5, lr: float = 1e-3
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstruction = model(crops)
        loss = loss_fn(reconstruction, crops)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
```

- [ ] **Step 5: Testin geçtiğini doğrula**

Run: `pytest tests/test_autoencoder.py -v`
Expected: PASS (2/2)

- [ ] **Step 6: Commit**

```bash
git add src/dronetrackingrl/encoder tests/test_autoencoder.py
git commit -m "feat: add shared masked-crop autoencoder for latent observations"
```

---

### Task 7: Uçtan Uca Sentetik Entegrasyon + PPO Duman Testi

**Files:**
- Test: `tests/test_integration_smoke.py`

**Interfaces:**
- Consumes: `BackgroundSubtractor` (Task 2), `MaskedCropAutoencoder` / `train_autoencoder` (Task 6), `CameraStepSignal` / `CameraSwitchEnv` (Task 4), `RandomPolicy` (Task 5).
- Produces: yok (bu, üretim kodu değil, tüm bileşenleri birbirine bağlayan doğrulama testidir).

Bu görev yeni üretim modülü eklemez; prosedürel olarak üretilmiş sentetik kamera görüntüleriyle tüm pipeline'ı (maskeleme → encoder → RL ortamı → PPO eğitimi) uçtan uca çalıştırıp, eğitilmiş PPO politikasının rastgele politikadan anlamlı ölçüde daha iyi ortalama ödül aldığını doğrular.

- [ ] **Step 1: Başarısız/eksik testi yaz**

`tests/test_integration_smoke.py`:
```python
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from dronetrackingrl.baselines.policies import RandomPolicy
from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder, train_autoencoder
from dronetrackingrl.masking.background_subtraction import BackgroundSubtractor
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv
from dronetrackingrl.rl_env.signal_provider import CameraStepSignal

FRAME_SIZE = 64
FRAMES_PER_PHASE = 20
NUM_CAMERAS = 3
CROP_SIZE = 32
LATENT_DIM = 8


def _make_synthetic_frames():
    """1 init frame (no camera has a blob) + FRAMES_PER_PHASE frames per
    camera, cycling which camera shows a bright blob: camera 0, then 1, then 2."""
    background_value = 50
    total_frames = 1 + FRAMES_PER_PHASE * NUM_CAMERAS
    frames = {camera: [] for camera in range(NUM_CAMERAS)}
    for frame_index in range(total_frames):
        active_camera = None if frame_index == 0 else (frame_index - 1) // FRAMES_PER_PHASE
        for camera in range(NUM_CAMERAS):
            frame = np.full((FRAME_SIZE, FRAME_SIZE), background_value, dtype=np.uint8)
            if camera == active_camera:
                frame[28:36, 28:36] = 255
            frames[camera].append(frame)
    return frames


class SyntheticSignalProvider:
    latent_dim = LATENT_DIM
    num_cameras = NUM_CAMERAS

    def __init__(self, frames_per_camera, encoder):
        self._frames_per_camera = frames_per_camera
        self._encoder = encoder
        self._subtractors = [BackgroundSubtractor(crop_size=CROP_SIZE) for _ in range(NUM_CAMERAS)]
        self._num_frames = len(frames_per_camera[0])
        self._frame_index = -1

    def reset(self):
        for subtractor in self._subtractors:
            subtractor.reset()
        self._frame_index = 0
        return self._signals_for_frame(self._frame_index)

    def step(self):
        self._frame_index += 1
        signals = self._signals_for_frame(self._frame_index)
        done = self._frame_index == self._num_frames - 1
        return signals, done

    def _signals_for_frame(self, frame_index):
        signals = []
        for camera in range(NUM_CAMERAS):
            frame = self._frames_per_camera[camera][frame_index]
            result = self._subtractors[camera].update(frame)
            crop_tensor = torch.tensor(result.mask_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                latent = self._encoder.encode(crop_tensor).squeeze(0).numpy()
            signals.append(
                CameraStepSignal(latent=latent, visible=result.visible, pixel=result.centroid)
            )
        return signals


def _train_encoder_on_synthetic_crops(frames_per_camera) -> MaskedCropAutoencoder:
    crops = []
    for camera in range(NUM_CAMERAS):
        subtractor = BackgroundSubtractor(crop_size=CROP_SIZE)
        for frame in frames_per_camera[camera]:
            result = subtractor.update(frame)
            crops.append(result.mask_crop)
    crops_tensor = torch.tensor(np.stack(crops), dtype=torch.float32).unsqueeze(1)
    model = MaskedCropAutoencoder(crop_size=CROP_SIZE, latent_dim=LATENT_DIM)
    train_autoencoder(model, crops_tensor, epochs=20)
    return model


def _average_reward_over_episode(env, policy) -> float:
    _, info = env.reset()
    total_reward = 0.0
    terminated = False
    steps = 0
    while not terminated:
        action = policy.select_action(info)
        _, reward, terminated, _, info = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward / steps


def test_trained_ppo_outperforms_random_policy_on_synthetic_pipeline():
    frames_per_camera = _make_synthetic_frames()
    encoder = _train_encoder_on_synthetic_crops(frames_per_camera)

    def make_env():
        provider = SyntheticSignalProvider(frames_per_camera, encoder)
        return CameraSwitchEnv(provider)

    train_env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", train_env, verbose=0, seed=0)
    model.learn(total_timesteps=4000)

    eval_env = make_env()
    obs, _ = eval_env.reset()
    total_reward = 0.0
    terminated = False
    steps = 0
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = eval_env.step(int(action))
        total_reward += reward
        steps += 1
    ppo_avg_reward = total_reward / steps

    random_avg_reward = _average_reward_over_episode(
        make_env(), RandomPolicy(num_cameras=NUM_CAMERAS, seed=0)
    )

    assert ppo_avg_reward > random_avg_reward + 0.5
```

- [ ] **Step 2: Testin başarısız olduğunu (ya da henüz koşulamadığını) doğrula**

Run: `pytest tests/test_integration_smoke.py -v`
Expected: Bu görev yeni üretim kodu eklemediği için önceki task'lar doğru tamamlandıysa test muhtemelen ilk seferde geçer. Yine de çalıştırıp gerçekten PASS ettiğini doğrulamak zorunlu adımdır — geçmezse (örn. import hatası, assertion hatası) kök nedeni araştırıp yukarıdaki task'lardaki ilgili implementasyonu düzelt.

- [ ] **Step 3: Testin geçtiğini doğrula**

Run: `pytest tests/test_integration_smoke.py -v -s`
Expected: PASS (1/1). Çıktıda `ppo_avg_reward`'ın `random_avg_reward`'dan en az 0.5 daha yüksek olduğu doğrulanır.

- [ ] **Step 4: Tüm test paketini çalıştır**

Run: `pytest -v`
Expected: tüm testler (projection, masking, metrics, env, policies, autoencoder, integration) PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_smoke.py
git commit -m "test: add synthetic end-to-end PPO integration smoke test"
```

---

## Sonraki Plan (kapsam dışı)

Veri seti indirmesi tamamlanınca, ayrı bir plan olarak: gerçek dosya/klasör yapısının incelenmesi, kalibrasyon dosyalarından `Camera` (Task 1) nesnelerinin üretilmesi, gerçek video karelerinin `BackgroundSubtractor`'a (Task 2) beslenmesi ve gerçek sekanslar üzerinde PPO eğitimi/değerlendirmesi ele alınacak.
