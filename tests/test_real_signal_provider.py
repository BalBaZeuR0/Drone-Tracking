from pathlib import Path

import numpy as np
import pytest

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


def test_signals_for_ref_frame_reports_not_visible_when_camera_not_recording():
    # cam3, cam0'ın 33875. karesine geldiğinde artık kayıt yapmıyor (el ile
    # doğrulandı: tests/test_real_sync.py::test_cam3_stops_recording_near_end_of_cam0_timeline).
    # Bu, is_recording()'in False döndüğü "not recording" dalını -- frame_reader
    # hiç çağrılmadan visible=False/pixel=None/sıfır-crop üretilen dalı -- gerçek
    # bir görüntü fixture'ı gerektirmeden test eder.
    def _frame_reader_should_not_be_called(frames_root, camera, frame_id):
        raise AssertionError(
            "frame_reader should not be called when the camera is not recording"
        )

    detections = {0: parse_detections_file(CAM0_DETECTIONS_PATH)}
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    provider = RealCameraSignalProvider(
        frames_root=FRAMES_ROOT,
        detections=detections,
        cameras=[3],
        encoder=encoder,
        latent_dim=LATENT_DIM,
        start_ref_frame=33875,
        end_ref_frame=33876,
        reference_camera=0,
        frame_reader=_frame_reader_should_not_be_called,
    )

    signals = provider.signals_for_ref_frame(33875)

    assert signals[0].visible is False
    assert signals[0].pixel is None
    assert signals[0].latent.shape == (LATENT_DIM,)


def test_signals_for_ref_frame_reports_not_visible_when_frame_read_returns_none(caplog):
    # Simulates a missing/corrupt JPEG on disk: frame_reader (e.g.
    # default_frame_reader backed by cv2.imread) returns None instead of
    # raising. The provider must degrade to the same visible=False/pixel=None
    # /zero-crop path used for the not-recording case, not crash trying to
    # call BackgroundSubtractor.update(None), and it must log a warning that
    # names the camera and frame id (spec: "skip + log, never silently
    # swallow").
    def _frame_reader_returns_none(frames_root, camera, frame_id):
        return None

    detections = {0: parse_detections_file(CAM0_DETECTIONS_PATH)}
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    provider = RealCameraSignalProvider(
        frames_root=FRAMES_ROOT,
        detections=detections,
        cameras=[0],
        encoder=encoder,
        latent_dim=LATENT_DIM,
        start_ref_frame=50,
        end_ref_frame=51,
        reference_camera=0,
        frame_reader=_frame_reader_returns_none,
    )

    with caplog.at_level("WARNING"):
        signals = provider.signals_for_ref_frame(50)

    assert signals[0].visible is False
    assert signals[0].pixel is None
    assert signals[0].latent.shape == (LATENT_DIM,)
    assert any(
        "0" in record.getMessage() and "50" in record.getMessage()
        for record in caplog.records
    ), "warning should name the camera and frame id"


def test_signals_for_ref_frame_raises_keyerror_when_camera_missing_from_detections():
    # cam3 is recording at ref_frame 0 (see
    # tests/test_real_sync.py::test_cam3_is_recording_near_start_of_cam0_timeline)
    # but is entirely absent from `detections` here -- a caller wiring
    # mistake. This must raise KeyError immediately rather than silently
    # reporting visible=False for every frame of the episode.
    def _frame_reader_returns_dummy_frame(frames_root, camera, frame_id):
        return np.zeros((64, 64), dtype=np.uint8)

    detections = {0: parse_detections_file(CAM0_DETECTIONS_PATH)}  # camera 3 missing
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    provider = RealCameraSignalProvider(
        frames_root=FRAMES_ROOT,
        detections=detections,
        cameras=[3],
        encoder=encoder,
        latent_dim=LATENT_DIM,
        start_ref_frame=0,
        end_ref_frame=1,
        reference_camera=0,
        frame_reader=_frame_reader_returns_dummy_frame,
    )

    with pytest.raises(KeyError):
        provider.signals_for_ref_frame(0)


def test_detector_factory_is_used_for_masking_but_never_for_visibility():
    from dronetrackingrl.masking.background_subtraction import MaskResult

    class StubDetector:
        resets = 0

        def __init__(self, crop_size):
            self.crop_size = crop_size

        def reset(self):
            StubDetector.resets += 1

        def update(self, frame):
            # Claims "not visible" -- the provider's visible flag must ignore this.
            return MaskResult(np.ones((self.crop_size, self.crop_size), dtype=np.float32), (5.0, 6.0), False)

    provider = RealCameraSignalProvider(
        frames_root=FRAMES_ROOT,
        detections={0: parse_detections_file(CAM0_DETECTIONS_PATH)},
        cameras=[0],
        encoder=MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM),
        latent_dim=LATENT_DIM,
        start_ref_frame=1,
        end_ref_frame=3,
        detector_factory=StubDetector,
    )
    signals = provider.reset()

    assert StubDetector.resets == 1
    assert signals[0].pixel == (5.0, 6.0)  # taken from the detector
    assert signals[0].visible is True  # ground truth (frame 1 is labeled visible), not the stub's False
