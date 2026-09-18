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
