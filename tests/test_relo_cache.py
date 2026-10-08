from pathlib import Path

import numpy as np
import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.masking.background_subtraction import MaskResult
from dronetrackingrl.real_data.experiment import (
    AUX_FEATURE_DIM,
    RELO_FEATURE_DIM,
    CachedSignalProvider,
    SignalCache,
    build_signal_cache,
    derive_relo_features,
    merge_camera_caches,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dataset3_sample"
LATENT_DIM = 8


class _FakeRelo:
    """İlk kare ARAMA (klasik gibi, RELO alanı yok), sonrası TAKİP."""

    def __init__(self, crop_size, **kwargs):
        self.crop_size = crop_size
        self.calls = 0

    def reset(self):
        self.calls = 0

    def update(self, frame):
        self.calls += 1
        crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        if self.calls == 1:
            return MaskResult(mask_crop=crop, centroid=None, visible=False)
        return MaskResult(mask_crop=crop, centroid=(5.0, 6.0), visible=True,
                          relo_score=0.75, relo_box_wh=(12.0, 14.0), relo_tracking=True)


def _relo_cache():
    return build_signal_cache(FIXTURE / "frames", FIXTURE / "detections", cameras=[0], ref_start=1, ref_end=3,
                              progress_every=0, detector_factory=_FakeRelo)


def test_build_records_relo_outputs_per_frame():
    cache = _relo_cache()
    assert cache.relo_tracking[:, 0].tolist() == [False, True, True]
    assert np.isnan(cache.relo_score[0, 0]) and cache.relo_score[1, 0] == np.float32(0.75)
    np.testing.assert_array_equal(cache.relo_box_wh[2, 0], [12.0, 14.0])
    np.testing.assert_array_equal(cache.mask_pixel[1, 0], [5.0, 6.0])


def test_classic_cache_has_empty_relo_fields():
    cache = build_signal_cache(FIXTURE / "frames", FIXTURE / "detections", cameras=[0], ref_start=1, ref_end=3,
                               progress_every=0)
    assert np.isnan(cache.relo_score).all() and np.isnan(cache.relo_box_wh).all()
    assert not cache.relo_tracking.any()


def test_v3_roundtrip_keeps_relo_fields(tmp_path):
    cache = _relo_cache()
    cache.save(tmp_path / "c.npz")
    loaded = SignalCache.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(loaded.relo_score, cache.relo_score)
    np.testing.assert_array_equal(loaded.relo_box_wh, cache.relo_box_wh)
    np.testing.assert_array_equal(loaded.relo_tracking, cache.relo_tracking)


def test_v2_caches_still_load_with_empty_relo_fields(tmp_path):
    cache = _relo_cache()
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path, cameras=np.array(cache.cameras), crops=cache.crops, recording=cache.recording, visible=cache.visible,
        mask_pixel=cache.mask_pixel, gt_pixel=cache.gt_pixel,
        meta=np.array([cache.reference_camera, cache.ref_start, cache.ref_end, cache.crop_size, 2]),
    )
    loaded = SignalCache.load(path)
    np.testing.assert_array_equal(loaded.crops, cache.crops)
    assert np.isnan(loaded.relo_score).all() and not loaded.relo_tracking.any()


def test_slice_pad_and_merge_carry_relo_fields():
    cache = _relo_cache()
    part = cache.slice(2, 3)
    assert part.relo_tracking[:, 0].tolist() == [True, True]
    padded = cache.pad_cameras(2)
    assert padded.relo_score.shape == (3, 2) and np.isnan(padded.relo_score[:, 1]).all()
    assert not padded.relo_tracking[:, 1].any()
    merged = merge_camera_caches([cache, cache])
    assert merged.relo_box_wh.shape == (3, 2, 2)
    np.testing.assert_array_equal(merged.relo_tracking[:, 1], cache.relo_tracking[:, 0])


def test_relo_features_are_flag_score_and_box_in_hundreds_of_pixels():
    features = derive_relo_features(_relo_cache())
    assert features.shape == (3, 1, RELO_FEATURE_DIM)
    assert features[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(features[1, 0], [1.0, 0.75, 0.12, 0.14], rtol=1e-6)


def test_provider_relo_mode_appends_relo_features_after_aux():
    torch.manual_seed(0)
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    provider = CachedSignalProvider(_relo_cache(), encoder, LATENT_DIM, obs_mode="latent+aux+relo")
    assert provider.latent_dim == LATENT_DIM + AUX_FEATURE_DIM + RELO_FEATURE_DIM
    provider.reset()
    latent = provider.step()[0][0].latent
    np.testing.assert_allclose(latent[-RELO_FEATURE_DIM:], [1.0, 0.75, 0.12, 0.14], rtol=1e-6)


def test_hybrid_factory_is_picklable_and_builds_a_hybrid_detector():
    import pickle

    from dronetrackingrl.masking.hybrid_detector import HybridReloDetector
    from dronetrackingrl.real_data.experiment import HybridDetectorFactory, ReloOptions

    options = ReloOptions(relo_root="/nonexistent", init_box_px=16.0, lost_patience=4)
    factory = pickle.loads(pickle.dumps(HybridDetectorFactory(options)))
    detector = factory(32, min_response=8.0)  # RELO modeli tembel: burada yüklenmez
    assert isinstance(detector, HybridReloDetector)
    assert detector.init_box_px == 16.0 and detector.lost_patience == 4
    assert detector.classic.min_response == 8.0 and detector.crop_size == 32
    assert options.tag() == "_relot256_b16_t0.2_p4_s3_m8_x60_r50"
