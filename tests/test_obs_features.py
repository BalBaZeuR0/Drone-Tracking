import numpy as np
import pytest
import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.real_data.experiment import (
    AUX_FEATURE_DIM,
    CachedSignalProvider,
    ExperimentConfig,
    SignalCache,
    derive_camera_features,
    run_experiment,
)

LATENT_DIM = 8


def _cache(mask_x, recording=None, visible=None, peak=None):
    """Tek kameralı sentetik önbellek; mask_x[i] = blob x'i (NaN = bulunamadı), y = 100."""
    n = len(mask_x)
    mask_pixel = np.full((n, 1, 2), np.nan)
    for i, x in enumerate(mask_x):
        if not np.isnan(x):
            mask_pixel[i, 0] = (x, 100.0)
    crops = np.zeros((n, 1, 64, 64), dtype=np.uint8)
    if peak is not None:
        for i, p in enumerate(peak):
            crops[i, 0, 5, 5] = p
    return SignalCache(
        cameras=[0], reference_camera=0, ref_start=1, ref_end=n, crop_size=64, crops=crops,
        recording=np.ones((n, 1), dtype=bool) if recording is None else np.array(recording, dtype=bool).reshape(n, 1),
        visible=np.ones((n, 1), dtype=bool) if visible is None else np.array(visible, dtype=bool).reshape(n, 1),
        mask_pixel=mask_pixel, gt_pixel=np.full((n, 1, 2), np.nan),
    )


def test_features_have_expected_shape_and_fired_flag():
    features = derive_camera_features(_cache([10, np.nan, 12]))
    assert features.shape == (3, 1, AUX_FEATURE_DIM)
    assert features[:, 0, 0].tolist() == [1.0, 0.0, 1.0]  # fired


def test_peak_feature_is_crop_maximum_scaled_to_unit_range():
    features = derive_camera_features(_cache([10, 10], peak=[255, 51]))
    np.testing.assert_allclose(features[:, 0, 1], [1.0, 0.2])


def test_track_length_grows_for_a_consistent_track_and_resets_on_a_jump():
    # x: 10 -> 12 -> 14 (tutarlı iz), sonra 400'e sıçrama (yanlış alarm gibi), sonra kayıp
    features = derive_camera_features(_cache([10, 12, 14, 400, np.nan]))
    track = features[:, 0, 2]
    assert track[0] < track[1] < track[2]  # iz uzuyor
    assert track[3] == track[0]            # sıçrama izi sıfırlar (yeni iz = 1 kare)
    assert track[4] == 0.0                 # kayıpta 0


def test_recent_rate_is_causal_and_recording_flag_is_passed_through():
    features = derive_camera_features(_cache([10, np.nan, np.nan, 10], recording=[1, 1, 0, 1]))
    rate = features[:, 0, 3]
    assert rate[0] == 1.0                   # sadece kendini görür
    assert rate[3] == pytest.approx(0.5)    # 4 karede 2 ateşleme
    assert features[:, 0, 4].tolist() == [1.0, 1.0, 0.0, 1.0]  # recording


def test_features_do_not_peek_at_the_future():
    base = derive_camera_features(_cache([10, 11, 12, 13]))
    changed = derive_camera_features(_cache([10, 11, 12, np.nan]))
    np.testing.assert_array_equal(base[:3], changed[:3])  # son kare değişse de önceki satırlar aynı


def test_provider_latent_plus_aux_widens_the_observation():
    torch.manual_seed(0)
    cache = _cache([10, 11, 12], peak=[255, 255, 255])
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    provider = CachedSignalProvider(cache, encoder, LATENT_DIM, obs_mode="latent+aux")
    assert provider.latent_dim == LATENT_DIM + AUX_FEATURE_DIM
    signals = provider.reset()
    assert signals[0].latent.shape == (LATENT_DIM + AUX_FEATURE_DIM,)
    assert signals[0].latent[LATENT_DIM] == 1.0  # fired


def test_provider_gt_mode_exposes_only_ground_truth_visibility_and_recording():
    cache = _cache([np.nan, np.nan], visible=[True, False], recording=[True, True])
    provider = CachedSignalProvider(cache, None, LATENT_DIM, obs_mode="gt")
    assert provider.latent_dim == 2
    first = provider.reset()[0]
    second = provider.step()[0][0]
    assert first.latent.tolist() == [1.0, 1.0]
    assert second.latent.tolist() == [0.0, 1.0]


def test_provider_rejects_unknown_obs_mode():
    with pytest.raises(ValueError):
        CachedSignalProvider(_cache([10]), None, LATENT_DIM, obs_mode="bogus")


@pytest.mark.parametrize("obs_mode", ["latent+aux", "gt"])
def test_run_experiment_supports_obs_modes(tmp_path, obs_mode):
    rng = np.random.default_rng(0)
    n, k = 40, 3
    cache = SignalCache(
        cameras=list(range(k)), reference_camera=0, ref_start=1, ref_end=n, crop_size=64,
        crops=np.zeros((n, k, 64, 64), dtype=np.uint8), recording=np.ones((n, k), dtype=bool),
        visible=rng.random((n, k)) > 0.4, mask_pixel=np.full((n, k, 2), np.nan), gt_pixel=np.full((n, k, 2), np.nan),
    )
    path = tmp_path / "c.npz"
    cache.save(path)
    results = run_experiment(
        ExperimentConfig(
            train_cache=str(path), eval_cache=str(path), out_dir=str(tmp_path / "run"), timesteps=64, seeds=[0],
            latent_dim=LATENT_DIM, encoder_epochs=1, ppo_n_steps=64, ppo_batch_size=64, random_seeds=1, pack=False,
            obs_mode=obs_mode,
        )
    )
    assert results["config"]["obs_mode"] == obs_mode
    assert "eval" in results["ppo"]["0"]["splits"]
