import json
from pathlib import Path

import numpy as np
import torch

from dronetrackingrl.encoder.autoencoder import MaskedCropAutoencoder
from dronetrackingrl.real_data.detections import parse_detections_file
from dronetrackingrl.real_data.experiment import (
    CachedSignalProvider,
    ExperimentConfig,
    SignalCache,
    build_signal_cache,
    observation_diagnostics,
    run_experiment,
)
from dronetrackingrl.real_data.real_signal_provider import RealCameraSignalProvider

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "dataset3_sample"
FRAMES_ROOT = FIXTURE / "frames"
DETECTIONS_DIR = FIXTURE / "detections"
LATENT_DIM = 8


def _cam0_cache() -> SignalCache:
    return build_signal_cache(
        FRAMES_ROOT, DETECTIONS_DIR, cameras=[0], ref_start=1, ref_end=3, progress_every=0
    )


def test_cache_has_expected_shapes_and_real_ground_truth():
    cache = _cam0_cache()
    assert cache.crops.shape == (3, 1, 64, 64)
    assert cache.crops.dtype == np.uint8
    assert cache.recording.all()
    assert cache.visible.all()  # real cam0 frames 1-3 are labeled visible
    assert cache.gt_pixel[0, 0].tolist() == [742.82211823, 897.10093596]


def test_cached_provider_matches_real_provider_exactly():
    torch.manual_seed(0)
    encoder = MaskedCropAutoencoder(crop_size=64, latent_dim=LATENT_DIM)
    cache = _cam0_cache()
    real = RealCameraSignalProvider(
        frames_root=FRAMES_ROOT,
        detections={0: parse_detections_file(str(DETECTIONS_DIR / "cam0.txt"))},
        cameras=[0],
        encoder=encoder,
        latent_dim=LATENT_DIM,
        start_ref_frame=1,
        end_ref_frame=3,
    )
    cached = CachedSignalProvider(cache, encoder, LATENT_DIM)

    real_signals, cached_signals = [real.reset()], [cached.reset()]
    for _ in range(2):
        real_step, real_done = real.step()
        cached_step, cached_done = cached.step()
        assert real_done == cached_done
        real_signals.append(real_step)
        cached_signals.append(cached_step)

    for real_step, cached_step in zip(real_signals, cached_signals):
        assert cached_step[0].visible == real_step[0].visible
        assert cached_step[0].pixel == real_step[0].pixel
        np.testing.assert_allclose(cached_step[0].latent, real_step[0].latent, atol=1e-5)


def test_cache_marks_non_recording_camera_as_not_visible():
    # cam3 has stopped recording by cam0 frame 33875 (verified in test_real_sync.py);
    # no frame files are read for it.
    cache = build_signal_cache(
        FRAMES_ROOT, DETECTIONS_DIR, cameras=[3], ref_start=33875, ref_end=33876, progress_every=0
    )
    assert not cache.recording.any()
    assert not cache.visible.any()
    assert not cache.crops.any()


def test_cache_save_load_roundtrip(tmp_path):
    cache = _cam0_cache()
    path = tmp_path / "cache.npz"
    cache.save(path)
    loaded = SignalCache.load(path)
    assert loaded.cameras == cache.cameras
    assert (loaded.ref_start, loaded.ref_end) == (cache.ref_start, cache.ref_end)
    np.testing.assert_array_equal(loaded.crops, cache.crops)
    np.testing.assert_array_equal(loaded.visible, cache.visible)
    np.testing.assert_array_equal(loaded.mask_pixel, cache.mask_pixel)


def test_observation_diagnostics_reports_per_camera_counts():
    diagnostics = observation_diagnostics(_cam0_cache())
    cam0 = diagnostics["0"]
    assert cam0["recording_rate"] == 1.0
    assert cam0["gt_visible_rate"] == 1.0
    assert cam0["tp"] + cam0["fn"] == 3  # all three frames are ground-truth visible


def test_run_experiment_writes_all_outputs_and_zip(tmp_path):
    cache_path = tmp_path / "train.npz"
    _cam0_cache().save(cache_path)
    out_dir = tmp_path / "run"

    results = run_experiment(
        ExperimentConfig(
            train_cache=str(cache_path),
            eval_cache=str(cache_path),
            out_dir=str(out_dir),
            timesteps=64,
            seeds=[0],
            latent_dim=LATENT_DIM,
            encoder_epochs=1,
            ppo_n_steps=64,
            ppo_batch_size=64,
            random_seeds=2,
        )
    )

    for name in ("results.json", "summary.txt", "run.log"):
        assert (out_dir / name).exists()
    for name in ("ppo_model.zip", "encoder.pt", "trace_train.csv", "trace_eval.csv", "progress.csv"):
        assert (out_dir / "seed_0" / name).exists()
    assert (tmp_path / "run.zip").exists()

    saved = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    assert saved["ppo"]["0"]["splits"]["train"]["visible_rate"] == 1.0  # single always-visible camera
    assert results["baselines"]["train"]["oracle"]["visible_rate"] == 1.0


def test_slice_returns_matching_subrange():
    cache = _cam0_cache()
    part = cache.slice(2, 3)
    assert (part.ref_start, part.ref_end, part.num_frames) == (2, 3, 2)
    np.testing.assert_array_equal(part.crops, cache.crops[1:3])
    np.testing.assert_array_equal(part.visible, cache.visible[1:3])


def test_slice_rejects_range_outside_cache():
    import pytest

    with pytest.raises(ValueError):
        _cam0_cache().slice(0, 3)


def test_rotating_provider_alternates_sources_per_episode():
    from dronetrackingrl.real_data.experiment import RotatingProvider

    cache = _cam0_cache()
    first, second = cache.slice(1, 2), cache.slice(2, 3)
    rotating = RotatingProvider(
        [CachedSignalProvider(first, None, LATENT_DIM), CachedSignalProvider(second, None, LATENT_DIM)]
    )

    rotating.reset()
    assert rotating.step()[1] is True  # first source: 2 frames -> done after one step
    rotating.reset()
    assert rotating._current is rotating.providers[1]
    rotating.reset()
    assert rotating._current is rotating.providers[0]  # wraps around


def test_pad_cameras_adds_invisible_ghost_cameras():
    cache = _cam0_cache()
    padded = cache.pad_cameras(3)
    assert padded.cameras == [0, -1, -2]
    assert padded.crops.shape == (3, 3, 64, 64)
    assert not padded.visible[:, 1:].any() and not padded.recording[:, 1:].any()
    assert np.isnan(padded.mask_pixel[:, 1:]).all()
    np.testing.assert_array_equal(padded.crops[:, 0], cache.crops[:, 0])
    assert cache.pad_cameras(1) is cache


def test_per_camera_parts_merge_into_the_serial_cache():
    serial = build_signal_cache(
        FRAMES_ROOT, DETECTIONS_DIR, cameras=[0, 3], ref_start=1, ref_end=3, progress_every=0
    )
    from dronetrackingrl.real_data.experiment import merge_camera_caches

    parts = [
        build_signal_cache(FRAMES_ROOT, DETECTIONS_DIR, cameras=[c], ref_start=1, ref_end=3, progress_every=0)
        for c in (0, 3)
    ]
    merged = merge_camera_caches(parts)
    assert merged.cameras == serial.cameras
    for field_name in ("crops", "recording", "visible"):
        np.testing.assert_array_equal(getattr(merged, field_name), getattr(serial, field_name))
    np.testing.assert_array_equal(merged.mask_pixel, serial.mask_pixel)
    np.testing.assert_array_equal(merged.gt_pixel, serial.gt_pixel)


def test_parallel_builder_matches_serial(tmp_path):
    from dronetrackingrl.real_data.experiment import build_signal_cache_parallel

    serial = build_signal_cache(
        FRAMES_ROOT, DETECTIONS_DIR, cameras=[0, 3], ref_start=1, ref_end=3, progress_every=0
    )
    parallel = build_signal_cache_parallel(
        FRAMES_ROOT, DETECTIONS_DIR, [0, 3], 1, 3, sync_name="dataset3", workers=2
    )
    np.testing.assert_array_equal(parallel.crops, serial.crops)
    np.testing.assert_array_equal(parallel.visible, serial.visible)


def test_run_experiment_trains_across_sources_with_different_camera_counts(tmp_path):
    def synthetic(cameras, seed):
        n, k = 30, len(cameras)
        rng = np.random.default_rng(seed)
        return SignalCache(
            cameras=cameras, reference_camera=0, ref_start=0, ref_end=n - 1, crop_size=64,
            crops=np.zeros((n, k, 64, 64), dtype=np.uint8), recording=np.ones((n, k), dtype=bool),
            visible=rng.random((n, k)) > 0.4,
            mask_pixel=np.full((n, k, 2), np.nan), gt_pixel=np.full((n, k, 2), np.nan),
        )

    six, seven = tmp_path / "six.npz", tmp_path / "seven.npz"
    synthetic(list(range(6)), 0).save(six)
    synthetic(list(range(7)), 1).save(seven)

    results = run_experiment(
        ExperimentConfig(
            train_cache=str(six), eval_cache=str(seven), out_dir=str(tmp_path / "run"),
            timesteps=64, seeds=[0], latent_dim=LATENT_DIM, encoder_epochs=1,
            ppo_n_steps=64, ppo_batch_size=64, random_seeds=1, pack=False,
        )
    )

    assert results["splits"]["train"]["cameras"] == list(range(6))
    assert results["splits"]["eval"]["cameras"] == list(range(7))
    # baselines see only the real cameras; PPO may also "see" ghost cameras
    assert set(results["baselines"]["train"]["fixed"]) == {str(c) for c in range(6)}
    assert set(results["ppo"]["0"]["splits"]["train"]["camera_share"]) == {"0", "1", "2", "3", "4", "5", "-1"}  # 6 real + 1 ghost


def test_parallel_builder_saves_parts_and_resumes_without_reading_frames(tmp_path):
    from dronetrackingrl.real_data.experiment import build_signal_cache_parallel

    parts = tmp_path / "parts"
    first = build_signal_cache_parallel(
        FRAMES_ROOT, DETECTIONS_DIR, [0, 3], 1, 3, sync_name="dataset3", workers=2, parts_dir=parts
    )
    assert len(list(parts.glob("*.npz"))) == 2  # one finished part per camera

    # Interrupted-run resume: the frames are gone, yet finished parts are reused as-is.
    resumed = build_signal_cache_parallel(
        tmp_path / "missing_frames", DETECTIONS_DIR, [0, 3], 1, 3, sync_name="dataset3", workers=1, parts_dir=parts
    )
    np.testing.assert_array_equal(resumed.crops, first.crops)
    np.testing.assert_array_equal(resumed.visible, first.visible)


def test_extra_eval_sources_are_reported_and_never_trained_on(tmp_path):
    cache_path = tmp_path / "c.npz"
    _cam0_cache().save(cache_path)
    results = run_experiment(
        ExperimentConfig(
            train_cache=str(cache_path), train_range=(1, 2),
            extra_eval=[(str(cache_path), 2, 3)], out_dir=str(tmp_path / "run"),
            timesteps=64, seeds=[0], latent_dim=LATENT_DIM, encoder_epochs=1,
            ppo_n_steps=64, ppo_batch_size=64, random_seeds=1, pack=False,
        )
    )
    assert set(results["splits"]) == {"train", "eval_extra1"}
    assert results["splits"]["eval_extra1"]["ref_start"] == 2
