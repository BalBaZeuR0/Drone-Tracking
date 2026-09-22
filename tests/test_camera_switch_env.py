import numpy as np

from dronetrackingrl.rl_env.camera_switch_env import STREAK_CAP, CameraSwitchEnv
from dronetrackingrl.rl_env.signal_provider import CameraStepSignal
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
    assert terminated is False  # no genuine terminal state in this env
    assert truncated is True  # running out of frames is time-limit truncation


def test_own_history_is_zero_before_any_action_is_taken():
    env = CameraSwitchEnv(FakeCameraSignalProvider(), include_own_history=True)
    obs, info = env.reset()
    assert obs.shape == (3 * 4,)  # (latent 2 + history 2) per camera, 3 cameras
    history = obs.reshape(3, 4)[:, 2:]
    np.testing.assert_array_equal(history, 0.0)


def test_own_history_flags_the_chosen_camera_and_grows_its_streak():
    env = CameraSwitchEnv(FakeCameraSignalProvider(), include_own_history=True)
    env.reset()
    obs, *_ = env.step(0)
    per_camera = obs.reshape(3, 4)
    np.testing.assert_array_equal(per_camera[:, 2], [1.0, 0.0, 0.0])  # only camera 0 flagged
    assert per_camera[0, 3] == 1.0 / STREAK_CAP

    obs, *_ = env.step(0)  # stays on camera 0 -> streak grows
    assert obs.reshape(3, 4)[0, 3] == 2.0 / STREAK_CAP


def test_own_history_streak_resets_on_switch():
    env = CameraSwitchEnv(FakeCameraSignalProvider(), include_own_history=True)
    env.reset()
    env.step(0)
    obs, *_ = env.step(1)  # switch to camera 1
    per_camera = obs.reshape(3, 4)
    np.testing.assert_array_equal(per_camera[:, 2], [0.0, 1.0, 0.0])
    assert per_camera[1, 3] == 1.0 / STREAK_CAP


class _SingleCameraFakeProvider:
    """Tek kameralı, çok karelik sahte sağlayıcı — streak tavanını test etmek için."""

    latent_dim = 1
    num_cameras = 1

    def __init__(self, num_frames: int):
        self._num_frames = num_frames
        self._frame_index = -1

    def reset(self):
        self._frame_index = 0
        return [CameraStepSignal(latent=np.array([0.0]), visible=True, pixel=None)]

    def step(self):
        self._frame_index += 1
        done = self._frame_index >= self._num_frames - 1
        return [CameraStepSignal(latent=np.array([0.0]), visible=True, pixel=None)], done


def test_own_history_streak_is_capped():
    env = CameraSwitchEnv(_SingleCameraFakeProvider(num_frames=STREAK_CAP + 10), include_own_history=True)
    env.reset()
    obs = None
    for _ in range(STREAK_CAP + 5):
        obs, *_ = env.step(0)
    assert obs[2] == 1.0  # streak capped, normalized to 1.0 instead of growing past it
