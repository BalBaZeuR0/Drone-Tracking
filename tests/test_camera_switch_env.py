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
    assert terminated is False  # no genuine terminal state in this env
    assert truncated is True  # running out of frames is time-limit truncation
