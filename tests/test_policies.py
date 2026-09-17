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
