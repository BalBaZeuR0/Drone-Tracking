from dronetrackingrl.baselines.policies import OraclePolicy
from dronetrackingrl.evaluation.episode_recording import record_episode
from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv
from fakes import FakeCameraSignalProvider


def test_record_episode_captures_pre_step_pixel_for_chosen_camera():
    # FakeCameraSignalProvider's script (see tests/fakes.py): camera 0 visible
    # at frame 0, camera 1 visible at frames 1-2. Its pixel for camera `c` at
    # frame `f` is always (float(c), float(f)).
    #
    # CameraSwitchEnv only ever scores/acts on frames 0 and 1 (the episode
    # truncates once the provider reports `done` on the step that reveals
    # frame 2 — frame 2 itself is observed only in the final `info`, never
    # acted on). OraclePolicy always picks the currently-visible camera:
    #   - at frame 0: camera 0 is visible -> action 0, pixel (0.0, 0.0)
    #   - at frame 1: camera 1 is visible -> action 1, pixel (1.0, 1.0)
    #
    # If record_episode captured pixels AFTER stepping (the off-by-one bug it
    # exists to avoid), the first entry would instead be camera 0's frame-1
    # pixel, (0.0, 1.0) -- wrong.
    env = CameraSwitchEnv(FakeCameraSignalProvider())
    policy = OraclePolicy()

    pixels, camera_indices, visible_flags = record_episode(env, policy)

    assert camera_indices == [0, 1]
    assert pixels == [(0.0, 0.0), (1.0, 1.0)]
    assert visible_flags == [True, True]
