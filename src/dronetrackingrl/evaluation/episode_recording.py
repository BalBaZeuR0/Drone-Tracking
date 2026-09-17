from typing import List, Optional, Tuple

from dronetrackingrl.rl_env.camera_switch_env import CameraSwitchEnv


def record_episode(env: CameraSwitchEnv, policy) -> Tuple[List[Optional[Tuple[float, float]]], List[int], List[bool]]:
    """Runs one episode, capturing the chosen camera's pixel BEFORE each step
    advances the env — matching CameraSwitchEnv's reward-timing contract, where
    `info` at call time describes the frame the next action will be scored
    against, not the one just acted on."""
    pixels: List[Optional[Tuple[float, float]]] = []
    camera_indices: List[int] = []
    visible_flags: List[bool] = []

    _, info = env.reset()
    terminated = False
    while not terminated:
        action = policy.select_action(info)
        pixels.append(info["pixel"][action])
        camera_indices.append(action)
        visible_flags.append(info["visible"][action])
        _, _, terminated, truncated, info = env.step(action)
        if truncated:
            terminated = True

    return pixels, camera_indices, visible_flags
