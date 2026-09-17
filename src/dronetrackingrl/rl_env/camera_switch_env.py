from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dronetrackingrl.rl_env.signal_provider import CameraSignalProvider


class CameraSwitchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, signal_provider: CameraSignalProvider, switch_penalty: float = 0.0):
        super().__init__()
        self.provider = signal_provider
        self.switch_penalty = switch_penalty
        self.num_cameras = signal_provider.num_cameras
        self.latent_dim = signal_provider.latent_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_cameras * self.latent_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_cameras)
        self._current_signals = None
        self._last_action: Optional[int] = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        signals = self.provider.reset()
        self._current_signals = signals
        self._last_action = None
        return self._signals_to_obs(signals), self._signals_to_info(signals)

    def step(self, action: int):
        reward = 1.0 if self._current_signals[action].visible else -1.0
        if self._last_action is not None and action != self._last_action:
            reward -= self.switch_penalty
        self._last_action = action

        signals, done = self.provider.step()
        self._current_signals = signals
        obs = self._signals_to_obs(signals)
        info = self._signals_to_info(signals)
        return obs, reward, done, False, info

    def _signals_to_obs(self, signals):
        return np.concatenate([signal.latent for signal in signals]).astype(np.float32)

    def _signals_to_info(self, signals):
        return {
            "visible": [signal.visible for signal in signals],
            "pixel": [signal.pixel for signal in signals],
        }
