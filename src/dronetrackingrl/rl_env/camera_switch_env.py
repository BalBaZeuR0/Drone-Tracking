from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dronetrackingrl.rl_env.signal_provider import CameraSignalProvider

# Kendi geçmişi (own-history) özellikleri, açıldığında her kameranın latent'ine
# eklenir: [bu kamerayı bir önceki adımda seçtim mi, kaç adımdır bu kamerada
# kaldım (normalize)]. Amaç: ajana kendi son kararının bilgisini vermek, böylece
# dedektörün kare kare titremesine (jitter) her seferinde tepki vermek yerine
# bir tür süreklilik/atalet (hysteresis) öğrenebilsin.
OWN_HISTORY_DIM = 2
STREAK_CAP = 50  # bu adım sayısının üstünde normalize streak 1.0'da sabitlenir


class CameraSwitchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        signal_provider: CameraSignalProvider,
        switch_penalty: float = 0.0,
        include_own_history: bool = False,
    ):
        super().__init__()
        self.provider = signal_provider
        self.switch_penalty = switch_penalty
        self.include_own_history = include_own_history
        self.num_cameras = signal_provider.num_cameras
        self.latent_dim = signal_provider.latent_dim + (OWN_HISTORY_DIM if include_own_history else 0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_cameras * self.latent_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_cameras)
        self._current_signals = None
        self._last_action: Optional[int] = None
        self._streak = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        signals = self.provider.reset()
        self._current_signals = signals
        self._last_action = None
        self._streak = 0
        return self._signals_to_obs(signals), self._signals_to_info(signals)

    def step(self, action: int):
        """Scores `action` against the frame already observed, then advances.

        Reward is computed against `self._current_signals` — the signals set by
        the prior `reset()`/`step()` call — BEFORE the provider is advanced. The
        `obs`/`info` this call returns describe the NEXT frame, not the one just
        scored; a caller reading `info["pixel"][action]` immediately after this
        call is reading the upcoming frame's pixel, not the one that earned
        `reward`.

        There is no genuine terminal state in this env: a provider running out
        of frames is a time-limit condition, not a state the agent "loses" from.
        That's why `done` maps to `truncated` and `terminated` is always False.
        """
        reward = 1.0 if self._current_signals[action].visible else -1.0
        if self._last_action is not None and action != self._last_action:
            reward -= self.switch_penalty
        self._streak = self._streak + 1 if action == self._last_action else 1
        self._last_action = action

        signals, done = self.provider.step()
        self._current_signals = signals
        obs = self._signals_to_obs(signals)
        info = self._signals_to_info(signals)
        return obs, reward, False, done, info

    def _signals_to_obs(self, signals):
        latents = [signal.latent for signal in signals]
        if not self.include_own_history:
            return np.concatenate(latents).astype(np.float32)
        streak_norm = min(self._streak, STREAK_CAP) / STREAK_CAP
        per_camera = [
            np.concatenate([latent, [1.0, streak_norm] if camera == self._last_action else [0.0, 0.0]])
            for camera, latent in enumerate(latents)
        ]
        return np.concatenate(per_camera).astype(np.float32)

    def _signals_to_info(self, signals):
        return {
            "visible": [signal.visible for signal in signals],
            "pixel": [signal.pixel for signal in signals],
        }
