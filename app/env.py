from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .feature_engineering import build_feature_report, risk_from_feature_report


class ExpertiseFraudEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_rows, max_steps: int = 5):
        super().__init__()
        self.data_rows = data_rows
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(3)  # PASS, FLAG, ASK_MORE
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.idx = -1
        self.t = 0
        self.current = None

    def _obs_from_row(self, row):
        fr = build_feature_report(row["profile_text"], row["answers"], row["web_signals"])
        return np.array([
            fr.timeline_anomaly,
            fr.answer_anomaly,
            fr.consistency_anomaly,
            fr.web_anomaly,
        ], dtype=np.float32), fr

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = (self.idx + 1) % len(self.data_rows)
        self.current = self.data_rows[self.idx]
        self.t = 0
        obs, _ = self._obs_from_row(self.current)
        return obs, {"candidate_id": self.current["candidate_id"]}

    def step(self, action):
        assert self.current is not None
        obs, fr = self._obs_from_row(self.current)
        label = int(self.current["label"])
        risk = risk_from_feature_report(fr)

        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps

        reward = 0.0
        if action == 1:  # FLAG
            reward = 3.0 if label == 1 else -1.0
            terminated = True
        elif action == 0:  # PASS
            reward = 1.0 if label == 0 else -5.0
            terminated = True
        elif action == 2:  # ASK_MORE
            reward = -0.2
            if self.t >= 2:
                reward += 0.25 * risk
        else:
            reward = -0.5

        info = {
            "label": label,
            "risk": float(risk),
            "feature_report": fr.model_dump(),
            "candidate_id": self.current["candidate_id"],
        }
        return obs, reward, terminated, truncated, info
