"""自建 gymnasium 环境：二维点质量到达原点。

状态 (4)：
  [x, y, vx, vy]
动作 (2)：
  [fx, fy]，范围 [-1, 1]
动力学：
  v = 0.9 v + 0.1 a
  p = p + v
奖励：
  -||p|| - 0.01 ||a||²
终止：
  ||p|| < 0.05
截断：
  100 步
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointMassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 100, dt: float = 0.1, damping: float = 0.9) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.damping = damping

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        self.vel = self.damping * self.vel + self.dt * action
        self.pos = self.pos + self.vel
        self.steps += 1

        distance = float(np.linalg.norm(self.pos))
        reward = -distance - 0.01 * float(np.dot(action, action))
        terminated = distance < 0.05
        truncated = self.steps >= self.max_steps
        info = {"distance": distance}
        return self._obs(), reward, terminated, truncated, info
