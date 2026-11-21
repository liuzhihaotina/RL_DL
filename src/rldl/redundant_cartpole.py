"""把 CartPole 的动作空间扩展成带冗余动作的版本。

例如 n_actions=4 时：
  0, 2 → 向左推
  1, 3 → 向右推

这样多个动作的 Q 值完全相同，用来检验 Dueling 的 V/A 分解是否有优势。
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


class RedundantActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_actions: int = 4) -> None:
        super().__init__(env)
        self.n_base_actions = env.action_space.n
        self.action_space = spaces.Discrete(n_actions)

    def action(self, action: int) -> int:
        return int(action) % self.n_base_actions
