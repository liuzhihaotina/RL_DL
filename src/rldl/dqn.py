"""DQN 的基础组件：Q 网络、经验回放、ε 衰减。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """输入状态，输出每个动作的 Q 值。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """固定容量的环形缓冲区，存 (s, a, r, s', done)。"""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.rng.integers(0, self.size, batch_size)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }


class DuelingQNetwork(nn.Module):
    """Dueling 结构：Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)。

    减去优势的均值是为了让 V 和 A 的分解唯一（否则同时给 V 加常数、
    给 A 减常数，Q 不变）。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    """线性衰减：step=0 时 start，step>=decay_steps 时 end。"""
    if decay_steps <= 0:
        return end
    fraction = min(1.0, step / decay_steps)
    return start + fraction * (end - start)
