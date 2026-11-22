"""PPO 用到的优势估计：GAE（Generalized Advantage Estimation）。"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (advantages, returns)。

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    A_t = δ_t + γλ A_{t+1}
    return_t = A_t + V(s_t)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    length = len(rewards)

    advantages = np.zeros(length, dtype=np.float32)
    last = 0.0
    for t in reversed(range(length)):
        if dones[t]:
            delta = rewards[t] - values[t]
            last = 0.0
        else:
            next_value = values[t + 1] if t + 1 < length else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last

    returns = advantages + values
    return advantages, returns



def compute_gae_bootstrap(
    rewards: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """处理截断回合（没有真正终止）的 GAE。

    最后一步用 last_value = V(s_{T}) 自举，而不是把 done 当成终止。
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    length = len(rewards)

    advantages = np.zeros(length, dtype=np.float32)
    last = 0.0
    next_value = float(last_value)
    for t in reversed(range(length)):
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
        next_value = values[t]

    returns = advantages + values
    return advantages, returns
