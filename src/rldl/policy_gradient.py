"""策略梯度基础组件：策略网络与 REINFORCE 损失。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ValueNetwork(nn.Module):
    """状态价值 V(s)，用作 REINFORCE 的 baseline。"""

    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class PolicyNetwork(nn.Module):
    """输入状态，输出每个动作的 logit。"""

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

    def distribution(self, state: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(state))


def compute_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """从后往前计算折扣回报 G_t。"""
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def reinforce_loss(
    log_probs: list[torch.Tensor],
    returns: np.ndarray,
    *,
    values: torch.Tensor | None = None,
    entropy: list[torch.Tensor] | None = None,
    normalize: bool = True,
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """REINFORCE 策略损失。

    - values=None：用原始回报（可标准化）
    - values=V(s)：用 advantage = G - V(s)，并做标准化
    - entropy_coef>0：减去熵奖励，防止策略过早塌缩成确定动作
    """
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    if values is not None:
        advantage = returns_t - values.detach()
    else:
        advantage = returns_t

    if normalize and len(advantage) > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    loss = -(torch.stack(log_probs) * advantage).mean()
    if entropy is not None and entropy_coef > 0:
        loss = loss - entropy_coef * torch.stack(entropy).mean()
    return loss



def n_step_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    n: int,
) -> np.ndarray:
    """n 步自举回报：前 n 步用真实奖励，之后用 V(s_{t+n}) 自举。

    如果窗口内遇到终止，就不再加 bootstrap。
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    length = len(rewards)
    out = np.zeros(length, dtype=np.float32)

    for t in range(length):
        total = 0.0
        discount = 1.0
        bootstrapped = False
        for k in range(n):
            if t + k >= length:
                break
            total += discount * rewards[t + k]
            discount *= gamma
            if dones[t + k]:
                bootstrapped = True
                break
        if not bootstrapped and t + n < length:
            total += discount * values[t + n]
        out[t] = total

    return out



class ContinuousPolicyNetwork(nn.Module):
    """连续动作策略：输出高斯分布的均值，标准差是可学习参数。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(self.feature(x))

    def distribution(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu = self.forward(x)
        std = self.log_std.clamp(-20.0, 2.0).exp()
        return torch.distributions.Normal(mu, std)
