"""视觉策略：CNN 提取特征，再接连续动作头和 V(s) 头。"""

from __future__ import annotations

import torch
from torch import nn


class VisualActorCritic(nn.Module):
    def __init__(self, action_dim: int, size: int = 48, hidden: int = 128) -> None:
        super().__init__()
        self.size = size
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        feature_dim = 32 * (size // 4) ** 2
        self.mu = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(feature_dim, 1)

        self._feature_dim = feature_dim
        self._hidden = hidden

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(self.features(x))

    def distribution(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu = self.forward(x)
        std = self.log_std.clamp(-20.0, 2.0).exp()
        return torch.distributions.Normal(mu, std)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(x)).squeeze(-1)
