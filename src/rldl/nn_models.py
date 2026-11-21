"""PyTorch 标准写法：用 nn.Module 搭一个二分类 MLP。"""

from __future__ import annotations

import torch
from torch import nn


class BinaryMLP(nn.Module):
    """2 → hidden → hidden → 1，输出 logit。"""

    def __init__(self, in_features: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (batch, 2)，输出 (batch, 1)。"""
        return self.net(x)
