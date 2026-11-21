"""MNIST 用的小型卷积网络。"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Conv→BN→ReLU→Pool 两层，再接全连接分类头。

    输入 (N, 1, 28, 28)：
      28×28 --conv/pool--> 14×14 --conv/pool--> 7×7
      64 个 7×7 特征图展平 → 3136 → 128 → 10
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
