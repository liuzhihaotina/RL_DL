"""手写 MLP：只用 torch.Tensor + autograd，不调用 torch.nn。

目的是把「层、参数初始化、前向、损失、反向、参数更新」全部摊开看清楚。
下一阶段再对比 torch.nn.Module 的写法（见 docs/roadmap.md）。
"""

from __future__ import annotations

import math

import torch


class Linear:
    """y = x @ W + b，参数用 Kaiming 均匀初始化。"""

    def __init__(self, in_features: int, out_features: int, *, generator: torch.Generator) -> None:
        bound = 1.0 / math.sqrt(in_features)
        self.weight = torch.empty(in_features, out_features, requires_grad=True)
        self.bias = torch.empty(out_features, requires_grad=True)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound, generator=generator)
            self.bias.uniform_(-bound, bound, generator=generator)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight, self.bias]


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


class MLP:
    """2 -> hidden -> hidden -> 1 的二分类网络。"""

    def __init__(
        self,
        in_features: int = 2,
        hidden: int = 32,
        *,
        seed: int = 0,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = Linear(in_features, hidden, generator=gen)
        self.fc2 = Linear(hidden, hidden, generator=gen)
        self.fc3 = Linear(hidden, 1, generator=gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = relu(self.fc1(x))
        h = relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)  # logits, (batch,)

    def parameters(self) -> list[torch.Tensor]:
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "fc1.weight": self.fc1.weight.detach().cpu(),
            "fc1.bias": self.fc1.bias.detach().cpu(),
            "fc2.weight": self.fc2.weight.detach().cpu(),
            "fc2.bias": self.fc2.bias.detach().cpu(),
            "fc3.weight": self.fc3.weight.detach().cpu(),
            "fc3.bias": self.fc3.bias.detach().cpu(),
        }
