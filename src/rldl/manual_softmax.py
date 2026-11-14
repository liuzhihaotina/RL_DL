"""手写 softmax、交叉熵和多分类 MLP（只用 Tensor + autograd）。

二分类用 sigmoid + BCE；多分类用 softmax + 交叉熵，本质上是同一件事的推广。
"""

from __future__ import annotations

import torch

from rldl.manual_nn import Linear, relu


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """数值稳定的 log(softmax(x))。

    先减去每行最大值，指数不会溢出；数学上结果完全等价。
    """
    shifted = logits - logits.max(dim=1, keepdim=True).values
    return shifted - torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """把每行分数变成概率：非负、和为 1。"""
    return torch.exp(log_softmax(logits))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """多分类交叉熵（均值）：-log p[正确类别]。

    Args:
        logits: (batch, n_classes) 未归一化分数
        targets: (batch,) int64 类别下标 0..n_classes-1
    """
    log_probs = log_softmax(logits)
    picked = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -picked.mean()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


class MLPClassifier:
    """in -> hidden -> hidden -> n_classes，输出 logits (batch, n_classes)。"""

    def __init__(
        self,
        in_features: int,
        hidden: int,
        n_classes: int,
        *,
        seed: int = 0,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = Linear(in_features, hidden, generator=gen)
        self.fc2 = Linear(hidden, hidden, generator=gen)
        self.fc3 = Linear(hidden, n_classes, generator=gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = relu(self.fc1(x))
        h = relu(self.fc2(h))
        return self.fc3(h)

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
