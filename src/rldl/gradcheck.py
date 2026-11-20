"""数值梯度：用中心差分估计导数，用来检查 autograd 的结果。

只适合小模型 / 抽查，因为每个参数都要额外做两次前向。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch


def numerical_gradients(
    params: Iterable[torch.Tensor],
    loss_fn: Callable[[], torch.Tensor],
    eps: float = 1e-5,
) -> list[torch.Tensor]:
    """对每个参数逐元素做中心差分：(L(w+ε) - L(w-ε)) / (2ε)。

    Args:
        params: 需要检查的参数（叶子张量，float64 最稳）
        loss_fn: 无参数函数，返回标量 loss（只做前向，不 backward）
        eps: 扰动大小
    """
    grads = []
    for p in params:
        flat = p.data.view(-1)
        g = torch.zeros_like(flat)
        for i in range(flat.numel()):
            original = flat[i].item()

            flat[i] = original + eps
            loss_plus = loss_fn().item()

            flat[i] = original - eps
            loss_minus = loss_fn().item()

            flat[i] = original
            g[i] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(g.view_as(p))
    return grads


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """逐元素相对误差的最大值；分母加 clamp 防止除以 0。"""
    denom = (a.abs() + b.abs()).clamp_min(1e-12)
    return ((a - b).abs() / denom).max().item()
