"""手写优化器：SGD（可选 momentum）与 Adam。

只用 Tensor 运算实现，不调用 torch.optim，便于看清每一步在改什么。
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


class SGD:
    """随机梯度下降。

    momentum=0 时：w = w - lr * grad
    momentum>0 时：v = momentum * v + grad; w = w - lr * v
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.1,
        momentum: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum > 0:
                self.velocity[i].mul_(self.momentum).add_(p.grad)
                p.add_(self.velocity[i], alpha=-self.lr)
            else:
                p.add_(p.grad, alpha=-self.lr)


class Adam:
    """Adam：一阶矩 + 二阶矩 + 偏差修正 + 逐参数步长。

    对应论文 "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)。
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            self.m[i].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)
