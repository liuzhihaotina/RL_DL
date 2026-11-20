"""实验 08：数值梯度检查——验证 autograd 算的梯度是否正确。

思路：
  解析梯度 = loss.backward() 得到的 p.grad
  数值梯度 = [L(w+ε) - L(w-ε)] / (2ε)   （中心差分）

两者应该几乎相等。数值梯度很慢（每个参数要 2 次前向），只适合检查。

用法：
  uv run python scripts/08_numerical_gradient.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.gradcheck import numerical_gradients, relative_error  # noqa: E402
from rldl.manual_softmax import MLPClassifier, cross_entropy  # noqa: E402
from rldl.utils import set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def build_model(hidden: int = 8, seed: int = 0) -> MLPClassifier:
    model = MLPClassifier(2, hidden, 3, seed=seed)
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.double()
        layer.bias.data = layer.bias.data.double()
    return model


def make_batch(n: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=gen, dtype=torch.float64)
    y = torch.randint(0, 3, (n,), generator=gen)
    return X, y


def analytic_gradients(
    model: MLPClassifier, X: torch.Tensor, y: torch.Tensor
) -> list[torch.Tensor]:
    """一次反向传播就得到所有参数的梯度。"""
    loss = cross_entropy(model.forward(X), y)
    for p in model.parameters():
        p.grad = None
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def check_scalar_function(eps: float) -> None:
    """最简单的情况：L(w) = w³ + 2w²，在 w=1 处解析梯度是 7。"""
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    loss = w**3 + 2 * w**2
    loss.backward()

    numeric = ((1 + eps) ** 3 + 2 * (1 + eps) ** 2 - ((1 - eps) ** 3 + 2 * (1 - eps) ** 2)) / (
        2 * eps
    )
    print("一维例子：L(w) = w³ + 2w²，w = 1")
    print(f"  解析梯度（autograd） = {w.grad.item():.10f}")
    print(f"  数值梯度（eps={eps:g}） = {numeric:.10f}")
    print(f"  相对误差 = {abs(numeric - w.grad.item()) / abs(w.grad.item()):.2e}\n")


def check_model(model: MLPClassifier, X: torch.Tensor, y: torch.Tensor, eps: float) -> None:
    analytic = analytic_gradients(model, X, y)
    numeric = numerical_gradients(
        model.parameters(), lambda: cross_entropy(model.forward(X), y), eps
    )

    errors = [relative_error(a, n) for a, n in zip(analytic, numeric, strict=True)]
    n_params = sum(a.numel() for a in analytic)
    print("三分类小网络：2 → 8 → 3，32 条样本")
    print(f"  检查参数总数 = {n_params}")
    print(f"  逐参数最大相对误差 = {max(errors):.2e}")
    print(f"  逐参数平均相对误差 = {np.mean(errors):.2e}\n")


def epsilon_sweep(
    model: MLPClassifier, X: torch.Tensor, y: torch.Tensor
) -> tuple[list[float], list[float]]:
    """固定一个参数，看 eps 太大/太小分别会怎样。"""
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # 解析梯度里取 fc1.weight[0,0] 的梯度
    analytic = analytic_gradients(model, X, y)
    target = model.fc1.weight
    true_grad = analytic[0][0, 0].item()

    errors = []
    flat = target.data.view(-1)
    original = flat[0].item()
    for eps in eps_list:
        flat[0] = original + eps
        loss_plus = cross_entropy(model.forward(X), y).item()
        flat[0] = original - eps
        loss_minus = cross_entropy(model.forward(X), y).item()
        flat[0] = original
        numeric = (loss_plus - loss_minus) / (2 * eps)
        errors.append(abs(numeric - true_grad) / max(abs(true_grad), 1e-12))

    print("同一个参数的梯度，随 eps 变化：")
    print(f"{'eps':>8} {'相对误差':>14}")
    for eps, err in zip(eps_list, errors, strict=True):
        print(f"{eps:>8.0e} {err:>14.2e}")
    return eps_list, errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/08_numerical_gradient"))
    args = p.parse_args()

    set_seed(args.seed)
    model = build_model(args.hidden, args.seed)
    X, y = make_batch(32, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    check_scalar_function(args.eps)
    check_model(model, X, y, 1e-4)
    print("注：eps=1e-4 时个别参数跨过了 ReLU 的折点，误差偏大。换小 eps 重试：\n")
    check_model(model, X, y, args.eps)
    eps_list, errors = epsilon_sweep(model, X, y)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(eps_list, errors, "o-", color="#4C78A8")
    ax.set_xlabel("eps")
    ax.set_ylabel("相对误差")
    ax.set_title("数值梯度的误差随 eps 变化（中心差分）")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out / "epsilon_sweep.png", dpi=150)
    print(f"\nsaved -> {args.out}/epsilon_sweep.png")


if __name__ == "__main__":
    main()
