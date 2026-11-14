"""实验 04：从零手写 softmax + 交叉熵，做三分类。

与实验 01 的对照：
  二分类：1 个 logit → sigmoid → BCE
  三分类：3 个 logit → softmax → 交叉熵

用法：
  uv run python scripts/04_softmax_3class.py
  uv run python scripts/04_softmax_3class.py --steps 3000 --hidden 64
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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_softmax import (  # noqa: E402
    MLPClassifier,
    accuracy,
    cross_entropy,
    softmax,
)
from rldl.softmax_data import make_spiral_3class, split_data  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
CLASS_LIGHT = ["#AEC7E8", "#FFBB78", "#98DF8A"]


def verify_ops(device: torch.device) -> None:
    """对照 PyTorch 官方实现，确认手写 softmax / 交叉熵没有写错。"""
    torch.manual_seed(0)
    logits = torch.randn(16, 3, device=device) * 5
    targets = torch.randint(0, 3, (16,), device=device)

    p_manual = softmax(logits)
    p_torch = torch.softmax(logits, dim=1)
    ce_manual = cross_entropy(logits, targets)
    ce_torch = F.cross_entropy(logits, targets)

    print("验证手写实现（与 PyTorch 对照）：")
    print(f"  softmax 最大误差 = {(p_manual - p_torch).abs().max().item():.2e}")
    print(f"  cross_entropy 误差 = {abs(ce_manual.item() - ce_torch.item()):.2e}")

    big = torch.tensor([[1000.0, 1001.0, 1002.0]], device=device)
    print(f"  大 logits 的稳定 softmax = {softmax(big).tolist()[0]}")
    print(f"  概率和 = {p_manual.sum(dim=1).mean().item():.6f}")


def plot_curves(history: list[tuple[int, float, float, float, float]], out: Path) -> None:
    steps = [h[0] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(steps, [h[1] for h in history], label="train loss")
    axes[0].plot(steps, [h[2] for h in history], label="val loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross entropy")
    axes[0].set_title("损失曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, [h[3] for h in history], label="train acc", color="#4C78A8")
    axes[1].plot(steps, [h[4] for h in history], label="val acc", color="#54A24B")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.3, 1.01)
    axes[1].set_title("准确率曲线")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)


def plot_boundary(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    out: Path,
) -> float:
    device = model.fc1.weight.device
    pad = 0.3
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    grid = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32))
    with torch.no_grad():
        pred = model.forward(grid.to(device)).argmax(dim=1).cpu().numpy().reshape(gx.shape)
        val_logits = model.forward(torch.from_numpy(Xva).to(device))
        val_acc = accuracy(val_logits, torch.from_numpy(yva).to(device))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.contourf(
        gx,
        gy,
        pred,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=CLASS_LIGHT,
        alpha=0.6,
    )
    for c in range(3):
        ax.scatter(
            X[y == c, 0],
            X[y == c, 1],
            s=14,
            color=CLASS_COLORS[c],
            edgecolor="white",
            linewidth=0.4,
            label=f"类别 {c}",
        )
    ax.set_title(f"三分类决策区域（验证准确率 {val_acc:.3f}）", fontsize=13)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return val_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--turns", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/04_softmax_3class"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")
    verify_ops(device)

    X, y = make_spiral_3class(
        n_per_class=args.per_class, noise=args.noise, turns=args.turns, seed=args.seed
    )
    X_tr, y_tr, X_val, y_val = split_data(X, y, val_ratio=0.2, seed=args.seed)
    print(f"data: train={len(X_tr)} val={len(X_val)} classes={np.bincount(y_tr).tolist()}")

    Xtr = torch.from_numpy(X_tr).to(device)
    ytr = torch.from_numpy(y_tr).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device)

    model = MLPClassifier(2, args.hidden, 3, seed=args.seed)
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.to(device)
        layer.bias.data = layer.bias.data.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    args.out.mkdir(parents=True, exist_ok=True)

    n = len(Xtr)
    history: list[tuple[int, float, float, float, float]] = []
    progress = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)

    for step in progress:
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        logits = model.forward(Xtr[idx])
        loss = cross_entropy(logits, ytr[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            with torch.no_grad():
                train_logits = model.forward(Xtr)
                val_logits = model.forward(Xva)
                train_loss = cross_entropy(train_logits, ytr).item()
                val_loss = cross_entropy(val_logits, yva).item()
                train_acc = accuracy(train_logits, ytr)
                val_acc = accuracy(val_logits, yva)
            history.append((step, train_loss, val_loss, train_acc, val_acc))
            progress.set_postfix(train_loss=f"{train_loss:.3f}", val_acc=f"{val_acc:.3f}")

    print(f"final: train_acc={history[-1][3]:.3f} val_acc={history[-1][4]:.3f}")
    print(f"final: train_loss={history[-1][1]:.4f} val_loss={history[-1][2]:.4f}")

    plot_curves(history, args.out)
    plot_boundary(model, X, y, X_val, y_val, args.out)

    torch.save(model.state_dict(), args.out / "model.pth")
    np.savez(args.out / "data.npz", X=X, y=y)
    print(f"saved -> {args.out}/curves.png, boundary.png, model.pth, data.npz")


if __name__ == "__main__":
    main()
