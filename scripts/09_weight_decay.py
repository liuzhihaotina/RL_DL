"""实验 09：过拟合与 weight decay。

场景：
  - 每类 30 个训练点（共 90），其中 20% 的标签被故意改错
  - 验证集每类 100 个（共 300），标签干净
  - 网络 2→128→128→3，约 1.7 万参数，足够把 90 个点连同错误标签一起背下来
  - 训练 6000 步，对比 weight decay = 0 与 3e-3

weight decay（L2 正则）：
  loss = 交叉熵 + 0.5 × wd × Σ w²
  梯度会多出一项 wd × w，把权重往 0 拉。

用法：
  uv run python scripts/09_weight_decay.py
  uv run python scripts/09_weight_decay.py --steps 8000 --wd 5e-3
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

from rldl.manual_optim import SGD  # noqa: E402
from rldl.manual_softmax import (  # noqa: E402
    MLPClassifier,
    accuracy,
    cross_entropy,
)
from rldl.softmax_data import make_spiral_3class  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
CLASS_LIGHT = ["#AEC7E8", "#FFBB78", "#98DF8A"]


def make_dataset(
    n_train_per_class: int,
    flip_ratio: float,
    noise: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 Xtr, ytr_noisy, flip_mask, Xva, yva。"""
    n_per_class = 130
    X, y = make_spiral_3class(n_per_class, noise=noise, turns=1.0, seed=0)

    train_idx, val_idx = [], []
    for c in range(3):
        idx = np.where(y == c)[0]
        train_idx.append(idx[:n_train_per_class])
        val_idx.append(idx[n_train_per_class:n_per_class])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)

    Xtr, ytr = X[train_idx], y[train_idx].copy()
    Xva, yva = X[val_idx], y[val_idx]

    rng = np.random.default_rng(seed)
    n_flip = int(len(ytr) * flip_ratio)
    flip_idx = rng.choice(len(ytr), n_flip, replace=False)
    ytr[flip_idx] = (ytr[flip_idx] + rng.integers(1, 3, n_flip)) % 3

    flip_mask = np.zeros(len(ytr), dtype=bool)
    flip_mask[flip_idx] = True
    return Xtr, ytr, flip_mask, Xva, yva


def move_to(model: MLPClassifier, device: torch.device) -> None:
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.to(device)
        layer.bias.data = layer.bias.data.to(device)


def evaluate(model, X, y) -> tuple[float, float]:
    with torch.no_grad():
        logits = model.forward(X)
        return float(cross_entropy(logits, y).item()), accuracy(logits, y)


def train_run(
    wd: float,
    seed: int,
    args: argparse.Namespace,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, MLPClassifier]:
    set_seed(seed)
    model = MLPClassifier(2, args.hidden, 3, seed=seed)
    move_to(model, device)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    weights = [model.fc1.weight, model.fc2.weight, model.fc3.weight]
    n = len(Xtr)
    history: list[tuple[int, float, float, float, float, float]] = []

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        loss = cross_entropy(model.forward(Xtr[idx]), ytr[idx])
        if wd > 0:
            reg = sum((w**2).sum() for w in weights)
            loss = loss + 0.5 * wd * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            train_loss, train_acc = evaluate(model, Xtr, ytr)
            val_loss, val_acc = evaluate(model, Xva, yva)
            weight_norm = float(sum((w**2).sum().item() for w in weights))
            history.append(
                (step, train_loss, val_loss, train_acc, val_acc, weight_norm)
            )

    return np.asarray(history), model


def plot_curves(
    results: dict[float, np.ndarray],
    stds: dict[float, np.ndarray],
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {0.0: "#8C8C8C", 3e-3: "#4C78A8"}
    steps = next(iter(results.values()))[:, 0]

    for wd, history in results.items():
        color = colors.get(wd, "#4C78A8")
        label = "无 weight decay" if wd == 0 else f"weight decay={wd:g}"
        std = stds[wd]

        axes[0].plot(steps, history[:, 3], color=color, linestyle="--", label=f"{label} / 训练")
        axes[0].plot(steps, history[:, 4], color=color, label=f"{label} / 验证")
        axes[0].fill_between(
            steps, history[:, 4] - std[:, 4], history[:, 4] + std[:, 4], color=color, alpha=0.15
        )

        axes[1].plot(steps, history[:, 2], color=color, label=label)
        axes[2].plot(steps, history[:, 5], color=color, label=label)

    axes[0].set_title("准确率：实线验证集，虚线训练集")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.3, 1.02)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("验证损失")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("cross entropy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].set_title("权重平方和 Σw²")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("||w||²")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"90 个训练点（20% 标签被改错），1.7 万参数网络，{int(steps[-1])} 步", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_boundaries(
    models: dict[float, MLPClassifier],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    flip_mask: np.ndarray,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    out: Path,
) -> None:
    pad = 0.25
    x_min, x_max = Xtr[:, 0].min() - pad, Xtr[:, 0].max() + pad
    y_min, y_max = Xtr[:, 1].min() - pad, Xtr[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (wd, model) in zip(axes, models.items(), strict=True):
        device = model.fc1.weight.device
        with torch.no_grad():
            pred = (
                model.forward(grid.to(device)).argmax(dim=1).cpu().numpy().reshape(gx.shape)
            )
            val_acc = accuracy(model.forward(Xva), yva)

        ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5, 2.5], colors=CLASS_LIGHT, alpha=0.6)
        for c in range(3):
            mask_c = ytr == c
            ax.scatter(
                Xtr[mask_c & ~flip_mask, 0],
                Xtr[mask_c & ~flip_mask, 1],
                s=40,
                color=CLASS_COLORS[c],
                edgecolor="white",
                linewidth=0.6,
                label=f"类别 {c}",
            )
        ax.scatter(
            Xtr[flip_mask, 0],
            Xtr[flip_mask, 1],
            s=60,
            facecolor="none",
            edgecolor="black",
            linewidth=1.2,
            marker="x",
            label="标签被改错",
        )
        title = "无 weight decay" if wd == 0 else f"weight decay={wd:g}"
        ax.set_title(f"{title}（验证准确率 {val_acc:.3f}）")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("训练点只有 90 个：不加正则的边界会去迁就错误标签", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-train-per-class", type=int, default=30)
    p.add_argument("--flip", type=float, default=0.2, help="训练标签被改错的比例")
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/09_weight_decay"))
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device = {device}")

    Xtr_np, ytr_np, flip_mask, Xva_np, yva_np = make_dataset(
        args.n_train_per_class, args.flip, args.noise
    )
    print(
        f"train = {len(Xtr_np)}（其中 {flip_mask.sum()} 个标签被改错），"
        f"val = {len(Xva_np)}"
    )

    Xtr = torch.from_numpy(Xtr_np).to(device)
    ytr = torch.from_numpy(ytr_np).to(device)
    Xva = torch.from_numpy(Xva_np).to(device)
    yva = torch.from_numpy(yva_np).to(device)

    wds = [0.0, args.wd]
    histories: dict[float, list[np.ndarray]] = {wd: [] for wd in wds}
    models: dict[float, MLPClassifier] = {}

    for wd in wds:
        for seed in range(args.seeds):
            history, model = train_run(wd, seed, args, Xtr, ytr, Xva, yva, device)
            histories[wd].append(history)
            if seed == 0:
                models[wd] = model
            print(
                f"[wd={wd:g} seed={seed}] train_acc={history[-1, 3]:.3f} "
                f"val_acc={history[-1, 4]:.3f} ||w||²={history[-1, 5]:.0f}"
            )

    mean = {wd: np.stack(histories[wd]).mean(axis=0) for wd in wds}
    std = {wd: np.stack(histories[wd]).std(axis=0) for wd in wds}

    print("\n最终结果（3 个种子均值）：")
    print(f"{'weight decay':<14}{'train_acc':>12}{'val_acc':>12}{'||w||²':>12}")
    for wd in wds:
        print(
            f"{wd:<14g}{mean[wd][-1, 3]:>12.3f}{mean[wd][-1, 4]:>12.3f}"
            f"{mean[wd][-1, 5]:>12.0f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(mean, std, args.out)
    plot_boundaries(models, Xtr_np, ytr_np, flip_mask, Xva, yva, args.out)

    np.savez(
        args.out / "history.npz",
        steps=mean[wds[0]][:, 0],
        **{f"wd{wd:g}": mean[wd] for wd in wds},
        **{f"wd{wd:g}_std": std[wd] for wd in wds},
    )
    print(f"\nsaved -> {args.out}/curves.png, boundary.png, history.npz")


if __name__ == "__main__":
    main()
