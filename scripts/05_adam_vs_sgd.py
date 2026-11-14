"""实验 05：手写 SGD / SGD+momentum / Adam 对比。

控制变量：
  - 同一份三分类螺旋数据、同一个 train/val 划分
  - 同一个初始模型（同一 seed）
  - 同样 1500 次参数更新、batch=64、hidden=64
  - 只改优化器（各自取合理的学习率）

用法：
  uv run python scripts/05_adam_vs_sgd.py
  uv run python scripts/05_adam_vs_sgd.py --seeds 5 --steps 2000
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

from rldl.manual_optim import SGD, Adam  # noqa: E402
from rldl.manual_softmax import (  # noqa: E402
    MLPClassifier,
    accuracy,
    cross_entropy,
)
from rldl.softmax_data import make_spiral_3class, split_data  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OPTIMIZERS = {
    "sgd": {"label": "SGD", "color": "#8C8C8C"},
    "momentum": {"label": "SGD + momentum", "color": "#F58518"},
    "adam": {"label": "Adam", "color": "#4C78A8"},
}


def move_to(model: MLPClassifier, device: torch.device) -> None:
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.to(device)
        layer.bias.data = layer.bias.data.to(device)


def make_optimizer(name: str, model: MLPClassifier, args: argparse.Namespace):
    if name == "sgd":
        return SGD(model.parameters(), lr=args.lr_sgd)
    if name == "momentum":
        return SGD(model.parameters(), lr=args.lr_momentum, momentum=0.9)
    if name == "adam":
        return Adam(model.parameters(), lr=args.lr_adam)
    raise ValueError(name)


def evaluate(model, X, y, loss_fn) -> tuple[float, float]:
    with torch.no_grad():
        logits = model.forward(X)
        return float(loss_fn(logits, y).item()), accuracy(logits, y)


def train_run(
    name: str,
    seed: int,
    args: argparse.Namespace,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
    model = MLPClassifier(2, args.hidden, 3, seed=seed)
    move_to(model, device)
    optimizer = make_optimizer(name, model, args)

    n = len(Xtr)
    history: list[tuple[int, float, float, float, float]] = []

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        loss = cross_entropy(model.forward(Xtr[idx]), ytr[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            train_loss, train_acc = evaluate(model, Xtr, ytr, cross_entropy)
            val_loss, val_acc = evaluate(model, Xva, yva, cross_entropy)
            history.append((step, train_loss, val_loss, train_acc, val_acc))

    return np.asarray(history)


def steps_to_threshold(history: np.ndarray, threshold: float) -> float:
    hits = history[history[:, 4] >= threshold]
    return float(hits[0, 0]) if len(hits) else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-sgd", type=float, default=0.1)
    p.add_argument("--lr-momentum", type=float, default=0.15)
    p.add_argument("--lr-adam", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/05_adam_vs_sgd"))
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device = {device}")

    X, y = make_spiral_3class(
        n_per_class=args.per_class, noise=args.noise, turns=1.0, seed=0
    )
    X_tr, y_tr, X_val, y_val = split_data(X, y, val_ratio=0.2, seed=0)
    Xtr = torch.from_numpy(X_tr).to(device)
    ytr = torch.from_numpy(y_tr).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device)
    print(f"train = {len(Xtr)}, val = {len(Xva)}, steps = {args.steps}")

    results: dict[str, list[np.ndarray]] = {k: [] for k in OPTIMIZERS}
    for name in OPTIMIZERS:
        for seed in range(args.seeds):
            history = train_run(name, seed, args, Xtr, ytr, Xva, yva, device)
            results[name].append(history)
            t95 = steps_to_threshold(history, 0.95)
            print(
                f"[{name:8s} seed={seed}] final val_acc = {history[-1, 4]:.4f}, "
                f"达到 0.95 的步数 = {t95:.0f}"
            )

    summary = {}
    for name in OPTIMIZERS:
        arr = np.stack(results[name])
        summary[name] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}

    print("\n最终结果（均值 ± 标准差）：")
    print(f"{'优化器':<18}{'val_acc':>18}{'val_loss':>18}{'达到0.95步数':>16}")
    for name, meta in OPTIMIZERS.items():
        mean = summary[name]["mean"][-1]
        std = summary[name]["std"][-1]
        t95 = np.nanmean([steps_to_threshold(h, 0.95) for h in results[name]])
        print(
            f"{meta['label']:<16}{mean[4]:>10.4f} ± {std[4]:.4f}"
            f"{mean[2]:>12.4f} ± {std[2]:.4f}{t95:>16.0f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    steps = summary["sgd"]["mean"][:, 0]

    for name, meta in OPTIMIZERS.items():
        mean = summary[name]["mean"]
        std = summary[name]["std"]
        color = meta["color"]

        axes[0].plot(steps, mean[:, 2], color=color, label=meta["label"])
        axes[0].fill_between(
            steps, mean[:, 2] - std[:, 2], mean[:, 2] + std[:, 2], color=color, alpha=0.18
        )

        axes[1].plot(steps, mean[:, 4], color=color, label=meta["label"])
        axes[1].fill_between(
            steps, mean[:, 4] - std[:, 4], mean[:, 4] + std[:, 4], color=color, alpha=0.18
        )

    axes[0].set_title("验证损失（均值 ± 标准差）")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross entropy")
    axes[0].set_ylim(0, 1.2)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("验证准确率（均值 ± 标准差）")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.3, 1.01)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"同一初始化 / 同一数据 / 同样 {args.steps} 次更新，只改优化器"
        f"（{args.seeds} 个种子）",
        fontsize=14,
    )
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "adam_vs_sgd.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        steps=steps,
        **{f"{name}_mean": summary[name]["mean"] for name in OPTIMIZERS},
        **{f"{name}_std": summary[name]["std"] for name in OPTIMIZERS},
    )
    print(f"\nsaved -> {args.out}/adam_vs_sgd.png, history.npz")


if __name__ == "__main__":
    main()
