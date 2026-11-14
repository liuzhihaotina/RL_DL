"""实验 03：有放回采样 vs 无放回（标准 epoch）采样。

控制变量：
  - 同一份数据、同一个 train/val 划分
  - 同一个初始模型（同一 seed）
  - 同样的 optimizer / lr / batch_size
  - 同样 1500 次参数更新（step）

唯一区别：每个 step 的 batch 怎么来。
  有放回：torch.randint 随机抽 64 个，可重复（实验 01 的写法）
  无放回：每个 epoch 先 randperm 洗牌，再按 64 切片，最后一批 32 条

用法：
  uv run python scripts/03_sampling_ab.py
  uv run python scripts/03_sampling_ab.py --seeds 5 --steps 1500
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

from rldl.manual_nn import MLP  # noqa: E402
from rldl.toy import make_moons, train_val_split  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_REPL = "#4C78A8"
C_EPOCH = "#54A24B"


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return ((torch.sigmoid(logits) >= 0.5).float() == y).float().mean().item()


def move_to(model: MLP, device: torch.device) -> None:
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.to(device)
        layer.bias.data = layer.bias.data.to(device)


def evaluate(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = model.forward(X)
        return float(loss_fn(logits, y).item()), accuracy(logits, y)


def train_run(
    mode: str,
    seed: int,
    args: argparse.Namespace,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    model = MLP(2, args.hidden, seed=seed)
    move_to(model, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = len(Xtr)
    counts = torch.zeros(n, dtype=torch.long, device=device)
    history: list[tuple[int, float, float, float]] = []
    step = 0

    def do_step(idx: torch.Tensor) -> None:
        nonlocal step
        logits = model.forward(Xtr[idx])
        loss = loss_fn(logits, ytr[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        counts.index_add_(0, idx, torch.ones_like(idx))
        step += 1

        if step == 1 or step % args.log_every == 0:
            train_loss, _ = evaluate(model, Xtr, ytr, loss_fn)
            val_loss, val_acc = evaluate(model, Xva, yva, loss_fn)
            history.append((step, train_loss, val_loss, val_acc))

    if mode == "replacement":
        while step < args.steps:
            idx = torch.randint(0, n, (args.batch_size,), device=device)
            do_step(idx)
    elif mode == "epoch":
        while step < args.steps:
            perm = torch.randperm(n, device=device)
            for i in range(0, n, args.batch_size):
                if step >= args.steps:
                    break
                do_step(perm[i : i + args.batch_size])
    else:
        raise ValueError(mode)

    return np.asarray(history), counts.cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3, help="跑几个随机种子取均值/方差")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/03_sampling_ab"))
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device = {device}")

    X, y = make_moons(n_samples=args.samples, noise=args.noise, seed=0)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=0)
    Xtr = torch.from_numpy(X_tr).to(device)
    ytr = torch.from_numpy(y_tr).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device)

    print(f"train = {len(Xtr)}, val = {len(Xva)}, steps = {args.steps}, batch = {args.batch_size}")

    results: dict[str, list[np.ndarray]] = {"replacement": [], "epoch": []}
    counts_by_mode: dict[str, np.ndarray] = {}
    for mode in ("replacement", "epoch"):
        for seed in range(args.seeds):
            history, counts = train_run(mode, seed, args, Xtr, ytr, Xva, yva, device)
            results[mode].append(history)
            if seed == 0:
                counts_by_mode[mode] = counts
            final_acc = history[-1, 3]
            print(f"[{mode:11s} seed={seed}] final val_acc = {final_acc:.4f}")

    # 汇总
    summary = {}
    for mode in ("replacement", "epoch"):
        arr = np.stack(results[mode])  # (seeds, points, 4)
        summary[mode] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
        }

    print("\n最终结果（均值 ± 标准差）：")
    print(f"{'采样方式':<16}{'val_acc':>18}{'val_loss':>18}{'每样本使用次数 std':>20}")
    for mode, label in (("replacement", "有放回"), ("epoch", "无放回 epoch")):
        final = summary[mode]["mean"][-1]
        final_std = summary[mode]["std"][-1]
        usage_std = float(counts_by_mode[mode].std())
        print(
            f"{label:<14}{final[3]:>10.4f} ± {final_std[3]:.4f}"
            f"{final[2]:>12.4f} ± {final_std[2]:.4f}"
            f"{usage_std:>16.2f}"
        )

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    steps = summary["replacement"]["mean"][:, 0]

    for mode, color, label in (
        ("replacement", C_REPL, "有放回（实验 01 写法）"),
        ("epoch", C_EPOCH, "无放回（标准 epoch）"),
    ):
        mean = summary[mode]["mean"]
        std = summary[mode]["std"]

        axes[0].plot(steps, mean[:, 2], color=color, label=label)
        axes[0].fill_between(
            steps, mean[:, 2] - std[:, 2], mean[:, 2] + std[:, 2], color=color, alpha=0.2
        )

        axes[1].plot(steps, mean[:, 3], color=color, label=label)
        axes[1].fill_between(
            steps, mean[:, 3] - std[:, 3], mean[:, 3] + std[:, 3], color=color, alpha=0.2
        )

    axes[0].set_title("验证损失（均值 ± 标准差）")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("验证准确率（均值 ± 标准差）")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.5, 1.01)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    bins = np.linspace(0, max(counts_by_mode[m].max() for m in counts_by_mode) + 1, 40)
    for mode, color, label in (
        ("replacement", C_REPL, "有放回"),
        ("epoch", C_EPOCH, "无放回"),
    ):
        counts = counts_by_mode[mode]
        axes[2].hist(counts, bins=bins, color=color, alpha=0.55, label=label)
        axes[2].axvline(
            counts.mean(),
            color=color,
            ls="--",
            lw=1.2,
            label=f"{label}平均 {counts.mean():.1f} 次",
        )
    lo = min(counts_by_mode[m].min() for m in counts_by_mode)
    hi = max(counts_by_mode[m].max() for m in counts_by_mode)
    axes[2].set_xlim(lo - 8, hi + 8)
    axes[2].set_title("每条训练数据被使用的次数（seed 0）")
    axes[2].set_xlabel("被抽到的次数")
    axes[2].set_ylabel("样本数")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.suptitle(
        f"同一初始化 / 同一数据 / 同样 {args.steps} 次更新，只改采样方式"
        f"（{args.seeds} 个种子）",
        fontsize=14,
    )
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "sampling_ab.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        steps=steps,
        replacement=summary["replacement"]["mean"],
        epoch=summary["epoch"]["mean"],
        replacement_counts=counts_by_mode["replacement"],
        epoch_counts=counts_by_mode["epoch"],
    )
    print(f"\nsaved -> {args.out}/sampling_ab.png, history.npz")


if __name__ == "__main__":
    main()
