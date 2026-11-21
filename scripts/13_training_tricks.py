"""实验 13：训练技巧对比（学习率调度 / 早停 / 梯度裁剪 / AMP）。

同一个 MNIST CNN、同一初始化，对比两种配置：
  baseline：恒定学习率，不裁剪、不早停、不用 AMP
  tricks  ：Cosine 学习率衰减 + 梯度裁剪 + 早停 + AMP

用法：
  uv run python scripts/13_training_tricks.py
  uv run python scripts/13_training_tricks.py --epochs 6 --patience 2
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.cnn import SmallCNN  # noqa: E402
from rldl.mnist import MNISTDataset, load_mnist  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TrainConfig:
    name: str
    scheduler: str = "none"  # none | cosine
    clip: float = 0.0  # 0 表示不裁剪
    amp: bool = False
    patience: int = 0  # 0 表示不早停
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


def train(
    cfg: TrainConfig,
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> TrainConfig:
    set_seed(args.seed)
    model = SmallCNN(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    start_all = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        last_grad_norm = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            if use_amp:
                scaler.unscale_(optimizer)
            last_grad_norm = grad_norm(model)
            if cfg.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_time = time.perf_counter() - start
        lr = optimizer.param_groups[0]["lr"]

        cfg.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": lr,
                "grad_norm": last_grad_norm,
                "epoch_time": epoch_time,
            }
        )

        improved = test_acc > best_acc + 1e-4
        if improved:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"[{cfg.name:8s}] epoch {epoch:2d} | lr={lr:.2e} grad_norm={last_grad_norm:5.2f} "
            f"| train_acc={train_acc:.4f} test_acc={test_acc:.4f} | {epoch_time:.1f}s"
        )

        if cfg.patience and bad_epochs >= cfg.patience:
            print(f"[{cfg.name}] 早停：连续 {cfg.patience} 轮没有提升")
            break

    model.load_state_dict(best_state)
    cfg.total_time = time.perf_counter() - start_all  # type: ignore[attr-defined]
    cfg.best_acc = best_acc  # type: ignore[attr-defined]
    cfg.epochs_run = len(cfg.history)  # type: ignore[attr-defined]
    return cfg


def plot(cfgs: list[TrainConfig], out: Path) -> None:
    colors = {"baseline": "#8C8C8C", "tricks": "#4C78A8"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for cfg in cfgs:
        h = cfg.history
        epochs = [r["epoch"] for r in h]
        color = colors.get(cfg.name, "#F58518")

        axes[0].plot(epochs, [r["test_acc"] for r in h], "o-", color=color, label=cfg.name)
        axes[1].plot(epochs, [r["lr"] for r in h], "o-", color=color, label=cfg.name)
        axes[2].plot(epochs, [r["grad_norm"] for r in h], "o-", color=color, label=cfg.name)

    axes[0].set_title("测试准确率")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.97, 1.0)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("学习率")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("lr")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    axes[2].set_title("梯度范数（裁剪前）")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("||grad||")
    axes[2].set_yscale("log")
    axes[2].axhline(1.0, color="#E45756", ls="--", lw=1, label="clip=1.0")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend()

    fig.suptitle("训练技巧对比：基线 vs 调度+裁剪+早停+AMP", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "tricks.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/13_training_tricks"))
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device = {device}")

    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    train_loader = DataLoader(
        MNISTDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(MNISTDataset(X_test, y_test), batch_size=512, shuffle=False)

    configs = [
        TrainConfig(name="baseline"),
        TrainConfig(
            name="tricks",
            scheduler="cosine",
            clip=args.clip,
            amp=True,
            patience=args.patience,
        ),
    ]

    for cfg in configs:
        print(f"\n===== {cfg.name} =====")
        train(cfg, args, train_loader, test_loader, device)

    print("\n汇总：")
    print(f"{'配置':<10}{'best_acc':>10}{'run_epochs':>12}{'总耗时(s)':>12}")
    for cfg in configs:
        print(
            f"{cfg.name:<10}{cfg.best_acc:>10.4f}{cfg.epochs_run:>12d}{cfg.total_time:>12.1f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    plot(configs, args.out)
    np.savez(
        args.out / "history.npz",
        **{
            f"{cfg.name}_{k}": np.array([r[k] for r in cfg.history])
            for cfg in configs
            for k in (
                "epoch",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "lr",
                "grad_norm",
            )
        },
    )
    print(f"\nsaved -> {args.out}/tricks.png, history.npz")


if __name__ == "__main__":
    main()
