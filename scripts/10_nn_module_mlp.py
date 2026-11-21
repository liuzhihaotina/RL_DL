"""实验 10：用 nn.Module + Dataset + DataLoader 重写实验 01。

对照实验 01 的手写版本，这里改成 PyTorch 的标准工程写法：
  - 模型：nn.Module + nn.Sequential
  - 数据：自定义 Dataset + DataLoader（每个 epoch 自动打乱、切 batch）
  - 训练：按 epoch 循环，model.train() / model.eval() 切换
  - 保存：model.state_dict()

用法：
  uv run python scripts/10_nn_module_mlp.py
  uv run python scripts/10_nn_module_mlp.py --epochs 80 --batch-size 32
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.nn_data import ArrayDataset  # noqa: E402
from rldl.nn_models import BinaryMLP  # noqa: E402
from rldl.toy import make_moons, train_val_split  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return ((torch.sigmoid(logits) >= 0.5).float() == y).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).squeeze(1)
        total_loss += criterion(logits, yb).item() * len(xb)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == yb).sum().item()
        total += len(xb)
    return total_loss / total, total_correct / total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/10_nn_module_mlp"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")

    X, y = make_moons(n_samples=args.samples, noise=args.noise, seed=args.seed)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=args.seed)

    train_ds = ArrayDataset(X_tr, y_tr)
    val_ds = ArrayDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    print(
        f"train = {len(train_ds)}, val = {len(val_ds)}, "
        f"batch = {args.batch_size}, steps/epoch = {steps_per_epoch}"
    )

    model = BinaryMLP(2, args.hidden).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    history: list[tuple[int, float, float, float, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append((epoch, train_loss, val_loss, train_acc, val_acc))

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"epoch {epoch:3d} | train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} | train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

    print(f"final: train_acc={history[-1][3]:.3f} val_acc={history[-1][4]:.3f}")
    print(f"total updates = {args.epochs * steps_per_epoch}")

    args.out.mkdir(parents=True, exist_ok=True)
    epochs = [h[0] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h[1] for h in history], label="train loss")
    ax1.plot(epochs, [h[2] for h in history], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("损失（每个 epoch）")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [h[3] for h in history], label="train acc")
    ax2.plot(epochs, [h[4] for h in history], label="val acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.5, 1.02)
    ax2.set_title("准确率（每个 epoch）")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out / "curves.png", dpi=150)
    torch.save(model.state_dict(), args.out / "model.pth")
    np.savez(args.out / "data.npz", X=X, y=y)
    print(f"saved -> {args.out}/curves.png, model.pth, data.npz")


if __name__ == "__main__":
    main()
