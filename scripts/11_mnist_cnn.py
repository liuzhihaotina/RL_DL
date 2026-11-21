"""实验 11：MNIST 手写数字识别（CNN）。

数据：60000 张训练图 + 10000 张测试图，28×28 灰度，10 类。
模型：两层卷积 + 池化 + 全连接，约 42 万参数。
目标：测试准确率 ≥ 0.99。

用法：
  uv run python scripts/11_mnist_cnn.py
  uv run python scripts/11_mnist_cnn.py --epochs 3
  uv run python scripts/11_mnist_cnn.py --limit-train 5000   # 快速冒烟
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
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.cnn import SmallCNN  # noqa: E402
from rldl.mnist import MNISTDataset, load_mnist  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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


def plot_curves(history: list[tuple[int, float, float, float, float]], out: Path) -> None:
    epochs = [h[0] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h[1] for h in history], label="train loss")
    ax1.plot(epochs, [h[2] for h in history], label="test loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross entropy")
    ax1.set_title("损失")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [h[3] for h in history], label="train acc")
    ax2.plot(epochs, [h[4] for h in history], label="test acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.8, 1.005)
    ax2.set_title("准确率")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_predictions(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out: Path,
    device: torch.device,
) -> None:
    model.eval()
    indices = np.arange(16)
    images = torch.from_numpy(X_test[indices]).float().unsqueeze(1)
    images = (images / 255.0 - 0.1307) / 0.3081
    labels = y_test[indices]

    logits = model(images.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for ax, img, pred, true in zip(axes.flat, X_test[indices], preds, labels, strict=True):
        ax.imshow(img, cmap="gray")
        color = "green" if pred == true else "red"
        ax.set_title(f"预测 {pred} / 真实 {true}", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("测试集前 16 张的预测（红色为错）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--limit-train", type=int, default=0, help="只用前 N 张训练图（0=全部）")
    p.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("outputs/11_mnist_cnn"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")

    X_train, y_train, X_test, y_test = load_mnist(args.data_dir)
    if args.limit_train:
        X_train, y_train = X_train[: args.limit_train], y_train[: args.limit_train]
    print(f"train = {len(X_train)}, test = {len(X_test)}, image = {X_train.shape[1:]}")

    train_ds = MNISTDataset(X_train, y_train)
    test_ds = MNISTDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = SmallCNN(10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params = {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[tuple[int, float, float, float, float]] = []
    best_acc = 0.0
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history.append((epoch, train_loss, test_loss, train_acc, test_acc))

        print(
            f"epoch {epoch:2d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out / "model.pth")

    print(f"best test acc = {best_acc:.4f}")
    plot_curves(history, args.out)
    plot_predictions(model, X_test, y_test, args.out, device)
    np.savez(args.out / "history.npz", history=np.asarray(history))
    print(f"saved -> {args.out}/curves.png, predictions.png, model.pth, history.npz")


if __name__ == "__main__":
    main()
