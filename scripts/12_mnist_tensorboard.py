"""实验 12：MNIST + TensorBoard + TOML 配置。

和实验 11 的区别：
  - 超参从 configs/mnist_cnn.toml 读取，命令行可覆盖
  - 每个 epoch 把 loss / acc / lr / 权重直方图写入 runs/
  - 用 tensorboard 网页查看训练过程

用法：
  uv run python scripts/12_mnist_tensorboard.py
  uv run python scripts/12_mnist_tensorboard.py --epochs 3 --lr 3e-3
  uv run tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.cnn import SmallCNN  # noqa: E402
from rldl.config import load_config  # noqa: E402
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


@torch.no_grad()
def log_predictions(
    writer: SummaryWriter,
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epoch: int,
) -> None:
    model.eval()
    indices = np.arange(8)
    images = torch.from_numpy(X_test[indices].copy()).float().unsqueeze(1)
    images = (images / 255.0 - 0.1307) / 0.3081
    preds = model(images.to(device)).argmax(dim=1).cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, img, pred, true in zip(
        axes.flat, X_test[indices], preds, y_test[indices], strict=True
    ):
        ax.imshow(img, cmap="gray")
        color = "green" if pred == true else "red"
        ax.set_title(f"预测 {pred} / 真实 {true}", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("测试集前 8 张")
    fig.tight_layout()
    writer.add_figure("predictions/test", fig, epoch)
    plt.close(fig)


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    out_cfg = cfg.get("output", {})

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/mnist_cnn.toml"))
    p.add_argument("--data-root", type=Path, default=Path(data_cfg.get("root", "data/mnist")))
    p.add_argument("--batch-size", type=int, default=data_cfg.get("batch_size", 128))
    p.add_argument("--limit-train", type=int, default=data_cfg.get("limit_train", 0))
    p.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 5))
    p.add_argument("--lr", type=float, default=train_cfg.get("lr", 1e-3))
    p.add_argument("--seed", type=int, default=train_cfg.get("seed", 0))
    p.add_argument("--device", default=train_cfg.get("device", "auto"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(out_cfg.get("out", "outputs/12_mnist_tensorboard")),
    )
    p.add_argument("--run-dir", type=Path, default=Path(out_cfg.get("run_dir", "runs/mnist_cnn")))
    return p


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("configs/mnist_cnn.toml"))
    known, _ = pre.parse_known_args()

    if not known.config.exists():
        raise SystemExit(f"找不到配置文件：{known.config}")
    cfg = load_config(known.config)

    args = build_parser(cfg).parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")
    print(f"config = {args.config}")

    X_train, y_train, X_test, y_test = load_mnist(args.data_root)
    if args.limit_train:
        X_train, y_train = X_train[: args.limit_train], y_train[: args.limit_train]
    print(f"train = {len(X_train)}, test = {len(X_test)}")

    train_loader = DataLoader(
        MNISTDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(MNISTDataset(X_test, y_test), batch_size=512, shuffle=False)

    model = SmallCNN(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, args.run_dir / "config.toml")
    writer = SummaryWriter(str(args.run_dir))
    writer.add_text(
        "config",
        "<br>".join(f"{k} = {v}" for k, v in sorted(vars(args).items()) if k != "config"),
    )

    best_acc = 0.0
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

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/test", test_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        for name, param in model.named_parameters():
            if param.dim() > 1:
                writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)

        print(
            f"epoch {epoch:2d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out / "model.pth")

    log_predictions(writer, model, X_test, y_test, device, args.epochs)
    writer.close()
    print(f"best test acc = {best_acc:.4f}")
    print(f"tensorboard: uv run tensorboard --logdir {args.run_dir.parent}")


if __name__ == "__main__":
    main()
