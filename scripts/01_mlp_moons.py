"""实验 01：从零手写 MLP 训练 two-moons 二分类。

只使用 torch.Tensor + autograd：
  - 参数初始化、Linear 层、ReLU、sigmoid 都在 src/rldl/manual_nn.py 里手写
  - loss 用 BCEWithLogits（等价于 sigmoid + 手写 BCE 的数值稳定版）
  - 反向传播由 autograd 完成，参数更新用 torch.optim.SGD

用法：
    uv run python scripts/01_mlp_moons.py
    uv run python scripts/01_mlp_moons.py --device cpu --steps 1000
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_nn import MLP  # noqa: E402
from rldl.toy import make_moons, train_val_split  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == y).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/01_mlp_moons"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")
    if device.type == "cuda":
        print(f"gpu    = {torch.cuda.get_device_name(device)}")

    X, y = make_moons(n_samples=args.samples, noise=args.noise, seed=args.seed)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=args.seed)

    Xtr = torch.from_numpy(X_tr).to(device)
    ytr = torch.from_numpy(y_tr).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    yva = torch.from_numpy(y_val).to(device)

    model = MLP(2, args.hidden, seed=args.seed)
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.to(device)
        layer.bias.data = layer.bias.data.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n = len(Xtr)
    history: list[tuple[int, float, float, float]] = []
    args.out.mkdir(parents=True, exist_ok=True)

    progress = tqdm(range(1, args.steps + 1), desc="train", dynamic_ncols=True)
    for step in progress:
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        logits = model.forward(Xtr[idx])
        loss = loss_fn(logits, ytr[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                val_logits = model.forward(Xva)
                val_loss = loss_fn(val_logits, yva).item()
                val_acc = accuracy(val_logits, yva)
            history.append((step, float(loss.item()), float(val_loss), val_acc))
            progress.set_postfix(train_loss=f"{loss.item():.4f}", val_acc=f"{val_acc:.3f}")

    train_acc = accuracy(model.forward(Xtr), ytr)
    print(f"final: train_acc={train_acc:.3f} val_acc={history[-1][3]:.3f}")

    steps = [h[0] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, [h[2] for h in history], label="val loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("validation loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(steps, [h[3] for h in history], label="val acc", color="tab:green")
    ax2.set_xlabel("step")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0.0, 1.02)
    ax2.set_title("validation accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(args.out / "curve.png", dpi=150)
    torch.save(model.state_dict(), args.out / "model.pth")
    np.savez(args.out / "data.npz", X=X, y=y)
    print(f"saved -> {args.out}/curve.png, model.pth, data.npz")


if __name__ == "__main__":
    main()
