"""最小例子：loss = w²，同一学习率下比较 SGD 与 Adam 的每一步。

解析梯度：dL/dw = 2w。两种优化器都从 w=1 出发，lr 都取 0.1。

用法：
  uv run python scripts/06_optimizer_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_optim import SGD, Adam  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run(optimizer, steps: int) -> list[tuple[int, float, float]]:
    """返回 [(step, 更新后 w, 本步移动量), ...]。"""
    w = torch.tensor([1.0], requires_grad=True)
    opt = optimizer([w])
    rows = []
    for t in range(1, steps + 1):
        loss = (w**2).sum()
        opt.zero_grad()
        loss.backward()
        before = w.item()
        opt.step()
        rows.append((t, w.item(), before - w.item()))
    return rows


def main() -> None:
    steps = 12
    sgd_rows = run(lambda p: SGD(p, lr=0.1), steps)
    adam_rows = run(lambda p: Adam(p, lr=0.1), steps)

    print("loss = w²，dL/dw = 2w，起点 w = 1.0，学习率都是 0.1\n")
    header = (
        f"{'step':>4} | {'SGD w':>8} {'SGD 步长':>9} {'SGD loss':>9} | "
        f"{'Adam w':>8} {'Adam 步长':>10} {'Adam loss':>10}"
    )
    print(header)
    print("-" * len(header))
    for (t, w_s, d_s), (_, w_a, d_a) in zip(sgd_rows, adam_rows, strict=True):
        print(
            f"{t:>4} | {w_s:>8.4f} {d_s:>9.4f} {w_s**2:>9.6f} | "
            f"{w_a:>8.4f} {d_a:>10.4f} {w_a**2:>10.6f}"
        )

    w_grid = torch.linspace(-0.3, 1.1, 200).numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(w_grid, w_grid**2, color="#333333", lw=1.5, label="loss = w²")
    ax.scatter(
        [r[1] for r in sgd_rows],
        [r[1] ** 2 for r in sgd_rows],
        color="#8C8C8C",
        s=30,
        label="SGD 轨迹",
    )
    ax.scatter(
        [r[1] for r in adam_rows],
        [r[1] ** 2 for r in adam_rows],
        color="#4C78A8",
        s=30,
        label="Adam 轨迹",
    )
    ax.set_xlabel("w")
    ax.set_ylabel("loss")
    ax.set_title("同一学习率 0.1：SGD 逐步衰减，Adam 每步约走 lr")
    ax.grid(alpha=0.3)
    ax.legend()

    out = Path("outputs/06_optimizer_demo")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=150)
    print(f"\nsaved -> {out}/trajectory.png")


if __name__ == "__main__":
    main()
