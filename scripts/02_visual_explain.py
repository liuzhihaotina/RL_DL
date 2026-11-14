"""生成实验 01 的图解：训练流程、MLP 结构、学到的决策边界。

输出到 docs/images/（会入库，方便在文档里引用）：
  01_pipeline.png          一次训练循环在做什么
  01_network.png           网络结构与参数形状
  01_decision_boundary.png 数据与训练后的分类边界

前置：先跑过一次
  python scripts/01_mlp_moons.py
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_nn import MLP  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_BLUE = "#4C78A8"
C_ORANGE = "#F58518"
C_GREEN = "#54A24B"
C_RED = "#E45756"
C_GRAY = "#8C8C8C"
C_BG = "#F7F7F7"


def _box(ax, x, y, w, h, title, body, color, body_size=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=2,
            edgecolor=color,
            facecolor=color + "18",
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + w / 2,
        y + h * 0.36,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#333333",
        linespacing=1.5,
    )


def _arrow(ax, p1, p2, color="#555555", style="-|>", rad=0.0, lw=2.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=18,
            linewidth=lw,
            linestyle=ls,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.98,
        "scripts/01_mlp_moons.py：一次训练循环在做什么",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )

    w, h, y = 0.17, 0.24, 0.47
    xs = [0.02 + i * 0.192 for i in range(5)]
    colors = [C_GRAY, C_BLUE, C_ORANGE, C_RED, C_GREEN]
    titles = ["① 造数据", "② 前向传播", "③ 算损失", "④ 反向传播", "⑤ 更新参数"]
    bodies = [
        "make_moons\nX(1000,2)  y(1000,)\n切 800 训练 / 200 验证",
        "x @ W + b → ReLU\n→ ReLU → logit\n形状 (64,2)→(64,)",
        "BCEWithLogitsLoss\n把 logit 和标签比\n得到 1 个标量",
        "loss.backward()\n链式法则，算出\n每个参数的 .grad",
        "SGD + momentum\nw ← w − lr × v\n沿梯度反方向走",
    ]

    for x, c, t, b in zip(xs, colors, titles, bodies, strict=True):
        _box(ax, x, y, w, h, t, b, c)
    for i in range(4):
        _arrow(ax, (xs[i] + w + 0.003, y + h / 2), (xs[i + 1] - 0.003, y + h / 2))

    # 循环箭头：⑤ -> ②
    _arrow(
        ax,
        (xs[4] + w / 2, y - 0.012),
        (xs[1] + w / 2, y - 0.012),
        color=C_BLUE,
        rad=-0.28,
        lw=2.2,
    )
    ax.text(
        0.5,
        0.19,
        "重复 1500 步：每次随机抽 64 条 → 前向 → 损失 → 反向 → 更新（这叫 1 step）",
        ha="center",
        fontsize=12,
        color=C_BLUE,
    )

    # 验证分支
    vx, vy, vw, vh = xs[3] - 0.015, 0.795, 0.34, 0.105
    ax.add_patch(
        FancyBboxPatch(
            (vx, vy),
            vw,
            vh,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.8,
            linestyle="--",
            edgecolor=C_GRAY,
            facecolor="#FFFFFF",
        )
    )
    ax.text(
        vx + vw / 2,
        vy + vh * 0.72,
        "⑥ 验证（每 50 步）",
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#444444",
    )
    ax.text(
        vx + vw / 2,
        vy + vh * 0.31,
        "200 条验证集前向一遍，算 val_loss / val_acc",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    _arrow(
        ax,
        (xs[4] + w / 2, y + h + 0.005),
        (vx + vw / 2, vy - 0.005),
        color=C_GRAY,
        ls="--",
        rad=-0.15,
    )
    ax.text(
        vx + vw + 0.025,
        vy + vh / 2,
        "只观察\n不学习",
        fontsize=9.5,
        color=C_GRAY,
        va="center",
    )

    # 底部说明
    ax.text(
        0.5,
        0.065,
        "核心：参数（1185 个数）一开始是随机的，靠「梯度指出 loss 上升方向」一步步往反方向挪，\n"
        "1500 步后就能把两条月牙分开。",
        ha="center",
        fontsize=11.5,
        color="#333333",
        linespacing=1.8,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": C_BG, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(out / "01_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_network(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "MLP 结构：2 → 32 → 32 → 1，共 1185 个可学习参数",
        ha="center",
        fontsize=17,
        fontweight="bold",
    )

    layers = [
        (0.09, 2, "输入层", "x1, x2\n(平面坐标)", C_GRAY),
        (0.36, 7, "隐藏层 1\nLinear(2,32)+ReLU", "96 个参数\n(2×32 + 32)", C_BLUE),
        (0.63, 7, "隐藏层 2\nLinear(32,32)+ReLU", "1056 个参数\n(32×32 + 32)", C_ORANGE),
        (0.90, 1, "输出层\nLinear(32,1)", "33 个参数\n(32×1 + 1)", C_GREEN),
    ]

    positions = []
    for cx, n, *_ in layers:
        ys = [0.48] if n == 1 else np.linspace(0.24, 0.72, n)
        positions.append([(cx, yy) for yy in ys])

    # 连接线（只画采样，避免太密）
    rng = np.random.default_rng(0)
    for li in range(len(layers) - 1):
        left, right = positions[li], positions[li + 1]
        for p1 in left:
            for p2 in right:
                if len(left) * len(right) > 24 and rng.random() > 0.28:
                    continue
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color="#B0B0B0",
                    lw=0.6,
                    alpha=0.5,
                    zorder=1,
                )

    for (cx, n, title, sub, color), pts in zip(layers, positions, strict=True):
        radius = 0.021 if n > 1 else 0.028
        for px, py in pts:
            ax.add_patch(
                Circle(
                    (px, py),
                    radius,
                    facecolor=color,
                    edgecolor="white",
                    lw=1.5,
                    zorder=3,
                    alpha=0.9,
                )
            )
        if n > 4:
            for dy in (-0.012, 0.0, 0.012):
                ax.add_patch(
                    Circle((cx + 0.052, 0.48 + dy), 0.0035, facecolor=color, edgecolor="none")
                )
        ax.text(
            cx,
            0.86,
            title,
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            linespacing=1.5,
        )
        ax.text(
            cx,
            0.13,
            sub,
            ha="center",
            fontsize=10.5,
            color="#444444",
            linespacing=1.6,
        )

    # 形状流
    shapes = [(0.09, "(64, 2)"), (0.36, "(64, 32)"), (0.63, "(64, 32)"), (0.90, "(64, 1) → (64,)")]
    for (cx, s), color in zip(shapes, [C_GRAY, C_BLUE, C_ORANGE, C_GREEN], strict=True):
        ax.text(cx, 0.035, s, ha="center", fontsize=11, color=color, fontweight="bold")

    ax.text(
        0.5,
        0.785,
        "每一列是一个「层」；线是权重 W。batch=64 条数据同时流过网络。",
        ha="center",
        fontsize=10.5,
        color="#666666",
    )
    ax.text(
        0.90,
        0.62,
        "logit（任意实数）\n↓ sigmoid\np ∈ (0, 1)",
        ha="center",
        fontsize=10,
        color="#666666",
        linespacing=1.5,
    )

    fig.tight_layout()
    fig.savefig(out / "01_network.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_model(exp_dir: Path) -> MLP:
    model = MLP(2, 32, seed=0)
    sd = torch.load(exp_dir / "model.pth", map_location="cpu", weights_only=True)
    with torch.no_grad():
        model.fc1.weight.copy_(sd["fc1.weight"])
        model.fc1.bias.copy_(sd["fc1.bias"])
        model.fc2.weight.copy_(sd["fc2.weight"])
        model.fc2.bias.copy_(sd["fc2.bias"])
        model.fc3.weight.copy_(sd["fc3.weight"])
        model.fc3.bias.copy_(sd["fc3.bias"])
    return model


def fig_boundary(out: Path, exp_dir: Path) -> None:
    data = np.load(exp_dir / "data.npz")
    X, y = data["X"], data["y"]
    model = load_model(exp_dir)

    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = torch.from_numpy(np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32))
    with torch.no_grad():
        prob = torch.sigmoid(model.forward(grid)).numpy().reshape(gx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["训练数据（1000 个点，2 类）", "训练后的分类边界（概率 ≥ 0.5 判为类别 1）"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("特征 x1")
        ax.set_ylabel("特征 x2")
        ax.set_aspect("equal")

    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], s=14, color=C_BLUE, label="类别 0", alpha=0.8)
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], s=14, color=C_ORANGE, label="类别 1", alpha=0.8)
    axes[0].legend(loc="upper right")
    axes[0].text(
        0.03,
        0.95,
        "make_moons(noise=0.1)\n两类交错，无法用一条直线分开",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    axes[1].contourf(gx, gy, prob, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.75)
    axes[1].contour(gx, gy, prob, levels=[0.5], colors="black", linewidths=1.2)
    axes[1].scatter(
        X[y == 0, 0], X[y == 0, 1], s=12, color=C_BLUE, edgecolor="white", linewidth=0.4
    )
    axes[1].scatter(
        X[y == 1, 0], X[y == 1, 1], s=12, color=C_ORANGE, edgecolor="white", linewidth=0.4
    )
    axes[1].text(
        0.03,
        0.05,
        "黑线 = 决策边界 p=0.5\n蓝/橙 = 真实类别",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.suptitle(
        "模型学到的不是一个公式，而是一片「哪里更可能是类别 1」的概率区域",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out / "01_decision_boundary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/images"))
    p.add_argument("--exp", type=Path, default=Path("outputs/01_mlp_moons"))
    args = p.parse_args()

    if not (args.exp / "model.pth").exists() or not (args.exp / "data.npz").exists():
        raise SystemExit(
            f"找不到 {args.exp}/model.pth 或 data.npz，请先运行：python scripts/01_mlp_moons.py"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    fig_pipeline(args.out)
    fig_network(args.out)
    fig_boundary(args.out, args.exp)
    print(f"saved -> {args.out}/01_pipeline.png, 01_network.png, 01_decision_boundary.png")


if __name__ == "__main__":
    main()
