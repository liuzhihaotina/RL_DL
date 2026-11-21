"""实验 15：GridWorld 上的策略迭代与价值迭代。

环境：5×4 格子，S 起点，G 终点(+1)，X 陷阱(-1)，# 墙。
每走一步 reward = -0.01，折扣因子 gamma = 0.95。

两种动态规划方法：
  策略迭代：评估当前策略 → 贪心改进 → 重复
  价值迭代：直接迭代贝尔曼最优方程 → 最后提取策略

用法：
  uv run python scripts/15_gridworld.py
  uv run python scripts/15_gridworld.py --gamma 0.99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.gridworld import (  # noqa: E402
    DEFAULT_GRID,
    MOVES,
    GridWorld,
    policy_iteration,
    value_iteration,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def draw_values(
    ax,
    env: GridWorld,
    V: np.ndarray,
    policy: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    values = V.reshape(env.n_rows, env.n_cols)
    state_grid = np.arange(env.n_states).reshape(env.n_rows, env.n_cols)
    masked = np.ma.masked_where(
        np.isin(state_grid, list(env.walls | env.terminals)),
        values,
    )
    ax.imshow(masked, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state = env.to_state((r, c))
            if state in env.walls:
                ax.text(c, r, "#", ha="center", va="center", fontsize=16, color="black")
                continue

            if state in env.terminals:
                color = "#2E7D32" if state == env.goal else "#C62828"
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, alpha=0.8))
                ax.text(c, r, "G" if state == env.goal else "X", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white")
                continue

            ax.text(c, r - 0.28, f"{V[state]:.2f}", ha="center", va="center", fontsize=8)
            if state == env.start:
                ax.text(c - 0.32, r - 0.32, "S", fontsize=9, fontweight="bold", color="#1D4E89")

            if not env.is_terminal(state):
                dr, dc = MOVES[policy[state]]
                ax.arrow(
                    c,
                    r + 0.08,
                    dc * 0.3,
                    dr * 0.3,
                    head_width=0.08,
                    head_length=0.08,
                    fc="black",
                    ec="black",
                    length_includes_head=True,
                    linewidth=1.2,
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--step-reward", type=float, default=-0.01)
    p.add_argument("--theta", type=float, default=1e-10)
    p.add_argument("--out", type=Path, default=Path("outputs/15_gridworld"))
    args = p.parse_args()

    env = GridWorld(DEFAULT_GRID, step_reward=args.step_reward, gamma=args.gamma)
    print("地图：")
    print(env.render_policy(np.zeros(env.n_states, dtype=int)))
    print(f"states={env.n_states}, actions={env.n_actions}, gamma={args.gamma}")

    pi_policy, pi_V, pi_iters = policy_iteration(env, env.gamma, theta=args.theta)
    vi_policy, vi_V, vi_iters, deltas = value_iteration(env, env.gamma, theta=args.theta)

    print(f"\n策略迭代：{pi_iters} 轮收敛")
    print(env.render_policy(pi_policy))
    print(f"\n价值迭代：{vi_iters} 轮收敛")
    print(env.render_policy(vi_policy))

    diff = np.abs(pi_V - vi_V).max()
    n_diff = int((pi_policy != vi_policy).sum())
    print(f"\n两种方法最大价值差 = {diff:.2e}，策略不同的状态数 = {n_diff}")

    print("\n状态价值 V(s)（策略迭代结果）：")
    for r in range(env.n_rows):
        row = []
        for c in range(env.n_cols):
            state = env.to_state((r, c))
            row.append(f"{pi_V[state]: .2f}")
        print("  ".join(row))

    args.out.mkdir(parents=True, exist_ok=True)
    vmin = min(pi_V.min(), vi_V.min())
    vmax = max(pi_V.max(), vi_V.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    draw_values(axes[0], env, pi_V, pi_policy, "策略迭代：V(s) 与最优策略", vmin, vmax)
    draw_values(axes[1], env, vi_V, vi_policy, "价值迭代：V(s) 与最优策略", vmin, vmax)
    positive = np.asarray(deltas)
    positive = positive[positive > 0]
    axes[2].semilogy(range(1, len(positive) + 1), positive, "o-", color="#4C78A8")
    axes[2].set_title("价值迭代的收敛过程")
    axes[2].text(
        0.98,
        0.95,
        f"{len(deltas)} 轮后 delta=0（精确收敛）",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#666666",
    )
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("max |V_new - V_old|")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle(f"GridWorld（gamma={args.gamma}，step reward={env.step_reward}）", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "gridworld.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "results.npz",
        pi_V=pi_V,
        vi_V=vi_V,
        pi_policy=pi_policy,
        vi_policy=vi_policy,
        deltas=np.asarray(deltas),
    )
    print(f"\nsaved -> {args.out}/gridworld.png, results.npz")


if __name__ == "__main__":
    main()
