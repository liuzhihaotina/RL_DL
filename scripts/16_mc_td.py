"""实验 16：无模型预测——蒙特卡洛 vs TD(0) vs TD(λ)。

在 GridWorld 上使用均匀随机策略采样轨迹，不读取环境的转移概率。
用动态规划算出的真实 V(s) 作为基准，比较三种方法的 RMSE。

用法：
  uv run python scripts/16_mc_td.py
  uv run python scripts/16_mc_td.py --episodes 5000 --runs 30
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
    GridWorld,
    policy_evaluation_probs,
)
from rldl.model_free import (  # noqa: E402
    generate_episode,
    mc_update,
    td0_update,
    td_lambda_update,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"mc": "#7B3294", "td0": "#0072B2", "tdl": "#009E73"}


def draw_value(ax, env: GridWorld, V: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    values = V.reshape(env.n_rows, env.n_cols)
    state_grid = np.arange(env.n_states).reshape(env.n_rows, env.n_cols)
    masked = np.ma.masked_where(
        np.isin(state_grid, list(env.walls | env.terminals)), values
    )
    ax.imshow(masked, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            state = env.to_state((r, c))
            if state in env.walls:
                ax.text(c, r, "#", ha="center", va="center", fontsize=16)
                continue
            if state in env.terminals:
                color = "#2E7D32" if state == env.goal else "#C62828"
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, alpha=0.8))
                ax.text(
                    c,
                    r,
                    "G" if state == env.goal else "X",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )
                continue
            ax.text(c, r, f"{V[state]:.2f}", ha="center", va="center", fontsize=11)
            if state == env.start:
                ax.text(c - 0.32, r - 0.32, "S", fontsize=9, fontweight="bold", color="#1D4E89")


def rmse(V: np.ndarray, true_V: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((V[mask] - true_V[mask]) ** 2)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/16_mc_td"))
    args = p.parse_args()

    env = GridWorld(DEFAULT_GRID, gamma=args.gamma)
    probs = np.ones((env.n_states, env.n_actions)) / env.n_actions
    true_V = policy_evaluation_probs(env, probs, env.gamma)
    mask = ~np.isin(np.arange(env.n_states), list(env.walls | env.terminals))

    print(f"states={env.n_states}, gamma={args.gamma}, episodes={args.episodes}, runs={args.runs}")
    print(f"真实 V(start) = {true_V[env.start]:.4f}")

    curves = {"mc": [], "td0": [], "tdl": []}
    final_V = {"mc": [], "td0": [], "tdl": []}
    eval_points = list(range(args.eval_every, args.episodes + 1, args.eval_every))

    for run in range(args.runs):
        rng = np.random.default_rng(args.seed + run)
        V_mc = np.zeros(env.n_states)
        counts = np.zeros(env.n_states, dtype=np.int64)
        V_td0 = np.zeros(env.n_states)
        V_tdl = np.zeros(env.n_states)
        eligibility = np.zeros(env.n_states)

        err_mc, err_td0, err_tdl = [], [], []
        for episode in range(1, args.episodes + 1):
            trajectory = generate_episode(env, rng, args.max_steps)
            mc_update(V_mc, counts, trajectory, env.gamma)
            td0_update(V_td0, trajectory, env.gamma, args.alpha)
            td_lambda_update(
                V_tdl, eligibility, trajectory, env.gamma, args.alpha, args.lam
            )

            if episode % args.eval_every == 0:
                err_mc.append(rmse(V_mc, true_V, mask))
                err_td0.append(rmse(V_td0, true_V, mask))
                err_tdl.append(rmse(V_tdl, true_V, mask))

        curves["mc"].append(err_mc)
        curves["td0"].append(err_td0)
        curves["tdl"].append(err_tdl)
        final_V["mc"].append(V_mc)
        final_V["td0"].append(V_td0)
        final_V["tdl"].append(V_tdl)

    for key in curves:
        curves[key] = np.mean(np.asarray(curves[key]), axis=0)
        final_V[key] = np.mean(np.asarray(final_V[key]), axis=0)

    print("\n最终 RMSE（相对真实 V）：")
    for key, label in (("mc", "Monte Carlo"), ("td0", "TD(0)"), ("tdl", f"TD(λ={args.lam})")):
        print(f"  {label:12s} {curves[key][-1]:.4f}")
    print(f"  V(start): 真实 {true_V[env.start]:.4f} | "
          f"MC {final_V['mc'][env.start]:.4f} | "
          f"TD0 {final_V['td0'][env.start]:.4f} | "
          f"TDλ {final_V['tdl'][env.start]:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    all_values = np.concatenate([true_V, final_V["mc"], final_V["tdl"]])
    vmin, vmax = float(all_values.min()), float(all_values.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    draw_value(axes[0, 0], env, true_V, "真实 V(s)（动态规划）", vmin, vmax)
    draw_value(axes[0, 1], env, final_V["mc"], "Monte Carlo 估计", vmin, vmax)
    draw_value(axes[1, 0], env, final_V["tdl"], f"TD(λ={args.lam}) 估计", vmin, vmax)

    axes[1, 1].semilogy(eval_points, curves["mc"], color=COLORS["mc"], label="Monte Carlo")
    axes[1, 1].semilogy(eval_points, curves["td0"], color=COLORS["td0"], label="TD(0)")
    axes[1, 1].semilogy(
        eval_points, curves["tdl"], color=COLORS["tdl"], label=f"TD(λ={args.lam})"
    )
    axes[1, 1].set_title("RMSE 随 episode 数变化")
    axes[1, 1].set_xlabel("episode")
    axes[1, 1].set_ylabel("RMSE")
    axes[1, 1].grid(alpha=0.3, which="both")
    axes[1, 1].legend()

    fig.suptitle(
        f"无模型预测：均匀随机策略，{args.episodes} episodes，{args.runs} 次重复", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(args.out / "mc_td.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        true_V=true_V,
        mc_V=final_V["mc"],
        td0_V=final_V["td0"],
        tdl_V=final_V["tdl"],
        eval_points=np.asarray(eval_points),
        mc_rmse=curves["mc"],
        td0_rmse=curves["td0"],
        tdl_rmse=curves["tdl"],
    )
    print(f"\nsaved -> {args.out}/mc_td.png, history.npz")


if __name__ == "__main__":
    main()
