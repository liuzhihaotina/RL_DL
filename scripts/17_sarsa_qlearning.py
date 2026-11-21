"""实验 17：SARSA（on-policy）vs Q-learning（off-policy）。

环境：CliffWalking（4×12），从起点走到终点，掉下悬崖 -100 并回到起点。
两种算法都学 Q(s,a)，区别只在更新目标：
  SARSA：      r + γ Q(s', a')   ← 用行为策略实际选出的 a'
  Q-learning： r + γ max Q(s', ·) ← 直接假设下一步选最优

用法：
  uv run python scripts/17_sarsa_qlearning.py
  uv run python scripts/17_sarsa_qlearning.py --episodes 1000 --epsilon 0.1
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

from rldl.cliffwalking import CliffWalking  # noqa: E402
from rldl.control import (  # noqa: E402
    evaluate_greedy,
    moving_average,
    train_q_learning,
    train_sarsa,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"sarsa": "#0072B2", "q": "#D55E00"}


def draw_map(ax, env: CliffWalking, path: list[int], title: str) -> None:
    ax.set_xlim(-0.5, env.n_cols - 0.5)
    ax.set_ylim(env.n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.grid(color="#CCCCCC", linewidth=0.5)

    for row in range(env.n_rows):
        for col in range(env.n_cols):
            state = env.to_state((row, col))
            face = "white"
            if state in env.cliffs:
                face = "#E45756"
            elif state == env.start:
                face = "#4C78A8"
            elif state == env.goal:
                face = "#54A24B"
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=face,
                    alpha=0.55,
                    edgecolor="none",
                )
            )

    ax.text(env.start % env.n_cols, env.start // env.n_cols, "S", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(env.goal % env.n_cols, env.goal // env.n_cols, "G", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text((env.n_cols - 1) / 2, env.n_rows - 1, "悬崖", ha="center", va="center",
            fontsize=10, color="white")

    for state, next_state in zip(path, path[1:], strict=False):
        r1, c1 = env.to_rc(state)
        r2, c2 = env.to_rc(next_state)
        if state == next_state:
            continue
        ax.annotate(
            "",
            xy=(c2, r2),
            xytext=(c1, r1),
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.4},
        )
    ax.set_title(title, fontsize=12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("outputs/17_sarsa_qlearning"))
    args = p.parse_args()

    env = CliffWalking()
    results = {
        "sarsa": {"history": [], "evaluations": [], "falls": [], "paths": []},
        "q": {"history": [], "evaluations": [], "falls": [], "paths": []},
    }

    for seed in range(args.seeds):
        sarsa = train_sarsa(
            env,
            args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            rng=np.random.default_rng(seed),
        )
        qlearn = train_q_learning(
            env,
            args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            rng=np.random.default_rng(seed),
        )

        for key, result in (("sarsa", sarsa), ("q", qlearn)):
            ret, success, path = evaluate_greedy(env, result["Q"])
            results[key]["history"].append(result["history"])
            results[key]["evaluations"].append(
                [(ep, r, s) for ep, r, s in result["evaluations"]]
            )
            results[key]["falls"].append(result["cliff_falls"])
            results[key]["paths"].append((ret, success, path))

        print(
            f"seed={seed} | SARSA eval_return={results['sarsa']['paths'][-1][0]:.1f} "
            f"success={results['sarsa']['paths'][-1][1]} falls={sarsa['cliff_falls']} | "
            f"Q-learning eval_return={results['q']['paths'][-1][0]:.1f} "
            f"success={results['q']['paths'][-1][1]} falls={qlearn['cliff_falls']}"
        )

    print("\n平均结果：")
    for key, label in (("sarsa", "SARSA"), ("q", "Q-learning")):
        rets = [x[0] for x in results[key]["paths"]]
        succ = np.mean([x[1] for x in results[key]["paths"]])
        falls = np.mean(results[key]["falls"])
        print(
            f"  {label:10s} eval_return={np.mean(rets):.1f} success={succ:.2f} "
            f"cliff_falls={falls:.0f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    for key, label in (("sarsa", "SARSA"), ("q", "Q-learning")):
        smoothed = np.mean(
            [moving_average(h, 20) for h in results[key]["history"]], axis=0
        )
        x = np.arange(20, 20 + len(smoothed))
        axes[0, 0].plot(x, smoothed, color=COLORS[key], label=label)

        evals = np.asarray([[r for _, r, _ in ev] for ev in results[key]["evaluations"]])
        eval_x = [ep for ep, _, _ in results[key]["evaluations"][0]]
        axes[0, 1].plot(eval_x, evals.mean(axis=0), "o-", color=COLORS[key], label=label)

    axes[0, 0].set_title("训练回报（ε-greedy 行为策略，20 轮滑动平均）")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("return")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("贪心策略的评估回报")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].set_ylabel("greedy return")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    sarsa_ret, sarsa_success, sarsa_path = results["sarsa"]["paths"][0]
    q_ret, q_success, q_path = results["q"]["paths"][0]

    draw_map(
        axes[1, 0],
        env,
        sarsa_path,
        f"SARSA 路径：return={sarsa_ret:.0f}，成功={sarsa_success}",
    )
    draw_map(
        axes[1, 1],
        env,
        q_path,
        f"Q-learning 路径：return={q_ret:.0f}，成功={q_success}",
    )

    fig.suptitle(
        f"CliffWalking：SARSA vs Q-learning（{args.episodes} episodes，ε={args.epsilon}）",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(args.out / "sarsa_vs_qlearning.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        sarsa_history=np.asarray(results["sarsa"]["history"]),
        q_history=np.asarray(results["q"]["history"]),
        sarsa_path=np.asarray(sarsa_path),
        q_path=np.asarray(q_path),
    )
    print(f"\nsaved -> {args.out}/sarsa_vs_qlearning.png, history.npz")


if __name__ == "__main__":
    main()
