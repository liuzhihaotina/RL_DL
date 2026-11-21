"""实验 14：多臂老虎机——ε-greedy / UCB / Thompson sampling。

10 个手臂，每个手臂中奖概率未知。跑 1000 步，重复 200 次取平均。
对比三个指标：
  1. 每步平均奖励：越高越好
  2. 选中最佳手臂的比例：越高越好
  3. 累计遗憾：越低越好（遗憾 = 最佳概率 - 实际选中手臂的概率）

用法：
  uv run python scripts/14_bandit.py
  uv run python scripts/14_bandit.py --steps 2000 --runs 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.bandit import (  # noqa: E402
    UCB,
    BernoulliBandit,
    EpsilonGreedy,
    ThompsonSampling,
    simulate,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TRUE_PROBS = [0.10, 0.50, 0.60, 0.20, 0.80, 0.30, 0.40, 0.70, 0.25, 0.55]

ALGOS = {
    "epsilon-greedy": {"label": "ε-greedy (ε=0.1)", "color": "#8C8C8C"},
    "ucb": {"label": "UCB (c=0.5)", "color": "#F58518"},
    "thompson": {"label": "Thompson sampling", "color": "#4C78A8"},
}


def make_agent(name: str, n_arms: int, seed: int, args):
    if name == "epsilon-greedy":
        return EpsilonGreedy(n_arms, epsilon=args.epsilon, seed=seed)
    if name == "ucb":
        return UCB(n_arms, c=args.ucb_c)
    if name == "thompson":
        return ThompsonSampling(n_arms, seed=seed)
    raise ValueError(name)


def run_algo(name: str, args) -> dict[str, np.ndarray]:
    rewards, optimal, regrets = [], [], []
    for run in range(args.runs):
        bandit = BernoulliBandit(TRUE_PROBS, seed=args.seed + run)
        agent = make_agent(name, bandit.n_arms, args.seed + run, args)
        r, o, g = simulate(agent, bandit, args.steps)
        rewards.append(r)
        optimal.append(o)
        regrets.append(g)

    return {
        "reward": np.mean(rewards, axis=0),
        "optimal": np.mean(optimal, axis=0),
        "regret": np.mean(regrets, axis=0),
    }


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--ucb-c", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smooth", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("outputs/14_bandit"))
    args = p.parse_args()

    print("真实中奖概率：")
    for i, prob in enumerate(TRUE_PROBS):
        mark = " ← 最佳" if i == int(np.argmax(TRUE_PROBS)) else ""
        print(f"  arm {i}: {prob:.2f}{mark}")

    results = {}
    for name in ALGOS:
        results[name] = run_algo(name, args)
        print(
            f"[{name:14s}] 平均奖励={results[name]['reward'].mean():.4f} "
            f"最佳臂比例={results[name]['optimal'].mean():.3f} "
            f"累计遗憾={results[name]['regret'][-1]:.1f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    steps = np.arange(1, args.steps + 1)

    for name, meta in ALGOS.items():
        res = results[name]
        color = meta["color"]
        x = steps[args.smooth - 1 :] if args.smooth > 1 else steps

        axes[0].plot(x, smooth(res["reward"], args.smooth), color=color, label=meta["label"])
        axes[1].plot(
            x, smooth(res["optimal"], args.smooth) * 100, color=color, label=meta["label"]
        )
        axes[2].plot(steps, res["regret"], color=color, label=meta["label"])

    axes[0].set_title("每步平均奖励")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("reward")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("选中最佳手臂的比例")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 102)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].set_title("累计遗憾（越低越好）")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("cumulative regret")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        f"多臂老虎机：10 个手臂，{args.steps} 步，{args.runs} 次重复的平均", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(args.out / "bandit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        **{
            f"{name}_{metric}": results[name][metric]
            for name in ALGOS
            for metric in ("reward", "optimal", "regret")
        },
    )
    print(f"\nsaved -> {args.out}/bandit.png, history.npz")


if __name__ == "__main__":
    main()
