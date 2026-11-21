"""实验 18：重要性采样（off-policy 评估）。

两个例子：
  1. 多臂老虎机：目标策略只拉 arm 4，行为策略均匀随机
  2. 两状态链：目标策略偏向 forward，行为策略 50/50

比较：
  - 朴素平均（把行为策略的数据当成目标策略）
  - 普通重要性采样（无偏，方差大）
  - 加权重要性采样（方差小，有限样本有偏）

用法：
  uv run python scripts/18_importance_sampling.py
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

from rldl.importance import (  # noqa: E402
    ordinary_is,
    trajectory_ratio,
    weighted_is,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def bandit_experiment(replications: int, samples: int, seed: int) -> dict[str, float]:
    """目标策略固定拉 arm 4，行为策略均匀随机。"""
    probs = np.array([0.10, 0.50, 0.60, 0.20, 0.80, 0.30, 0.40, 0.70, 0.25, 0.55])
    target_arm = 4
    rng = np.random.default_rng(seed)

    naive, is_est = [], []
    for _ in range(replications):
        arms = rng.integers(0, len(probs), samples)
        rewards = (rng.random(samples) < probs[arms]).astype(float)
        weights = np.where(arms == target_arm, len(probs), 0.0)
        naive.append(rewards.mean())
        is_est.append((weights * rewards).mean())

    return {
        "true": float(probs[target_arm]),
        "naive_mean": float(np.mean(naive)),
        "naive_std": float(np.std(naive)),
        "is_mean": float(np.mean(is_est)),
        "is_std": float(np.std(is_est)),
    }


def chain_experiment(episodes: int, seed: int) -> dict[str, np.ndarray | float]:
    """0 → 1 → terminal，动作 0=forward（+1），1=quit（0）。

    目标策略：state0 一定 forward；state1 以 0.8 概率 forward。
    行为策略：两个状态都 50/50。
    """
    target = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 0.0],
        ]
    )
    behavior = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.0, 0.0],
        ]
    )

    # 真实价值（gamma=1）
    V1 = 0.8 * 1.0 + 0.2 * 0.0
    V0 = 1.0 + V1
    true_value = V0

    rng = np.random.default_rng(seed)
    returns, ratios = [], []
    for _ in range(episodes):
        state = 0
        states, actions = [], []
        total = 0.0
        for _ in range(10):
            action = int(rng.random() < 0.5)  # 行为策略 50/50
            states.append(state)
            actions.append(action)
            if action == 0:
                total += 1.0
                state += 1
                if state == 2:
                    break
            else:
                break
        returns.append(total)
        ratios.append(trajectory_ratio(target, behavior, states, actions))

    returns_arr = np.asarray(returns)
    ratios_arr = np.asarray(ratios)

    return {
        "true": true_value,
        "naive_mean": float(returns_arr.mean()),
        "ordinary": ordinary_is(returns_arr, ratios_arr),
        "weighted": weighted_is(returns_arr, ratios_arr),
        "returns": returns_arr,
        "ratios": ratios_arr,
    }


def running_curves(
    returns: np.ndarray, ratios: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回朴素平均、普通 IS、加权 IS 的逐 episode 累计曲线。"""
    n = np.arange(1, len(returns) + 1)
    naive = np.cumsum(returns) / n
    is_num = np.cumsum(returns * ratios)
    is_den = np.cumsum(ratios)
    ordinary = is_num / n
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = np.where(is_den > 0, is_num / is_den, np.nan)
    return naive, ordinary, weighted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--replications", type=int, default=500)
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/18_importance_sampling"))
    args = p.parse_args()

    bandit = bandit_experiment(args.replications, args.samples, args.seed)
    print("老虎机：目标策略固定拉 arm 4（p=0.80），行为策略均匀随机")
    print(
        f"  真实值 {bandit['true']:.3f} | 朴素平均 {bandit['naive_mean']:.3f} "
        f"| 普通 IS {bandit['is_mean']:.3f} ± {bandit['is_std']:.3f}"
    )

    chain = chain_experiment(args.episodes, args.seed)
    print("\n两状态链：目标策略偏向 forward，行为策略 50/50")
    print(f"  真实 V(target) = {chain['true']:.3f}")
    print(f"  朴素平均 = {chain['naive_mean']:.3f}（估计的是行为策略）")
    print(f"  普通 IS  = {chain['ordinary']:.3f}")
    print(f"  加权 IS  = {chain['weighted']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ["真实值", "朴素平均", "普通 IS"]
    values = [bandit["true"], bandit["naive_mean"], bandit["is_mean"]]
    errors = [0.0, bandit["naive_std"], bandit["is_std"]]
    colors = ["#54A24B", "#8C8C8C", "#4C78A8"]
    axes[0].bar(labels, values, yerr=errors, color=colors, capsize=5)
    axes[0].axhline(bandit["true"], color="#54A24B", ls="--", lw=1)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("平均奖励")
    axes[0].set_title(f"老虎机：{args.replications} 次重复，每次 {args.samples} 个样本")
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)

    naive, ordinary, weighted = running_curves(chain["returns"], chain["ratios"])
    x = np.arange(1, args.episodes + 1)
    axes[1].semilogx(x, naive, color="#8C8C8C", label="朴素平均（行为策略）")
    axes[1].semilogx(x, ordinary, color="#4C78A8", label="普通 IS")
    axes[1].semilogx(x, weighted, color="#E45756", label="加权 IS")
    axes[1].axhline(chain["true"], color="#54A24B", ls="--", lw=1, label="目标策略真实值")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("V(s0)")
    axes[1].set_title("两状态链：累计估计")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.suptitle("重要性采样：用行为策略的数据估计目标策略", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "importance_sampling.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        bandit_true=bandit["true"],
        bandit_naive=bandit["naive_mean"],
        bandit_is=bandit["is_mean"],
        chain_returns=chain["returns"],
        chain_ratios=chain["ratios"],
        chain_true=chain["true"],
    )
    print(f"\nsaved -> {args.out}/importance_sampling.png, history.npz")


if __name__ == "__main__":
    main()
