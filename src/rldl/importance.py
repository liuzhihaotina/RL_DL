"""重要性采样：用行为策略的数据估计目标策略的期望。

对一条轨迹，重要性权重是每一步动作概率比的乘积：

  ρ = Π_t  π(a_t | s_t) / b(a_t | s_t)

要求：只要 π(a|s) > 0，就必须有 b(a|s) > 0（行为策略要覆盖目标策略）。
"""

from __future__ import annotations

import numpy as np


def trajectory_ratio(
    target_probs: np.ndarray,
    behavior_probs: np.ndarray,
    states: list[int],
    actions: list[int],
) -> float:
    """计算整条轨迹的重要性权重。"""
    ratio = 1.0
    for state, action in zip(states, actions, strict=True):
        p_target = target_probs[state, action]
        p_behavior = behavior_probs[state, action]
        if p_behavior <= 0:
            raise ValueError(f"行为策略在 state={state} 没有覆盖 action={action}")
        ratio *= p_target / p_behavior
    return ratio


def ordinary_is(returns: np.ndarray, ratios: np.ndarray) -> float:
    """普通重要性采样：mean(ρ * G)。无偏，但方差大。"""
    return float(np.mean(returns * ratios))


def weighted_is(returns: np.ndarray, ratios: np.ndarray) -> float:
    """加权重要性采样：sum(ρG) / sum(ρ)。方差小，但有限样本有偏。"""
    denominator = float(np.sum(ratios))
    if denominator == 0:
        return float("nan")
    return float(np.sum(returns * ratios) / denominator)
