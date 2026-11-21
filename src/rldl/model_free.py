"""无模型预测：蒙特卡洛（MC）与时序差分（TD）。

只通过和环境交互采样，不读取 GridWorld 的转移概率与奖励函数。
"""

from __future__ import annotations

import numpy as np

from rldl.gridworld import GridWorld


def generate_episode(
    env: GridWorld,
    rng: np.random.Generator,
    max_steps: int = 100,
) -> list[tuple[int, int, float, int, bool]]:
    """用均匀随机策略跑一条轨迹，返回 (s, a, r, s', done) 列表。"""
    state = env.start
    trajectory = []
    for _ in range(max_steps):
        action = int(rng.integers(env.n_actions))
        next_state, reward, done = env.step(state, action)
        trajectory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break
    return trajectory


def mc_update(
    V: np.ndarray,
    counts: np.ndarray,
    trajectory: list[tuple[int, int, float, int, bool]],
    gamma: float,
) -> None:
    """首次访问蒙特卡洛：从后往前算回报 G，更新第一次出现的状态。"""
    returns = []
    G = 0.0
    for _, _, reward, _, _ in reversed(trajectory):
        G = reward + gamma * G
        returns.append(G)
    returns.reverse()

    visited = set()
    for (state, _, _, _, _), G in zip(trajectory, returns, strict=True):
        if state not in visited:
            visited.add(state)
            counts[state] += 1
            V[state] += (G - V[state]) / counts[state]


def td0_update(
    V: np.ndarray,
    trajectory: list[tuple[int, int, float, int, bool]],
    gamma: float,
    alpha: float,
) -> None:
    """TD(0)：每走一步，用 r + γV(s') 作为目标更新 V(s)。"""
    for state, _, reward, next_state, done in trajectory:
        target = reward if done else reward + gamma * V[next_state]
        V[state] += alpha * (target - V[state])


def td_lambda_update(
    V: np.ndarray,
    eligibility: np.ndarray,
    trajectory: list[tuple[int, int, float, int, bool]],
    gamma: float,
    alpha: float,
    lam: float,
) -> None:
    """TD(λ)：用资格迹 e(s) 把当前 TD 误差分配给最近访问过的状态。

    λ=0 时退化为 TD(0)；λ=1 时接近蒙特卡洛。
    """
    for state, _, reward, next_state, done in trajectory:
        target = reward if done else reward + gamma * V[next_state]
        delta = target - V[state]
        eligibility *= gamma * lam
        eligibility[state] += 1.0
        V += alpha * delta * eligibility
