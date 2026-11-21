"""多臂老虎机：环境与三种最基础的策略。

老虎机有 n 个手臂，每个手臂的奖励服从伯努利分布（中奖概率未知）。
目标：在有限步数内最大化总奖励。

策略只依赖两个量：
  counts[i]  手臂 i 被拉过的次数
  values[i]  手臂 i 的平均奖励
"""

from __future__ import annotations

import numpy as np


class BernoulliBandit:
    """n 个手臂，每个手臂以概率 p_i 给奖励 1，否则 0。"""

    def __init__(self, probs: np.ndarray | list[float], seed: int = 0) -> None:
        self.probs = np.asarray(probs, dtype=np.float64)
        self.n_arms = len(self.probs)
        self.best_arm = int(np.argmax(self.probs))
        self.best_prob = float(self.probs[self.best_arm])
        self.rng = np.random.default_rng(seed)

    def step(self, arm: int) -> int:
        return int(self.rng.random() < self.probs[arm])


class EpsilonGreedy:
    """以概率 ε 随机探索，否则选当前平均奖励最高的手臂。"""

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 0) -> None:
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        best = np.flatnonzero(self.values == self.values.max())
        return int(self.rng.choice(best))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB:
    """Upper Confidence Bound：平均奖励 + 不确定性奖励。

    score = values[i] + c * sqrt(ln(t) / counts[i])
    没拉过的手臂 counts=0，给无穷大的分数，保证先各试一次。
    """

    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.values = np.zeros(n_arms, dtype=np.float64)
        self.total = 0

    def select(self) -> int:
        scores = self.values + self.c * np.sqrt(
            np.log(max(self.total, 1)) / np.maximum(self.counts, 1)
        )
        scores[self.counts == 0] = np.inf
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.total += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class ThompsonSampling:
    """每个手臂维护一个 Beta(α, β) 后验，采样后选采样值最大的。

    α 记录成功次数 + 1，β 记录失败次数 + 1。
    一个手臂被选中的概率 = 它「当前看起来最好」的概率。
    """

    def __init__(self, n_arms: int, seed: int = 0) -> None:
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms, dtype=np.float64)
        self.beta = np.ones(n_arms, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


def simulate(
    agent,
    bandit: BernoulliBandit,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跑 steps 步，返回每步奖励、是否选中最佳手臂、累计遗憾。"""
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    regrets = np.zeros(steps)
    cumulative = 0.0

    for t in range(steps):
        arm = agent.select()
        reward = bandit.step(arm)
        agent.update(arm, reward)

        rewards[t] = reward
        optimal[t] = arm == bandit.best_arm
        cumulative += bandit.best_prob - bandit.probs[arm]
        regrets[t] = cumulative

    return rewards, optimal, regrets
