"""多臂老虎机环境与策略测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.bandit import (
    UCB,
    BernoulliBandit,
    EpsilonGreedy,
    ThompsonSampling,
    simulate,
)


def test_bernoulli_bandit_deterministic() -> None:
    bandit = BernoulliBandit([1.0, 0.0], seed=0)
    assert bandit.best_arm == 0
    assert bandit.step(0) == 1
    assert bandit.step(1) == 0


def test_epsilon_greedy_exploits_when_epsilon_zero() -> None:
    agent = EpsilonGreedy(5, epsilon=0.0, seed=0)
    agent.values[:] = [0.1, 0.2, 0.9, 0.3, 0.4]
    assert all(agent.select() == 2 for _ in range(20))


def test_epsilon_greedy_explores() -> None:
    agent = EpsilonGreedy(5, epsilon=1.0, seed=0)
    arms = {agent.select() for _ in range(100)}
    assert arms == {0, 1, 2, 3, 4}


def test_ucb_tries_each_arm_once_first() -> None:
    agent = UCB(4, c=0.5)
    chosen = []
    for _ in range(4):
        arm = agent.select()
        chosen.append(arm)
        agent.update(arm, 0)
    assert chosen == [0, 1, 2, 3]


def test_thompson_updates_posterior() -> None:
    agent = ThompsonSampling(3, seed=0)
    arm = agent.select()
    assert 0 <= arm < 3
    before = agent.alpha.sum() + agent.beta.sum()
    agent.update(arm, 1)
    after = agent.alpha.sum() + agent.beta.sum()
    assert after == before + 1


def test_simulate_shapes_and_regret() -> None:
    bandit = BernoulliBandit([0.2, 0.8], seed=0)
    agent = EpsilonGreedy(2, epsilon=0.1, seed=0)
    rewards, optimal, regret = simulate(agent, bandit, steps=50)

    assert rewards.shape == (50,)
    assert optimal.shape == (50,)
    assert regret.shape == (50,)
    assert np.all(np.diff(regret) >= -1e-12)
    assert set(np.unique(optimal)) <= {0.0, 1.0}
