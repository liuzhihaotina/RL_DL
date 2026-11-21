"""SARSA / Q-learning 的基础函数测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.cliffwalking import CliffWalking
from rldl.control import (
    epsilon_greedy,
    evaluate_greedy,
    greedy_action,
    moving_average,
)


def test_greedy_action() -> None:
    Q = np.array([[0.0, 2.0, 1.0, -1.0]])
    assert greedy_action(Q, 0) == 1


def test_epsilon_greedy_no_exploration() -> None:
    Q = np.array([[0.0, 2.0, 1.0, -1.0]])
    rng = np.random.default_rng(0)
    assert all(epsilon_greedy(Q, 0, 0.0, rng) == 1 for _ in range(20))


def test_epsilon_greedy_explores() -> None:
    Q = np.zeros((1, 4))
    rng = np.random.default_rng(0)
    arms = {epsilon_greedy(Q, 0, 1.0, rng) for _ in range(200)}
    assert arms == {0, 1, 2, 3}


def test_evaluate_greedy_with_handcrafted_q() -> None:
    env = CliffWalking()
    Q = np.zeros((env.n_states, env.n_actions))

    # 起点向上，第 2 行一路向右，最后一列向下进终点
    Q[env.start, 0] = 1.0
    for col in range(env.n_cols - 1):
        Q[env.to_state((2, col)), 1] = 1.0
    Q[env.to_state((2, env.n_cols - 1)), 2] = 1.0

    total, success, path = evaluate_greedy(env, Q)
    assert success
    assert total == -13.0
    assert path[0] == env.start
    assert path[-1] == env.goal


def test_moving_average_length() -> None:
    values = list(range(10))
    smoothed = moving_average(values, window=3)
    assert len(smoothed) == 8
    assert np.isclose(smoothed[0], 1.0)
