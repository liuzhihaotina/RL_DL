"""无模型预测（MC / TD）公式测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.gridworld import DEFAULT_GRID, GridWorld, policy_evaluation_probs
from rldl.model_free import (
    generate_episode,
    mc_update,
    td0_update,
    td_lambda_update,
)


def test_mc_first_visit_uses_first_occurrence() -> None:
    trajectory = [
        (0, 0, 1.0, 1, False),
        (0, 0, 3.0, 1, False),
        (1, 0, 2.0, 2, True),
    ]
    V = np.zeros(3)
    counts = np.zeros(3, dtype=np.int64)
    mc_update(V, counts, trajectory, gamma=0.5)

    # 第一次访问 state 0 的回报：1 + 0.5*(3 + 0.5*2) = 3
    # 最后一次访问的回报是 3 + 0.5*2 = 4，不应该被使用
    assert V[0] == 3.0
    assert V[1] == 2.0
    assert counts[0] == 1 and counts[1] == 1


def test_td0_update() -> None:
    trajectory = [
        (0, 0, 1.0, 1, False),
        (1, 0, 2.0, 2, True),
    ]
    V = np.array([0.0, 2.0, 0.0])
    td0_update(V, trajectory, gamma=0.5, alpha=0.5)

    # 第一步：target = 1 + 0.5*2 = 2，V0 = 0 + 0.5*(2-0) = 1
    # 第二步：target = 2，V1 已经是 2，不变
    assert V[0] == 1.0
    assert V[1] == 2.0


def test_td_lambda_zero_equals_td0() -> None:
    trajectory = [
        (0, 0, 1.0, 1, False),
        (1, 0, 0.0, 2, False),
        (2, 0, 2.0, 3, True),
    ]
    V_td = np.zeros(4)
    V_lam = np.zeros(4)
    eligibility = np.zeros(4)

    td0_update(V_td, trajectory, gamma=0.9, alpha=0.1)
    td_lambda_update(V_lam, eligibility, trajectory, gamma=0.9, alpha=0.1, lam=0.0)
    assert np.allclose(V_td, V_lam)


def test_generate_episode_starts_at_start() -> None:
    env = GridWorld(DEFAULT_GRID)
    rng = np.random.default_rng(0)
    trajectory = generate_episode(env, rng, max_steps=200)

    assert trajectory[0][0] == env.start
    for _, _, _, next_state, _ in trajectory:
        assert 0 <= next_state < env.n_states
    assert trajectory[-1][4] or len(trajectory) == 200


def test_uniform_policy_value() -> None:
    env = GridWorld(DEFAULT_GRID, gamma=0.95)
    probs = np.ones((env.n_states, env.n_actions)) / env.n_actions
    V = policy_evaluation_probs(env, probs, env.gamma)

    assert V.shape == (env.n_states,)
    assert V[env.goal] == 0.0
    assert V[env.trap] == 0.0
    assert V[env.start] < 0  # 随机走很容易掉进陷阱
