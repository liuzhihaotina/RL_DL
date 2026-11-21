"""GridWorld 环境与动态规划测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.gridworld import (
    DEFAULT_GRID,
    GridWorld,
    policy_iteration,
    value_iteration,
)


def test_move_into_wall_stays() -> None:
    env = GridWorld(DEFAULT_GRID)
    next_state, reward, done = env.step(env.start, 0)  # 向上越界
    assert next_state == env.start
    assert reward == env.step_reward
    assert not done


def test_normal_move_has_step_cost() -> None:
    env = GridWorld(DEFAULT_GRID)
    next_state, reward, done = env.step(env.start, 1)  # 向右
    assert next_state == env.to_state((0, 1))
    assert reward == env.step_reward
    assert not done


def test_trap_and_goal_are_terminal() -> None:
    env = GridWorld(DEFAULT_GRID)

    next_state, reward, done = env.step(env.to_state((2, 1)), 1)  # 进入陷阱
    assert next_state == env.trap
    assert reward == env.trap_reward
    assert done

    next_state, reward, done = env.step(env.to_state((3, 3)), 2)  # 向下进入终点
    assert next_state == env.goal
    assert reward == env.goal_reward
    assert done


def test_terminal_state_step_is_zero() -> None:
    env = GridWorld(DEFAULT_GRID)
    next_state, reward, done = env.step(env.goal, 0)
    assert next_state == env.goal
    assert reward == 0.0
    assert done


def test_value_iteration_and_policy_iteration_agree() -> None:
    env = GridWorld(DEFAULT_GRID, gamma=0.95)
    pi_policy, pi_V, _ = policy_iteration(env, env.gamma)
    vi_policy, vi_V, _, deltas = value_iteration(env, env.gamma)

    assert pi_policy.shape == (env.n_states,)
    assert len(deltas) > 0
    assert np.allclose(pi_V, vi_V, atol=1e-8)
    assert np.array_equal(pi_policy, vi_policy)

    assert pi_V[env.start] > 0
    assert pi_V[env.goal] == 0.0  # 终止状态的价值定义为 0
    assert pi_V.max() <= 1.0 + 1e-9


def test_render_policy_shape() -> None:
    env = GridWorld(DEFAULT_GRID)
    policy = np.zeros(env.n_states, dtype=int)
    text = env.render_policy(policy)
    assert len(text.splitlines()) == env.n_rows
