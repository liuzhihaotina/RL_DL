"""CliffWalking 环境测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.cliffwalking import CliffWalking


def test_dimensions_and_positions() -> None:
    env = CliffWalking()
    assert env.n_rows == 4 and env.n_cols == 12
    assert env.start == env.to_state((3, 0))
    assert env.goal == env.to_state((3, 11))
    assert env.to_state((3, 1)) in env.cliffs
    assert env.to_state((3, 10)) in env.cliffs


def test_step_into_cliff_resets_to_start() -> None:
    env = CliffWalking()
    next_state, reward, done = env.step(env.start, 1)  # 向右掉下悬崖
    assert next_state == env.start
    assert reward == env.cliff_reward
    assert not done


def test_normal_step() -> None:
    env = CliffWalking()
    next_state, reward, done = env.step(env.start, 0)  # 向上
    assert next_state == env.to_state((2, 0))
    assert reward == env.step_reward
    assert not done


def test_reach_goal() -> None:
    env = CliffWalking()
    next_state, reward, done = env.step(env.to_state((3, 10)), 1)
    assert next_state == env.goal
    assert reward == env.step_reward
    assert done
