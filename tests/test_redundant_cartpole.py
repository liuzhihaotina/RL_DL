"""冗余动作 CartPole 包装器测试。"""

import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.redundant_cartpole import RedundantActionWrapper


def test_action_mapping() -> None:
    env = gym.make("CartPole-v1")
    wrapped = RedundantActionWrapper(env, n_actions=4)

    assert wrapped.action_space.n == 4
    assert wrapped.action(0) == 0
    assert wrapped.action(1) == 1
    assert wrapped.action(2) == 0
    assert wrapped.action(3) == 1
    wrapped.close()
