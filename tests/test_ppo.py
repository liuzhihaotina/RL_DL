"""GAE 测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.ppo import compute_gae, compute_gae_bootstrap


def test_gae_without_terminal() -> None:
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False])
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.5, lam=0.5)

    # t=1：δ=1，A=1
    # t=0：δ=1，A=1 + 0.5*0.5*1 = 1.25
    assert advantages[1] == pytest.approx(1.0)
    assert advantages[0] == pytest.approx(1.25)
    assert np.allclose(returns, advantages + values)


def test_gae_resets_at_terminal() -> None:
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([False, True])
    advantages, _ = compute_gae(rewards, values, dones, gamma=0.5, lam=0.5)

    assert advantages[1] == pytest.approx(1.0)
    assert advantages[0] == pytest.approx(1.0 + 0.5 * 0.5 * 1.0)


def test_gae_bootstraps_next_value() -> None:
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.0], dtype=np.float32)
    dones = np.array([False])
    advantages, _ = compute_gae(rewards, values, dones, gamma=0.9, lam=0.9)

    # rollout 结束时没有下一个状态，next_value 按 0 处理
    assert advantages[0] == pytest.approx(1.0)



def test_gae_bootstrap_uses_last_value() -> None:
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    advantages, returns = compute_gae_bootstrap(
        rewards, values, last_value=2.0, gamma=0.5, lam=0.5
    )

    # t=1：δ=1+0.5*2=2，A=2
    # t=0：δ=1+0.5*0=1，A=1+0.25*2=1.5
    assert advantages[1] == pytest.approx(2.0)
    assert advantages[0] == pytest.approx(1.5)
    assert np.allclose(returns, advantages + values)
