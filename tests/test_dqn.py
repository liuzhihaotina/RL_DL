"""DQN 组件测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.dqn import QNetwork, ReplayBuffer, epsilon_by_step


def test_qnetwork_forward_shape() -> None:
    net = QNetwork(obs_dim=4, n_actions=2, hidden=32)
    out = net.forward(torch.randn(8, 4))
    assert out.shape == (8, 2)


def test_replay_buffer_push_and_sample() -> None:
    buffer = ReplayBuffer(capacity=5, obs_dim=3, seed=0)
    for i in range(3):
        buffer.push(
            state=np.full(3, i, dtype=np.float32),
            action=i % 2,
            reward=float(i),
            next_state=np.full(3, i + 1, dtype=np.float32),
            done=(i == 2),
        )
    assert buffer.size == 3

    batch = buffer.sample(2)
    assert batch["states"].shape == (2, 3)
    assert batch["actions"].shape == (2,)
    assert batch["rewards"].shape == (2,)
    assert batch["next_states"].shape == (2, 3)
    assert batch["dones"].shape == (2,)


def test_replay_buffer_wraps_around() -> None:
    buffer = ReplayBuffer(capacity=3, obs_dim=2, seed=0)
    for i in range(5):
        buffer.push(np.zeros(2), i % 2, 0.0, np.zeros(2), False)
    assert buffer.size == 3
    assert buffer.pos == 2


def test_epsilon_schedule() -> None:
    assert epsilon_by_step(0, 1.0, 0.05, 100) == pytest.approx(1.0)
    assert epsilon_by_step(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert epsilon_by_step(100, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert epsilon_by_step(200, 1.0, 0.05, 100) == pytest.approx(0.05)
