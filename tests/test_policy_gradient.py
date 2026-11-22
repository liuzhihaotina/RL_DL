"""策略梯度组件测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.policy_gradient import (
    PolicyNetwork,
    ValueNetwork,
    compute_returns,
    n_step_returns,
    reinforce_loss,
)


def test_compute_returns() -> None:
    rewards = [1.0, 1.0, 1.0]
    returns = compute_returns(rewards, gamma=0.5)
    assert np.allclose(returns, [1.75, 1.5, 1.0])


def test_policy_network_output() -> None:
    policy = PolicyNetwork(obs_dim=4, n_actions=2, hidden=16)
    logits = policy.forward(torch.randn(8, 4))
    assert logits.shape == (8, 2)

    dist = policy.distribution(torch.randn(1, 4))
    probs = dist.probs
    assert probs.shape == (1, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)


def test_value_network_output() -> None:
    value = ValueNetwork(obs_dim=4, hidden=16)
    out = value.forward(torch.randn(8, 4))
    assert out.shape == (8,)


def test_baseline_equal_returns_gives_zero_advantage() -> None:
    log_probs = [torch.tensor(-0.7), torch.tensor(-0.6)]
    returns = np.array([1.0, 2.0], dtype=np.float32)
    values = torch.tensor([1.0, 2.0])
    loss = reinforce_loss(log_probs, returns, values=values, normalize=False)
    assert abs(loss.item()) < 1e-6


def test_entropy_bonus_reduces_loss() -> None:
    log_probs = [torch.tensor(-0.7), torch.tensor(-0.6)]
    returns = np.array([1.0, 2.0], dtype=np.float32)
    entropy = [torch.tensor(0.5), torch.tensor(0.5)]

    without = reinforce_loss(log_probs, returns, normalize=False)
    with_bonus = reinforce_loss(
        log_probs, returns, entropy=entropy, normalize=False, entropy_coef=0.1
    )
    assert with_bonus.item() < without.item()



def test_n_step_returns_with_bootstrap() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    dones = np.array([False, False, False])
    returns = n_step_returns(rewards, values, dones, gamma=0.5, n=2)

    assert returns[0] == pytest.approx(1.0 + 0.5 * 1.0 + 0.25 * 10.0)
    assert returns[1] == pytest.approx(1.0 + 0.5 * 1.0)
    assert returns[2] == pytest.approx(1.0)


def test_n_step_returns_stops_at_terminal() -> None:
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([10.0, 10.0], dtype=np.float32)
    dones = np.array([False, True])
    returns = n_step_returns(rewards, values, dones, gamma=0.5, n=2)

    assert returns[0] == pytest.approx(1.0 + 0.5 * 1.0)
    assert returns[1] == pytest.approx(1.0)
