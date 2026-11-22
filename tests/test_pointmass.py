"""PointMass 自建环境测试。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.pointmass import PointMassEnv


def test_spaces() -> None:
    env = PointMassEnv()
    assert env.observation_space.shape == (4,)
    assert env.action_space.shape == (2,)
    assert env.action_space.low.tolist() == [-1.0, -1.0]
    assert env.action_space.high.tolist() == [1.0, 1.0]


def test_reset_is_deterministic_with_seed() -> None:
    env = PointMassEnv()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2)
    assert obs1.shape == (4,)


def test_step_moves_and_penalizes_distance() -> None:
    env = PointMassEnv()
    env.reset(seed=0)
    env.pos = np.array([1.0, 0.0], dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    assert obs.shape == (4,)
    assert reward < 0
    assert not terminated and not truncated
    assert info["distance"] > 0


def test_terminates_near_origin() -> None:
    env = PointMassEnv()
    env.reset(seed=0)
    env.pos = np.array([0.01, 0.0], dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)

    _, _, terminated, _, info = env.step(np.array([0.0, 0.0]))
    assert terminated
    assert info["distance"] < 0.05


def test_truncates_at_max_steps() -> None:
    env = PointMassEnv(max_steps=3)
    env.reset(seed=0)
    env.pos = np.array([1.0, 1.0], dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)

    done = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        done = terminated or truncated
    assert done
    assert truncated
