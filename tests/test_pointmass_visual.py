"""视觉 PointMass 环境与 CNN 策略测试。"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.pointmass import PointMassEnv
from rldl.pointmass_visual import PointMassPixelEnv
from rldl.visual_policy import VisualActorCritic


def test_pixel_observation_shape_and_range() -> None:
    env = PointMassPixelEnv(size=32)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (3, 32, 32)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0
    assert obs.max() > 0.5  # 至少能看到点和目标


def test_visual_network_shapes() -> None:
    net = VisualActorCritic(action_dim=2, size=32)
    x = torch.randn(4, 3, 32, 32)
    mu = net.forward(x)
    assert mu.shape == (4, 2)

    dist = net.distribution(x)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4, 2)
    assert torch.isfinite(log_prob).all()

    value = net.value(x)
    assert value.shape == (4,)


def test_bounds_clip_position_and_velocity() -> None:
    env = PointMassEnv(bounds=1.0)
    env.reset(seed=0)
    env.pos = np.array([0.99, 0.0], dtype=np.float32)
    env.vel = np.array([0.5, 0.0], dtype=np.float32)

    env.step(np.array([0.0, 0.0]))
    assert env.pos[0] == 1.0
    assert env.vel[0] == 0.0
