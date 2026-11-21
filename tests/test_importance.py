"""重要性采样工具测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.importance import ordinary_is, trajectory_ratio, weighted_is


def test_ratio_is_one_when_policies_match() -> None:
    probs = np.array([[0.3, 0.7], [1.0, 0.0]])
    ratio = trajectory_ratio(probs, probs, [0, 0], [1, 0])
    assert ratio == 1.0


def test_ratio_product() -> None:
    target = np.array([[1.0, 0.0], [0.8, 0.2]])
    behavior = np.array([[0.5, 0.5], [0.5, 0.5]])
    ratio = trajectory_ratio(target, behavior, [0, 1], [0, 0])
    assert ratio == pytest.approx(2.0 * 1.6)


def test_behavior_must_cover_target() -> None:
    target = np.array([[1.0, 0.0]])
    behavior = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        trajectory_ratio(target, behavior, [0], [1])


def test_is_estimators() -> None:
    returns = np.array([1.0, 2.0, 3.0])
    ratios = np.array([1.0, 0.0, 2.0])

    assert ordinary_is(returns, ratios) == pytest.approx(7.0 / 3.0)
    assert weighted_is(returns, ratios) == pytest.approx(7.0 / 3.0)
