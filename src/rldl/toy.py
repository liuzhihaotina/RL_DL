"""玩具数据集：不依赖 sklearn 的两月牙（two moons）分类数据。"""

from __future__ import annotations

import numpy as np


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成两个交错的半月形，标签 0/1。

    Returns:
        X: (n, 2) float32
        y: (n,)  float32（BCE 用）
    """
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    theta_out = rng.uniform(0.0, np.pi, n_out)
    theta_in = rng.uniform(0.0, np.pi, n_in)

    outer = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)
    inner = np.stack([1.0 - np.cos(theta_in), 0.5 - np.sin(theta_in)], axis=1)

    X = np.concatenate([outer, inner], axis=0)
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)])
    return X.astype(np.float32), y.astype(np.float32)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """随机打乱后按比例切分训练/验证集。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
