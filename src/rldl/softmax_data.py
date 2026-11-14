"""三分类螺旋数据：三条互相缠绕的臂，必须用非线性边界才能分开。"""

from __future__ import annotations

import numpy as np


def make_spiral_3class(
    n_per_class: int = 300,
    noise: float = 0.08,
    turns: float = 1.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 3 类螺旋点。

    Returns:
        X: (3*n_per_class, 2) float32
        y: (3*n_per_class,)  int64，取值 0/1/2
    """
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.05, 1.0, n_per_class)
    angle = radius * turns * 2 * np.pi

    xs, ys = [], []
    for c in range(3):
        offset = c * 2 * np.pi / 3
        arm = np.stack(
            [radius * np.cos(angle + offset), radius * np.sin(angle + offset)],
            axis=1,
        )
        xs.append(arm)
        ys.append(np.full(n_per_class, c, dtype=np.int64))

    X = np.concatenate(xs, axis=0)
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate(ys)

    idx = rng.permutation(len(X))
    return X[idx].astype(np.float32), y[idx]


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """随机切分训练/验证集（三分类数据量均衡，简单随机切即可）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
