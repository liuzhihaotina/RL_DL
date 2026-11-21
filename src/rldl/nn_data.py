"""自定义 Dataset：把 NumPy 数组包装成 PyTorch 数据集。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    """最简单的 Dataset：按索引返回一条 (x, y)。

    DataLoader 会负责：
      - 打乱顺序（shuffle=True）
      - 切成 batch
      - 把多条样本拼成一个张量
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]
