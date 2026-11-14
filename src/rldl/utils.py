"""设备与随机种子的统一入口。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def pick_device(preferred: str = "auto") -> torch.device:
    """auto: 有 CUDA 用 CUDA，否则 CPU。"""
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    return torch.device(preferred)


def set_seed(seed: int = 0) -> None:
    """固定 python / numpy / torch 的随机性，保证实验可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
