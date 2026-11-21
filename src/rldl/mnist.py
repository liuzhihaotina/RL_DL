"""MNIST 数据：下载、解析 IDX 文件、封装成 Dataset。

不依赖 torchvision，直接读原始文件，顺便看清数据格式。

训练集：60000 张 28×28 灰度图
测试集：10000 张
标签：0~9
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

MIRRORS = [
    "https://dataset.bj.bcebos.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train-images": "train-images-idx3-ubyte.gz",
    "train-labels": "train-labels-idx1-ubyte.gz",
    "test-images": "t10k-images-idx3-ubyte.gz",
    "test-labels": "t10k-labels-idx1-ubyte.gz",
}

MEAN, STD = 0.1307, 0.3081


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        with tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := response.read(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def ensure_files(root: Path) -> None:
    """缺少哪个文件就下载哪个，依次尝试多个镜像。"""
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        dest = root / name
        if dest.exists():
            continue
        last_error: Exception | None = None
        for base in MIRRORS:
            try:
                print(f"downloading {name} from {base}")
                _download(base + name, dest)
                break
            except Exception as exc:  # noqa: BLE001 - 镜像失败就换下一个
                last_error = exc
                print(f"  failed: {exc}")
        else:
            raise RuntimeError(f"无法下载 {name}，最后错误：{last_error}")


def read_images(path: Path) -> np.ndarray:
    """IDX 图像文件：magic=2051, n, rows, cols，随后是 uint8 像素。"""
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"{path} 不是图像文件（magic={magic}）")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_labels(path: Path) -> np.ndarray:
    """IDX 标签文件：magic=2049, n，随后是 uint8 标签。"""
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"{path} 不是标签文件（magic={magic}）")
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)


def load_mnist(
    root: Path = Path("data/mnist"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X_train, y_train, X_test, y_test)，图像形状 (N, 28, 28)。"""
    ensure_files(root)
    X_train = read_images(root / FILES["train-images"])
    y_train = read_labels(root / FILES["train-labels"])
    X_test = read_images(root / FILES["test-images"])
    y_test = read_labels(root / FILES["test-labels"])
    return X_train, y_train, X_test, y_test


class MNISTDataset(Dataset):
    """把 (N, 28, 28) uint8 图像转成 (N, 1, 28, 28) float 张量。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images.copy()).float().unsqueeze(1)
        self.images = (self.images / 255.0 - MEAN) / STD
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]
