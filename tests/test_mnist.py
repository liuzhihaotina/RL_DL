"""MNIST 解析 / Dataset / CNN 前向测试。"""

import gzip
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.cnn import SmallCNN
from rldl.mnist import MNISTDataset, read_images, read_labels


def _write_images(path: Path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, *images.shape))
        f.write(images.tobytes())


def _write_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(labels.tobytes())


def test_read_images_and_labels(tmp_path: Path) -> None:
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    labels = np.array([5, 9], dtype=np.uint8)
    img_path, lbl_path = tmp_path / "images.gz", tmp_path / "labels.gz"
    _write_images(img_path, images)
    _write_labels(lbl_path, labels)

    loaded_images = read_images(img_path)
    loaded_labels = read_labels(lbl_path)
    assert loaded_images.shape == (2, 3, 4)
    assert np.array_equal(loaded_images, images)
    assert np.array_equal(loaded_labels, labels)


def test_mnist_dataset_item() -> None:
    images = np.random.randint(0, 255, (4, 28, 28), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3], dtype=np.uint8)
    ds = MNISTDataset(images, labels)
    assert len(ds) == 4

    x, y = ds[0]
    assert x.shape == (1, 28, 28)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_small_cnn_forward_and_params() -> None:
    model = SmallCNN(10)
    out = model.forward(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 421_834
