"""nn.Module / Dataset / DataLoader 基础测试。"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.nn_data import ArrayDataset
from rldl.nn_models import BinaryMLP
from rldl.toy import make_moons, train_val_split


def test_dataset_len_and_getitem() -> None:
    X, y = make_moons(100, seed=0)
    ds = ArrayDataset(X, y)
    assert len(ds) == 100
    x0, y0 = ds[0]
    assert x0.shape == (2,)
    assert y0.shape == ()
    assert x0.dtype == torch.float32


def test_dataloader_batches_and_coverage() -> None:
    X, y = make_moons(800, seed=0)
    ds = ArrayDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    batches = list(loader)
    assert len(batches) == 13  # 12 个 64 + 1 个 32
    assert batches[0][0].shape == (64, 2)
    assert batches[-1][0].shape == (32, 2)

    seen = torch.cat([xb for xb, _ in batches])
    assert len(torch.unique(seen, dim=0)) == 800  # 一个 epoch 每条恰好一次


def test_model_forward_and_params() -> None:
    model = BinaryMLP(2, 32)
    out = model.forward(torch.randn(8, 2))
    assert out.shape == (8, 1)

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 1185  # 2*32+32 + 32*32+32 + 32*1+1


def test_train_val_split_for_loader() -> None:
    X, y = make_moons(1000, seed=0)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, 0.2, seed=0)
    assert len(ArrayDataset(X_tr, y_tr)) == 800
    assert len(ArrayDataset(X_val, y_val)) == 200
