"""环境冒烟测试：不训练，只验证依赖、设备与手写模型前向可用。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_nn import MLP
from rldl.toy import make_moons, train_val_split
from rldl.utils import pick_device


def test_torch_cuda_report() -> None:
    assert torch.__version__
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")


def test_toy_data() -> None:
    X, y = make_moons(200, noise=0.1, seed=0)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert set(y.tolist()) == {0.0, 1.0}

    X_tr, y_tr, X_val, y_val = train_val_split(X, y, 0.2, seed=0)
    assert len(X_tr) == 160 and len(X_val) == 40
    assert y_tr.shape == (160,) and y_val.shape == (40,)


def test_manual_mlp_forward_and_backward() -> None:
    device = pick_device("auto")
    model = MLP(2, 16, seed=0)
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.to(device)
        layer.bias.data = layer.bias.data.to(device)

    x = torch.randn(8, 2, device=device)
    logits = model.forward(x)
    assert logits.shape == (8,)

    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
