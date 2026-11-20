"""数值梯度检查：验证 autograd 与中心差分一致。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.gradcheck import numerical_gradients, relative_error
from rldl.manual_softmax import MLPClassifier, cross_entropy


def test_scalar_function() -> None:
    w = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    loss = w**3 + 2 * w**2
    loss.backward()

    (numeric,) = numerical_gradients([w], lambda: w**3 + 2 * w**2, eps=1e-6)
    assert relative_error(w.grad, numeric) < 1e-8


def test_mlp_gradients_match_numerical() -> None:
    torch.manual_seed(0)
    model = MLPClassifier(2, 4, 3, seed=0)
    for layer in (model.fc1, model.fc2, model.fc3):
        layer.weight.data = layer.weight.data.double()
        layer.bias.data = layer.bias.data.double()

    gen = torch.Generator().manual_seed(0)
    X = torch.randn(8, 2, generator=gen, dtype=torch.float64)
    y = torch.randint(0, 3, (8,), generator=gen)

    loss = cross_entropy(model.forward(X), y)
    for p in model.parameters():
        p.grad = None
    loss.backward()
    analytic = [p.grad.clone() for p in model.parameters()]

    numeric = numerical_gradients(
        model.parameters(), lambda: cross_entropy(model.forward(X), y), eps=1e-6
    )
    for a, n in zip(analytic, numeric, strict=True):
        assert relative_error(a, n) < 1e-5
