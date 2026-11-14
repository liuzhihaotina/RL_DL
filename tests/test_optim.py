"""手写优化器的正确性测试：与 torch.optim 逐步对照。"""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_optim import SGD, Adam
from rldl.manual_softmax import MLPClassifier, cross_entropy


def _make_model(seed: int = 0) -> MLPClassifier:
    return MLPClassifier(2, 16, 3, seed=seed)


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(32, 2, generator=gen)
    y = torch.randint(0, 3, (32,), generator=gen)
    return x, y


def _run(model, optimizer, steps: int = 5) -> None:
    x, y = _batch()
    for _ in range(steps):
        loss = cross_entropy(model.forward(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_sgd_matches_torch() -> None:
    model = _make_model()
    ref = copy.deepcopy(model)
    _run(model, SGD(model.parameters(), lr=0.1))
    _run(ref, torch.optim.SGD(ref.parameters(), lr=0.1))
    for a, b in zip(model.parameters(), ref.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-7)


def test_sgd_momentum_matches_torch() -> None:
    model = _make_model()
    ref = copy.deepcopy(model)
    _run(model, SGD(model.parameters(), lr=0.05, momentum=0.9))
    _run(ref, torch.optim.SGD(ref.parameters(), lr=0.05, momentum=0.9))
    for a, b in zip(model.parameters(), ref.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-7)


def test_adam_matches_torch() -> None:
    model = _make_model()
    ref = copy.deepcopy(model)
    _run(model, Adam(model.parameters(), lr=0.01))
    _run(ref, torch.optim.Adam(ref.parameters(), lr=0.01))
    for a, b in zip(model.parameters(), ref.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-7)
