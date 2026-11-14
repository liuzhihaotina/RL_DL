"""softmax / 交叉熵 / 三分类数据的正确性测试。"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.manual_softmax import MLPClassifier, cross_entropy, log_softmax, softmax
from rldl.softmax_data import make_spiral_3class, split_data


def test_softmax_is_a_distribution() -> None:
    logits = torch.randn(32, 5)
    p = softmax(logits)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=1), torch.ones(32), atol=1e-6)
    assert torch.allclose(p, torch.softmax(logits, dim=1), atol=1e-6)


def test_log_softmax_is_numerically_stable() -> None:
    logits = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    out = log_softmax(logits)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.log_softmax(logits, dim=1), atol=1e-6)


def test_cross_entropy_matches_torch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(16, 4) * 3
    targets = torch.randint(0, 4, (16,))
    manual = cross_entropy(logits, targets)
    builtin = F.cross_entropy(logits, targets)
    assert torch.allclose(manual, builtin, atol=1e-6)

    with torch.no_grad():
        log_probs = torch.log_softmax(logits, dim=1)
        expected = -log_probs.gather(1, targets[:, None]).mean()
    assert torch.allclose(manual, expected, atol=1e-6)


def test_mlp_classifier_shapes_and_grads() -> None:
    model = MLPClassifier(2, 16, 3, seed=0)
    x = torch.randn(8, 2)
    logits = model.forward(x)
    assert logits.shape == (8, 3)

    loss = cross_entropy(logits, torch.randint(0, 3, (8,)))
    loss.backward()
    assert all(p.grad is not None and p.grad.shape == p.shape for p in model.parameters())


def test_spiral_data() -> None:
    X, y = make_spiral_3class(100, noise=0.05, seed=0)
    assert X.shape == (300, 2)
    assert y.shape == (300,)
    assert set(y.tolist()) == {0, 1, 2}

    X_tr, y_tr, X_val, y_val = split_data(X, y, 0.2, seed=0)
    assert len(X_tr) == 240 and len(X_val) == 60
    assert set(y_tr.tolist()) == {0, 1, 2}
