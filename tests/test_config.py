"""TOML 配置加载测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.config import load_config


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "exp.toml"
    path.write_text(
        """
[data]
root = "data/mnist"
batch_size = 64

[train]
epochs = 3
lr = 0.001
""",
        encoding="utf-8",
    )

    cfg = load_config(path)
    assert cfg["data"]["root"] == "data/mnist"
    assert cfg["data"]["batch_size"] == 64
    assert cfg["train"]["epochs"] == 3
    assert cfg["train"]["lr"] == 0.001
