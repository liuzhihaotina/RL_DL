"""读取 TOML 实验配置。

Python 3.11 自带 tomllib，不需要额外依赖。

配置优先级：命令行参数 > TOML 文件 > 代码里的默认值。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """读取 TOML 文件，返回嵌套字典。"""
    with open(path, "rb") as f:
        return tomllib.load(f)
