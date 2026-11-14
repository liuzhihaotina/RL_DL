# 环境配置（实测记录）

> 本文记录在本机实际验证通过的步骤，不是通用教程。换机器时请重跑「验证」一节。

## 本机实测环境

| 项 | 值 |
| --- | --- |
| 系统 | Windows 11 |
| CPU / 内存 | Intel i7-14650HX / 15.7 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop，6 GB，算力 sm_89 |
| 驱动 | 616.56（CUDA UMD 13.4） |
| Python | 3.11.0（uv 管理的 `.venv`） |
| uv | 0.11.21 |
| torch | 2.10.0+cu128 |

## 为什么用镜像

本机直连 `download.pytorch.org` 和 `pypi.org` 会间歇性出现 TLS 握手被重置（`tls handshake eof`），
`uv sync` 下载 2.6 GB 的 torch 轮子失败。实测阿里云镜像稳定（首个 MB 约 0.8 s，整包约 13 min）。

因此 `pyproject.toml` 做了两处固定：

1. 默认 PyPI index 指向阿里云：`https://mirrors.aliyun.com/pypi/simple`
2. Windows 上 torch 的 cu128 轮子用阿里云文件镜像的**完整 URL**（`[tool.uv.sources]`，带 `sys_platform` marker）

> 注意：torch 的镜像 URL 绑定 `cp311 / win_amd64 / 2.10.0+cu128`，所以本项目把
> `requires-python` 限定为 `>=3.11,<3.12`。换 Python 版本或系统时，需要同步改这条 URL。

## 从零复现

```bash
# 1. 安装 uv（已安装可跳过）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 在项目根目录安装依赖（首次会下载约 2.7 GB 的 torch）
uv sync

# 3. 验证
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.10.0+cu128 True NVIDIA GeForce RTX 4050 Laptop GPU

uv run pytest
# 期望：3 passed

uv run python scripts/01_mlp_moons.py
# 期望：device = cuda，约 2 秒跑完 1500 步，val_acc ≈ 0.99
```

## 网络回退方案（镜像 URL 失效时）

如果阿里云上该文件 404，可以手动下载同版本轮子再本地安装：

```bash
# 1. 只同步其余依赖（不装 torch）
uv sync --no-install-package torch

# 2. 从可用镜像 / 官方源下载 torch 轮子（文件名必须匹配 Python 版本和平台）
curl -L -C - -o torch-2.10.0+cu128-cp311-cp311-win_amd64.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl"

# 3. 安装本地轮子（依赖已由上一步装好）
uv pip install --no-deps ./torch-2.10.0+cu128-cp311-cp311-win_amd64.whl
```

之后运行命令统一加 `--no-sync`（例如 `uv run --no-sync python ...`），避免 uv 自动同步时
按 lockfile 重新下载 torch。

## 常用命令

```bash
uv sync                 # 按 uv.lock 同步环境
uv add <pkg>            # 新增依赖并更新 lock
uv run ...              # 在项目环境里执行命令（自动同步）
uv run ruff check .
uv run pytest
uv run --no-sync ...    # 跳过自动同步（本地手装 torch 时用）
```
