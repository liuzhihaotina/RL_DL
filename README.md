# RL_DL

从零学习深度学习与强化学习：手写模型和训练循环，逐步实现 RL 算法，而不是只调库。

## 环境

| 项 | 值 |
| --- | --- |
| 系统 | Windows 11 |
| Python | 3.11（由 uv 管理） |
| 包管理 | [uv](https://docs.astral.sh/uv/) |
| GPU | NVIDIA RTX 4050 Laptop，6 GB 显存 |
| PyTorch | 2.10.0+cu128（Windows 走阿里云轮子镜像，见 [docs/setup.md](docs/setup.md)） |

> 环境从零配置、镜像说明和验证命令见 [docs/setup.md](docs/setup.md)。

## 快速开始

```bash
# 1. 安装依赖（含 CUDA 版 torch，首次约 2~3 GB）
uv sync

# 2. 跑通第一个实验：手写 MLP 分类 two-moons（GPU 上约 2 秒）
uv run python scripts/01_mlp_moons.py

# 强制用 CPU / 换超参
uv run python scripts/01_mlp_moons.py --device cpu --steps 1000 --lr 0.1

# 3. 运行测试
uv run pytest
```

实验产物（曲线图、模型权重、数据快照）写到 `outputs/`，已被 `.gitignore` 忽略。

## 目录结构

```text
RL_DL/
├── pyproject.toml          # 依赖、uv 配置、ruff/pytest 配置
├── README.md
├── docs/
│   └── roadmap.md          # 学习路线与阶段验收标准
├── configs/                # 后续放实验配置（TOML/YAML）
├── src/rldl/
│   ├── toy.py              # 玩具数据集（two-moons 等）
│   ├── manual_nn.py        # 手写 Linear / ReLU / MLP（只用 Tensor + autograd）
│   └── utils.py            # 设备选择、随机种子
├── scripts/
│   ├── 01_mlp_moons.py       # 实验 01：手写 MLP 训练与评估
│   ├── 02_visual_explain.py  # 生成实验 01 的三张图解
│   └── 03_sampling_ab.py     # 实验 03：有放回 vs 无放回采样对比
├── tests/
│   └── test_smoke.py       # 环境冒烟测试
└── outputs/                # 实验产物（本地生成，不入库）
```

## 图解

实验 01 的流程图、网络结构、决策边界见 [docs/01_walkthrough.md](docs/01_walkthrough.md)。
图由脚本生成，改完超参后重新跑：

```bash
uv run python scripts/02_visual_explain.py
```

## 学习路线

见 [docs/roadmap.md](docs/roadmap.md)。当前进度：

- [x] 阶段 0：环境与仓库脚手架
- [ ] 阶段 1：深度学习基础（张量、autograd、手写 MLP/优化器）
- [ ] 阶段 2：PyTorch 工程化（nn.Module、DataLoader、CNN）
- [ ] 阶段 3：强化学习基础（bandit、MDP、DP、MC/TD）
- [ ] 阶段 4：深度强化学习（DQN、REINFORCE、PPO）
- [ ] 阶段 5：综合项目

## 约定

- 每个实验一个脚本，编号递增（`01_`、`02_`…），超参用 argparse 暴露。
- 固定随机种子，训练/验证集划分可复现。
- 数据、权重、日志、`outputs/` 一律不入库；只提交代码、文档和配置。
- 每完成一个阶段，在 roadmap 里勾选并写一句结论（学到了什么、卡在哪里）。
