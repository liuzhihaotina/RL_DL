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
│   ├── roadmap.md          # 学习路线与阶段验收标准
│   ├── setup.md            # 实测环境、镜像原因、复现与回退命令
│   ├── 01_walkthrough.md   # 实验 01 图解与逐段讲解
│   ├── 03_sampling_ab.md   # 有放回 vs 无放回采样的对比结论
│   ├── 04_softmax_3class.md  # 手写 softmax + 交叉熵讲解
│   ├── 05_adam_vs_sgd.md   # 手写 Adam 与 SGD 的对比
│   ├── 08_numerical_gradient.md  # 数值梯度检查
│   ├── 09_weight_decay.md  # 过拟合与 weight decay
│   ├── 10_nn_module_rewrite.md  # nn.Module / DataLoader 重写实验 01
│   ├── 11_mnist_cnn.md     # MNIST 手写数字 CNN
│   ├── 12_tensorboard_config.md  # TensorBoard 与 TOML 配置
│   └── images/             # 文档用图
├── configs/                # 实验配置（mnist_cnn.toml）
├── src/rldl/
│   ├── toy.py              # 玩具数据集（two-moons）
│   ├── manual_nn.py        # 手写 Linear / ReLU / MLP（只用 Tensor + autograd）
│   ├── manual_softmax.py   # 手写 softmax / 交叉熵 / 多分类 MLP
│   ├── manual_optim.py     # 手写 SGD / momentum / Adam
│   ├── gradcheck.py        # 数值梯度检查工具
│   ├── softmax_data.py     # 三分类螺旋数据
│   ├── nn_models.py        # nn.Module 版 MLP
│   ├── nn_data.py          # 自定义 Dataset
│   ├── cnn.py              # MNIST 用的小型 CNN
│   ├── mnist.py            # MNIST 下载、解析、Dataset
│   ├── config.py           # TOML 配置加载
│   └── utils.py            # 设备选择、随机种子
├── scripts/
│   ├── 01_mlp_moons.py       # 实验 01：二分类 MLP
│   ├── 02_visual_explain.py  # 生成实验 01 的图解
│   ├── 03_sampling_ab.py     # 实验 03：有放回 vs 无放回采样
│   ├── 04_softmax_3class.py  # 实验 04：手写 softmax + 交叉熵（三分类）
│   ├── 05_adam_vs_sgd.py     # 实验 05：SGD / momentum / Adam 对比
│   ├── 06_optimizer_demo.py  # 最小例子：w² 上逐步对比两种优化器
│   ├── 07_adam_step_by_step.py  # 逐步打印 Adam 的每个中间量
│   ├── 08_numerical_gradient.py # 实验 08：数值梯度检查
│   ├── 09_weight_decay.py       # 实验 09：过拟合与 weight decay
│   ├── 10_nn_module_mlp.py      # 实验 10：标准的 nn.Module + DataLoader 写法
│   ├── 11_mnist_cnn.py          # 实验 11：MNIST CNN
│   └── 12_mnist_tensorboard.py  # 实验 12：TensorBoard + TOML 配置
├── tests/
│   ├── test_smoke.py         # 环境与手写 MLP 冒烟测试
│   ├── test_softmax.py       # softmax / 交叉熵正确性测试
│   ├── test_optim.py         # 手写优化器与 torch.optim 对照
│   ├── test_gradcheck.py     # 数值梯度与 autograd 对照
│   ├── test_nn_basics.py     # nn.Module / Dataset / DataLoader
│   ├── test_mnist.py         # MNIST 解析 / Dataset / CNN
│   └── test_config.py        # TOML 配置加载
└── outputs/                # 实验产物（本地生成，不入库）
```

## 图解

实验 01 的流程图、网络结构、决策边界见 [docs/01_walkthrough.md](docs/01_walkthrough.md)；
三分类的手写 softmax 讲解见 [docs/04_softmax_3class.md](docs/04_softmax_3class.md)，
优化器对比见 [docs/05_adam_vs_sgd.md](docs/05_adam_vs_sgd.md)，
数值梯度检查见 [docs/08_numerical_gradient.md](docs/08_numerical_gradient.md)，
过拟合与 weight decay 见 [docs/09_weight_decay.md](docs/09_weight_decay.md)，
标准写法重写实验 01 见 [docs/10_nn_module_rewrite.md](docs/10_nn_module_rewrite.md)，
MNIST CNN 见 [docs/11_mnist_cnn.md](docs/11_mnist_cnn.md)，
TensorBoard 与配置见 [docs/12_tensorboard_config.md](docs/12_tensorboard_config.md)。
图由脚本生成，改完超参后重新跑：

```bash
uv run python scripts/02_visual_explain.py
```

## 学习路线

见 [docs/roadmap.md](docs/roadmap.md)。当前进度：

- [x] 阶段 0：环境与仓库脚手架
- [x] 阶段 1：深度学习基础（张量、autograd、手写 MLP/优化器）
- [ ] 阶段 2：PyTorch 工程化（nn.Module、DataLoader、CNN）
- [ ] 阶段 3：强化学习基础（bandit、MDP、DP、MC/TD）
- [ ] 阶段 4：深度强化学习（DQN、REINFORCE、PPO）
- [ ] 阶段 5：综合项目

## 约定

- 每个实验一个脚本，编号递增（`01_`、`02_`…），超参用 argparse 暴露。
- 固定随机种子，训练/验证集划分可复现。
- 数据、权重、日志、`outputs/` 一律不入库；只提交代码、文档和配置。
- 每完成一个阶段，在 roadmap 里勾选并写一句结论（学到了什么、卡在哪里）。
