# 学习路线

原则：**先手写，再用库，再做对比**。每个阶段都要有一个能跑的实验脚本和可量化的验收标准。

---

## 阶段 0：环境与脚手架 ✅

- [x] uv + Python 3.11 + CUDA 版 PyTorch
- [x] `src/rldl` 包结构、pytest、ruff、`.gitignore`
- [x] 冒烟测试：设备检测、手写 MLP 前向/反向

验收：`uv sync && uv run pytest && uv run python scripts/01_mlp_moons.py` 全部通过。

---

## 阶段 1：深度学习基础（手写）

目标：把「一层网络到底在算什么」彻底搞清楚。

- [x] 张量、广播、autograd 计算图
- [x] 手写 `Linear` + Kaiming 初始化 + ReLU + sigmoid
- [x] 手写 MLP 前向、BCEWithLogits 损失、反向传播、SGD + momentum 更新
- [x] 数据集生成与 train/val 划分、固定种子
- [x] 对比有放回 / 无放回采样，理解 epoch（`scripts/03_sampling_ab.py`）
- [x] 手写 softmax + 交叉熵，做三分类玩具数据（`scripts/04_softmax_3class.py`）
- [x] 手写 Adam 优化器，与 SGD 对比收敛曲线（`scripts/05_adam_vs_sgd.py`）
- [x] 手写数值梯度检查，验证 autograd 结果（`scripts/08_numerical_gradient.py`）
- [x] 过拟合实验：观察 train/val 曲线分离，加入 weight decay（`scripts/09_weight_decay.py`）

验收：`scripts/01_mlp_moons.py` 在 1500 步内 val_acc ≥ 0.95；能解释每个超参的作用。

---

## 阶段 2：PyTorch 工程化

目标：从「教学代码」过渡到「能长期迭代的工程」。

- [x] `nn.Module` / `Dataset` / `DataLoader` 重写实验 01（`scripts/10_nn_module_mlp.py`）
- [x] MNIST：手写 CNN（conv/pool/BN/dropout），训练到 test_acc ≥ 0.99（`scripts/11_mnist_cnn.py`，0.9909）
- [x] 训练技巧：学习率调度、早停、梯度裁剪、混合精度（AMP）（`scripts/13_training_tricks.py`）
- [x] TensorBoard 记录 loss/acc/学习率/权重直方图（`scripts/12_mnist_tensorboard.py`）
- [x] 实验配置化：`configs/*.toml`，命令行可覆盖

验收：一条命令完成训练 + 验证 + 产物落盘；TensorBoard 能看到完整曲线。

---

## 阶段 3：强化学习基础

目标：不依赖库，手写 RL 的核心循环。

- [x] 多臂老虎机：ε-greedy、UCB、Thompson sampling（`scripts/14_bandit.py`）
- [ ] MDP、贝尔曼方程、策略迭代 / 价值迭代（GridWorld）
- [ ] 无模型预测：蒙特卡洛、TD(0)、TD(λ)
- [ ] 无模型控制：SARSA、Q-learning（FrozenLake / CliffWalking）
- [ ] 重要性采样与 off-policy 的概念验证

验收：GridWorld 上策略收敛到最优；能画出价值函数热力图。

---

## 阶段 4：深度强化学习

目标：在 gymnasium 环境中实现并对比深度 RL 算法。

- [ ] DQN + target network + replay buffer（CartPole）
- [ ] Double DQN / Dueling DQN 消融对比
- [ ] REINFORCE + baseline（CartPole）
- [ ] Actor-Critic / A2C
- [ ] PPO（连续动作：Pendulum 或 LunarLander）
- [ ] 训练稳定性：多 seed 均值 ± 方差曲线

验收：每个算法 ≥ 3 个 seed，报告 return 曲线和最终性能；代码复用同一套训练循环。

---

## 阶段 5：综合项目

任选其一：

- [ ] 从零复现一篇 RL 论文的核心实验
- [ ] 自建环境（自定义 gymnasium Env）+ 训练 + 可视化
- [ ] 把视觉输入（CNN）接入 RL（如 CartPole 像素版 / Atari 单游戏）

验收：README 写清方法、超参、结果与失败尝试；他人可按文档复现。

---

## 记录模板

每完成一个实验，在下方追加：

```text
### YYYY-MM-DD 实验名
- 目标：
- 设置（关键超参）：
- 结果（指标 + 曲线位置）：
- 结论 / 踩坑：
```
