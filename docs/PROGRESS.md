# 项目进展（阶段 1–4 回顾）

单一进展文档：做到哪、结果是什么、下一步做什么。

---

## 阶段总览

| 阶段 | 主题 | 状态 | 核心产出 |
| --- | --- | --- | --- |
| 1 | 深度学习基础（手写） | ✅ | MLP / softmax / SGD+Adam / 梯度检查 / 正则化 |
| 2 | PyTorch 工程化 | ✅ | nn.Module / DataLoader / MNIST CNN / TensorBoard / AMP |
| 3 | 强化学习基础 | ✅ | bandit / MDP / DP / MC-TD / SARSA-Q / 重要性采样 |
| 4 | 深度强化学习 | ✅ | DQN / Double / Dueling / REINFORCE / A2C / PPO |
| 5 | 综合项目 | 🔧 | 连续动作 PPO + 自建环境已跑通；下一步多 seed 报告 |

---

## 实验清单

### 阶段 1：深度学习基础

| # | 实验 | 关键结果 |
| --- | --- | --- |
| 01 | 手写 MLP 分类 two-moons | val_acc 0.995，1185 参数 |
| 02 | 实验 01 图解 | 流程图 / 网络结构 / 决策边界 |
| 03 | 有放回 vs 无放回采样 | val_acc 0.9933 vs 0.9917；使用次数 std 10.61 vs 0.49 |
| 04 | 手写 softmax + 交叉熵（三分类） | val_acc 0.989，4547 参数 |
| 05 | SGD / momentum / Adam | Adam 150 步到 0.95；momentum 200；SGD ~1400 |
| 06 | w² 上的优化器轨迹 | Adam 步长≈lr，会在最低点附近跳 |
| 07 | Adam 逐步手算 | m/v/偏差修正的每一步数字 |
| 08 | 数值梯度检查 | 与 autograd 误差 7e-8；epsilon U 形曲线；ReLU 折点 |
| 09 | 过拟合与 weight decay | val_acc 0.723 → 0.789；‖w‖² 1248 → 134 |

### 阶段 2：PyTorch 工程化

| # | 实验 | 关键结果 |
| --- | --- | --- |
| 10 | nn.Module + Dataset + DataLoader 重写实验 01 | val_acc 0.990 |
| 11 | MNIST 手写 CNN | test_acc 0.9909，42 万参数，GPU 16 秒 |
| 12 | TensorBoard + TOML 配置 | 标量 / 权重直方图 / 配置快照落盘 |
| 13 | 训练技巧 | baseline 0.9922 / 24.3s；tricks 0.9928 / 20.1s；AMP 小模型无提速 |

### 阶段 3：强化学习基础

| # | 实验 | 关键结果 |
| --- | --- | --- |
| 14 | 多臂老虎机 | UCB 0.761 > Thompson 0.756 > ε-greedy 0.725；UCB c 需匹配奖励尺度 |
| 15 | GridWorld + MDP + DP | 策略迭代 / 价值迭代均 8 轮收敛，价值差 0 |
| 16 | MC / TD(0) / TD(λ) | MC 0.0165，TD0 0.0493，TDλ 0.0634；λ=0 在该任务最好 |
| 17 | SARSA vs Q-learning | SARSA return -17 / 掉崖 45；Q-learning -13 / 掉崖 174 |
| 18 | 重要性采样 | 老虎机朴素 0.439 vs IS 0.800±0.058；链 0.752 vs 1.795/1.796 |

### 阶段 4：深度强化学习

| # | 实验 | 关键结果 |
| --- | --- | --- |
| 19 | DQN / Double DQN | Double 在 episode 320 起连续评估 500；普通 DQN 最好 469.7 且不稳 |
| 20 | Dueling DQN | 负面：标准 320 达标，Dueling 400 才首次 500 |
| 21 | 冗余动作版 Dueling | 仍无优势：lr=5e-4/600 轮，标准 420 达标，Dueling 460 |
| 22 | REINFORCE + baseline | 无 baseline 时塌缩到 return 10；加 baseline 后 ~1150 episodes 达标 |
| 23 | Actor-Critic（n=5） | ~1000 episodes 达标，CPU 约 60 秒；advantage 标准化曾导致梯度 1e-8 |
| 24 | PPO | 27,578 环境步达标，CPU 约 36 秒，曲线最稳 |

---

### 阶段 5：综合项目（准备中）

| # | 实验 | 关键结果 |
| --- | --- | --- |
| 25 | Pendulum 连续动作入门 | 随机 -876.5，零扭矩 -680.0，手调 PD -0.4；确认连续动作接口 |
| 26 | 连续动作 PPO（Pendulum） | 110k 步，best eval -83.7；σ 从 0.95 降到 0.68 |
| 27 | 自建 PointMass 环境 + PPO | 31k 步，best eval -1.5；四个起点都收敛到原点 |

## 方法论总结

1. **控制变量**：只改一个因素，其他固定；比较时看均值 ± 标准差，而不是单次结果。
2. **负面结果要记录**：Dueling 在 CartPole 无优势、AMP 在小模型无提速，都是有效结论。
3. **实现要对照**：手写 softmax/CE、优化器、数值梯度都与 PyTorch 对照过。
4. **指标要选对**：DQN 的 loss 不单调；REINFORCE/A2C 训练回报低但贪心评估可能满分。
5. **方差是主线**：baseline、n 步回报、GAE、PPO 裁剪，本质上都在降方差或限制更新幅度。

---

## 当前仓库状态

```text
提交：28 个
脚本：23 个
文档：23 篇
测试：17 个文件，67 个测试全部通过
工作区：干净
远程：git@github.com:liuzhihaotina/RL_DL.git
```

---

## 下一步（阶段 5 候选）

1. **连续动作 PPO**：Pendulum / LunarLander，策略网络输出高斯均值与标准差；
2. **自建 gymnasium 环境**：自定义状态/动作/奖励，训练 DQN 或 PPO；
3. **多 seed 实验报告**：对 DQN、A2C、PPO 做 3–5 个种子的均值 ± 标准差对比，写成可复现报告。
