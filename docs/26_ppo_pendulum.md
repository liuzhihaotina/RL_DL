# 实验 26：连续动作 PPO 解决 Pendulum

脚本：`scripts/26_ppo_pendulum.py`
代码：`src/rldl/policy_gradient.py`（ContinuousPolicyNetwork）、`src/rldl/ppo.py`（GAE bootstrap）
图：`docs/images/26_ppo_pendulum.png`

---

## 1. 和 CartPole PPO 的区别

| | CartPole | Pendulum |
| --- | --- | --- |
| 动作 | 2 个离散 | 1 个连续实数 |
| 策略分布 | Categorical（softmax） | Normal（μ, σ） |
| 动作范围 | 0/1 | [-2, 2] |
| log_prob | log softmax | 高斯密度 |
| 回合结束 | 真终止 | 200 步截断 |

网络：

```text
共享特征：3 → 128 → 128
μ head：128 → 1
logσ：一个可学习参数（初始 0）
a ~ Normal(μ, exp(logσ))
```

σ 也会被训练：需要探索时大，收敛后变小。

---

## 2. 三个实现细节

**① 动作裁剪**

```text
a_sample = Normal(μ, σ).sample()
a_env = clip(a_sample, -2, 2)
```

环境只接受 [-2,2]，但 `log_prob` 用**未裁剪**的采样值计算，保证和策略分布一致。

**② 截断回合的 GAE**

Pendulum 没有真正终止，只是 200 步截断。不能把最后一步当成 `done`，否则会丢掉之后的 V(s)。

```text
A_t = δ_t + γλ A_{t+1}
最后一步用 V(s_T) 自举，而不是 0
```

**③ 奖励缩放**

Pendulum 每回合回报约 -1000。直接做 value loss 会让梯度很大：

```python
reward = reward * 0.1
```

训练用缩放后的奖励，评估仍然报告原始回报。

---

## 3. 实验设置

| 项 | 值 |
| --- | --- |
| rollout | 2048 步（约 10 个 episode） |
| update epochs | 10 |
| minibatch | 64 |
| γ / λ | 0.99 / 0.95 |
| clip ε | 0.2 |
| lr | 3e-4 |
| entropy | 0 |
| reward scale | 0.1 |
| 总步数上限 | 150000 |
| 评估 | 每 10000 步，贪心跑 3 局 |
| 设备 | CPU |

---

## 4. 结果

```text
step  11000 | train_return=-1193.9 | eval= -895.1 | std=0.946
step  22000 | train_return= -874.3 | eval= -725.0 | std=0.884
step  33000 | train_return= -391.9 | eval=  -83.7 | std=0.838
step  55000 | train_return= -429.5 | eval= -401.0 | std=0.754
step  88000 | train_return= -147.9 | eval=  -87.6 | std=0.719
step 110000 | train_return= -199.7 | eval=  -85.9 | std=0.680
连续 3 次评估达到 -200，提前停止
best eval return = -83.7
CPU 总耗时约 2 分 48 秒
```

![PPO Pendulum](images/26_ppo_pendulum.png)

- 训练回报从 -1800 上升到 -150 左右；
- 评估在 -84 附近波动，最好 -83.7；
- 学习到的 σ 从 0.95 降到 0.68，探索变小。

---

## 5. 和手调 PD 的对比

实验 25 的 PD 控制器：

```text
return = -0.4
```

PPO：

```text
return ≈ -84
```

PD 更好，因为它是人根据「角度、角速度」手工设计的；PPO 完全从 reward 里学，没有得到动力学公式。

这不是 PPO 失败：

- -84 已经远好于随机（-876）和零扭矩（-680）；
- 继续训练/调参可能进一步接近 0；
- PD 的 -0.4 是「已知模型 + 人工调参」的上界参考。

---

## 6. 常见坑

1. **log_prob 用裁剪后的动作**：应该用采样时的原始动作；
2. **截断当终止**：Pendulum 每回合 200 步截断，最后一步要 bootstrap；
3. **不缩放奖励**：value loss 会非常大，训练不稳；
4. **logσ 不限制**：可能变得极大或极小，导致 log_prob 数值问题；
5. **评估用随机策略**：应该用 μ（确定性动作）评估。

---

## 7. 自己动手

```bash
# 默认
uv run python scripts/26_ppo_pendulum.py

# 训练更久，看能否接近 0
uv run python scripts/26_ppo_pendulum.py --total-steps 300000

# 不加奖励缩放
uv run python scripts/26_ppo_pendulum.py --reward-scale 1.0

# 加一点熵奖励
uv run python scripts/26_ppo_pendulum.py --entropy-coef 0.001
```

观察点：

1. `--reward-scale 1.0` 时 value loss 会不会爆炸？
2. 训练 300k 步，eval 能到多少？
3. σ 最终收敛到多少？太小会怎样？

---

## 8. 下一步

连续动作 PPO 已经跑通，下一步可以：

1. **自建 gymnasium 环境**：PointMass 二维到达任务，用同一个 PPO 训练；
2. **多 seed 对比**：PPO / A2C / DQN 在多个环境上的均值±标准差；
3. **离散化对比**：把 Pendulum 动作离散成 9 个格子，用 DQN，看精度损失。
