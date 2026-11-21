# 实验 12：TensorBoard + TOML 配置

脚本：`scripts/12_mnist_tensorboard.py`
配置：`configs/mnist_cnn.toml`

实验 11 把所有指标打印在终端、曲线存成 PNG。实验 12 换成更常用的工程做法：

- 训练过程写入 **TensorBoard**，网页里实时看曲线、直方图、图片；
- 超参写在 **TOML 文件**里，命令行可以临时覆盖。

---

## 1. TensorBoard 是什么

TensorBoard 原本是 TensorFlow 的可视化工具，PyTorch 也能用。训练脚本把数据写进 `runs/` 下的事件文件，TensorBoard 启动一个本地网页读取这些文件：

```bash
uv run tensorboard --logdir runs
# 浏览器打开 http://localhost:6006
```

能看的东西：

| 类型 | 用途 |
| --- | --- |
| Scalars | loss、accuracy、学习率随 epoch 的变化 |
| Histograms | 每层权重的分布，看是否在正常范围 |
| Images / Figures | 预测结果、输入样本 |
| Text | 本次运行的配置 |
| Graphs | 计算图（本实验未启用） |

和存 PNG 的区别：TensorBoard 可以同时比较多个 run、放大、悬停看数值，不用改代码。

---

## 2. 本实验记录了什么

每个 epoch：

```python
writer.add_scalar("loss/train", train_loss, epoch)
writer.add_scalar("loss/test", test_loss, epoch)
writer.add_scalar("acc/train", train_acc, epoch)
writer.add_scalar("acc/test", test_acc, epoch)
writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

for name, param in model.named_parameters():
    if param.dim() > 1:
        writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)
```

训练结束：

```python
writer.add_figure("predictions/test", fig, epoch)
```

实际写入的标签（已用 EventAccumulator 验证）：

```text
scalars:
  loss/train, loss/test, acc/train, acc/test, lr
histograms:
  weights/features.0.weight    （第一个卷积）
  weights/features.4.weight    （第二个卷积）
  weights/classifier.1.weight  （全连接 3136→128）
  weights/classifier.4.weight  （全连接 128→10）
```

`runs/mnist_cnn/` 下会生成：

```text
events.out.tfevents.xxxxx      # 训练数据
config.toml                    # 本次运行的配置快照
```

`config.toml` 快照很重要：以后回看某个 run 时，能知道当时用的超参。

---

## 3. TOML 配置

`configs/mnist_cnn.toml`：

```toml
[data]
root = "data/mnist"
batch_size = 128
limit_train = 0

[train]
epochs = 5
lr = 0.001
seed = 0
device = "auto"

[output]
out = "outputs/12_mnist_tensorboard"
run_dir = "runs/mnist_cnn"
```

Python 3.11 自带 `tomllib`，不需要装 `toml` 包。

加载逻辑：

```python
cfg = load_config(Path("configs/mnist_cnn.toml"))
p.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
```

优先级：**命令行 > TOML > 代码默认值**。

例如临时改用 3 个 epoch、学习率 3e-3：

```bash
uv run python scripts/12_mnist_tensorboard.py --epochs 3 --lr 3e-3
```

不需要改文件。

---

## 4. 结果

默认配置（5 epoch，Adam lr=1e-3，batch=128）：

```text
epoch 1 | train acc=0.9516 | test acc=0.9855
epoch 2 | train acc=0.9819 | test acc=0.9881
epoch 3 | train acc=0.9865 | test acc=0.9898
epoch 4 | train acc=0.9883 | test acc=0.9893
epoch 5 | train acc=0.9905 | test acc=0.9903
best test acc = 0.9903
```

和实验 11 的 0.9909 基本一致，差异来自 GPU 卷积的数值非确定性。

TensorBoard 里应该看到：

- `loss/train` 从 0.155 降到 0.030；
- `acc/test` 从 0.9855 升到 0.9903；
- `lr` 是一条水平线（本实验还没有学习率调度）；
- 权重直方图集中在 0 附近，没有爆炸或全部归零。

---

## 5. 常用命令

```bash
# 训练 + 记录
uv run python scripts/12_mnist_tensorboard.py

# 看面板
uv run tensorboard --logdir runs

# 指定端口
uv run tensorboard --logdir runs --port 6007

# 只用 2000 张图快速验证流程
uv run python scripts/12_mnist_tensorboard.py --limit-train 2000 --epochs 1

# 换一个 run 目录，方便对比两次实验
uv run python scripts/12_mnist_tensorboard.py --run-dir runs/mnist_lr3e-3 --lr 3e-3
```

TensorBoard 启动后，把 `runs/` 作为根目录，左侧会自动列出所有子 run，可以勾选多个一起比较。

---

## 6. 为什么值得这么做

- **不用反复改 print 和画图代码**：指标自动进面板；
- **对比实验**：不同学习率、不同 batch 的 run 可以叠在一张图上；
- **可追溯**：`config.toml` 快照 + 时间戳，知道每个结果从哪来；
- **为后面铺路**：强化学习的 return、episode 长度、探索率都适合用标量记录。

下一项是训练技巧：学习率调度、早停、梯度裁剪、混合精度（AMP）。
