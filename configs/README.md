# 实验配置

- `mnist_cnn.toml`：实验 12 的 MNIST CNN 配置。

优先级：**命令行参数 > TOML 文件 > 代码默认值**。

示例：

```bash
uv run python scripts/12_mnist_tensorboard.py
uv run python scripts/12_mnist_tensorboard.py --epochs 3 --lr 3e-3
```
