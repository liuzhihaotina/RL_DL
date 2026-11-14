# 实验配置目录

后续把可复用的实验超参写成 TOML，例如：

```toml
# configs/dqn_cartpole.toml
[env]
id = "CartPole-v1"

[train]
total_steps = 100_000
lr = 2.5e-4
gamma = 0.99
batch_size = 64
```

脚本通过 `--config configs/xxx.toml` 读取，命令行参数优先覆盖。
