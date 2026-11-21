"""实验 19：DQN 解决 CartPole。

DQN 的四个关键部件：
  1. Q 网络：输入 4 维状态，输出 2 个动作的 Q 值
  2. 经验回放：把 (s, a, r, s', done) 存起来，随机抽 batch
  3. 目标网络：单独一份参数，定期从 Q 网络同步，稳定 TD 目标
  4. ε-greedy：先大量探索，再逐渐利用

用法：
  uv run python scripts/19_dqn_cartpole.py
  uv run python scripts/19_dqn_cartpole.py --double      # Double DQN
  uv run python scripts/19_dqn_cartpole.py --episodes 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.dqn import QNetwork, ReplayBuffer, epsilon_by_step  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def select_action(
    qnet: QNetwork,
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(qnet.net[-1].out_features))
    with torch.no_grad():
        q_values = qnet(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
    return int(q_values.argmax(dim=1).item())


@torch.no_grad()
def evaluate(qnet: QNetwork, device: torch.device, episodes: int = 3, seed: int = 0) -> float:
    env = gym.make("CartPole-v1")
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            q_values = qnet(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            state, reward, terminated, truncated, _ = env.step(int(q_values.argmax(dim=1).item()))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def moving_average(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--solved-return", type=float, default=475.0)
    p.add_argument(
        "--double",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 Double DQN 目标（--no-double 关闭）",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cpu",
        help="CartPole 网络很小，CPU 通常比 GPU 快；需要时可传 cuda",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/19_dqn_cartpole"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device} | double={args.double}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    rng = np.random.default_rng(args.seed)

    policy_net = QNetwork(obs_dim, n_actions, args.hidden).to(device)
    target_net = QNetwork(obs_dim, n_actions, args.hidden).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, seed=args.seed)

    episode_returns: list[float] = []
    episode_losses: list[float] = []
    eval_episodes: list[int] = []
    eval_returns: list[float] = []
    best_eval = 0.0
    solved_count = 0
    global_step = 0

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        total = 0.0
        losses = []

        for _ in range(args.max_steps):
            epsilon = epsilon_by_step(
                global_step, args.eps_start, args.eps_end, args.eps_decay
            )
            action = select_action(policy_net, state, epsilon, rng, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total += reward
            global_step += 1

            if global_step >= args.warmup and buffer.size >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
                actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
                rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
                next_states = torch.as_tensor(
                    batch["next_states"], dtype=torch.float32, device=device
                )
                dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)

                with torch.no_grad():
                    if args.double:
                        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                        next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
                    else:
                        next_q = target_net(next_states).max(dim=1).values
                    target = rewards + args.gamma * next_q * (1.0 - dones)

                predicted = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(predicted, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
                losses.append(loss.item())

                if global_step % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_returns.append(total)
        episode_losses.append(float(np.mean(losses)) if losses else 0.0)

        if episode % args.eval_every == 0:
            eval_return = evaluate(policy_net, device, episodes=3, seed=args.seed + 1000)
            eval_episodes.append(episode)
            eval_returns.append(eval_return)
            if eval_return >= args.solved_return:
                solved_count += 1
            else:
                solved_count = 0

            print(
                f"episode {episode:3d} | return={total:5.1f} | eval={eval_return:6.1f} "
                f"| eps={epsilon:.3f} | loss={episode_losses[-1]:.3f} | steps={global_step}"
            )

            if eval_return > best_eval:
                best_eval = eval_return
                args.out.mkdir(parents=True, exist_ok=True)
                torch.save(policy_net.state_dict(), args.out / "best_model.pth")

            if solved_count >= 3:
                print(f"\n已连续 3 次评估达到 {args.solved_return}，提前停止。")
                break

    env.close()
    print(f"\nbest eval return = {best_eval:.1f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    episodes = np.arange(1, len(episode_returns) + 1)

    axes[0].plot(episodes, episode_returns, color="#CCCCCC", lw=0.8, label="episode return")
    if len(episode_returns) >= 20:
        smooth = moving_average(episode_returns, 20)
        axes[0].plot(np.arange(20, 20 + len(smooth)), smooth, color="#4C78A8", label="20 轮平均")
    axes[0].set_title("训练回报")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(eval_episodes, eval_returns, "o-", color="#54A24B")
    axes[1].axhline(args.solved_return, color="#E45756", ls="--", lw=1, label="solved 线")
    axes[1].set_title("贪心策略评估回报")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("return")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(episodes, episode_losses, color="#F58518", lw=0.8)
    axes[2].set_title("Huber loss（每 episode 平均）")
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("loss")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"DQN on CartPole（{'Double DQN' if args.double else 'DQN'}，"
        f"{len(episode_returns)} episodes）",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(args.out / "dqn_cartpole.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        episode_returns=np.asarray(episode_returns),
        episode_losses=np.asarray(episode_losses),
        eval_episodes=np.asarray(eval_episodes),
        eval_returns=np.asarray(eval_returns),
    )
    print(f"saved -> {args.out}/dqn_cartpole.png, best_model.pth, history.npz")


if __name__ == "__main__":
    main()
