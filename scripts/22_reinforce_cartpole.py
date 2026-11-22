"""实验 22：REINFORCE 解决 CartPole。

REINFORCE 直接参数化策略 π(a|s)：
  1. 用当前策略跑一整局，记录每步的 log π(a|s)
  2. 计算每步的折扣回报 G_t
  3. loss = -Σ log π(a_t|s_t) × G_t
  4. 反向传播，提高「回报高」的动作概率

用法：
  uv run python scripts/22_reinforce_cartpole.py
  uv run python scripts/22_reinforce_cartpole.py --episodes 1500
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.policy_gradient import (  # noqa: E402
    PolicyNetwork,
    ValueNetwork,
    compute_returns,
    reinforce_loss,
)
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def evaluate(
    policy: PolicyNetwork, device: torch.device, episodes: int = 3, seed: int = 0
) -> float:
    env = gym.make("CartPole-v1")
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            logits = policy(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            state, reward, terminated, truncated, _ = env.step(int(logits.argmax(dim=1).item()))
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
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--solved-return", type=float, default=475.0)
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="对回报/优势做标准化",
    )
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/22_reinforce_cartpole"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(
        f"device = {device} | normalize={args.normalize} | "
        f"baseline={not args.no_baseline}"
    )

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNetwork(obs_dim, n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value_net = None
    value_optimizer = None
    if not args.no_baseline:
        value_net = ValueNetwork(obs_dim, args.hidden).to(device)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)

    episode_returns: list[float] = []
    entropies: list[float] = []
    eval_episodes, eval_returns = [], []
    best_eval = 0.0
    solved_count = 0

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        log_probs = []
        entropy_terms = []
        states = []
        rewards = []
        done = False

        for _ in range(args.max_steps):
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dist = policy.distribution(state_t)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropy_terms.append(dist.entropy())
            states.append(state_t.squeeze(0).detach())

            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(reward)
            done = terminated or truncated
            if done:
                break

        returns = compute_returns(rewards, args.gamma)
        values = None
        if value_net is not None:
            values = value_net(torch.stack(states))

        policy_loss = reinforce_loss(
            log_probs,
            returns,
            values=values,
            entropy=entropy_terms,
            normalize=args.normalize,
            entropy_coef=args.entropy_coef,
        )

        optimizer.zero_grad(set_to_none=True)
        if value_net is not None and value_optimizer is not None:
            value_optimizer.zero_grad(set_to_none=True)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
            value_loss = F.mse_loss(values, returns_t)
            loss = policy_loss + args.value_coef * value_loss
        else:
            loss = policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if value_net is not None:
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        optimizer.step()
        if value_optimizer is not None:
            value_optimizer.step()

        total_return = float(sum(rewards))
        episode_returns.append(total_return)
        entropies.append(float(torch.stack(entropy_terms).mean().item()))

        if episode % args.eval_every == 0:
            eval_return = evaluate(policy, device, episodes=3, seed=args.seed + 1000)
            eval_episodes.append(episode)
            eval_returns.append(eval_return)

            if eval_return > best_eval:
                best_eval = eval_return
                args.out.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), args.out / "best_model.pth")

            if eval_return >= args.solved_return:
                solved_count += 1
            else:
                solved_count = 0

            print(
                f"episode {episode:4d} | return={total_return:5.1f} | "
                f"eval={eval_return:6.1f} | entropy={entropies[-1]:.3f}"
            )

            if solved_count >= 3:
                print(f"\n连续 3 次评估达到 {args.solved_return}，提前停止。")
                break

    env.close()
    print(f"\nbest eval return = {best_eval:.1f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    episodes = np.arange(1, len(episode_returns) + 1)

    axes[0].plot(episodes, episode_returns, color="#CCCCCC", lw=0.8, label="episode return")
    if len(episode_returns) >= 50:
        smooth = moving_average(episode_returns, 50)
        axes[0].plot(np.arange(50, 50 + len(smooth)), smooth, color="#4C78A8", label="50 轮平均")
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

    fig.suptitle(f"REINFORCE on CartPole（{len(episode_returns)} episodes）", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "reinforce_cartpole.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        episode_returns=np.asarray(episode_returns),
        entropies=np.asarray(entropies),
        eval_episodes=np.asarray(eval_episodes),
        eval_returns=np.asarray(eval_returns),
    )
    print(f"saved -> {args.out}/reinforce_cartpole.png, best_model.pth, history.npz")


if __name__ == "__main__":
    main()
