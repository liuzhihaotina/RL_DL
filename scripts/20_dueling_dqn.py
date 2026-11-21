"""实验 20：Dueling DQN vs 标准 DQN（CartPole）。

两个网络都用 Double DQN 目标，只改网络结构：
  标准：  features → Q(s,·)
  Dueling：features → V(s) + A(s,·) - mean(A)

用法：
  uv run python scripts/20_dueling_dqn.py
  uv run python scripts/20_dueling_dqn.py --seeds 2 --episodes 500
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

from rldl.dqn import (  # noqa: E402
    DuelingQNetwork,
    QNetwork,
    ReplayBuffer,
    epsilon_by_step,
)
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"standard": "#8C8C8C", "dueling": "#4C78A8"}


def select_action(net, state, epsilon, rng, device, n_actions: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q = net(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
    return int(q.argmax(dim=1).item())


@torch.no_grad()
def evaluate(net, device, episodes: int = 3, seed: int = 0) -> float:
    env = gym.make("CartPole-v1")
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            q = net(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
            state, reward, terminated, truncated, _ = env.step(int(q.argmax(dim=1).item()))
            total += reward
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def train(arch: str, seed: int, args, device: torch.device) -> dict:
    set_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)

    net_cls = QNetwork if arch == "standard" else DuelingQNetwork
    policy_net = net_cls(obs_dim, n_actions, args.hidden).to(device)
    target_net = net_cls(obs_dim, n_actions, args.hidden).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, seed=seed)

    episode_returns = []
    eval_episodes, eval_returns = [], []
    global_step = 0
    best_eval = 0.0
    solved_count = 0
    first_solved = None

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0

        for _ in range(args.max_steps):
            epsilon = epsilon_by_step(global_step, args.eps_start, args.eps_end, args.eps_decay)
            action = select_action(policy_net, state, epsilon, rng, device, n_actions)
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
                    next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                    next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
                    target = rewards + args.gamma * next_q * (1.0 - dones)

                predicted = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(predicted, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                if global_step % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_returns.append(total)

        if episode % args.eval_every == 0:
            eval_return = evaluate(policy_net, device, 3, seed=seed + 1000)
            eval_episodes.append(episode)
            eval_returns.append(eval_return)
            if eval_return > best_eval:
                best_eval = eval_return
            if eval_return >= args.solved_return:
                solved_count += 1
                if first_solved is None:
                    first_solved = episode
            else:
                solved_count = 0

            print(
                f"[{arch:8s} seed={seed}] episode {episode:3d} | "
                f"return={total:5.1f} | eval={eval_return:6.1f} | steps={global_step}"
            )
            if solved_count >= 3:
                break

    env.close()
    return {
        "episode_returns": episode_returns,
        "eval_episodes": eval_episodes,
        "eval_returns": eval_returns,
        "best_eval": best_eval,
        "first_solved": first_solved,
        "episodes_run": len(episode_returns),
    }


def moving_average(values: list[float], window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--archs", choices=["both", "standard", "dueling"], default="both")
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
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/20_dueling_dqn"))
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device = {device}")

    archs = ["standard", "dueling"] if args.archs == "both" else [args.archs]
    results = {"standard": [], "dueling": []}
    for arch in archs:
        for seed in range(args.seeds):
            results[arch].append(train(arch, seed, args, device))

    print("\n汇总：")
    print(f"{'结构':<10}{'best_eval':>10}{'首次达标':>10}{'episodes':>10}")
    for arch in archs:
        runs = results[arch]
        best = np.mean([r["best_eval"] for r in runs])
        first = [r["first_solved"] for r in runs if r["first_solved"] is not None]
        first_text = f"{np.mean(first):.0f}" if first else "未达标"
        episodes = np.mean([r["episodes_run"] for r in runs])
        print(f"{arch:<10}{best:>10.1f}{first_text:>10}{episodes:>10.0f}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for arch in archs:
        color = COLORS[arch]
        for run in results[arch]:
            episodes = np.arange(1, len(run["episode_returns"]) + 1)
            smooth = moving_average(run["episode_returns"], 20)
            axes[0].plot(
                np.arange(20, 20 + len(smooth)),
                smooth,
                color=color,
                alpha=0.35,
                lw=1,
            )
            axes[1].plot(run["eval_episodes"], run["eval_returns"], "o-", color=color, alpha=0.5)

        # 图例只画一次
        axes[0].plot([], [], color=color, label=arch)
        axes[1].plot([], [], color=color, label=arch)

    axes[0].set_title("训练回报（20 轮滑动平均）")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axhline(args.solved_return, color="#E45756", ls="--", lw=1, label="solved 线")
    axes[1].set_title("贪心策略评估回报")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("return")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Dueling DQN vs 标准 DQN（都用 Double 目标，CartPole）", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "dueling_vs_standard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        **{
            f"{arch}_{i}_{key}": np.asarray(run[key])
            for arch in results
            for i, run in enumerate(results[arch])
            for key in ("episode_returns", "eval_episodes", "eval_returns")
        },
    )
    print(f"\nsaved -> {args.out}/dueling_vs_standard.png, history.npz")


if __name__ == "__main__":
    main()
