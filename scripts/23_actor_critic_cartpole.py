"""实验 23：Actor-Critic（A2C）解决 CartPole。

和 REINFORCE 的区别：
  - Critic 学 V(s)；
  - Actor 用 n 步回报减去 V(s) 的 advantage 更新；
  - 不再等整条轨迹结束才更新，方差更小。

用法：
  uv run python scripts/23_actor_critic_cartpole.py
  uv run python scripts/23_actor_critic_cartpole.py --n-step 10
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
    n_step_returns,
)
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def evaluate(actor: PolicyNetwork, device: torch.device, episodes: int = 3, seed: int = 0) -> float:
    env = gym.make("CartPole-v1")
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            logits = actor(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
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
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--batch-episodes", type=int, default=4)
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--solved-return", type=float, default=475.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/23_actor_critic_cartpole"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device} | n_step={args.n_step}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = PolicyNetwork(obs_dim, n_actions, args.hidden).to(device)
    critic = ValueNetwork(obs_dim, args.hidden).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    episode_returns: list[float] = []
    losses: list[float] = []
    eval_episodes, eval_returns = [], []
    best_eval = 0.0
    solved_count = 0
    episode = 0

    while episode < args.episodes:
        batch = []
        for _ in range(args.batch_episodes):
            episode += 1
            state, _ = env.reset(seed=args.seed + episode)
            states, log_probs, entropies, rewards, dones = [], [], [], [], []
            done = False

            for _ in range(args.max_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                dist = actor.distribution(state_t)
                action = dist.sample()

                states.append(state_t.squeeze(0))
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

                state, reward, terminated, truncated, _ = env.step(int(action.item()))
                rewards.append(reward)
                done = terminated or truncated
                dones.append(done)
                if done:
                    break

            episode_returns.append(float(sum(rewards)))
            batch.append(
                {
                    "states": states,
                    "log_probs": log_probs,
                    "entropies": entropies,
                    "rewards": rewards,
                    "dones": dones,
                }
            )

        all_states, all_log_probs, all_entropies, all_returns = [], [], [], []
        for item in batch:
            states_t = torch.stack(item["states"])
            with torch.no_grad():
                values_np = critic(states_t).cpu().numpy()
            returns = n_step_returns(
                np.asarray(item["rewards"], dtype=np.float32),
                values_np,
                np.asarray(item["dones"], dtype=bool),
                args.gamma,
                args.n_step,
            )
            all_states.extend(item["states"])
            all_log_probs.extend(item["log_probs"])
            all_entropies.extend(item["entropies"])
            all_returns.extend(returns.tolist())

        states_t = torch.stack(all_states)
        values = critic(states_t)
        returns_t = torch.as_tensor(all_returns, dtype=torch.float32, device=device)
        advantages = returns_t - values.detach()

        policy_loss = -(
            torch.stack(all_log_probs) * advantages
        ).mean() - args.entropy_coef * torch.stack(all_entropies).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + args.value_coef * value_loss

        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        actor_opt.step()
        critic_opt.step()
        losses.append(float(loss.item()))
        entropy_mean = float(torch.stack(all_entropies).mean().item())

        if episode % args.eval_every == 0:
            eval_return = evaluate(actor, device, episodes=3, seed=args.seed + 1000)
            eval_episodes.append(episode)
            eval_returns.append(eval_return)

            if eval_return > best_eval:
                best_eval = eval_return
                args.out.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"actor": actor.state_dict(), "critic": critic.state_dict()},
                    args.out / "best_model.pth",
                )

            if eval_return >= args.solved_return:
                solved_count += 1
            else:
                solved_count = 0

            print(
                f"episode {episode:4d} | return={episode_returns[-1]:5.1f} | "
                f"eval={eval_return:6.1f} | loss={losses[-1]:.3f} | entropy={entropy_mean:.3f}"
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

    fig.suptitle(
        f"Actor-Critic（n={args.n_step}）on CartPole（{len(episode_returns)} episodes）",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(args.out / "actor_critic_cartpole.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        episode_returns=np.asarray(episode_returns),
        losses=np.asarray(losses),
        eval_episodes=np.asarray(eval_episodes),
        eval_returns=np.asarray(eval_returns),
    )
    print(f"saved -> {args.out}/actor_critic_cartpole.png, best_model.pth, history.npz")


if __name__ == "__main__":
    main()
