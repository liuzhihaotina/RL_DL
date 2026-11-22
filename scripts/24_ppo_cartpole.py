"""实验 24：PPO 解决 CartPole。

在 A2C 基础上：
  1. 收集一批完整 episode 的数据
  2. 用 GAE 计算优势和回报
  3. 对同一批数据做多轮 minibatch 更新
  4. 用 ratio 裁剪限制每次策略变化的幅度

用法：
  uv run python scripts/24_ppo_cartpole.py
  uv run python scripts/24_ppo_cartpole.py --total-steps 50000
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

from rldl.policy_gradient import PolicyNetwork, ValueNetwork  # noqa: E402
from rldl.ppo import compute_gae  # noqa: E402
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


def collect_rollout(
    env,
    actor: PolicyNetwork,
    critic: ValueNetwork,
    min_steps: int,
    device: torch.device,
    max_steps: int,
) -> dict:
    """收集完整 episode，直到总步数 >= min_steps。"""
    states, actions, log_probs, values = [], [], [], []
    rewards, dones = [], []
    episode_returns = []
    total = 0

    while total < min_steps:
        state, _ = env.reset()
        episode_return = 0.0
        while True:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = actor.distribution(state_t)
                action = dist.sample()
                value = critic(state_t)

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            states.append(state_t.squeeze(0))
            actions.append(int(action.item()))
            log_probs.append(dist.log_prob(action).squeeze(0))
            values.append(float(value.item()))
            rewards.append(reward)
            dones.append(done)
            episode_return += reward
            total += 1

            state = next_state
            if done or total >= max_steps * 4:
                break

        episode_returns.append(episode_return)

    return {
        "states": torch.stack(states),
        "actions": torch.as_tensor(actions, dtype=torch.int64, device=device),
        "log_probs": torch.stack(log_probs),
        "values": np.asarray(values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "episode_returns": episode_returns,
        "num_steps": total,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=80000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--solved-return", type=float, default=475.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/24_ppo_cartpole"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = PolicyNetwork(obs_dim, n_actions).to(device)
    critic = ValueNetwork(obs_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    episode_returns: list[float] = []
    eval_steps, eval_returns = [], []
    best_eval = 0.0
    solved_count = 0
    global_step = 0
    last_eval = 0
    update = 0

    while global_step < args.total_steps:
        rollout = collect_rollout(
            env, actor, critic, args.rollout_steps, device, args.max_episode_steps
        )
        global_step += rollout["num_steps"]
        episode_returns.extend(rollout["episode_returns"])

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            args.gamma,
            args.lam,
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"].detach()
        n = len(states)

        for _ in range(args.update_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.minibatch_size):
                mb = perm[start : start + args.minibatch_size]
                dist = actor.distribution(states[mb])
                new_log_probs = dist.log_prob(actions[mb])
                ratio = (new_log_probs - old_log_probs[mb]).exp()

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(critic(states[mb]), ret_t[mb])
                entropy = dist.entropy().mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                actor_opt.step()
                critic_opt.step()

        update += 1

        if global_step - last_eval >= args.eval_every:
            last_eval = global_step
            eval_return = evaluate(actor, device, episodes=3, seed=args.seed + 1000)
            eval_steps.append(global_step)
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

            recent = np.mean(episode_returns[-20:])
            print(
                f"step {global_step:6d} | update {update:3d} | train_return(20)={recent:6.1f} "
                f"| eval={eval_return:6.1f}"
            )

            if solved_count >= 3:
                print(f"\n连续 3 次评估达到 {args.solved_return}，提前停止。")
                break

    env.close()
    print(f"\nbest eval return = {best_eval:.1f}, total steps = {global_step}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    episodes = np.arange(1, len(episode_returns) + 1)
    axes[0].plot(episodes, episode_returns, color="#CCCCCC", lw=0.8, label="episode return")
    if len(episode_returns) >= 20:
        kernel = np.ones(20) / 20
        smooth = np.convolve(episode_returns, kernel, mode="valid")
        axes[0].plot(np.arange(20, 20 + len(smooth)), smooth, color="#4C78A8", label="20 轮平均")
    axes[0].set_title("训练回报")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(eval_steps, eval_returns, "o-", color="#54A24B")
    axes[1].axhline(args.solved_return, color="#E45756", ls="--", lw=1, label="solved 线")
    axes[1].set_title("贪心策略评估回报")
    axes[1].set_xlabel("environment step")
    axes[1].set_ylabel("return")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"PPO on CartPole（{global_step} steps）", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "ppo_cartpole.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        episode_returns=np.asarray(episode_returns),
        eval_steps=np.asarray(eval_steps),
        eval_returns=np.asarray(eval_returns),
    )
    print(f"saved -> {args.out}/ppo_cartpole.png, best_model.pth, history.npz")


if __name__ == "__main__":
    main()
