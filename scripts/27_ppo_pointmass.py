"""实验 27：自建 PointMass 环境 + 连续动作 PPO。

环境：二维点质量，动作是二维推力，目标是回到原点。
用实验 26 的连续 PPO 训练，观察策略是否学会「推到目标」。

用法：
  uv run python scripts/27_ppo_pointmass.py
  uv run python scripts/27_ppo_pointmass.py --total-steps 60000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rldl.pointmass import PointMassEnv  # noqa: E402
from rldl.policy_gradient import ContinuousPolicyNetwork, ValueNetwork  # noqa: E402
from rldl.ppo import compute_gae_bootstrap  # noqa: E402
from rldl.utils import pick_device, set_seed  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def evaluate(actor, device, episodes: int = 5, seed: int = 0) -> float:
    env = PointMassEnv()
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor(state_t).squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


@torch.no_grad()
def evaluate_paths(actor, device, seeds: list[int]) -> list[np.ndarray]:
    env = PointMassEnv()
    paths = []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        path = [state[:2].copy()]
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor(state_t).squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            state, _, terminated, truncated, _ = env.step(action)
            path.append(state[:2].copy())
            done = terminated or truncated
        paths.append(np.asarray(path))
    return paths


def collect_rollout(env, actor, critic, min_steps: int, device, reward_scale: float) -> dict:
    episodes = []
    total = 0

    while total < min_steps:
        state, _ = env.reset()
        states, actions, log_probs, values = [], [], [], []
        rewards, episode_return = [], 0.0
        done = False

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = actor.distribution(state_t)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = critic(state_t)

            action_np = action.squeeze(0).cpu().numpy()
            clipped = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(clipped)
            done = terminated or truncated

            states.append(state_t.squeeze(0))
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            values.append(float(value.item()))
            rewards.append(float(reward) * reward_scale)
            episode_return += float(reward)
            total += 1
            state = next_state

        if terminated:
            last_value = 0.0
        else:
            with torch.no_grad():
                last_value = float(
                    critic(
                        torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    ).item()
                )

        episodes.append(
            {
                "states": states,
                "actions": actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": rewards,
                "last_value": last_value,
                "return": episode_return,
            }
        )

    return {"episodes": episodes, "num_steps": total}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=100000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--solved-return", type=float, default=-15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("outputs/27_pointmass"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device = {device}")

    env = PointMassEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = ContinuousPolicyNetwork(obs_dim, action_dim).to(device)
    critic = ValueNetwork(obs_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    episode_returns: list[float] = []
    eval_steps, eval_returns = [], []
    best_eval = -float("inf")
    solved_count = 0
    global_step = 0
    last_eval = 0

    while global_step < args.total_steps:
        rollout = collect_rollout(
            env, actor, critic, args.rollout_steps, device, args.reward_scale
        )
        global_step += rollout["num_steps"]

        all_states, all_actions, all_log_probs = [], [], []
        all_advantages, all_returns = [], []
        for episode in rollout["episodes"]:
            episode_returns.append(episode["return"])
            advantages, returns = compute_gae_bootstrap(
                np.asarray(episode["rewards"], dtype=np.float32),
                np.asarray(episode["values"], dtype=np.float32),
                episode["last_value"],
                args.gamma,
                args.lam,
            )
            all_states.extend(episode["states"])
            all_actions.extend(episode["actions"])
            all_log_probs.extend(episode["log_probs"])
            all_advantages.extend(advantages.tolist())
            all_returns.extend(returns.tolist())

        states = torch.stack(all_states)
        actions = torch.stack(all_actions)
        old_log_probs = torch.stack(all_log_probs).detach()
        adv_t = torch.as_tensor(all_advantages, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(all_returns, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(states)
        for _ in range(args.update_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.minibatch_size):
                mb = perm[start : start + args.minibatch_size]
                dist = actor.distribution(states[mb])
                new_log_probs = dist.log_prob(actions[mb]).sum(dim=-1)
                ratio = (new_log_probs - old_log_probs[mb]).exp()

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(critic(states[mb]), ret_t[mb])
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                actor_opt.step()
                critic_opt.step()

        if global_step - last_eval >= args.eval_every:
            last_eval = global_step
            eval_return = evaluate(actor, device, episodes=5, seed=args.seed + 1000)
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

            recent = np.mean(episode_returns[-10:])
            print(
                f"step {global_step:6d} | train_return(10)={recent:7.1f} "
                f"| eval={eval_return:7.1f} | std={actor.log_std.exp().mean().item():.3f}"
            )

            if solved_count >= 3:
                print(f"\n连续 3 次评估达到 {args.solved_return}，提前停止。")
                break

    print(f"\nbest eval return = {best_eval:.1f}, total steps = {global_step}")

    args.out.mkdir(parents=True, exist_ok=True)
    paths = evaluate_paths(actor, device, seeds=[101, 202, 303, 404])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    episodes = np.arange(1, len(episode_returns) + 1)
    axes[0].plot(episodes, episode_returns, color="#CCCCCC", lw=0.8, label="episode return")
    if len(episode_returns) >= 10:
        kernel = np.ones(10) / 10
        smooth = np.convolve(episode_returns, kernel, mode="valid")
        axes[0].plot(np.arange(10, 10 + len(smooth)), smooth, color="#4C78A8", label="10 轮平均")
    axes[0].set_title("训练回报")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(eval_steps, eval_returns, "o-", color="#54A24B")
    axes[1].axhline(args.solved_return, color="#E45756", ls="--", lw=1, label="solved 线")
    axes[1].set_title("评估回报")
    axes[1].set_xlabel("environment step")
    axes[1].set_ylabel("return")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for i, path in enumerate(paths):
        axes[2].plot(path[:, 0], path[:, 1], marker="o", ms=2, label=f"起点 {i + 1}")
        axes[2].scatter(path[0, 0], path[0, 1], s=30)
    axes[2].scatter(0, 0, marker="*", s=200, color="#E45756", label="目标")
    axes[2].set_title("学到的轨迹：从随机起点回到原点")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"自建 PointMass 环境 + 连续 PPO（{global_step} steps）", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "pointmass.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        args.out / "history.npz",
        episode_returns=np.asarray(episode_returns),
        eval_steps=np.asarray(eval_steps),
        eval_returns=np.asarray(eval_returns),
    )
    print(f"saved -> {args.out}/pointmass.png, best_model.pth, history.npz")


if __name__ == "__main__":
    main()
