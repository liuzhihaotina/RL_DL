"""实验 25：Pendulum 入门——先看清连续动作环境长什么样。

不做训练，只做四件事：
  1. 打印 observation_space / action_space
  2. 随机动作跑几步，打印 obs / action / reward
  3. 对比三种策略：随机、零扭矩、简单 PD 控制器
  4. 画角度、扭矩、累计奖励、相图

用法：
  uv run python scripts/25_pendulum_intro.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def random_policy(rng: np.random.Generator):
    def act(obs):
        return rng.uniform(-2.0, 2.0, size=1).astype(np.float32)

    return act


def zero_policy(obs):
    return np.zeros(1, dtype=np.float32)


def pd_policy(obs, kp: float = 10.0, kd: float = 2.0):
    """简单 PD：角度偏了就往反方向推，角速度大也反向阻尼。"""
    cos_theta, sin_theta, theta_dot = obs
    theta = np.arctan2(sin_theta, cos_theta)
    torque = -kp * theta - kd * theta_dot
    return np.clip([torque], -2.0, 2.0).astype(np.float32)


def rollout(env, action_fn, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    thetas, theta_dots, actions, rewards = [], [], [], []

    while True:
        action = np.asarray(action_fn(obs), dtype=np.float32).reshape(1)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        thetas.append(float(np.arctan2(obs[1], obs[0])))
        theta_dots.append(float(obs[2]))
        actions.append(float(action[0]))
        rewards.append(float(reward))

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "thetas": np.asarray(thetas),
        "theta_dots": np.asarray(theta_dots),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
    }


def main() -> None:
    env = gym.make("Pendulum-v1")
    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)
    print("max_episode_steps:", env.spec.max_episode_steps)

    obs, info = env.reset(seed=0)
    print("\nreset 后的 obs:", np.round(obs, 4))
    print("obs[0]=cosθ, obs[1]=sinθ, obs[2]=θdot")

    print("\n随机动作的前 5 步：")
    header = (
        f"{'step':>4} {'cosθ':>8} {'sinθ':>8} {'θdot':>8} "
        f"{'θ(rad)':>8} {'torque':>8} {'reward':>8}"
    )
    print(header)
    rng = np.random.default_rng(0)
    for step in range(5):
        action = rng.uniform(-2, 2, size=1).astype(np.float32)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        theta = np.arctan2(obs[1], obs[0])
        print(
            f"{step:>4} {obs[0]:>8.3f} {obs[1]:>8.3f} {obs[2]:>8.3f} "
            f"{theta:>8.3f} {action[0]:>8.3f} {reward:>8.2f}"
        )
        obs = next_obs
        if terminated or truncated:
            break

    results = {
        "随机": rollout(env, random_policy(np.random.default_rng(1)), seed=1),
        "零扭矩": rollout(env, zero_policy, seed=1),
        "PD 控制": rollout(env, pd_policy, seed=1),
    }

    print("\n完整 episode（200 步）总奖励：")
    for name, res in results.items():
        print(
            f"  {name:6s} return={res['rewards'].sum():8.1f} "
            f"平均扭矩={np.abs(res['actions']).mean():.2f}"
        )

    args_out = Path("outputs/25_pendulum_intro")
    args_out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for name, res in results.items():
        steps = np.arange(len(res["thetas"]))
        axes[0, 0].plot(steps, res["thetas"], label=name)
        axes[0, 1].plot(steps, res["actions"], label=name, alpha=0.8)
        axes[1, 0].plot(steps, np.cumsum(res["rewards"]), label=name)

    axes[0, 0].axhline(0, color="#999999", lw=0.8, ls="--")
    axes[0, 0].set_title("角度 θ（0 = 竖直向上）")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("θ (rad)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("施加的扭矩（连续动作）")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("torque")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_title("累计奖励（越接近 0 越好）")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("cumulative reward")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    pd_res = results["PD 控制"]
    axes[1, 1].plot(pd_res["thetas"], pd_res["theta_dots"], color="#54A24B")
    axes[1, 1].scatter(
        pd_res["thetas"][0],
        pd_res["theta_dots"][0],
        color="black",
        zorder=3,
        label="起点",
    )
    axes[1, 1].set_title("PD 控制的相图：θ vs θdot")
    axes[1, 1].set_xlabel("θ (rad)")
    axes[1, 1].set_ylabel("θdot")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("Pendulum：连续动作环境长什么样", fontsize=14)
    fig.tight_layout()
    fig.savefig(args_out / "pendulum_intro.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    env.close()

    print(f"\nsaved -> {args_out}/pendulum_intro.png")


if __name__ == "__main__":
    main()
