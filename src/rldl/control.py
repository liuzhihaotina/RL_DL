"""表格型控制：SARSA（on-policy）与 Q-learning（off-policy）。"""

from __future__ import annotations

import numpy as np


def epsilon_greedy(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """以 ε 概率随机，否则选 Q 最大的动作（并列时随机）。"""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[state] == Q[state].max())
    return int(rng.choice(best))


def greedy_action(Q: np.ndarray, state: int) -> int:
    return int(np.argmax(Q[state]))


def evaluate_greedy(
    env,
    Q: np.ndarray,
    max_steps: int = 100,
) -> tuple[float, bool, list[int]]:
    """从起点按贪心策略走一遍，返回总奖励、是否到达终点、路径。"""
    state = env.start
    total = 0.0
    path = [state]
    for _ in range(max_steps):
        action = greedy_action(Q, state)
        state, reward, done = env.step(state, action)
        total += reward
        path.append(state)
        if done:
            return total, True, path
        if len(path) > max_steps:
            break
    return total, False, path


def train_sarsa(
    env,
    episodes: int,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    max_steps: int = 200,
    eval_every: int = 20,
    rng: np.random.Generator | None = None,
) -> dict:
    """SARSA：用行为策略实际选出的 a' 作为目标。

    Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
    """
    rng = rng or np.random.default_rng(0)
    Q = np.zeros((env.n_states, env.n_actions))
    history: list[float] = []
    evaluations: list[tuple[int, float, bool]] = []
    cliff_falls = 0

    for episode in range(1, episodes + 1):
        state = env.start
        action = epsilon_greedy(Q, state, epsilon, rng)
        total = 0.0

        for _ in range(max_steps):
            next_state, reward, done = env.step(state, action)
            total += reward
            if reward == env.cliff_reward:
                cliff_falls += 1

            next_action = epsilon_greedy(Q, next_state, epsilon, rng)
            target = reward if done else reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])

            state, action = next_state, next_action
            if done:
                break

        history.append(total)
        if episode % eval_every == 0:
            ret, success, _ = evaluate_greedy(env, Q)
            evaluations.append((episode, ret, success))

    return {
        "Q": Q,
        "history": history,
        "evaluations": evaluations,
        "cliff_falls": cliff_falls,
    }


def train_q_learning(
    env,
    episodes: int,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    max_steps: int = 200,
    eval_every: int = 20,
    rng: np.random.Generator | None = None,
) -> dict:
    """Q-learning：直接用 max_a' Q(s',a') 作为目标，与行为策略无关。

    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
    """
    rng = rng or np.random.default_rng(0)
    Q = np.zeros((env.n_states, env.n_actions))
    history: list[float] = []
    evaluations: list[tuple[int, float, bool]] = []
    cliff_falls = 0

    for episode in range(1, episodes + 1):
        state = env.start
        total = 0.0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done = env.step(state, action)
            total += reward
            if reward == env.cliff_reward:
                cliff_falls += 1

            target = reward if done else reward + gamma * Q[next_state].max()
            Q[state, action] += alpha * (target - Q[state, action])

            state = next_state
            if done:
                break

        history.append(total)
        if episode % eval_every == 0:
            ret, success, _ = evaluate_greedy(env, Q)
            evaluations.append((episode, ret, success))

    return {
        "Q": Q,
        "history": history,
        "evaluations": evaluations,
        "cliff_falls": cliff_falls,
    }


def moving_average(values: list[float], window: int = 20) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="valid")
