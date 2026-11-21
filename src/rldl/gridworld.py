"""GridWorld：一个 5×4（20 个格子）的格子世界，用来学 MDP 与动态规划。

图例：
  S 起点    G 终点(+1)    X 陷阱(-1)    # 墙
  . 普通格子（每走一步 reward = -0.01）

动作：0=上 1=右 2=下 3=左
状态编号：state = row * n_cols + col
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = ["↑", "→", "↓", "←"]

DEFAULT_GRID = [
    "S...",
    ".#..",
    "..X.",
    ".#..",
    "...G",
]


@dataclass
class GridWorld:
    grid: list[str]
    step_reward: float = -0.01
    gamma: float = 0.95
    goal_reward: float = 1.0
    trap_reward: float = -1.0

    def __post_init__(self) -> None:
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.start = self._find("S")
        self.goal = self._find("G")
        self.trap = self._find("X")
        self.walls = self._find_all("#")
        self.terminals = {self.goal, self.trap}

        self.transitions = self._build_transitions()

    # ---------- 坐标与状态编号 ----------
    def _find(self, ch: str) -> int:
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == ch:
                    return self.to_state((r, c))
        raise ValueError(f"grid 里找不到 {ch}")

    def _find_all(self, ch: str) -> set[int]:
        return {
            self.to_state((r, c))
            for r, row in enumerate(self.grid)
            for c, cell in enumerate(row)
            if cell == ch
        }

    def to_state(self, rc: tuple[int, int]) -> int:
        return rc[0] * self.n_cols + rc[1]

    def to_rc(self, state: int) -> tuple[int, int]:
        return divmod(state, self.n_cols)

    def is_terminal(self, state: int) -> bool:
        return state in self.terminals

    # ---------- MDP 模型 ----------
    def _build_transitions(self) -> list[list[tuple[float, int, float, bool]]]:
        transitions: list[list[tuple[float, int, float, bool]]] = []
        for state in range(self.n_states):
            row, col = self.to_rc(state)
            if state in self.terminals:
                transitions.append([[(1.0, state, 0.0, True)] for _ in range(4)])
                continue

            per_action = []
            for dr, dc in MOVES:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                    next_state = state
                else:
                    candidate = self.to_state((nr, nc))
                    next_state = state if candidate in self.walls else candidate

                if next_state == self.goal:
                    reward, done = self.goal_reward, True
                elif next_state == self.trap:
                    reward, done = self.trap_reward, True
                else:
                    reward, done = self.step_reward, False
                per_action.append([(1.0, next_state, reward, done)])
            transitions.append(per_action)
        return transitions

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        _, next_state, reward, done = self.transitions[state][action][0]
        return next_state, reward, done

    def render_policy(self, policy: np.ndarray) -> str:
        lines = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                state = self.to_state((r, c))
                ch = self.grid[r][c]
                if ch == "#":
                    cells.append(" # ")
                elif ch == "X":
                    cells.append(" X ")
                elif ch == "G":
                    cells.append(" G ")
                elif ch == "S":
                    cells.append(" S ")
                else:
                    cells.append(f" {ARROWS[policy[state]]} ")
            lines.append("".join(cells))
        return "\n".join(lines)


def policy_evaluation(
    env: GridWorld,
    policy: np.ndarray,
    gamma: float,
    theta: float = 1e-10,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int]:
    """给定策略，迭代贝尔曼期望方程，求出 V(s)。"""
    V = np.zeros(env.n_states)
    for iteration in range(1, max_iter + 1):
        delta = 0.0
        for state in range(env.n_states):
            if env.is_terminal(state):
                continue
            value = 0.0
            for prob, next_state, reward, _ in env.transitions[state][policy[state]]:
                value += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(value - V[state]))
            V[state] = value
        if delta < theta:
            return V, iteration
    return V, max_iter


def policy_evaluation_probs(
    env: GridWorld,
    action_probs: np.ndarray,
    gamma: float,
    theta: float = 1e-10,
    max_iter: int = 10000,
) -> np.ndarray:
    """随机策略（每状态一个动作分布）的贝尔曼期望方程迭代。"""
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        delta = 0.0
        for state in range(env.n_states):
            if env.is_terminal(state):
                continue
            value = 0.0
            for action in range(env.n_actions):
                prob_action = action_probs[state, action]
                if prob_action == 0:
                    continue
                for prob, next_state, reward, _ in env.transitions[state][action]:
                    value += prob_action * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(value - V[state]))
            V[state] = value
        if delta < theta:
            break
    return V


def policy_improvement(env: GridWorld, V: np.ndarray, gamma: float) -> np.ndarray:
    """对每个状态选 Q(s,a) 最大的动作（贪心）。"""
    policy = np.zeros(env.n_states, dtype=int)
    for state in range(env.n_states):
        if env.is_terminal(state):
            continue
        q_values = []
        for action in range(env.n_actions):
            q = 0.0
            for prob, next_state, reward, _ in env.transitions[state][action]:
                q += prob * (reward + gamma * V[next_state])
            q_values.append(q)
        policy[state] = int(np.argmax(q_values))
    return policy


def policy_iteration(
    env: GridWorld,
    gamma: float,
    theta: float = 1e-10,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """策略评估 + 策略改进，交替直到策略不再变化。"""
    policy = np.zeros(env.n_states, dtype=int)
    V = np.zeros(env.n_states)
    for iteration in range(1, max_iter + 1):
        V, _ = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if np.array_equal(new_policy, policy):
            return policy, V, iteration
        policy = new_policy
    return policy, V, max_iter


def value_iteration(
    env: GridWorld,
    gamma: float,
    theta: float = 1e-10,
    max_iter: int = 10000,
) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """直接迭代贝尔曼最优方程，最后从 V 里提取贪心策略。"""
    V = np.zeros(env.n_states)
    deltas = []
    for _iteration in range(1, max_iter + 1):
        V_new = V.copy()
        delta = 0.0
        for state in range(env.n_states):
            if env.is_terminal(state):
                continue
            q_values = []
            for action in range(env.n_actions):
                q = 0.0
                for prob, next_state, reward, _ in env.transitions[state][action]:
                    q += prob * (reward + gamma * V[next_state])
                q_values.append(q)
            V_new[state] = max(q_values)
            delta = max(delta, abs(V_new[state] - V[state]))
        V = V_new
        deltas.append(delta)
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V, _iteration, deltas
