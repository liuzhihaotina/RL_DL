"""CliffWalking：经典悬崖地图。

4 行 × 12 列：
  S . . . . . . . . . . .
  . . . . . . . . . . . .
  . . . . . . . . . . . .
  S C C C C C C C C C C G

S 起点(3,0)，G 终点(3,11)，C 悬崖(3,1)~(3,10)。
每走一步 -1；掉下悬崖 -100，并回到起点，回合继续。
"""

from __future__ import annotations

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class CliffWalking:
    n_rows = 4
    n_cols = 12
    n_states = n_rows * n_cols
    n_actions = 4

    start = 3 * n_cols + 0
    goal = 3 * n_cols + 11
    cliffs = set(range(3 * n_cols + 1, 3 * n_cols + 11))

    step_reward = -1.0
    cliff_reward = -100.0

    def to_state(self, rc: tuple[int, int]) -> int:
        return rc[0] * self.n_cols + rc[1]

    def to_rc(self, state: int) -> tuple[int, int]:
        return divmod(state, self.n_cols)

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        if state == self.goal:
            return self.goal, 0.0, True

        row, col = self.to_rc(state)
        dr, dc = MOVES[action]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            return state, self.step_reward, False

        next_state = self.to_state((nr, nc))
        if next_state in self.cliffs:
            return self.start, self.cliff_reward, False
        if next_state == self.goal:
            return self.goal, self.step_reward, True
        return next_state, self.step_reward, False
