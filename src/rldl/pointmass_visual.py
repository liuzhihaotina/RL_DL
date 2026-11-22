"""PointMass 的视觉版本：把状态渲染成 48×48 RGB 图像。

图像内容：
  蓝点：当前位置
  绿圈：目标（原点）
  红线：速度方向和大小

观测形状 (3, 48, 48)，取值 [0, 1]。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rldl.pointmass import PointMassEnv


class PointMassPixelEnv(gym.ObservationWrapper):
    def __init__(self, size: int = 32, world: float = 1.5) -> None:
        super().__init__(PointMassEnv(bounds=1.5))
        self.size = size
        self.world = world
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, size, size), dtype=np.float32
        )

    def _to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int((x + self.world) / (2 * self.world) * (self.size - 1))
        row = int((self.world - y) / (2 * self.world) * (self.size - 1))
        return int(np.clip(row, 0, self.size - 1)), int(np.clip(col, 0, self.size - 1))

    def _circle(self, img: np.ndarray, center: tuple[int, int], radius: int, color) -> None:
        row, col = center
        yy, xx = np.ogrid[: self.size, : self.size]
        mask = (xx - col) ** 2 + (yy - row) ** 2 <= radius**2
        img[mask] = color

    def observation(self, obs: np.ndarray) -> np.ndarray:
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        base = self.unwrapped
        px, py = self._to_pixel(float(base.pos[0]), float(base.pos[1]))

        # 目标：绿色圆环
        self._circle(img, self._to_pixel(0.0, 0.0), 5, (0.0, 1.0, 0.0))
        self._circle(img, self._to_pixel(0.0, 0.0), 3, (0.0, 0.0, 0.0))

        # 速度：从当前点出发的红色线段
        vx, vy = float(base.vel[0]), float(base.vel[1])
        end_x = float(base.pos[0]) + vx * 0.5
        end_y = float(base.pos[1]) + vy * 0.5
        for t in np.linspace(0.0, 1.0, 12):
            x = float(base.pos[0]) + (end_x - float(base.pos[0])) * t
            y = float(base.pos[1]) + (end_y - float(base.pos[1])) * t
            self._circle(img, self._to_pixel(x, y), 1, (1.0, 0.0, 0.0))

        # 当前点：蓝色
        self._circle(img, (px, py), 2, (0.3, 0.6, 1.0))

        return np.transpose(img, (2, 0, 1)).astype(np.float32)
