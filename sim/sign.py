from __future__ import annotations
import math

import numpy as np


class Sign:
    x: float  # 世界座標系での位置
    y: float  # 世界座標系での位置
    name: str

    def __init__(
        self,
        x: float,
        y: float,
        name: str,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
    ):
        self.initial_x = x
        self.initial_y = y
        self.name = name
        self.dx = dx
        self.dy = dy

        # 仮の値。relocateで最終決定される。
        self.x = x
        self.y = y

    def relocate(self):
        self.x = self.initial_x + np.random.uniform(-self.dx, self.dx)
        self.y = self.initial_y + np.random.uniform(-self.dy, self.dy)


class DetectedSign:
    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        conf: float = 1.0,
        *,
        gt: Sign | None = None,
    ):
        self.name = name
        self.x = x  # 車座標系での位置
        self.y = y  # 車座標系での位置
        self.sign = gt
        self.r = math.sqrt(x**2 + y**2)
        self.theta_rad = math.atan2(y, x)
        self.theta = math.degrees(self.theta_rad)
        self.conf = conf
