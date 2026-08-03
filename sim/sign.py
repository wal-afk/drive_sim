from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class Sign:
    x: float  # 世界座標系での位置
    y: float  # 世界座標系での位置
    name: str


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
