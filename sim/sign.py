from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class Sign:
    x: float
    y: float
    name: str


class DetectedSign:
    def __init__(self, sign: Sign, x: float, y: float, conf: float = 1.0):
        self.x = x
        self.y = y
        self.sign = sign
        self.name = sign.name
        self.r = math.sqrt(x**2 + y**2)
        self.theta_rad = math.atan2(y, x)
        self.theta = math.degrees(self.theta_rad)
        self.conf = conf
