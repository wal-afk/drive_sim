from __future__ import annotations
import math

import numpy as np

from .sign import Sign
from .goal import GoalBase
from .world import World
from .calc import Box
from .vehicle import VehicleState


class MissionBase:
    def __init__(
        self,
        world: World,
        *,
        t_max: float | None = None,
    ):
        """
        Args:
            t_max: シミュレーションの最大時間[s]。Noneの場合は無制限
        """

        self.t_max = t_max

        self.signs: list[Sign] = []
        self.signs_pos_world: np.ndarray = np.array([], dtype=np.float32)
        self.signs_box: Box | None = None

        self.sign_to_symbol: dict[str, tuple[str, str]] = {}
        self.goals: list[GoalBase] = []
        self.world = world

        self.initial_cam_pitch_deg = 30  # 30度
        self.initial_x = 0
        self.initial_y = 0
        self.initial_yaw_deg = 0
        self.randomize_initial_state: bool = True

    def set_world(self, world: World):
        self.world = world

    def command_func(self):
        raise NotImplementedError("command_func must be implemented in subclass")

    def get_initial_state(self) -> VehicleState:
        d_x = np.random.uniform(-0.1, 0.1) if self.randomize_initial_state else 0
        d_y = np.random.uniform(-0.1, 0.1) if self.randomize_initial_state else 0
        d_yaw_deg = (
            np.random.uniform(-10.0, 10.0) if self.randomize_initial_state else 0
        )

        return VehicleState(
            x=self.initial_x + d_x,
            y=self.initial_y + d_y,
            yaw=math.radians(self.initial_yaw_deg + d_yaw_deg),
            cam_pitch=math.radians(self.initial_cam_pitch_deg),
        )

    def set_signs(
        self,
        signs: list[Sign],
        sign_to_symbol: dict[str, tuple[str, str]] | None = None,
    ):
        self.signs = signs
        self.signs_pos_world = np.array([[m.x, m.y] for m in signs], dtype=np.float32)
        self.signs_box: Box | None = (
            Box(
                np.min(self.signs_pos_world[:, 0]),
                np.min(self.signs_pos_world[:, 1]),
                np.max(self.signs_pos_world[:, 0]),
                np.max(self.signs_pos_world[:, 1]),
            )
            if len(signs) > 0
            else None
        )

        if sign_to_symbol is not None:
            self.sign_to_symbol = sign_to_symbol
        else:
            default_symbols = [
                "triangle-up",
                "triangle-down",
                "diamond",
                "square",
                "star",
                "triangle-up-open",
                "triangle-down-open",
                "diamond-open",
                "square-open",
                "star-open",
            ]
            names = set([m.name for m in signs])
            self.sign_to_symbol = {
                name: (default_symbols[i % len(default_symbols)], "orange")
                for i, name in enumerate(names)
            }
