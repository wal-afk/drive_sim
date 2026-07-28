from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np


def calc_circle_points(
    xy: tuple[float, float],
    r: float,
    start_deg: float = 0,
    deg: float = 360,
) -> np.ndarray:
    """
    Args:
        xy: 円弧の中心座標
        r: 円弧の半径
        start_deg: 円弧の開始角度[deg]
        deg: 反時計回りに進む円弧の角度[deg]
    Returns:
        円弧の座標点の列 shape=(N,2)を反時計回りに返す
    """

    POINTS_PER_DEG = 0.125
    theta = np.deg2rad(
        np.linspace(start_deg, start_deg + deg, math.ceil(deg * POINTS_PER_DEG))
    )
    return np.asarray(xy) + r * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def vehicle_coord_to_world_coord(
    pts: np.ndarray, x: float, y: float, yaw: float
) -> np.ndarray:
    """
    車両座標系の点群をワールド座標系に変換する。
    Args:
        pts: 車両座標系の点群 shape=(N, 2)
        x: 車両のx座標
        y: 車両のy座標
        yaw: 車両のヨー角（ラジアン）反時計回りが正
    Returns:
        ワールド座標系の点群 shape=(N, 2)
    """
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return pts @ R.T + np.array([x, y])


def world_coord_to_vehicle_coord(
    pts: np.ndarray, x: float, y: float, yaw: float
) -> np.ndarray:
    """
    ワールド座標系の点群を車両座標系に変換する。
    Args:
        pts: ワールド座標系の点群 shape=(N, 2)
        x: 車両のx座標
        y: 車両のy座標
        yaw: 車両のヨー角（ラジアン）
    Returns:
        車両座標系の点群 shape=(N, 2)
    """
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (pts - np.array([x, y])) @ R


@dataclass
class Box:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def get_x_range(self) -> tuple[float, float]:
        return self.min_x, self.max_x

    def get_y_range(self) -> tuple[float, float]:
        return self.min_y, self.max_y

    def get_pos_rel(self, x_ratio: float, y_ratio: float) -> tuple[float, float]:
        """
        Args:
            x_ratio: 0.0~1.0の範囲でx方向の位置を指定する。0.0でmin_x、1.0でmax_x
            y_ratio: 0.0~1.0の範囲でy方向の位置を指定する。0.0でmin_y、1.0でmax_y
        Returns:
            (x,y)の座標
        """
        x = self.min_x + (self.max_x - self.min_x) * x_ratio
        y = self.min_y + (self.max_y - self.min_y) * y_ratio
        return x, y

    def expand(self, margin: float):
        self.min_x -= margin
        self.min_y -= margin
        self.max_x += margin
        self.max_y += margin

    def rectify(self, top_mergin_ratio: float = 0.0):
        """
        矩形を正方形になるように短辺を伸ばす。
        さらに、上部にtop_mergin_ratioの比率でマージンを追加する。
        このマージンは、元の矩形が横長の時は正方形になる時に伸ばした分の長さと相殺する

        Args:
            top_mergin_ratio: 上部に追加するマージンの比率
        """
        w = self.max_x - self.min_x
        h = self.max_y - self.min_y
        xy_ratio = w / h
        if xy_ratio > 1:  # 横長の場合
            expnad_max_y = max(
                h * top_mergin_ratio,
                (w - h) / 2,
            )
            self.max_y += expnad_max_y
            self.min_y -= (w - h) / 2
        else:  # 縦長の場合
            self.max_x += (h - w) / 2
            self.min_x -= (h - w) / 2
            self.max_y += (self.max_y - self.min_y) * top_mergin_ratio

    @staticmethod
    def merge(boxes: list[Box | None]) -> Box | None:
        _boxes = [box for box in boxes if box is not None]
        if len(_boxes) == 0:
            return None
        return Box(
            min([box.min_x for box in _boxes]),
            min([box.min_y for box in _boxes]),
            max([box.max_x for box in _boxes]),
            max([box.max_y for box in _boxes]),
        )
