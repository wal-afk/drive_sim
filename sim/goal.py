from __future__ import annotations

import numpy as np


class GoalBase:
    def __init__(self, *, should_stop: bool = True):
        self.should_stop = should_stop  # Trueの場合はゴール到達後に停止する必要がある。Falseの場合はゴールを通過すればよい

    def ok(self, xy: tuple[float, float], v: float, w: float) -> bool:
        raise NotImplementedError("ok must be implemented in subclass")


class GoalLine(GoalBase):
    """
    直線のゴールを表す。ゴール線の内側もしくは外側に達したらゴールとする
    """

    def __init__(
        self, xy: tuple[float, float], outside_is_goal: bool = True, *, should_stop=True
    ):
        """
        Args:
            xy: 原点からゴール直線に垂線を下ろした点のxy座標
            outside_is_goal: Trueの場合ゴール線の外側を、Falseの場合ゴール線の内側に達したらゴールとする（外側とは原点から遠い方を指す）
        """
        super().__init__(should_stop=should_stop)
        self.xy = np.asarray(xy, dtype=np.float32)
        self.norm2 = np.inner(self.xy, self.xy)
        self.outside_is_goal = outside_is_goal

    def ok(self, xy: tuple[float, float], v: float, w: float) -> bool:
        if self.should_stop:
            if v != 0 or w != 0:
                return False
        s = np.inner(np.asarray(xy, dtype=np.float32), self.xy) - self.norm2
        if self.outside_is_goal:
            return s >= 0
        else:
            return s <= 0


class GoalCircle(GoalBase):
    """
    円形のゴールを表す
    """

    def __init__(self, xy: tuple[float, float], r: float, *, should_stop=True):
        """
        指定の円の内側で停止したら成功とするゴールを作成する。
        should_stopがFalseの場合は円の内側を通過したら成功とする。
        """
        super().__init__(should_stop=should_stop)
        self.xy = xy
        self.r = r

    def ok(self, xy: tuple[float, float], v: float, w: float) -> bool:
        if self.should_stop:
            if v != 0 or w != 0:
                return False
        p = np.asarray(xy, dtype=np.float32) - np.asarray(self.xy)
        s = np.inner(p, p) - self.r * self.r
        return s <= 0
