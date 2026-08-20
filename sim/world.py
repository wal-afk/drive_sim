from __future__ import annotations

import numpy as np

from .calc import Box, calc_arc_points


class WorldEdge:
    def __init__(
        self,
        name: str,
        color: str,
        width: float,
        dash: str = "solid",
        fill: bool = False,
        *,
        center_xy: tuple[float, float] = (0, 0),
        rotate_deg: float = 0,
    ):
        """
        Args:
            name: 図形の名前
            color: 図形の色
            width: 線の太さ
            dash: 線の種類。solid, dash, dot, dashdot等
            fill:塗つぶす場合はTrue
            center_xy: 図形を平行移動する中心座標。デフォルトは(0,0)
            rotate_deg: 図形を回転する角度（時計回り）。デフォルトは0
        線分と円弧によって閉じたポリゴンを表現する。
        閉曲線になるようにadd_line()とadd_arc()を用いて線分と円弧を順番に追加すること。
        順番は、時計周り・反時計回りを問わないが連続になるようにすること

        center_xyやrotate_degを与えた場合、
        add_line()とadd_arc()で記述した図形は-center_xy平行移動した後に時計回りにrotate_deg度回転した
        図形として記録される。
        """
        self.name = name
        self.color = color
        self.width = width
        self.dash = dash
        self.fill = fill

        self.center_xy = center_xy
        self.rotate_deg = rotate_deg
        rotate_rad = np.deg2rad(rotate_deg)
        self.R = np.array(
            [
                [np.cos(rotate_rad), -np.sin(rotate_rad)],
                [np.sin(rotate_rad), np.cos(rotate_rad)],
            ]
        )
        self.order: list[int] = []  # 線分と円弧の追加順序を記録する。0:線分, 1:円弧

        self.line_start: list[tuple[float, float]] = []
        self.line_end: list[tuple[float, float]] = []

        self.arc_center: list[tuple[float, float]] = []
        self.arc_radius: list[float] = []
        self.arc_inverse: list[bool] = []
        self.arc_smaller_deg: list[float] = []  # -180<=start_deg<180
        self.arc_bigger_deg: list[float] = []  # -180<end_deg<540, start_def<end_deg

    def _trans(self, xy: tuple[float, float]) -> tuple[float, float]:
        x, y = (np.asarray(xy) - np.asarray(self.center_xy)) @ self.R
        return (x, y)

    def add_line(self, start: tuple[float, float], end: tuple[float, float]):
        self.line_start.append(self._trans(start))
        self.line_end.append(self._trans(end))
        self.order.append(0)

    def add_lines(self, xy: list[tuple[float, float]], loop=False):
        for i in range(len(xy) - 1):
            self.add_line(xy[i], xy[i + 1])
        if loop:
            self.add_line(xy[-1], xy[0])

    def add_arc(
        self,
        center: tuple[float, float],
        radius: float,
        start_deg: float,
        end_deg: float,
        *,
        inverse=False,
    ):
        """
        start_degから反時計回りにend_degまでの円弧を表す。
        inverse=Trueの場合は、start_degから時計回りにend_degまでの円弧を表す。
        """
        self.arc_center.append(self._trans(center))
        self.arc_radius.append(radius)
        self.arc_inverse.append(inverse)

        start_deg -= self.rotate_deg
        end_deg -= self.rotate_deg

        _start_deg = (start_deg + 180) % 360 - 180  # -180<=value<180

        _end_deg = (end_deg + 180) % 360 - 180  # -180<=value<180
        if _end_deg <= _start_deg:
            _end_deg += 360  # -180<value<540

        if inverse:
            self.arc_smaller_deg.append(_end_deg)
            self.arc_bigger_deg.append(_start_deg)
        else:
            self.arc_smaller_deg.append(_start_deg)
            self.arc_bigger_deg.append(_end_deg)
        self.order.append(1)

    def _calc_lines_box(self) -> Box | None:
        if len(self.line_start) == 0:
            return None

        xy_start = np.asarray(self.line_start)  # shape=(N,2)
        xy_end = np.asarray(self.line_end)  # shape=(N,2)

        xy = np.concatenate([xy_start, xy_end], axis=0)  # shape=(2N,2)
        max_xy = np.max(xy, axis=0)
        min_xy = np.min(xy, axis=0)
        return Box(min_xy[0], min_xy[1], max_xy[0], max_xy[1])

    @staticmethod
    def _contain_angle(angle: float, angle_range: np.ndarray) -> np.ndarray:
        """
        指定角度が指定の角度レンジの中に含まれるか判定します
        Args:
            angle: -pi<=angle<pi
            angle_range: shape=(N,2)で、各行が[start_rad,end_rad]。-pi<=start_rad<pi, -pi<end_rad<3pi, start_rad<end_rad
        """
        contain1 = (angle_range[:, 0] <= angle) & (angle <= angle_range[:, 1])
        contain2 = (angle_range[:, 0] <= angle + 2 * np.pi) & (
            angle + 2 * np.pi <= angle_range[:, 1]
        )
        return contain1 | contain2

    def _calc_arcs_box(self) -> Box | None:
        if len(self.arc_center) == 0:
            return None
        xy_center = np.asarray(self.arc_center)
        range_rad = np.deg2rad(
            np.stack([self.arc_smaller_deg, self.arc_bigger_deg], axis=1)
        )
        r = np.asarray(self.arc_radius)

        min_x = xy_center[:, 0] + r * np.where(
            self._contain_angle(-np.pi, range_rad),
            -1,
            np.min(np.cos(range_rad), axis=1),
        )
        max_x = xy_center[:, 0] + r * np.where(
            self._contain_angle(0, range_rad), 1, np.max(np.cos(range_rad), axis=1)
        )
        min_y = xy_center[:, 1] + r * np.where(
            self._contain_angle(-np.pi / 2, range_rad),
            -1,
            np.min(np.sin(range_rad), axis=1),
        )
        max_y = xy_center[:, 1] + r * np.where(
            self._contain_angle(np.pi / 2, range_rad),
            1,
            np.max(np.sin(range_rad), axis=1),
        )
        return Box(np.min(min_x), np.min(min_y), np.max(max_x), np.max(max_y))

    def _calc_cross_lines(self, xy: tuple[float, float]) -> int:
        """
        指定点から右に伸ばした半直線がlineと何回交わるかを求める
        指定点が線上にいる場合はカウントしない
        """
        xy_start = np.asarray(self.line_start)
        xy_end = np.asarray(self.line_end)

        x, y = xy
        x0 = xy_start[:, 0]
        y0 = xy_start[:, 1]
        x1 = xy_end[:, 0]
        y1 = xy_end[:, 1]

        x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        is_cross = (x < x_cross) & ((y0 > y) != (y1 > y))
        return int(np.sum(is_cross))

    def _calc_cross_arcs(self, xy: tuple[float, float]) -> int:
        """
        指定点から右に伸ばした半直線がarcと何回交わるかを求める
        指定点が線上にいる場合はカウントしない
        半直線があるarcと接する場合は2回と数える
        """
        xy_center = np.asarray(self.arc_center)
        smaller_rad = np.deg2rad(np.asarray(self.arc_smaller_deg))  # -pi<=start_rad<pi
        bigger_rad = np.deg2rad(
            np.asarray(self.arc_bigger_deg)
        )  # -pi<end_rad<3pi, start_rad<end_rad
        r = np.asarray(self.arc_radius)

        x, y = xy
        cx = xy_center[:, 0]
        cy = xy_center[:, 1]

        d_cross_y = y - cy

        with np.errstate(
            invalid="ignore"
        ):  # 円と交わらない場合にはnanとなるが、警告を抑制
            d_cross_x_plus = np.sqrt(r * r - d_cross_y * d_cross_y)

        cross_x_plus = cx + d_cross_x_plus
        cross_x_minus = cx - d_cross_x_plus

        theta_plus = np.atan2(d_cross_y, d_cross_x_plus)  # -pi/2<=value<=pi/2
        theta_minus = np.pi - theta_plus  # pi/2<=value<=3pi/2

        is_cross_1_plus = (smaller_rad < theta_plus) & (theta_plus < bigger_rad)
        is_cross_2_plus = (smaller_rad < theta_plus + 2 * np.pi) & (
            theta_plus + 2 * np.pi < bigger_rad
        )
        is_cross_plus = (x < cross_x_plus) & (is_cross_1_plus | is_cross_2_plus)

        is_cross_1_minus = (smaller_rad < theta_minus) & (theta_minus < bigger_rad)
        is_cross_2_minus = (smaller_rad < theta_minus + 2 * np.pi) & (
            theta_minus + 2 * np.pi < bigger_rad
        )
        is_cross_minus = (x < cross_x_minus) & (is_cross_1_minus | is_cross_2_minus)
        return int(np.sum(is_cross_plus) + np.sum(is_cross_minus))

    def contains(self, xy: tuple[float, float]) -> bool:
        """
        レイキャスティング法で指定点が多角形の内部にあるかを判定する。
        """
        num_cross_lines = self._calc_cross_lines(xy)
        num_cross_arcs = self._calc_cross_arcs(xy)
        return (num_cross_lines + num_cross_arcs) % 2 == 1

    def get_bounding_box(self) -> Box | None:
        return Box.merge([self._calc_lines_box(), self._calc_arcs_box()])

    def _get_arc_points(self, i: int) -> np.ndarray:
        smaller_deg = self.arc_smaller_deg[i]
        bigger_deg = self.arc_bigger_deg[i]
        points = calc_arc_points(
            self.arc_center[i],
            self.arc_radius[i],
            smaller_deg,
            (bigger_deg - smaller_deg) % 360,
        )
        if self.arc_inverse[i]:
            points = points[::-1]
        return points

    def _get_line_points(self, i: int) -> np.ndarray:
        return np.stack([self.line_start[i], self.line_end[i]], axis=0)

    def get_points(self):
        num_line = 0
        num_arc = 0
        points_list = []
        for kind in self.order:
            if kind == 0:
                points = self._get_line_points(num_line)
                num_line += 1
            elif kind == 1:
                points = self._get_arc_points(num_arc)
                num_arc += 1
            else:
                raise ValueError(f"invalid kind={kind}")
            if len(points_list) > 0:
                points_list.append(points[1:])
            else:
                points_list.append(points)
        return np.concatenate(points_list, axis=0)

    def calc_rms_distance(self, points: np.ndarray) -> np.ndarray:
        """
        指定点からWorldEdgeまでの距離を求める。
        Args:
            points: xy座標の配列。M個の点群を1セットとして、Nセットの情報を一度に扱う。shape=(N,M,2)
        Returns:
            各セットの点群のWorldEdgeまでの距離の二乗平均の平方根を返す。 shape=(N)
        """
        n = points.shape[0]
        m = points.shape[1]

        dist_lines = self._calc_distance_to_lines(points.reshape(-1, 2))  # shape=(N*M)
        dist_arcs = self._calc_distance_to_arcs(points.reshape(-1, 2))  # shape=(N*M)
        if dist_lines is None:
            dist = dist_arcs
        elif dist_arcs is None:
            dist = dist_lines
        else:
            dist = np.minimum(dist_lines, dist_arcs)

        d = dist.reshape(n, m)  # shape=(N,M)
        rms = np.sqrt(np.mean(d**2, axis=1))
        return rms

    def _calc_distance_to_lines(self, points: np.ndarray) -> np.ndarray | None:
        """
        指定点から全線分までの距離を求める（各線分への距離の内で最小の距離を全線分への距離とする）
        Args:
            points: xy座標の配列。shape=(N,2)
        Returns:
            全線分までの距離 shape=(N)
        """
        if len(self.line_start) == 0:
            return None

        line_start = np.asarray(self.line_start)  # shape=(K,2)
        line_end = np.asarray(self.line_end)  # shape=(K,2)
        line_vec = line_end - line_start  # shape=(K,2)
        pos_vec = (
            points[:, np.newaxis, :] - line_start[np.newaxis, :, :]
        )  # shape=(N,K,2)

        t = np.dot(pos_vec, line_vec) / np.dot(line_vec, line_vec)  # shape=(N,K)
        t = np.clip(t, 0.0, 1.0)  # 線分なので 0～1 に制限

        # 線分上で最も近い点
        closest = line_start + t * line_vec  # shape=(N,K,2)
        dist = np.linalg.norm(points[:, np.newaxis, :] - closest, axis=2)  # shape=(N,K)
        min_dist = dist.min(axis=1)  # shape=(N)

        return min_dist

    def _calc_distance_to_arcs(self, points: np.ndarray) -> np.ndarray | None:
        """
        指定点から全弧までの距離を求める（各弧への距離の内で最小の距離を全弧への距離とする）
        Args:
            points: xy座標の配列。shape=(N,2)
        Returns:
            全弧までの距離 shape=(N)
        """
        if len(self.arc_center) == 0:
            return None

        center = np.asarray(self.arc_center, dtype=float)  # shape=(K,2)
        radius = np.asarray(self.arc_radius, dtype=float)  # shape=(K)
        start_rad = np.radians(
            np.asarray(self.arc_smaller_deg, dtype=float)
        )  # shape=(K)
        end_rad = np.radians(np.asarray(self.arc_bigger_deg, dtype=float))  # shape=(K)

        v = points[:, None, :] - center[None, :, :]  # shape=(N,K,2)

        angle_rad = np.arctan2(v[..., 1], v[..., 0])  # shape=(N,K)

        # 点の方向が円弧の範囲内か
        half_arc_rad = (end_rad - start_rad) / 2  # shape=(K)
        mid_rad = (start_rad + end_rad) / 2  # shape=(K)
        relative_rad = (angle_rad - mid_rad[None, :] + np.pi) % (
            2 * np.pi
        ) - np.pi  # - np.pi<=relative_rad<=np.pi. shape=(N,K)

        closest_rad = mid_rad[None, :] + np.clip(
            relative_rad, -half_arc_rad, half_arc_rad
        )  # 円弧なので start_rad～end_radの範囲に制限. shape=(N,K)

        closest_point = center[None, :] + radius[None, :, None] * np.stack(
            [
                np.cos(closest_rad),
                np.sin(closest_rad),
            ],
            axis=2,
        )  # shape=(N,K,2)

        dist = np.linalg.norm(
            points[:, None, :] - closest_point,
            axis=2,
        )  # shape=(N,K)
        min_dist = np.min(dist, axis=1)  # shape=(N)
        return min_dist


class World:
    def __init__(self):
        self.edges: dict[str, WorldEdge] = {}

    def add_edge(self, edge: WorldEdge):
        self.edges[edge.name] = edge

    def add_edges(self, edges: list[WorldEdge]):
        for edge in edges:
            self.add_edge(edge)

    def get_bounding_box(self) -> Box | None:
        boxes = [edge.get_bounding_box() for edge in self.edges.values()]
        return Box.merge(boxes)
