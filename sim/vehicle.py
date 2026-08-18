from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np

from .sign import DetectedSign


@dataclass
class VehicleState:

    # commandシミュレーターからのみ更新（更新も読取もlockは不要）
    _time_cmd_issued: int = 0  # 有効時間ありで発行されたコマンドの累計数

    # driveシミュレーションからのみ更新（更新にlockが必要。複数の値を同時に読む場合はlockが必要）
    _time_cmd_ended: int = 0  # 有効時間ありで発行されたコマンドの累計完了数
    _t_cancel: float | None = None  # 命令キャンセル予定のシミュレーション時刻
    _t: float = (
        0.0  # シミュレーション時刻（=driveシミュレーションにおける最終更新時刻）
    )
    x: float = 0.0  # 車両の世界座標系でのx座標
    y: float = 0.0  # 車両の世界座標系でのy座標
    yaw: float = 0.0  # 車両の世界座標系でのヨー角（ラジアン）
    cam_pitch: float = 0.0  # カメラのピッチ角（ラジアン）
    v: float = 0.0  # 車両の前進速度(m/s)
    w: float = 0.0  # 車両の回転速度(rad/s)
    _goal_cnt: int = 0  # ゴール到達数

    # detectシミュレーションからのみ更新（更新も読取もlockは不要）
    _detection: list[DetectedSign] = field(
        default_factory=list
    )  # 車両座標系での路面標識(xが近い順にソート済)
    _t_last_detect: float = 0.0  # detectシミュレーションにおける最終更新時刻
    _t_stop: float | None = 0.0  # 停止状態になった時刻


@dataclass
class VehicleProp:
    car_length: float = 0.3
    car_width: float = 0.2
    camera_height: float = 0.2
    camera_vertical_fov_deg: float = 60
    camera_horizontal_fov_deg: float = 90
    max_velocity: float = 1
    max_rotate_deg: float = 180

    def get_vehicle_polygon(self, loop=True) -> np.ndarray:
        """
        Args:
            loop: Trueの場合、車両のポリゴンの最初の点と最後の点を同じにする
        Returns:
            車両のポリゴンの頂点座標を車両座標系で時計回り(左前, 右前, 右後, 左後)に返す。shape=(4, 2)
            loop=Trueの場合は、左前, 右前, 右後, 左後、左前を返す。shape=(5, 2)

        """
        half_l = self.car_length / 2
        half_w = self.car_width / 2

        car_frame = [
            [half_l, half_w],  # 左前
            [half_l, -half_w],  # 右前
            [-half_l, -half_w],  # 右後
            [-half_l, half_w],  # 左後
        ]
        return np.array(car_frame if not loop else car_frame + [car_frame[0]])

    def get_camera_view_polygon(self, pitch_rad: float, loop=True) -> np.ndarray:
        """
        履歴のある状態におけるカメラの地面視野の台形の頂点座標を車両座標系(x,y)で返す。
        車両座標系はx軸が正面方向、y軸が左方向。
        Args:
            pitch_rad: カメラのピッチ角（ラジアン）
            loop: Trueの場合、ポリゴンの最初の点と最後の点を同じにする
        Returns:
            カメラ地面視野のポリゴンの頂点座標を時計回り（near_left, far_left, far_right, near_right）に返す。 shape=(4, 2)
            loop=Trueの場合は、near_left, far_left, far_right, near_right, near_leftを返す。shape=(5, 2)
        """
        MAX_DISTANCE = 2
        half_vertical_fov_rad = math.radians(self.camera_vertical_fov_deg / 2)
        half_horizontal_fov_rad = math.radians(self.camera_horizontal_fov_deg / 2)
        lower_rad = pitch_rad + half_vertical_fov_rad
        upper_rad = pitch_rad - half_vertical_fov_rad
        if upper_rad <= 0:
            upper_distance = MAX_DISTANCE
        else:
            upper_distance = min(self.camera_height / math.tan(upper_rad), MAX_DISTANCE)

        lower_distance = self.camera_height / math.tan(lower_rad)
        lower_depth = (
            self.camera_height
            * math.cos(half_vertical_fov_rad)
            / math.sin(pitch_rad + half_vertical_fov_rad)
        )
        lower_half_width = lower_depth * math.tan(half_horizontal_fov_rad)

        # tan(projected_hfov/2)=cos(pitch)*tan(hfov/2)の関係を使用する
        tan_projected_half_horizontal_fov_rad = math.cos(pitch_rad) * math.tan(
            half_horizontal_fov_rad
        )
        upper_half_width = lower_half_width + tan_projected_half_horizontal_fov_rad * (
            upper_distance - lower_distance
        )

        # x,yはrosと同じ座標系。xは正面方向、yは左方向
        near_left = (lower_distance, lower_half_width)
        near_right = (lower_distance, -lower_half_width)
        far_left = (upper_distance, upper_half_width)
        far_right = (upper_distance, -upper_half_width)

        points = [near_left, far_left, far_right, near_right]
        pts = np.array(points if not loop else points + [points[0]])

        pts[:, 0] += self.car_length / 2  # カメラは車両の前端につているものとする

        return pts

    def calc_recomended_dt(self, max_overshoot=0.1, max_overshoot_deg=5) -> float:
        """
        シミュレーションにおける推奨更新間隔を求める。
        例えばmax_overshoot=0.1,max_overshoot_deg=5とした場合は、
        最悪ケースでも更新までに0.1mを超えて進むことなく、かつ5度を超えて回転しまうことはない範囲での、最大の時間を返す。
        """
        return min(
            max_overshoot / self.max_velocity,
            max_overshoot_deg / self.max_rotate_deg,
        )
