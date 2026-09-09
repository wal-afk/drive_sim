from __future__ import annotations
import time
import threading
import math
from collections import deque
import traceback

import numpy as np

from .sign import DetectedSign
from .vehicle import VehicleState, VehicleProp
from .mission_base import MissionBase
from .calc import Box, world_coord_to_vehicle_coord, vehicle_coord_to_world_coord


class BaseCommand:
    def __init__(
        self,
        name: str,
        *,
        t_created: float | None = None,
    ):
        self.name = name
        self.t_created = t_created

    def equal(self, command: BaseCommand) -> bool:
        raise Exception("implement equal method in subclass")


class SpeedCommand(BaseCommand):
    """
    車両の速度指令を表す
    auto_w_edge_nameにNone以外を指定した場合、wの指定は無視され、自動的に適切なwが選ばれる
    """

    def __init__(
        self,
        v: float = 0.0,
        w: float = 0.0,
        t: float | None = None,
        *,
        auto_w_edge_name: str | None = None,
        t_created: float | None = None,
    ):
        super().__init__("speed", t_created=t_created)
        self.v = v  # [m/s]
        self.w = w  # [rad/s]

        # コマンドの有効時間。有効時間経過後にはv=0,w=0に戻る。0を指定しても1フレーム分はコマンドが有効になる。
        # Noneを指定した場合、永遠にコマンドは有効になる。
        self.t = t

        self.auto_w_edge_name = auto_w_edge_name

    def equal(self, command: BaseCommand) -> bool:
        if isinstance(command, SpeedCommand):
            return (
                self.name == command.name
                and self.v == command.v
                and self.w == command.w
                and self.t == command.t
                and self.t is None
                and self.auto_w_edge_name == command.auto_w_edge_name
            )
        return False


class CameraCommand(BaseCommand):

    def __init__(self, pitch: float = 0.0, *, t_created: float | None = None):
        super().__init__("camera", t_created=t_created)
        self.pitch = pitch  # [rad]

    def equal(self, command: BaseCommand) -> bool:
        if isinstance(command, CameraCommand):
            return self.name == command.name and self.pitch == command.pitch
        return False


class DummyCommand(BaseCommand):

    def __init__(self, *, t_created: float | None = None):
        super().__init__("dummy", t_created=t_created)

    def equal(self, command: BaseCommand) -> bool:
        return self.name == command.name


class History:
    def __init__(self):
        self.ts: list[float] = []
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.yaws: list[float] = []
        self.cam_pitchs: list[float] = []
        self.vs: list[float] = []
        self.ws: list[float] = []
        self.predict_ws: list[np.ndarray | None] = []
        self.detections: list[list[DetectedSign]] = []
        self.goal_cnt: list[int] = []

    def record(self, state: VehicleState):
        self.ts.append(state._t)
        self.xs.append(state.x)
        self.ys.append(state.y)
        self.yaws.append(state.yaw)
        self.cam_pitchs.append(state.cam_pitch)
        self.vs.append(state.v)
        self.ws.append(state.w)
        self.predict_ws.append(state.predict_w)
        self.detections.append(state._detection)
        self.goal_cnt.append(state._goal_cnt)

    def update_latest_vw(self, v: float, w: float):
        self.vs[-1] = v
        self.ws[-1] = w

    def update_latest_goal_cnt(self, goal_cnt: int):
        self.goal_cnt[-1] = goal_cnt

    def skip(self, n: int) -> History:
        new_history = History()
        new_history.ts = self.ts[::n]
        new_history.xs = self.xs[::n]
        new_history.ys = self.ys[::n]
        new_history.yaws = self.yaws[::n]
        new_history.cam_pitchs = self.cam_pitchs[::n]
        new_history.vs = self.vs[::n]
        new_history.ws = self.ws[::n]
        new_history.predict_ws = self.predict_ws[::n]
        new_history.detections = self.detections[::n]
        new_history.goal_cnt = self.goal_cnt[::n]
        if len(self.ts) % n != 1:  # 最後のフレームを加える
            new_history.ts.append(self.ts[-1])
            new_history.xs.append(self.xs[-1])
            new_history.ys.append(self.ys[-1])
            new_history.yaws.append(self.yaws[-1])
            new_history.cam_pitchs.append(self.cam_pitchs[-1])
            new_history.vs.append(self.vs[-1])
            new_history.ws.append(self.ws[-1])
            new_history.predict_ws.append(self.predict_ws[-1])
            new_history.detections.append(self.detections[-1])
            new_history.goal_cnt.append(self.goal_cnt[-1])
        return new_history

    def get_bounding_box(self) -> Box | None:
        if len(self.xs) == 0:
            return None
        else:
            return Box(min(self.xs), min(self.ys), max(self.xs), max(self.ys))


class ControllerSharedData:
    """
    コマンドスレッドとシミュレーションスレッドの間で共有する情報をまとめて管理する
    """

    def __init__(self):
        self.reset(VehicleState())

    def reset(self, initial_state: VehicleState):
        self.state = initial_state
        self.commands: deque = deque(maxlen=1)  # 暫定：コマンドは最大1件まで保持
        self.stop_event = threading.Event()


class Commander:
    def __init__(self, sim: CarSim):
        self.sim = sim
        self.last_auto_fail_pos = None
        self.last_command: BaseCommand | None = None

    def _put_command(self, command: BaseCommand) -> bool:
        if self.last_command is not None and command.equal(self.last_command):
            if self.sim.real_sim_mode:
                time.sleep(
                    0.03 / self.sim.throttle
                )  # コマンドはsim時間内で0.03秒置きに律速する
                return False
            else:
                if len(self.sim.share.commands) == 0:
                    self.sim.share.commands.append(
                        DummyCommand(t_created=self.sim.share.state._t)
                    )
                time.sleep(0)  # GILを開放

                return False

        self.sim.share.commands.append(command)
        self.last_command = command

        state = self.sim.share.state
        if isinstance(command, SpeedCommand):
            if command.t is not None:
                state._time_cmd_issued += 1
            if self.sim.debug_log:
                print(
                    "[{:.3f}] put command name={}, v={}, w={}, t={}, auto={}".format(
                        state._t,
                        command.name,
                        command.v,
                        (command.w if command.auto_w_edge_name is None else "auto"),
                        command.t,
                        (
                            command.auto_w_edge_name
                            if command.auto_w_edge_name is not None
                            else "None"
                        ),
                    )
                )
        elif isinstance(command, CameraCommand):
            if self.sim.debug_log:
                print(
                    "[{:.3f}] put camera command pitch={}".format(
                        state._t, command.pitch
                    )
                )
        elif isinstance(command, DummyCommand):
            pass
        else:
            raise Exception("Unknown command type")

        if self.sim.real_sim_mode:
            time.sleep(
                0.03 / self.sim.throttle
            )  # コマンドはsim時間内で0.03秒置きに律速する
        return True

    def _send_auto_speed_cmd(self, v: float, edge_name: str, t: float | None = None):
        return self._put_command(
            SpeedCommand(
                v, 0, t, auto_w_edge_name=edge_name, t_created=self.sim.share.state._t
            )
        )

    def _send_manual_speed_cmd(self, v: float, w_rad: float, t: float | None = None):
        return self._put_command(
            SpeedCommand(v, w_rad, t, t_created=self.sim.share.state._t)
        )

    def _send_camera_cmd(self, pitch_rad: float):
        return self._put_command(
            CameraCommand(pitch_rad, t_created=self.sim.share.state._t)
        )

    def _send_dummy_cmd(self):
        return self._put_command(DummyCommand(t_created=self.sim.share.state._t))

    def move(self, v: float, r: float | None = None, t: float | None = None):
        """
        Args:
            v: 前進速度[m/秒]。正の値は前進、負の値は後退
            r: 回転半径[m]。Noneの場合は直進する。正の値は左カーブ、負の値は右カーブ
            t: 継続秒数[秒]
        """
        if r is None:
            self._send_manual_speed_cmd(v, 0, t)
        else:
            self._send_manual_speed_cmd(v, v / r, t)

    def auto(self, v: float, t: float | None = None):
        """
        自動走行を開始を指示する。自動走行開始可能な条件を満たさない場合、Falseを返す。
        Args:
            v: 前進速度[m/秒]。正の値は前進、負の値は後退
            t: 継続秒数[秒]

        Returns:
            bool: 自動走行開始を指示した場合はTrue、指示できなかった場合はFalse
        """
        state = self.sim.share.state
        auto_edges = self.sim.mission.world.get_auto_edges()
        if auto_edges is None:
            print(
                f"[{state._t:.3f}] cannot execute auto command because world has no auto_edge."
            )
            return False
        center, inner, outer = auto_edges

        pos = (state.x, state.y)
        if not outer.contains(pos):
            if self.last_auto_fail_pos != pos:
                print(
                    f"[{state._t:.3f}] cannot execute auto command because car is outside the outer edge."
                )
            self.last_auto_fail_pos = pos
            return False
        if inner.contains(pos):
            if self.last_auto_fail_pos != pos:
                print(
                    f"[{state._t:.3f}] cannot execute auto command because car is inside the inner edge."
                )
            self.last_auto_fail_pos = pos
            return False
        self.last_auto_fail_pos = None
        self._send_auto_speed_cmd(v, center.name, t)
        return True

    def rotate(self, w: float, t: float | None = None):
        """
        Args:
            w: 回転速度[度/秒]。正の値は反時計回り
            t: 継続秒数[秒]
        """
        self._send_manual_speed_cmd(0, math.radians(w), t)

    def camera(self, pitch: float):
        """
        Args:
            pitch: カメラのピッチ角[度]。正の値は下向き
        """
        self._send_camera_cmd(math.radians(pitch))

    def search(self, name: str | None = None, conf: float = 0.0) -> DetectedSign | None:
        limited_detections = self.search_all(name, conf)
        if len(limited_detections) == 0:
            return None
        return limited_detections[0]

    def search_all(
        self, name: str | None = None, conf: float = 0.0
    ) -> list[DetectedSign]:
        state = self.sim.share.state
        if len(state._detection) == 0:
            return []
        return (
            [d for d in state._detection if d.name == name and d.conf >= conf]
            if name is not None or conf > 0.0
            else state._detection
        )

    def alive(self) -> bool:
        return not self.sim.share.stop_event.is_set()

    def wait(self):
        state = self.sim.share.state
        if self.sim.debug_log:
            print(f"[{state._t:.3f}] start wait")
        if self.sim.real_sim_mode:
            while state._time_cmd_issued > state._time_cmd_ended and self.alive():
                time.sleep(0)  # GILを開放し他のスレッドの処理を進める
        else:
            while state._time_cmd_issued > state._time_cmd_ended and self.alive():
                if len(self.sim.share.commands) == 0:
                    self._send_dummy_cmd()
                if len(self.sim.share.commands) != 0:
                    time.sleep(0)  # GILを開放し他のスレッドの処理を進める

        if self.sim.debug_log:
            print(f"[{state._t:.3f}] exit wait")


class TimeStat:
    def __init__(self):
        self.cnt = 0
        self.sum = 0.0
        self.sum2 = 0.0

    def add(self, t: float):
        self.cnt += 1
        self.sum += t
        self.sum2 += t * t

    def get_avg(self):
        return self.sum / self.cnt

    def get_stddev(self):
        return math.sqrt(self.sum2 / self.cnt - self.get_avg() ** 2)

    def get_msg(self):
        return f"{self.get_avg()/1000:.3f}ms ± std={self.get_stddev()/1000:.3f} cnt={self.cnt}"


class CarSim:
    def __init__(
        self,
        prop: VehicleProp,
        mission: MissionBase,
        drive_dt: float | None = None,
        detect_dt: float | None = None,
        auto_estimate_dt: float = 1.0,
        throttle: float = 20,
        real_sim_mode=False,
        *,
        debug_log=False,
    ):
        """
        Args:
            prop: 車両の定義
            mission: ミッションの定義
            drive_dt: 車両の位置・姿勢の更新間隔[秒]
            detect_dt: 認識結果の更新間隔[秒]
            auto_estimate_dt: 自動的に適切なwを計算する際の予測区間間隔[秒]
            throttle: シミュレーションの実行速度。1.0の場合、シミュレーション時間と実時間は同じ。10の場合、10倍の速さで処理される。
            real_sim_mode: Trueの場合、実時間のthrottle倍の速さでリアルタイムシミュレーションを行う（スレッド間の計算速度に影響される）。Flaseの場合、待ち合わせ処理でシミュレーションする。
        """
        self.debug_log = debug_log
        self.real_sim_mode = real_sim_mode
        if drive_dt is None:
            drive_dt = prop.calc_recomended_dt(
                0.01, 5
            )  # 1stepの長さを自動調整。最大速度で0.01m進むもしくは最大回転速度で5度回転する時間幅とする
        if detect_dt is None:
            detect_dt = prop.calc_recomended_dt(
                0.1, 10
            )  # 認識間隔を自動調整。最大速度で0.1m進むもしくは最大回転速度で10度回転する時間幅とする

        print(
            f"drive_dt={drive_dt:.3f}, detect_dt={detect_dt:.3f}, throttle={throttle}"
        )

        if drive_dt > detect_dt:
            raise ValueError(f"drive_dt={drive_dt} must be <= detect_dt={detect_dt}")
        self.prop = prop

        self.drive_dt = drive_dt
        self.detect_dt = detect_dt
        self.auto_estimate_dt = max(
            auto_estimate_dt, drive_dt
        )  # auto_dtはdrive_dtより短くできない
        self.auto_control_dt: float | None = None
        self.throttle = throttle
        self.share = ControllerSharedData()

        self._reset(mission)
        self.com = Commander(self)

    def _reset(
        self,
        mission: MissionBase,
    ):
        self.mission = mission
        self.history = History()
        self.drive_stat = TimeStat()
        self.detect_stat = TimeStat()
        self.auto_stat = TimeStat()
        self.predict_w: np.ndarray | None = None

        self._stopped_from: float | None = None
        self._command_fail = False

    @staticmethod
    def contains(polygon: np.ndarray, x: float, y: float) -> bool:
        edges = np.roll(polygon, -1, axis=0) - polygon  # (N,2)
        vec = np.array([x, y]) - polygon  # (N,2)
        cross = edges[:, 0] * vec[:, 1] - edges[:, 1] * vec[:, 0]  # (N,)
        return bool(np.all(cross >= 0) or np.all(cross <= 0))

    def _update_detect_state(self):
        """
        認識のシミュレーションを行い、state.merkersを更新する。
        """
        state = self.share.state
        poly = self.prop.get_camera_view_polygon(state.cam_pitch)
        found: list[DetectedSign] = []

        if len(self.mission.signs_pos_world) > 0:
            signs_pos_vehicle = world_coord_to_vehicle_coord(
                self.mission.signs_pos_world,
                state.x,
                state.y,
                state.yaw,
            )
            for idx in range(len(self.mission.signs)):
                sign = self.mission.signs[idx]
                sign_pos_vehicle = signs_pos_vehicle[idx]
                if self.contains(poly, sign_pos_vehicle[0], sign_pos_vehicle[1]):
                    found.append(
                        DetectedSign(  # 車座標系での位置を設定する
                            name=sign.name,
                            x=sign_pos_vehicle[0],
                            y=sign_pos_vehicle[1],
                            gt=sign,
                        )
                    )
        state._detection = sorted(found, key=lambda d: d.x)
        state._t_last_detect = state._t

    @staticmethod
    def _limit_two_sides(value: float, limit: float) -> float:
        return max(-limit, min(value, limit))

    def _calc_auto_w(self, *, num_waypoints=5) -> np.ndarray:
        """
        self.share.state.auto_w_edge_nameに従って、適切なwを計算する
        Args:
            num_waypoints: 経由点を何点設定するか
        Returns:
            予測経路の各区間での角速度 shape=(NUM_WAYPOINTS,)
        """
        state = self.share.state
        assert state.auto_w_edge_name is not None

        # stateが変わる可能性があるのでコピーする。本来はLockが必要
        _w = state.w
        _x = state.x
        _y = state.y
        _v = state.v
        _yaw = state.yaw

        target_edge = self.mission.world.edges.get(state.auto_w_edge_name)

        if target_edge is None:
            raise ValueError(f"edge {state.auto_w_edge_name} not found in world")

        # 角加速度（角速度の変化量）の候補値[deg/s^2]
        candidate_a_deg = [-50, -10, 0, 10, 50]

        num_candidates = len(candidate_a_deg)

        indices = (
            np.indices([num_candidates] * num_waypoints).reshape(num_waypoints, -1).T
        )  # shape=(N_patterns, NUM_WAYPOINTS)

        # 各区間開始時の角速度の変化
        dw_patterns = (
            self.auto_estimate_dt * np.deg2rad(np.array(candidate_a_deg))[indices]
        )  # shape=(N_patterns, NUM_WAYPOINTS)

        # 各区間での角速度
        w_patterns = _w + np.cumsum(
            dw_patterns, axis=1
        )  # shape=(N_patterns, NUM_WAYPOINTS)
        max_rotate_rad = math.radians(self.prop.max_rotate_deg) / 4
        w_patterns = np.clip(w_patterns, -max_rotate_rad, max_rotate_rad)
        xy_patterns_local = self._predict_xy_from_w(
            w_patterns, _v
        )  # local座標系での軌跡
        xy_patterns_wprld = vehicle_coord_to_world_coord(
            xy_patterns_local,
            _x,
            _y,
            _yaw,
        )

        score_patterns = target_edge.calc_rms_distance(
            xy_patterns_wprld
        )  # shape=(N_patterns)
        best_idx = np.argmin(score_patterns)

        return w_patterns[best_idx]

    def _predict_xy_from_w(self, w_patterns: np.ndarray, v: float) -> np.ndarray:
        """
        各パターンの角速度から予測経路の各区間での位置を計算する
        Args:
            w_patterns: 各区間での角速度 shape=(N_patterns, NUM_WAYPOINTS)
            v :速度[m/秒]
        Returns:
            予測経路の各区間での位置 shape=(N_patterns, NUM_WAYPOINTS, 2)
            車の初期位置からの相対座標で表される
        """
        ds = v * self.auto_estimate_dt

        # 各区間終了時（=経由点）の姿勢角
        yaw_end_patterns = np.cumsum(
            w_patterns * self.auto_estimate_dt,
            axis=1,
        )  # shape=(N_patterns, NUM_WAYPOINTS)

        # 各区間開始時の姿勢角
        yaw_start_patterns = np.concatenate(
            [
                np.full(
                    (yaw_end_patterns.shape[0], 1),
                    0,
                ),
                yaw_end_patterns[:, :-1],
            ],
            axis=1,
        )  # shape=(N_patterns, NUM_WAYPOINTS)

        # 各区間での曲率
        k_patterns = w_patterns / v  # shape=(N_patterns, NUM_WAYPOINTS)

        # 各区間での移動量
        dx_patterns = np.where(
            np.abs(k_patterns) > 1e-10,
            (np.sin(yaw_end_patterns) - np.sin(yaw_start_patterns)) / k_patterns,
            ds * np.cos(yaw_start_patterns),
        )

        dy_patterns = np.where(
            np.abs(k_patterns) > 1e-10,
            (np.cos(yaw_start_patterns) - np.cos(yaw_end_patterns)) / k_patterns,
            ds * np.sin(yaw_start_patterns),
        )

        # 初期位置からの位置
        x_patterns = np.cumsum(dx_patterns, axis=1)  # shape=(N_patterns, NUM_WAYPOINTS)
        y_patterns = np.cumsum(dy_patterns, axis=1)  # shape=(N_patterns, NUM_WAYPOINTS)
        xy_patterns = np.stack(
            [x_patterns, y_patterns], axis=2
        )  # shape=(N_patterns, NUM_WAYPOINTS, 2)
        return xy_patterns

    def _set_auto_drive(self, v: float, edge_name: str):
        state = self.share.state
        state.auto_w_edge_name = edge_name

        if self.real_sim_mode:
            AUTO_CONTROL_PER_METER = 0.1
        else:
            AUTO_CONTROL_PER_METER = 0.2

        # 自動制御間隔を自動調整。現在速度でAUTO_CONTROL_PER_METER進む時間幅とする
        self.auto_control_dt = (
            None
            if v == 0
            else min(AUTO_CONTROL_PER_METER / abs(v), self.auto_estimate_dt)
        )

        if state.v != v:
            # 加速度無限大で即座に反映
            state.v = self._limit_two_sides(v, self.prop.max_velocity)
            if self.debug_log:
                print(
                    f"[{state._t:.3f}] changed v={state.v:.3f}, auto={edge_name}:dt={self.auto_control_dt}"
                )

    def _set_manual_drive(self, v: float, w: float):
        state = self.share.state
        state.auto_w_edge_name = None
        self.auto_control_dt = None
        if state.v != v or state.w != w:
            # 加速度無限大で即座に反映
            state.v = self._limit_two_sides(v, self.prop.max_velocity)
            state.w = self._limit_two_sides(w, math.radians(self.prop.max_rotate_deg))
            if self.debug_log:
                print(f"[{state._t:.3f}] changed v={state.v:.3f}, w={state.w:.3f}")

    def _step(self):
        """
        車の位置・姿勢の更新を1step(=1微小区間分)だけ行いstateの時刻をself.drive_dtだけ進める。
        関数呼び出し時点のstateを微小区間の始まり時点の状態とし、
        微小区間の終わりの時点の状態を計算でもとめ、stateに上書きする。

        コマンドを受けていない場合、vとwは関数呼び出し時点のstateの値を用いるが
        コマンドを受けている場合、vとwはコマンドの値がそのまま用いられる(加速度無限大で即座に反映される)

        - この関数は、微小区間の終わりの時刻が経過した瞬間に呼ぶこと。
          - その結果、ある微小区間の間に受けたコマンドは、その微小区間の始まりに遡って計算に反映される。
          - 例えば、シミュレーション開始時刻0からself.drive_dt秒以内に秒速vで動けとのコマンドが来た場合
          - 速度指示はシミュレーション時刻0から有効であり、最初から速度vで動くことになる
        - コマンドは１つの微小区間で最大で１つのみ処理される
          - ある微小区間で複数のコマンドが来た場合でキューの最大サイズが1より大きい場合、処理されなかったコマンドは次の微小区間で順次処理される。
        """
        state = self.share.state
        if state._t_cancel is not None and state._t_cancel <= state._t:
            # コマンドの有効時間が終了したので停止する
            state.v = 0
            state.w = 0
            state.auto_w_edge_name = None
            state._t_cancel = None
            state._time_cmd_ended += 1
            if self.debug_log:
                print(
                    f"[{state._t:.3f}] stopped, _time_cmd_ended={state._time_cmd_ended}"
                )

        # commandの処理
        if len(self.share.commands) > 0:
            command = self.share.commands.pop()

            if self.debug_log and command.name != "dummy":
                print(
                    "[{:.3f}] recv command: {} delay={:.3f}".format(
                        state._t,
                        command.name,
                        (
                            state._t - command.t_created
                            if command.t_created is not None
                            else "N/A"
                        ),
                    )
                )

            if isinstance(command, SpeedCommand):
                if state._t_cancel is not None:
                    # 有効時間ありのコマンドが実行中に次のコマンドが来たら、有効時間ありのコマンドは終了扱いとする
                    state._time_cmd_ended += 1

                # コマンドキャンセル時刻の予約
                state._t_cancel = (
                    state._t + command.t if command.t is not None else None
                )
                if command.auto_w_edge_name is not None:
                    self._set_auto_drive(command.v, command.auto_w_edge_name)
                else:
                    self._set_manual_drive(command.v, command.w)
            elif isinstance(command, CameraCommand):
                state.cam_pitch = command.pitch
            elif isinstance(command, DummyCommand):
                pass
            else:
                raise ValueError(f"invalid command type: {type(command)}")

        # auto_wの処理（自動的に決定されたwを採用する）
        best_w = self._get_best_w()
        if best_w is not None:
            state.w = best_w
            if (
                state._t_last_auto_w_calc_start is not None
                and state._t_last_auto_w_calc_start == state._t
            ):
                state.predict_w = self.predict_w
            else:
                state.predict_w = None
        else:
            state.predict_w = None

        if state.v == 0 and state.w == 0:
            if state._t_stop is None:
                state._t_stop = state._t
        else:
            state._t_stop = None

        state.yaw += state.w * self.drive_dt
        state.x += state.v * math.cos(state.yaw) * self.drive_dt
        state.y += state.v * math.sin(state.yaw) * self.drive_dt
        state._t += self.drive_dt
        goal_newly_reached = self._check_goal_completion(
            (state._t - state._t_stop) if state._t_stop is not None else 0.0
        )
        if goal_newly_reached:
            self._increase_goal_cnt()

    def _get_best_w(self) -> float | None:
        state = self.share.state
        if state.auto_w_edge_name is None or state.v == 0 or self.predict_w is None:
            return None

        elapsed_time = state._t - state._t_last_auto_w_calc_end
        index = math.floor(elapsed_time / self.auto_estimate_dt)
        if index >= len(self.predict_w):
            print("WARN: auto_w calculation time span is too long")
            return None
        return self.predict_w[index]

    def _increase_goal_cnt(self, update_latest_history=False):
        state = self.share.state
        state._goal_cnt += 1
        if len(self.mission.goals) == state._goal_cnt:
            print(f"[{state._t:.3f}] all goals reached")
            self.share.stop_event.set()
        else:
            print(f"[{state._t:.3f}] goal {state._goal_cnt} reached")

    def _check_goal_completion(self, stopping_duration: float) -> bool:
        state = self.share.state
        num_goals = len(self.mission.goals)
        if num_goals > state._goal_cnt:
            target_goal = self.mission.goals[state._goal_cnt]
            if target_goal.ok(
                (state.x, state.y),
                stopping_duration,
            ):
                return True
        return False

    def _sim_all(self):
        """
        シミュレーションを行いstateを更新する。
        更新は微小区間(シミュレーション時間におけるself.drive_dt秒)ごとに行う。
        (ただし、認識結果の更新はシミュレーション時間におけるself.detect_dt秒ごとに行う)

        self.throttle=1の場合、シミュレーション時間と実時間は同じになる。
        self.throttle!=1の場合、実時間の1秒はシミュレーション時間のself.throttle秒に相当する。
        (つまり、実時間でself.drive_dt/self.throttle秒毎に位置・姿勢の更新が行われる)
        処理能力が十分であれば、シミュレーションは「総シミュレーション時間」/self.throttle秒で完了するが、
        更新処理が実時間でself.drive_dt/self.throttle秒内に完了しない場合、
        シミュレーションの実行にかかる時間が延びる。

        - 車の状態はself.drive_dt秒ごとのスナップショットとしてself.historyに記録される
           - historyには開始時に初期状態が書き込まれれ、以降drive_dt秒ごとに状態が追記されていく
        """
        try:
            state = self.share.state
            self._update_detect_state()
            self.history.record(state)  # 初期状態
            t_start = time.perf_counter()

            if self.real_sim_mode:
                self._run_sim_loop_real_mode()
            else:
                self._run_sim_loop_sync_mode()

            if not self._command_fail:
                # 停止状態で終了する場合は、停止継続時間を無限大と見做してゴール判定をしなおす
                if state.v == 0 and state.w == 0:
                    goal_newly_reached = self._check_goal_completion(float("inf"))
                    if goal_newly_reached:
                        self._increase_goal_cnt(True)
                        self.history.update_latest_goal_cnt(state._goal_cnt)

                print(f"[{state._t:.3f}] simulation_func finished")
                print(f"    takes {time.perf_counter() - t_start:.3f}s")
                if self.real_sim_mode:
                    print(f"    ideal {state._t / self.throttle:.3f}s")

        except Exception:
            # 例外で終了した場合もcommand_thread側のwaitが永久に残らないよう停止を通知する
            self.share.stop_event.set()
            print("プログラム（sim_all）にエラーが発生しました")
            traceback.print_exc()

    def _run_sim_loop_real_mode(self):
        state = self.share.state
        t_start = time.perf_counter()
        while self.alive():
            if self._should_stop_sim():
                break

            elapsed_sim_time = (time.perf_counter() - t_start) * self.throttle
            if elapsed_sim_time < state._t + self.drive_dt:
                time.sleep(0)  # GILを開放し他のスレッドの処理を進める
            else:
                t0 = time.thread_time()
                self._step()
                t1 = time.thread_time()
                self.drive_stat.add(t1 - t0)

                self.history.update_latest_vw(state.v, state.w)  # v,wの変更は遡って反映

                # 認識結果の更新はself.detect_dt秒ごとに行う
                if state._t >= state._t_last_detect + self.detect_dt:
                    self._update_detect_state()
                t2 = time.thread_time()
                self.detect_stat.add(t2 - t1)

                self.history.record(state)

    def _run_sim_loop_sync_mode(self):
        state = self.share.state
        t_last_cmd_recv = None
        while self.alive():
            if self._should_stop_sim():
                break

            # auto_w計算スレッドの待ち合わせ
            if self.auto_control_dt is not None:
                if (
                    state._t_last_auto_w_calc_end is None
                    or state._t >= state._t_last_auto_w_calc_end + self.auto_control_dt
                ):
                    time.sleep(0)  # GILを開放
                    continue

            # userプログラム実行スレッドの待ち合わせ
            if len(self.share.commands) == 0:
                if t_last_cmd_recv is None:
                    time.sleep(0)  # GILを開放
                    continue
                else:
                    if time.time() - t_last_cmd_recv < 1:
                        time.sleep(0)  # GILを開放
                        continue
                    else:
                        # 1秒以上userプログラムからコマンドが来ない場合、step()に進む
                        pass
            else:
                t_last_cmd_recv = time.time()

            t0 = time.thread_time()
            self._step()
            t1 = time.thread_time()
            self.drive_stat.add(t1 - t0)

            self.history.update_latest_vw(state.v, state.w)  # v,wの変更は遡って反映

            # 認識結果の更新はself.detect_dt秒ごとに行う
            if state._t >= state._t_last_detect + self.detect_dt:
                self._update_detect_state()
            t2 = time.thread_time()
            self.detect_stat.add(t2 - t1)

            self.history.record(state)

    def _should_stop_sim(self):
        state = self.share.state
        if self.mission.t_max is not None and state._t > self.mission.t_max:
            print(f"!!!!!! time limit {self.mission.t_max}s: stop simulation")
            return True

        # force_exit_stopping_sec秒間、車が停車している場合はシミュレーションを終了する
        if state.v == 0 and state.w == 0:
            if self._stopped_from is None:
                self._stopped_from = state._t
            else:
                if self._stopped_from + self.mission.force_exit_stopping_sec < state._t:
                    print(
                        f"!!!!!! force exit because stopping for {self.mission.force_exit_stopping_sec}sec"
                    )
                    return True
        else:
            self._stopped_from = None
        return False

    def _call_command_func(self) -> bool:
        # simlationが始まるまで待つ（sim_threadが例外等で終了した場合もここで止まり続けないようalive()も見る）
        while len(self.history.ts) == 0:
            if not self.alive():
                return False
            time.sleep(0)  # GILを開放し他のスレッドの処理を進める

        commands = {
            "move": self.com.move,
            "auto": self.com.auto,
            "rotate": self.com.rotate,
            "camera": self.com.camera,
            "search": self.com.search,
            "search_all": self.com.search_all,
            "wait": self.com.wait,
        }
        try:
            self.mission.command_func(self.com.alive, **commands)
            print(f"[{self.share.state._t:.3f}] command_func finished")
            return True
        except Exception as e:
            self._command_fail = True
            print("プログラム（command_func）にエラーが発生しました")
            print("------ エラー：", e)
            print("")
            traceback.print_exc()
        return False

    def _update_auto_w(self):
        while not self.share.stop_event.is_set():
            state = self.share.state
            # auto_wの更新はself.auto_control_dt秒ごとに行う
            if self.auto_control_dt is None:
                self.predict_w = None
                state._t_last_auto_w_calc_start = None
                state._t_last_auto_w_calc_end = None
                time.sleep(0)  # GILを開放し他のスレッドの処理を進める
            else:
                if self._wait_for_next_calc_auto_w():
                    continue
                state._t_last_auto_w_calc_start = state._t
                self.predict_w = self._calc_auto_w()
                if self.debug_log:
                    print(
                        f"[{self.share.state._t:.3f}] predict auto_w: {self.predict_w}"
                    )
                state._t_last_auto_w_calc_end = state._t

    def _wait_for_next_calc_auto_w(self):
        state = self.share.state
        if self.real_sim_mode:
            if state._t_last_auto_w_calc_start is not None:
                should_wait_sim_sec = (
                    state._t_last_auto_w_calc_start + self.auto_control_dt - state._t
                )
                if should_wait_sim_sec < -self.auto_control_dt * 0.2:
                    print(
                        f"[{self.share.state._t:.3f}] WARN: auto_w calc takes time. should decrease throttle"
                    )
                if should_wait_sim_sec > 0:
                    time.sleep(
                        should_wait_sim_sec / self.throttle / 10
                        if self.real_sim_mode
                        else 0
                    )
                    return True
        else:
            if (
                state._t_last_auto_w_calc_end is not None
                and state._t_last_auto_w_calc_end + self.auto_control_dt > state._t
            ):
                time.sleep(0)
                return True
        return False

    def alive(self) -> bool:
        return not self.share.stop_event.is_set()

    def run(self) -> bool:
        """
        シミュレーションの実行を開始し、２つのスレッドを起動する。
        - simスレッド: 車両の位置・姿勢の更新、detection結果の更新を行う。
        - autoスレッド: 指定のedgeに沿って走る為のwを計算する
        - commandスレッド: ユーザが定義したcommand_funcを実行する。

        """
        self.share.reset(self.mission.get_initial_state())
        self.mission.relocate_signs()

        command_thread = threading.Thread(target=self._call_command_func)
        auto_thread = threading.Thread(target=self._update_auto_w)
        sim_thread = threading.Thread(target=self._sim_all)

        try:
            for thread in [command_thread, auto_thread, sim_thread]:
                thread.start()

            while command_thread.is_alive() and sim_thread.is_alive():
                time.sleep(
                    1 / self.throttle
                )  # シミュレーション時間内で1秒以内に終了に気づく
        finally:
            self.share.stop_event.set()
            while (
                command_thread.is_alive()
                or sim_thread.is_alive()
                or auto_thread.is_alive()
            ):
                time.sleep(1 / self.throttle)  # GILを開放し他のスレッドの処理を進める
            if self._command_fail:
                return False
            else:
                print(f"Trajectory points : {len(self.history.ts)}")
                print(f"Active threads : {threading.active_count()}")
                return True
