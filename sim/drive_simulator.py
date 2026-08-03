from __future__ import annotations
import time
import threading
from dataclasses import dataclass
import math
from queue import Queue

import numpy as np

from .sign import DetectedSign
from .vehicle import VehicleState, VehicleProp
from .mission_base import MissionBase
from .calc import Box, world_coord_to_vehicle_coord


@dataclass
class SpeedCommand:
    v: float = 0.0
    w: float = 0.0

    # コマンドの有効時間。有効時間経過後にはv=0,w=0に戻る。0を指定しても1フレーム分はコマンドが有効になる。
    # Noneを指定した場合、永遠にコマンドは有効になる。
    t: float | None = None


@dataclass
class CameraCommand:
    pitch: float = 0.0


class History:
    def __init__(self):
        self.ts: list[float] = []
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.yaws: list[float] = []
        self.cam_pitchs: list[float] = []
        self.vs: list[float] = []
        self.ws: list[float] = []
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
        self.detections.append(state._detection)
        self.goal_cnt.append(state._goal_cnt)

    def update_latest(self, v: float, w: float):
        self.vs[-1] = v
        self.ws[-1] = w

    def skip(self, n: int) -> History:
        new_history = History()
        new_history.ts = self.ts[::n]
        new_history.xs = self.xs[::n]
        new_history.ys = self.ys[::n]
        new_history.yaws = self.yaws[::n]
        new_history.cam_pitchs = self.cam_pitchs[::n]
        new_history.vs = self.vs[::n]
        new_history.ws = self.ws[::n]
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
            new_history.detections.append(self.detections[-1])
            new_history.goal_cnt.append(self.goal_cnt[-1])
        return new_history

    def get_bounding_box(self) -> Box | None:
        if len(self.xs) == 0:
            return None
        else:
            return Box(min(self.xs), min(self.ys), max(self.xs), max(self.ys))


class SimSharedData:
    """
    コマンドスレッドとシミュレーションスレッドの間で共有する情報をまとめて管理する
    """

    def __init__(self):
        self.reset(VehicleState())

    def reset(self, state: VehicleState):
        self.state = state
        self.commands: Queue = Queue(maxsize=1)  # 暫定：コマンドは最大1件まで保持
        self.lock = threading.Lock()
        self.stop_event = threading.Event()


class Commander:
    def __init__(self, sim: CarSim):
        self.sim = sim

    def _send_speed_cmd(self, v: float, w_rad: float, t: float | None = None):
        self.sim.share.commands.put(SpeedCommand(v=v, w=w_rad, t=t))
        if t is not None:
            self.sim.share.state._time_cmd_issued += 1
        print(
            f"[{self.sim.share.state._t:.3f}] put speed command v={v}, w={w_rad}, t={t}"
        )

    def _send_camera_cmd(self, pitch_rad: float):
        self.sim.share.commands.put(CameraCommand(pitch_rad))
        print(f"[{self.sim.share.state._t:.3f}] put camera command pitch={pitch_rad}")

    def move(self, v: float, r: float | None = None, t: float | None = None):
        """
        Args:
            v: 前進速度[m/s]。正の値は前進、負の値は後退
            r: 回転半径[m]。Noneの場合は直進する。正の値は左カーブ、負の値は右カーブ
            t: 継続秒数[s]
        """
        if r is None:
            self._send_speed_cmd(v, 0, t)
        else:
            self._send_speed_cmd(v, v / r, t)

    def rotate(self, w: float, t: float | None = None):
        """
        Args:
            w: 回転速度[度/s]。正の値は反時計回り
            t: 継続秒数[s]
        """
        self._send_speed_cmd(0, math.radians(w), t)

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
        if len(self.sim.share.state._detection) == 0:
            return []
        return (
            [
                d
                for d in self.sim.share.state._detection
                if d.name == name and d.conf >= conf
            ]
            if name is not None or conf > 0.0
            else self.sim.share.state._detection
        )

    def alive(self) -> bool:
        return not self.sim.share.stop_event.is_set()

    def wait(self):
        print(f"[{self.sim.share.state._t:.3f}] start wait")
        while (
            self.sim.share.state._time_cmd_issued > self.sim.share.state._time_cmd_ended
            and self.alive()
        ):
            time.sleep(0)  # GILを開放し他のスレッドの処理を進める
        print(f"[{self.sim.share.state._t:.3f}] exit wait")


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
        drive_dt=None,
        detect_dt=None,
        throttle: float = 10,
    ):
        """
        Args:
            prop: 車両の定義
            drive_dt: 車両の位置・姿勢の更新間隔[s]
            detect_dt: 認識結果の更新間隔[s]
            throttle: シミュレーションの実行速度。1.0の場合、シミュレーション時間と実時間は同じ。10の場合、10倍の速さで処理される。
        """
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
        self.throttle = throttle
        self.share = SimSharedData()
        self._stopped_from: float | None = None

        self.mission = None
        self.history = History()
        self.drive_stat = TimeStat()
        self.detect_stat = TimeStat()

    def set_mission(
        self,
        mission: MissionBase,
    ):
        self.mission = mission
        self.history = History()
        self.drive_stat = TimeStat()
        self.detect_stat = TimeStat()

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
        if self.mission is None:
            raise Exception("mission is not set")

        poly = self.prop.get_camera_view_polygon(self.share.state.cam_pitch)
        found: list[DetectedSign] = []

        if len(self.mission.signs_pos_world) > 0:
            signs_pos_vehicle = world_coord_to_vehicle_coord(
                self.mission.signs_pos_world,
                self.share.state.x,
                self.share.state.y,
                self.share.state.yaw,
            )
            for idx in range(len(self.mission.signs)):
                sign = self.mission.signs[idx]
                sign_pos_vehicle = signs_pos_vehicle[idx]
                if self.contains(poly, sign_pos_vehicle[0], sign_pos_vehicle[1]):
                    found.append(
                        DetectedSign(  # 車座標系での位置を設定する
                            sign,
                            x=sign_pos_vehicle[0],
                            y=sign_pos_vehicle[1],
                        )
                    )
        self.share.state._detection = sorted(found, key=lambda d: d.x)
        self.share.state._t_last_detect = self.share.state._t

    @staticmethod
    def _limit_two_sides(value: float, limit: float) -> float:
        return max(-limit, min(value, limit))

    def _step(self):
        """
        車の位置・姿勢の更新を1step(=1微小区間分)だけ行いstateの時刻をself.drive_dtだけ進める。
        関数呼び出し時点のstateを微小区間の始まり時点の状態とし、
        微小区間の終わりの時点の状態を計算でもとめ、stateに上書きする。

        コマンドを受けていない場合、vとwは関数呼び出し時点のstateの値を用いるが
        コマンドを受けている場合、vとwはコマンドの値がそのまま用いられる(加速度無限大で即座に反映される)

        - この関数は、微小区間の終わりの時刻が経過した瞬間に呼ぶこと。
          - その結果、ある微小区間の間に受けたコマンドは、その微小区間の始まりに遡って計算に反映される。
          - 例えば、シミュレーション開始からself.drive_dt秒以内に秒速vで動けとのコマンドが来た場合
          - 速度指示はシミュレーション時刻0から有効であり、最初から速度vで動くことになる
        - コマンドは１つの微小区間で最大で１つのみ処理される
          - ある微小区間で複数のコマンドが来た場合でキューの最大サイズが1より大きい場合、処理されなかったコマンドは次の微小区間で順次処理される。
        """
        if self.mission is None:
            raise Exception("mission is not set")

        if (
            self.share.state._t_cancel is not None
            and self.share.state._t_cancel <= self.share.state._t
        ):
            # コマンドの有効時間が終了したので停止する
            self.share.state.v = 0
            self.share.state.w = 0
            self.share.state._t_cancel = None
            self.share.state._time_cmd_ended += 1
            print(f"[{self.share.state._t:.3f}] stopped")

        if not self.share.commands.empty():
            command = self.share.commands.get(block=False)
            print(
                f"[{self.share.state._t:.3f}] recv command remained:{self.share.commands.qsize()}"
            )
            if isinstance(command, SpeedCommand):
                if self.share.state._t_cancel is not None:
                    # 有効時間ありのコマンドが実行中に次のコマンドが来たら、有効時間ありのコマンドは終了扱いとする
                    self.share.state._time_cmd_ended += 1
                self.share.state._t_cancel = (
                    self.share.state._t + command.t if command.t is not None else None
                )
                if self.share.state.v != command.v or self.share.state.w != command.w:
                    # 加速度無限大で即座に反映
                    self.share.state.v = self._limit_two_sides(
                        command.v, self.prop.max_velocity
                    )
                    self.share.state.w = self._limit_two_sides(
                        command.w, math.radians(self.prop.max_rotate_deg)
                    )
                    print(
                        f"[{self.share.state._t:.3f}] changed v={self.share.state.v:.3f}, w={self.share.state.w:.3f}"
                    )
            elif isinstance(command, CameraCommand):
                self.share.state.cam_pitch = command.pitch
            else:
                raise ValueError(f"invalid command type: {type(command)}")

        self.share.state.yaw += self.share.state.w * self.drive_dt
        self.share.state.x += (
            self.share.state.v * math.cos(self.share.state.yaw) * self.drive_dt
        )
        self.share.state.y += (
            self.share.state.v * math.sin(self.share.state.yaw) * self.drive_dt
        )
        self.share.state._t += self.drive_dt

        num_goals = len(self.mission.goals)
        if num_goals > self.share.state._goal_cnt:
            target_goal = self.mission.goals[self.share.state._goal_cnt]
            if target_goal.ok(
                (self.share.state.x, self.share.state.y),
                self.share.state.v,
                self.share.state.w,
            ):
                self.share.state._goal_cnt += 1
                print(
                    f"[{self.share.state._t:.3f}] goal {self.share.state._goal_cnt} reached"
                )

        if num_goals >= 1 and num_goals == self.share.state._goal_cnt:
            print(f"[{self.share.state._t:.3f}] all goals reached")
            self.share.stop_event.set()

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
        """
        if self.mission is None:
            raise Exception("mission is not set")

        self._update_detect_state()
        self.history.record(self.share.state)  # 初期状態
        t_start = time.perf_counter()
        while not self.share.stop_event.is_set():
            if (
                self.mission.t_max is not None
                and self.share.state._t > self.mission.t_max
            ):
                print(f"!!!!!! time limit {self.mission.t_max}s: stop simulation")
                break

            if self.share.state.v == 0 and self.share.state.w == 0:
                if self._stopped_from is None:
                    self._stopped_from = self.share.state._t
                else:
                    if (
                        self._stopped_from + self.mission.force_exit_stopping_sec
                        < self.share.state._t
                    ):
                        print(
                            f"!!!!!! force exit because stopping for {self.mission.force_exit_stopping_sec}sec"
                        )
                        break
            else:
                self._stopped_from = None

            elapsed_sim_time = (time.perf_counter() - t_start) * self.throttle
            if elapsed_sim_time < self.share.state._t + self.drive_dt:
                time.sleep(0)  # GILを開放し他のスレッドの処理を進める
            else:
                with self.share.lock:
                    t0 = time.thread_time()
                    self._step()
                    self.history.update_latest(
                        self.share.state.v, self.share.state.w
                    )  # v,wの変更は遡って反映
                    t1 = time.thread_time()
                    self.drive_stat.add(t1 - t0)
                    if (
                        self.share.state._t
                        >= self.share.state._t_last_detect + self.detect_dt
                    ):
                        self._update_detect_state()
                    t2 = time.thread_time()
                    self.detect_stat.add(t2 - t1)
                    self.history.record(self.share.state)

        print(f"[{self.share.state._t:.3f}] simulation_func finished")
        print(f"    takes {time.perf_counter() - t_start:.3f}s")
        print(f"    ideal {self.share.state._t / self.throttle:.3f}s")

    def _call_command_func(self):
        if self.mission is None:
            raise Exception("mission is not set")

        self.mission.command_func()
        print(f"[{self.share.state._t:.3f}] command_func finished")

    def run(self):
        """
        シミュレーションの実行を開始し、２つのスレッドを起動する。
        - simスレッド: 車両の位置・姿勢の更新、detection結果の更新を行う。
        - commandスレッド: ユーザが定義したcommand_funcを実行する。

        """
        if self.mission is None:
            raise Exception("mission is not set")

        self.share.reset(self.mission.get_initial_state())

        command_thread = threading.Thread(target=self.mission.command_func)
        sim_thread = threading.Thread(target=self._sim_all)

        try:
            for thread in [command_thread, sim_thread]:
                thread.start()

            while command_thread.is_alive() and sim_thread.is_alive():
                time.sleep(
                    1 / self.throttle
                )  # シミュレーション時間内で1秒以内に終了に気づく
        finally:
            self.share.stop_event.set()
            print(f"Trajectory points : {len(self.history.ts)}")
