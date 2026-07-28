from __future__ import annotations
import math

from tqdm.auto import tqdm
import plotly.graph_objects as go

from .drive_simulator import (
    CarSim,
    History,
)

from .goal import GoalLine, GoalCircle
from .calc import Box, vehicle_coord_to_world_coord, calc_circle_points


class VehicleDrawer:

    def __init__(self, sim: CarSim):
        self.sign_size = 16
        self.sim = sim

        self.world_scatters = self._create_world_scatters()
        self.sign_scatters = self._create_sign_scatters()

    def show(self, draw_target_dt=0.1):
        skip = int(draw_target_dt / self.sim.drive_dt)
        draw_dt = self.sim.drive_dt * skip
        fig = self.create_figure(self.sim.history.skip(skip), draw_dt)
        fig.show()

    def _create_world_scatters(self) -> list[go.Scatter]:
        """
        worldを描画する為のscatterを作成する。
        同じnameのsignを1グループにして1つのscatterにまとめる。
        """
        if self.sim.mission is None:
            raise Exception("mission is not set")

        world_scatters = []
        for world_edge in self.sim.mission.world.edges:
            common_args = {
                "showlegend": False,
                "mode": "lines",
                "fill": "toself" if world_edge.fill else "none",
                "line": {
                    "width": world_edge.width,
                    "color": world_edge.color,
                    "dash": world_edge.dash,
                },
                "name": world_edge.name,
            }
            pts = world_edge.get_points()
            world_scatters.append(go.Scatter(x=pts[:, 0], y=pts[:, 1], **common_args))
        return world_scatters

    def _create_sign_scatters(self) -> list[go.Scatter]:
        """
        signを描画する為のscatterを作成する。
        同じnameのsignを1グループにして1つのscatterにまとめる。
        """
        if self.sim.mission is None:
            raise Exception("mission is not set")

        unknown_symbol = ("?", "red")
        sign_scatters = []
        for name in set([m.name for m in self.sim.mission.signs]):
            target_signs = [m for m in self.sim.mission.signs if m.name == name]
            symbol, color = self.sim.mission.sign_to_symbol.get(name, unknown_symbol)

            common_args = {
                "legendrank": 5,
                "x": [m.x for m in target_signs],
                "y": [m.y for m in target_signs],
                "name": name,
            }

            if len(symbol) == 1:
                sign_scatters.append(
                    go.Scatter(
                        **common_args,
                        mode="text",
                        text=[symbol] * len(target_signs),
                        textfont={
                            "size": self.sign_size,
                            "color": color,
                        },
                    )
                )
            else:
                sign_scatters.append(
                    go.Scatter(
                        **common_args,
                        mode="markers",
                        marker={
                            "symbol": symbol,
                            "size": self.sign_size - 4,
                            "color": color,
                        },
                    )
                )
        return sign_scatters

    def _create_detected_sign_scatters(
        self, his: History, idx: int
    ) -> tuple[go.Scatter, go.Scatter]:
        """
        検出されたsignを描画する為のscatterを作成する。
        """
        detections = his.detections[idx]

        first = go.Scatter(
            legendrank=6,
            x=[d.sign.x for d in detections[0:1]],
            y=[d.sign.y for d in detections[0:1]],
            mode="markers",
            name="search",
            marker={
                "symbol": "star",
                "size": self.sign_size,
                "color": "yellow",
                "line": {"width": 3},
            },
        )
        others = go.Scatter(
            legendrank=7,
            x=[d.sign.x for d in detections[1:]],
            y=[d.sign.y for d in detections[1:]],
            mode="markers",
            name="search_all",
            marker={
                "symbol": "star",
                "size": self.sign_size,
                "color": "yellow",
                "line": {"width": 1},
            },
        )
        return first, others

    def _create_goal_scatters(self, his: History, idx: int) -> list[go.Scatter]:
        """
        goalを描画する為のscatterを作成する。
        """
        if self.sim.mission is None:
            raise Exception("mission is not set")

        goal_cnt = his.goal_cnt[idx]

        scatters = []
        for i, goal in enumerate(self.sim.mission.goals):
            color = "gray" if i > goal_cnt else ("gray" if i == goal_cnt else "pink")
            dash = "solid" if goal.should_stop else "dot"

            common_args = {
                "legendrank": 4,
                "mode": "lines",
                "line": {"width": 2, "color": color, "dash": dash},
                "name": f"goal{i+1}",
            }

            if isinstance(goal, GoalLine):
                x = goal.xy[0]
                y = goal.xy[1]
                if y != 0:
                    a = -x / y
                    b = goal.norm2 / y
                    scatters.append(
                        go.Scatter(
                            x=[self.box.min_x, self.box.max_x],
                            y=[a * self.box.min_x + b, a * self.box.max_x + b],
                            **common_args,
                        )
                    )
                else:
                    scatters.append(
                        go.Scatter(
                            x=[x, x], y=[self.box.min_y, self.box.max_y], **common_args
                        )
                    )
            elif isinstance(goal, GoalCircle):
                points = calc_circle_points(goal.xy, goal.r)
                scatters.append(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        **common_args,
                    )
                )
        return scatters

    def _create_trajectory_scatter(self, his: History, idx: int) -> go.Scatter:
        """
        車の軌跡を描画する為のscatterを作成する。
        """
        return go.Scatter(
            legendrank=1,
            x=his.xs[: idx + 1],
            y=his.ys[: idx + 1],
            mode="lines",
            line={"width": 3, "color": "brown"},
            name="軌跡",
        )

    def _create_vehicle_scatter(self, his: History, idx: int) -> go.Scatter:
        """
        車体を描画する為のscatterを作成する。
        """
        poly = self.sim.prop.get_vehicle_polygon()
        poly_world = vehicle_coord_to_world_coord(
            poly, his.xs[idx], his.ys[idx], his.yaws[idx]
        )
        return go.Scatter(
            legendrank=2,
            x=poly_world[:, 0],
            y=poly_world[:, 1],
            fill="toself",
            mode="lines",
            line={"width": 2, "color": "brown"},
            name="車体",
        )

    def _create_camera_view_scatter(self, his: History, idx: int) -> go.Scatter:
        """
        カメラの地面視野を描画する為のscatterを作成する。
        """
        poly = self.sim.prop.get_camera_view_polygon(his.cam_pitchs[idx])
        poly_world = vehicle_coord_to_world_coord(
            poly, his.xs[idx], his.ys[idx], his.yaws[idx]
        )
        return go.Scatter(
            legendrank=3,
            x=poly_world[:, 0],
            y=poly_world[:, 1],
            fill="toself",
            mode="lines",
            line={"width": 2, "color": "gray"},
            name="カメラの視界",
        )

    def _create_vehicle_annotation(self, his: History, idx: int) -> dict:
        """
        グラフの車の位置にhistoryの情報からx,yの内容を表示する為のannotationを作成する。
        """
        return {
            "x": his.xs[idx],
            "y": his.ys[idx],
            "showarrow": True,
            "text": f"x={his.xs[idx]:.2f}, y={his.ys[idx]:.2f}, Θ={math.degrees(his.yaws[idx]):.2f}",
            "font": {"size": 14},
        }

    def _create_fixed_first_detection_annotation(
        self, his: History, idx: int, x: float, y: float
    ) -> dict:
        """
        グラフの固定位置にhistoryの情報からdetectionsの1個目の内容を表示する為のannotationを作成する。
        """
        return {
            "x": x,
            "y": y,
            "xanchor": "left",
            "yanchor": "top",
            "showarrow": False,
            "text": (
                "{}: r={:.2f} m, theta={:.2f} 度".format(
                    his.detections[idx][0].name,
                    his.detections[idx][0].r,
                    math.degrees(his.detections[idx][0].theta_rad),
                )
                if len(his.detections[idx]) >= 1
                else ""
            ),
            "font": {"size": 12, "color": "red"},
        }

    def _create_fixed_basic_annotation(
        self, his: History, idx: int, x: float, y: float
    ) -> dict:
        """
        グラフの固定位置にhistoryの情報からv,w,goal個数を表示する為のannotationを作成する。
        """
        if self.sim.mission is None:
            raise Exception("mission is not set")
        return {
            "x": x,
            "y": y,
            "xanchor": "left",
            "yanchor": "top",
            "showarrow": False,
            "text": "v={:.2f} m/s, w={:.2f} 度/s, goal {}/{} ".format(
                his.vs[idx],
                math.degrees(his.ws[idx]),
                his.goal_cnt[idx],
                len(self.sim.mission.goals),
            ),
            "font": {"size": 12, "color": "black"},
        }

    def _create_frames(
        self,
        his: History,
    ) -> list[go.Frame]:
        first_msg_xy = self.box.get_pos_rel(0, 1)
        second_msg_xy = self.box.get_pos_rel(0, 0.9)

        fixed_top_trace_num = len(self.world_scatters)
        frames = []
        for i in tqdm(range(len(his.xs))):
            data = [
                self._create_trajectory_scatter(his, i),
                self._create_vehicle_scatter(his, i),
                self._create_camera_view_scatter(his, i),
                *self._create_goal_scatters(his, i),
                *self._create_detected_sign_scatters(his, i),
            ]
            frames.append(
                go.Frame(
                    data=data,
                    traces=list(
                        range(fixed_top_trace_num, fixed_top_trace_num + len(data))
                    ),
                    name=str(i),
                    layout=go.Layout(
                        annotations=[
                            self._create_vehicle_annotation(his, i),
                            self._create_fixed_basic_annotation(
                                his, i, first_msg_xy[0], first_msg_xy[1]
                            ),
                            self._create_fixed_first_detection_annotation(
                                his, i, second_msg_xy[0], second_msg_xy[1]
                            ),
                        ],
                    ),
                )
            )
        return frames

    def _calc_axis_range(self, his: History) -> Box:
        if self.sim.mission is None:
            raise Exception("mission is not set")

        base_margin = max(self.sim.prop.car_length / 2, self.sim.prop.car_width / 2)

        traj_box = his.get_bounding_box()
        signs_box = self.sim.mission.signs_box
        world_box = self.sim.mission.world.get_bounding_box()
        all_box = Box.merge([traj_box, signs_box, world_box])
        if all_box is None:
            raise Exception("there is no data to draw")
        all_box.expand(base_margin)

        # 上部にannotation表示をする為にマージンを設ける
        all_box.rectify(0.2)
        return all_box

    @staticmethod
    def _create_play_button(label: str, duration_msec: float) -> dict:
        return {
            "label": label,
            "method": "animate",
            "args": [
                None,
                {
                    "frame": {
                        "duration": duration_msec,
                        "redraw": True,
                    },
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                },
            ],
        }

    def create_figure(self, his: History, frame_sec: float) -> go.Figure:
        self.box = self._calc_axis_range(his)
        fig = go.Figure(
            data=[
                *self.world_scatters,
                self._create_trajectory_scatter(his, 0),
                self._create_vehicle_scatter(his, 0),
                self._create_camera_view_scatter(his, 0),
                *self._create_goal_scatters(his, 0),
                *self._create_detected_sign_scatters(his, 0),
                *self.sign_scatters,
            ],
            frames=self._create_frames(his),
            layout=go.Layout(
                width=600,
                height=600,
                xaxis={
                    "scaleanchor": "y",
                    "range": self.box.get_x_range(),
                },
                yaxis={"range": self.box.get_y_range()},
                sliders=[
                    {
                        "steps": [
                            {
                                "method": "animate",
                                "args": [
                                    [str(k)],
                                    {
                                        "mode": "immediate",
                                        "frame": {"duration": 0, "redraw": True},
                                    },
                                ],
                                "label": f"{his.ts[k]:.1f}",
                            }
                            for k in range(len(his.ts))
                        ]
                    }
                ],
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            self._create_play_button("▶ Play", frame_sec * 1000),
                            self._create_play_button("▶▶ Play", frame_sec * 1000 / 2),
                            {
                                "label": "■ Stop",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {"frame": {"duration": 0}, "mode": "immediate"},
                                ],
                            },
                        ],
                    }
                ],
            ),
        )

        return fig
