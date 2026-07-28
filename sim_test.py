# %%
import yaml
from sim.drive_simulator import (
    VeihicleDrawer,
    MissionBase,
    GoalLine,
    GoalCircle,
    Sign,
    VehicleProp,
    CarSim,
    Commander,
)
from world.type_b_world import type_b_circuit

with open("config/type-b.yaml", "r") as f:
    vehicle_config = yaml.safe_load(f)

prop = VehicleProp(**vehicle_config)
sim = CarSim(prop)
com = Commander(sim)
move = com.move
rotate = com.rotate
wait = com.wait
search = com.search


class Mission0(MissionBase):
    def __init__(self):
        super().__init__(type_b_circuit, t_max=20)
        self.goals = [
            GoalLine((0.8, 0.0), should_stop=False),
            GoalCircle((0.0, 0.0), 0.2),
        ]

    def command_func(self):
        move(v=0.2, t=5)
        wait()
        rotate(w=90, t=2)
        wait()
        move(v=0.2, t=5)
        wait()


class Mission2(MissionBase):
    def __init__(self):
        super().__init__(type_b_circuit, t_max=10)
        self.goals = [
            GoalCircle((2, 0.0), 0.2),
        ]
        self.set_signs(
            [
                Sign(x=0.7, y=0.1, name="sign1"),
                Sign(x=1.5, y=-0.1, name="sign1"),
                Sign(x=2.3, y=0, name="sign1"),
            ]
        )

    def command_func(self):
        """回答例１：回転と直進を交互にして認識対象に近づく"""
        while True:
            pos = search()
            if pos is None:
                move(v=0)
            else:
                if pos.theta > 5:
                    rotate(w=45)
                elif pos.theta < -5:
                    rotate(w=-45)
                else:
                    move(v=1)


class Mission3(MissionBase):
    def __init__(self):
        super().__init__(type_b_circuit, t_max=80)
        self.goals = [
            GoalCircle((2, 0.0), 0.2, should_stop=False),
            GoalCircle((4.3, 1.0), 0.2, should_stop=False),
            GoalCircle((3.2, 2.2), 0.2, should_stop=False),
            GoalCircle((1.7, 1.0), 0.2, should_stop=False),
            GoalCircle((0.0, 0.8), 0.2, should_stop=False),
            GoalCircle((0.0, 0.0), 0.2),
        ]
        self.set_signs(
            [
                Sign(x=1.1, y=-0.1, name="sign2"),
                Sign(x=2.2, y=-0.1, name="sign2"),
                Sign(x=3.3, y=-0.1, name="sign2"),
                Sign(x=4.1, y=0.3, name="sign2"),
                Sign(x=4.5, y=1.3, name="sign2"),
                Sign(x=3.9, y=2.3, name="sign2"),
                Sign(x=2.8, y=2.3, name="sign2"),
                Sign(x=3.1, y=1.3, name="sign3"),
                Sign(x=2.2, y=1.0, name="sign2"),
                Sign(x=1.2, y=0.9, name="sign2"),
                Sign(x=0.2, y=0.8, name="sign2"),
                Sign(x=-0.6, y=0.4, name="sign2"),
                Sign(x=0.2, y=-0.2, name="sign1"),
            ]
        )

    def command_func(self):
        """回答例１：
        回転と直進を使用して標識に近づく
        - sign2を最後に見た後に標識を見失ったら反時計回りに回転する
        - sign3を最後に見た後に標識を見失ったら時計回りに回転する
        """
        last_name = None
        while True:
            pos = search()
            if pos is None:
                if last_name == "sign2":
                    rotate(w=45)
                elif last_name == "sign3":
                    rotate(w=-45)
                else:
                    move(v=0)
            else:
                last_name = pos.name
                if pos.theta > 5:
                    rotate(w=45)
                elif pos.theta < -5:
                    rotate(w=-45)
                else:
                    move(v=1)


mission0 = Mission0()
mission2 = Mission2()
mission3 = Mission3()
mission = mission3

sim.set_mission(mission)
sim.run()

drawer = VeihicleDrawer(sim)

drawer.show()

# %%
