"""
Рассмотрим возможные варианты:

1. Запрещается переливать (допускается недолив):
    - Насосы будут отключаться при необходимости работать на меньшей скорости чем они могут.

2. Запрет на недолив (допускается перелив):
    - Насосы будут переливать (работать на своем минимуме, даже когда требуется меньшая скорость)
    - Максимальная скорость дрона будет снижена

"""
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np

from SpeedPredictor import MotionConstraints, PureLateralLimitTurnPolicy, SpeedPredictor
from PumpController import BoundaryAction, PumpConstraints, PumpController
from utils import point_on_path
from parse_log import Plan
from plots import plot_density_profile


class PumpingPolicy(StrEnum):
    NoOverflowPolicy = "NoOverflowPolicy"  # запрет на перелив
    NoUnderfillPolicy = "NoUnderfillPolicy"  # запрет на недолив


class PumpFacade:

    def __init__(self,
                 plan: Plan,
                 motion_constraints: MotionConstraints,
                 pump_constraints: PumpConstraints,
                 norma: float,
                 pumping_policy: PumpingPolicy,
                 ):

        self.plan = plan
        self.pumping_policy = pumping_policy
        self.norma = norma
        self.pump_constraints = pump_constraints
        self.motion_constraints = motion_constraints

        self.predictor = SpeedPredictor(self.motion_constraints, PureLateralLimitTurnPolicy())

        self._calculate_excepted_and_fact_volume()

        print(f"{self.volume_total=}, {self.total_dispensed_by_pump_plan=}")

        """        
        1. Запрещается переливать (допускается недолив):
            - Насосы будут отключаться в случаях когда требуется работать на меньшей скорости чем они могут.

        2. Запрет на недолив (допускается перелив):
            - Насосы будут переливать (работать на своем минимуме, даже когда требуется меньшая скорость)
            - Максимальная скорость дрона будет снижена
        """

        # если первый случай
        if self.pumping_policy is PumpingPolicy.NoOverflowPolicy:

            # Если переливаем
            if self.total_dispensed_by_pump_plan > self.volume_total:
                print("Мы перелили")

        # если второй случай
        if self.pumping_policy is PumpingPolicy.NoUnderfillPolicy:

            # Если мы недоливаем
            if self.total_dispensed_by_pump_plan < self.volume_total:
                print("Попробуйте уменьшить максимальную скорость дрона")

    def _calculate_excepted_and_fact_volume(self, dt=0.1):

        self.profile = self.predictor.build_profile(self.plan.waypoints_xy)

        self.volume_total = self.plan.calculate_total_volume(self.norma)

        self.t_list, self.s_list, self.speed_list = self.profile.simulate_time_param(dt=dt)
        self.points_list = [point_on_path(self.plan.waypoints_xy, s) for s in self.s_list]



        self.pump_controller = PumpController(self.pump_constraints)

        self.pump_plan = self.pump_controller.compute_flow_series(
            t=self.t_list,
            v_motion=self.speed_list,
            length=self.profile.total_distance,
            volume_total=self.volume_total,
        )

        # мгновенная плотность внесения л/м
        self.instant_introduction_density = PumpController.instantaneous_densities(np.array(self.speed_list), np.array(self.pump_plan.q))

        self.density = self.volume_total / self.profile.total_distance # ожидаемая средняя плотность внесения [л/м]
        self.total_dispensed_by_pump_plan = self.pump_controller.total_dispensed(self.pump_plan)

        self.diff_density = self.instant_introduction_density - self.density

    def plot(self):
        plot_density_profile(self.points_list,
                             self.pump_plan.q,
                             self.speed_list,
                             self.density)

    @classmethod
    def from_simple_params(cls,
                           plan: Plan,
                           norma: float,
                           max_drone_speed: float,
                           pump_min_speed: float,
                           pump_max_speed: float,
                           tank_volume: float,
                           pumping_policy: PumpingPolicy,
                           ) -> Self:

        motion_constraints = MotionConstraints(
            v_max=max_drone_speed,
            a_max=1.0,
            d_max=1.0,
            yaw_rate=90.0,
            turn_radius=0,
            a_lat_max=2.0,
            angle_eps_deg=10.0,
            start_speed=0.0,
            end_speed=0.0
        )

        pump_low_mode = BoundaryAction.ZERO if pumping_policy is PumpingPolicy.NoOverflowPolicy else BoundaryAction.CLAMP

        pump_constraints = PumpConstraints(
            q_min=pump_min_speed / 60,  # л/с
            q_max=pump_max_speed / 60,  # л/с
            tank_volume=tank_volume,  # л
            low_mode=pump_low_mode,
            high_mode=BoundaryAction.CLAMP,
        )

        return cls(motion_constraints=motion_constraints,
                   pump_constraints=pump_constraints,
                   plan=plan,
                   norma=norma,
                   pumping_policy=pumping_policy
                   )

class AutoCompensation:
    def compensate(self):
        pass

def main():
    req_norma = 5
    drone_v_max = 10.0

    pump_min_speed = 1  # л/мин
    pump_max_speed = 13  # л/мин

    tank_volume = 40  # л

    log_file_path = Path(
        r"C:\Users\nodax\Downloads\Telegram Desktop\spraying_20250908_080420_485_24a89b48_9ecb_4956_bdb3_4d01f9de0f4a.json")

    plan = Plan.get_plan_from_log_file(log_file_path)

    pf = PumpFacade.from_simple_params(plan=plan,
                                       norma=req_norma,
                                       max_drone_speed=drone_v_max,
                                       pump_min_speed=pump_min_speed,
                                       pump_max_speed=pump_max_speed,
                                       tank_volume=tank_volume,
                                       pumping_policy=PumpingPolicy.NoOverflowPolicy)

    pf.plot()


if __name__ == '__main__':
    main()
