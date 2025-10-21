from functools import cache

from dto import SimulationResult
from parse_log import Plan
from use_cases import PumpFacade, PumpingPolicy


@cache
def calc_simulation_result(plan: Plan,
                           norma: float,
                           max_drone_speed: float,
                           pump_min_speed: float,
                           pump_max_speed: float,
                           tank_volume: float,
                           pumping_policy: PumpingPolicy) -> SimulationResult:
    pf = PumpFacade.from_simple_params(
        plan=plan,
        norma=norma,
        max_drone_speed=max_drone_speed,
        pump_min_speed=pump_min_speed,
        pump_max_speed=pump_max_speed,
        tank_volume=tank_volume,
        pumping_policy=pumping_policy,
    )
    return pf.build_result()
