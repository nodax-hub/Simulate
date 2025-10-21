
from typing import Callable

from use_cases import PumpFacade


class EvalCache:
    """Обёртка над func с кэшированием и счётчиком вызовов."""

    def __init__(self, func: Callable[[float], float]):
        self.func = func
        self.cache = {}
        self.n_calls = 0

    def __call__(self, x: float) -> float:
        self.n_calls += 1

        if x <= 0:
            raise ValueError("x должен быть > 0")

        key = float(x)
        if key in self.cache:
            return self.cache[key]

        y = self.func(x)
        self.cache[key] = y

        return y


@EvalCache
def fabs(x):
    pf = PumpFacade.from_simple_params(plan=plan,
                                       norma=x,
                                       max_drone_speed=2,
                                       pump_min_speed=pump_min_speed,
                                       pump_max_speed=pump_max_speed,
                                       tank_volume=tank_volume,
                                       pumping_policy=PumpingPolicy.NoUnderfillPolicy)
    return float(abs(pf.total_dispensed_by_pump_plan - pf.volume_total))