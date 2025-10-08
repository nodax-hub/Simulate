from __future__ import annotations

from functools import cache
from typing import Callable

from ErrorStrategy import ErrorStrategy
from IntervalFinder import BrentMin, RootBounds, ErrorIntervalFinder
from parse_log import Plan
from use_cases import PumpingPolicy, PumpFacade


def make_error_func_by_norma(plan: Plan,
                             max_drone_speed: float,
                             pump_min_speed: float,
                             pump_max_speed: float,
                             tank_volume: float,
                             pumping_policy: PumpingPolicy,
                             strategy: ErrorStrategy) -> Callable[[float], float]:
    """
    Возвращает f(norma)->error, обёрнутую кэшем.
    """

    @cache
    def _err(norma: float) -> float:
        pf = PumpFacade.from_simple_params(
            plan=plan,
            norma=norma,
            max_drone_speed=max_drone_speed,
            pump_min_speed=pump_min_speed,
            pump_max_speed=pump_max_speed,
            tank_volume=tank_volume,
            pumping_policy=pumping_policy,
        )
        res = pf.build_result()
        return float(strategy(res))

    return _err  # с проверкой x>0 как раньше


def make_error_func_by_max_drone_speed(plan: Plan,
                                       norma: float,
                                       pump_min_speed: float,
                                       pump_max_speed: float,
                                       tank_volume: float,
                                       pumping_policy: PumpingPolicy,
                                       strategy: ErrorStrategy) -> Callable[[float], float]:
    """
    Возвращает f(norma)->error, обёрнутую кэшем.
    """

    @cache
    def _err(max_drone_speed: float) -> float:
        pf = PumpFacade.from_simple_params(
            plan=plan,
            norma=norma,
            max_drone_speed=max_drone_speed,
            pump_min_speed=pump_min_speed,
            pump_max_speed=pump_max_speed,
            tank_volume=tank_volume,
            pumping_policy=pumping_policy,
        )
        res = pf.build_result()
        return float(strategy(res))

    return _err  # с проверкой x>0 как раньше


class ParamSearcher:
    """
    Обёртка над ErrorIntervalFinder, позволяющая подменять стратегию ошибки.
    """

    def __init__(self) -> None:
        self._finder = ErrorIntervalFinder(BrentMin(), RootBounds())  # как у тебя

    def find_interval(self,
                      error_func: Callable[[float], float],
                      start_x: float,
                      stop_x: float,
                      eps: float) -> tuple[tuple[float, float], float, int]:
        interval, optimum, calls = self._finder.find_interval(error_func, start_x, stop_x, eps=eps)
        # При желании можно добавить доступ к error_func.n_calls, если нужен внешний счётчик кэша
        return interval, optimum, calls
