from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np


class Point(NamedTuple):
    x: float
    y: float


class BoundPoints(NamedTuple):
    start: Point
    end: Point


class GeoPoint(NamedTuple):
    lat: float
    lon: float
    alt: Optional[float] = None


@dataclass(frozen=True)
class SimulationResult:
    """
    Набор данных, необходимых для расчёта ошибочных метрик.
    """
    # агрегаты
    volume_total: float
    total_dispensed: float
    dt: float

    # траектория/профиль
    s_list: list[float]
    t_list: list[float]
    speed_list: list[float]  # v_motion

    # насос (расход по времени, л/с)
    q_list: list[float]

    # плотности
    target_density: float  # [л/м], средняя ожидаемая (volume_total / distance)
    instant_density: np.ndarray  # [л/м] по моментам (только на движении, остальное 0)
