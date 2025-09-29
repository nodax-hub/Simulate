from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np

from dto import SimulationResult


class ErrorStrategy(ABC):
    @abstractmethod
    def __call__(self, res: SimulationResult) -> float:
        ...


class VolumeAbsError(ErrorStrategy):
    """
    Абсолютная ошибка по общему объёму (л).
    optionally normalize=True => делим на volume_total (безразмерная доля).
    """
    def __init__(self, normalize: bool = False) -> None:
        self.normalize = normalize

    def __call__(self, res: SimulationResult) -> float:
        err = abs(res.total_dispensed - res.volume_total)
        if self.normalize and res.volume_total > 0:
            return err / res.volume_total
        return err


class DensityMSE(ErrorStrategy):
    """
    MSE отклонения мгновенной плотности от целевой, считается только на движении.
    normalize=True => делим на (target_density^2) если target_density>0, иначе без нормализации.
    """
    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def __call__(self, res: SimulationResult) -> float:
        v = np.asarray(res.speed_list, float)
        d = res.instant_density
        target = res.target_density

        moving = np.abs(v) > 1e-9
        if not np.any(moving):
            return float('inf')

        mse = float(((d[moving] - target) ** 2).mean())

        if self.normalize and target > 0:
            return mse / (target ** 2)
        return mse


class StopOverflowRatio(ErrorStrategy):
    """
    Относительная доля перелива на стоянках: (объём/volume_total).
    """
    def __call__(self, res: SimulationResult) -> float:
        v = np.asarray(res.speed_list, float)
        q = np.asarray(res.q_list, float)
        stopped = np.abs(v) <= 1e-9

        overflow_volume = float(np.sum(q[stopped]) * res.dt)  # л
        if res.volume_total > 0:
            return overflow_volume / res.volume_total
        return float('inf')


class CompositeError(ErrorStrategy):
    """
    Линейная комбинация стратегий с весами.
    Пример: CompositeError([(VolumeAbsError(True), 0.7), (DensityMSE(True), 0.3)])
    """
    def __init__(self, items: Iterable[tuple[ErrorStrategy, float]]) -> None:
        self.items = list(items)

    def __call__(self, res: SimulationResult) -> float:
        total = 0.0
        for strat, w in self.items:
            total += w * float(strat(res))
        return total
