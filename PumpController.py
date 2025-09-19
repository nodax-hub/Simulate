from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class BoundaryAction(StrEnum):
    CLAMP = "clamp"  # прижимать к границе (мин/макс)
    ZERO = "zero"  # установить 0 л/с (экв. "выключить")
    ERROR = "error"  # выброосить исключение


@dataclass(frozen=True)
class PumpConstraints:
    q_min: float  # л/с
    q_max: float  # л/с
    tank_volume: float  # л
    low_mode: BoundaryAction = BoundaryAction.CLAMP
    high_mode: BoundaryAction = BoundaryAction.CLAMP


@dataclass
class PumpPlan:
    """
    План работы насоса во времени.
    t: список времени [с]
    q: соответствующие подачи насоса [л/с] после применения граничной логики
    empty_events: список моментов времени опустошения бака (с кумулят. литрами)
    """
    t: list[float]
    q: list[float]
    empty_events: list[tuple[float, float]]  # (t_empty, V_cumulative)


class PumpError(Exception):
    """Базовое исключение для ошибок насоса."""


class FlowBelowMinimumError(PumpError):
    """Требуемый расход меньше допустимого минимума."""

    def __init__(self, value: float, q_min: float):
        super().__init__(f"Запрошенный расход {value:.6f} < минимального {q_min:.6f}")
        self.value = value
        self.q_min = q_min


class FlowAboveMaximumError(PumpError):
    """Требуемый расход больше допустимого максимума."""

    def __init__(self, value: float, q_max: float):
        super().__init__(f"Запрошенный расход {value:.6f} > максимального {q_max:.6f}")
        self.value = value
        self.q_max = q_max


class PumpController:
    def __init__(self, constraints: PumpConstraints):
        self.c = constraints

    def _apply_low(self, x: float) -> float:
        m = self.c.low_mode
        if m is BoundaryAction.CLAMP:
            return max(x, self.c.q_min)
        if m is BoundaryAction.ZERO:
            return 0.0
        if m is BoundaryAction.ERROR:
            raise FlowBelowMinimumError(x, self.c.q_min)
        return x

    def _apply_high(self, x: float) -> float:
        m = self.c.high_mode
        if m is BoundaryAction.CLAMP:
            return min(x, self.c.q_max)
        if m is BoundaryAction.ZERO:
            return 0.0
        if m is BoundaryAction.ERROR:
            raise FlowAboveMaximumError(x, self.c.q_max)
        return x

    def compute_flow_series(self, t, v_motion, length: float, volume_total: float, eps=1e-9) -> PumpPlan:
        assert len(t) == len(v_motion)

        if length <= 0 or volume_total <= 0:
            return PumpPlan(t=t, q=[0.0] * len(t), empty_events=[])
        """
        плотность = v_total / length = л / м
        
        q(t) = плотность * speed_motion(t) = л/м * м/с = [л/с]
        """
        q_req = [volume_total * max(vm, 0.0) / max(length, eps) for vm in v_motion]

        q = []
        for x in q_req:
            if x <= eps:
                q.append(0.0)
                continue

            if x < self.c.q_min:
                q.append(self._apply_low(x))

            elif x > self.c.q_max:
                q.append(self._apply_high(x))

            else:
                q.append(x)

        empty_events = []
        V_cum = 0.0
        V_capacity = self.c.tank_volume

        for i in range(1, len(t)):
            dt = max(t[i] - t[i - 1], 0.0)
            V_cum_next = V_cum + q[i - 1] * dt

            if V_capacity > 0 and int(V_cum / V_capacity) < int(V_cum_next / V_capacity):
                accumulate_v = int(V_cum_next / V_capacity) * V_capacity
                t_empty = t[i - 1] + ((accumulate_v - V_cum) / max(q[i - 1], eps))
                empty_events.append((t_empty, accumulate_v))

            V_cum = V_cum_next

        return PumpPlan(t=t, q=q, empty_events=empty_events)

    @staticmethod
    def instantaneous_densities(v_motion: np.ndarray, v_pump: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """
        Возвращает:
          lambda_lperm (л/м)
        Примечание: когда v_motion ~ 0, считаем плотность 0 при условии v_pump тоже 0.
        Иначе — помечаем NaN, чтобы увидеть проблемные места (насос не выключен при остановке).
        """
        v_motion = np.asarray(v_motion, dtype=float)
        v_pump = np.asarray(v_pump, dtype=float)
        assert v_motion.shape == v_pump.shape

        lambda_lperm = np.empty_like(v_pump)
        stopped = np.abs(v_motion) < eps

        # где движемся:
        moving = ~stopped
        lambda_lperm[moving] = v_pump[moving] / v_motion[moving]  # л/с  /  м/с = л/м

        # где стоим: если насос выключен — плотность 0, иначе NaN как индикатор ошибки
        lambda_lperm[stopped] = np.where(np.abs(v_pump[stopped]) < eps, 0.0, np.nan)

        return lambda_lperm

    @staticmethod
    def total_dispensed(plan: PumpPlan) -> float:
        V = 0.0
        for i in range(1, len(plan.t)):
            dt = max(plan.t[i] - plan.t[i - 1], 0.0)
            V += plan.q[i - 1] * dt
        return V
