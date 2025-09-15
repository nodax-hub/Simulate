from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Mapping, Optional, Callable


# ---------- Базовые протоколы ----------

class Loss(Protocol):
    """Функция потерь: принимает ошибку e = value - target, возвращает стоимость."""

    def cost(self, error: float) -> float: ...


class Constraint(Protocol):
    """Ограничение на знак ошибки (или его отсутствие)."""

    def feasible(self, target: float, value: float) -> bool: ...

    def clip(self, target: float, value: float) -> float: ...


# ---------- Реализации потерь (L1) ----------

@dataclass(frozen=True)
class AsymmetricL1(Loss):
    """Линейный штраф |e| с разными весами.
    w_over  — цена перелива  (e > 0)
    w_under — цена недолива  (e < 0)
    """
    w_over: float = 1.0
    w_under: float = 1.0

    def __post_init__(self):
        if self.w_over <= 0 or self.w_under <= 0:
            raise ValueError("w_over и w_under должны быть > 0")

    def cost(self, error: float) -> float:
        return self.w_over * error if error > 0 else self.w_under * (-error)


@dataclass(frozen=True)
class PinballLoss(Loss):
    """Квантильная (pinball) потеря: tau ∈ (0,1).
    Чем больше tau, тем дороже недолив (e<0)."""
    tau: float

    def __post_init__(self):
        if not (0.0 < self.tau < 1.0):
            raise ValueError("tau должен быть в (0,1)")

    def cost(self, error: float) -> float:
        return (1 - self.tau) * error if error >= 0 else self.tau * (-error)


# ---------- Ограничения ----------

class NoConstraint(Constraint):
    """Без ограничений."""

    def feasible(self, target: float, value: float) -> bool:  # noqa: ARG002
        return True

    def clip(self, target: float, value: float) -> float:
        return value


class NoOverfill(Constraint):
    """Запретить перелив: value <= target."""

    def feasible(self, target: float, value: float) -> bool:
        return value <= target

    def clip(self, target: float, value: float) -> float:
        return min(value, target)


class NoUnderfill(Constraint):
    """Запретить недолив: value >= target."""

    def feasible(self, target: float, value: float) -> bool:
        return value >= target

    def clip(self, target: float, value: float) -> float:
        return max(value, target)


@dataclass(frozen=True)
class ToleranceConstraint(Constraint):
    """Допуск по абсолютным отклонениям"""
    over_tol: float = math.inf
    under_tol: float = math.inf

    def __post_init__(self):
        if self.over_tol < 0 or self.under_tol < 0:
            raise ValueError("over_tol и under_tol должны быть >= 0")

    def feasible(self, target: float, value: float) -> bool:
        return (target - self.under_tol) <= value <= (target + self.over_tol)

    def clip(self, target: float, value: float) -> float:
        lo = target - self.under_tol
        hi = target + self.over_tol
        return min(max(value, lo), hi)


# ---------- Политика (композиция потерь и ограничений) ----------

class ViolationMode(StrEnum):
    IGNORE = "ignore"  # игнорируем оставляем всё как есть
    ERROR = "error"  # кидаем исключение при нарушении
    REJECT = "reject"  # исключаем этот случай из рассмотрения
    CLIP = "clip"  # проецируем к границе


@dataclass
class DecisionPolicy:
    loss: Loss
    constraint: Constraint = NoConstraint()
    violation: ViolationMode = ViolationMode.IGNORE

    # --- Внутренние исключения ---
    class PolicyError(Exception):
        """Базовое исключение для политик."""

    class ConstraintViolation(PolicyError):
        """Нарушение ограничения (violation=ERROR)."""

    class NoFeasibleActions(PolicyError):
        """Нет допустимых действий (все отфильтрованы)."""

    class UnknownViolationMode(PolicyError):
        """Неизвестный режим обработки нарушения."""

    # --- Методы ---
    def _handle_violation(
            self,
            target: float,
            value: float,
            on_valid: Callable[[float], float]
    ) -> Optional[float]:
        """
        Универсальная обработка значений с учётом ограничений.
        on_valid — функция, применяемая к корректному значению.
        """
        if self.constraint.feasible(target, value):
            return on_valid(value)

        match self.violation:
            case ViolationMode.IGNORE:
                return on_valid(value)
            case ViolationMode.CLIP:
                clipped = self.constraint.clip(target, value)
                return on_valid(clipped)
            case ViolationMode.REJECT:
                return None
            case ViolationMode.ERROR:
                raise self.ConstraintViolation("Нарушено ограничение политики")
            case _:
                raise self.UnknownViolationMode(f"Неизвестный режим нарушения: {self.violation}")

    def cost(self, target: float, value: float) -> Optional[float]:
        return self._handle_violation(target, value, lambda v: self.loss.cost(v - target))

    def apply_setpoint(self, target: float, value: float) -> Optional[float]:
        return self._handle_violation(target, value, lambda v: v)

    def choose_best(self, target: float, actions: Mapping[str, float]) -> set[str]:
        costs_by_names = {name: self.cost(target, value) for name, value in actions.items()}
        minimum_cost_value = min(costs_by_names.values())
        if math.isinf(minimum_cost_value):
            raise self.NoFeasibleActions("Нет допустимых действий под текущую политику")
        return {k for k, v in costs_by_names.items() if v == minimum_cost_value}


# ---------- Фабрики политик ----------
class BanMode(StrEnum):
    OVER = "over"  # Запрет перелива
    UNDER = "under"  # Запрет недолива


class PolicyFactory:
    @staticmethod
    def symmetric() -> DecisionPolicy:
        """Равнозначность перелива/недолива."""
        return PolicyFactory.ratio(w_over=1, w_under=1)

    @staticmethod
    def ratio(w_over, w_under, violation: ViolationMode = ViolationMode.IGNORE) -> DecisionPolicy:
        """Соотношение важности (недолив / перелив).
        under_over=5.0 => недолив в 5 раз дороже перелива."""
        loss = AsymmetricL1(w_over=w_over, w_under=w_under)
        return DecisionPolicy(loss=loss, violation=violation)

    @staticmethod
    def quantile_tau(tau: float, violation: ViolationMode = ViolationMode.IGNORE) -> DecisionPolicy:
        """Эквивалент через τ-квантильную потерю."""
        loss = PinballLoss(tau=tau)
        return DecisionPolicy(loss=loss, violation=violation)

    @staticmethod
    def with_tolerance(
            w_over: float = 1.0,
            w_under: float = 1.0,
            over_tol: float = math.inf,
            under_tol: float = math.inf,
            violation: ViolationMode = ViolationMode.IGNORE,
    ) -> DecisionPolicy:
        """Политика с допуском на отклонения.
        over_tol, under_tol – максимально допустимые ошибки по модулю.
        violation – что делать при нарушении: CLIP, ERROR, REJECT, IGNORE.
        """
        loss = AsymmetricL1(w_over=w_over, w_under=w_under)
        constraint = ToleranceConstraint(over_tol=over_tol, under_tol=under_tol)
        return DecisionPolicy(loss=loss, constraint=constraint, violation=violation)


def main():
    target = 100.0
    values = {"A": 99.0,
              # "B": 100.0,
              "C": 101.0}
    # 1) Равнозначность (симметричный штраф), без запретов
    p1 = PolicyFactory.symmetric()
    print(1, p1.choose_best(target, values))  # ('B', 0.0)
    # 2) Недолив в 5 раз дороже перелива (ratio)
    p2 = PolicyFactory.ratio(w_over=1.0, w_under=5.0)
    print(2, p2.choose_best(target, values))  # предпочтёт 'C' (перелив), т.к. недолив дорог
    # 3) Жёсткий запрет на перелив (дискретный выбор)
    p3 = PolicyFactory.ratio(w_over=1, w_under=5)
    print(3, p3.choose_best(target, values))  # из допустимых {'A','B'} выберет 'B'
    # 4) Жёсткий запрет на недолив (дискретный выбор)
    p4 = PolicyFactory.symmetric()
    print(4, p4.choose_best(target, values))  # из допустимых {'B','C'} выберет 'B'
    # 5) Непрерывный setpoint с клиппингом (строгий запрет на перелив)
    p5 = PolicyFactory.symmetric()
    print(p5.apply_setpoint(target, 101.7))  # 100.0 (спроецировано к границе)
    print(p5.apply_setpoint(target, 98.2))  # 98.2  (допустимо)

    # Допустим: перелив не более +2, недолив не более -1
    p_tol = PolicyFactory.with_tolerance(over_tol=.0,
                                         under_tol=1.0,
                                         violation=ViolationMode.IGNORE)

    print("Выбор лучшего:", p_tol.choose_best(target, values))
    print("Setpoint 103 →", p_tol.apply_setpoint(target, 103.0))  # спроецирует в 102
    print("Setpoint 97 →", p_tol.apply_setpoint(target, 97.0))  # допустимо
    print("Setpoint 96 →", p_tol.apply_setpoint(target, 96.0))  # спроецирует в 99
    print(p_tol.cost(target, 0))


if __name__ == '__main__':
    main()
