from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------- Абстракции ----------

class ErrorFunction(ABC):
    """Интерфейс детерминированной функции ошибки e(x), x>0."""

    @abstractmethod
    def evaluate(self, x: float) -> float:
        pass


class DomainTransform(ABC):
    """Преобразование области: x <-> u (например, u = log x)."""

    @abstractmethod
    def to_u(self, x: float) -> float: ...

    @abstractmethod
    def to_x(self, u: float) -> float: ...


@dataclass(frozen=True)
class IdentityTransform(DomainTransform):
    def to_u(self, x: float) -> float: return x

    def to_x(self, u: float) -> float: return u


@dataclass(frozen=True)
class LogTransform(DomainTransform):
    """u = ln(x), x = exp(u). Подходит при x>0 и широком диапазоне."""

    def to_u(self, x: float) -> float:
        if x <= 0:
            raise ValueError("LogTransform: x must be > 0")
        return math.log(x)

    def to_x(self, u: float) -> float: return math.exp(u)


class Bracketer(ABC):
    """Стратегия захвата минимума тройкой (uL<uM<uR, f(uM) < f(uL), f(uM) < f(uR))."""

    @abstractmethod
    def bracket(self, f: Callable[[float], float], u0: float) -> Tuple[float, float, float]:
        pass


class UnimodalMinimizer(ABC):
    """Стратегия поиска минимума на [uL, uR] без производных."""

    @abstractmethod
    def minimize(self, f: Callable[[float], float], uL: float, uR: float,
                 tol_u: float, tol_f: float) -> Tuple[float, float]:
        """Возврат: (u*, f(u*))"""
        pass


class BoundaryFinder(ABC):
    """Стратегия нахождения границ по порогу e*+Δ, слева и справа от u*."""

    @abstractmethod
    def find_left(self, f: Callable[[float], float], u_star: float, f_star: float,
                  threshold: float, tol_u: float) -> float: ...

    @abstractmethod
    def find_right(self, f: Callable[[float], float], u_star: float, f_star: float,
                   threshold: float, tol_u: float) -> float: ...


# ---------- Реализации ----------

@dataclass
class GeometricBracketer(Bracketer):
    """Симметричное геом. расширение до выполнения условия бранкета."""
    initial_span: float = 1.0  # стартовая ширина по u
    expand_factor: float = 2.0  # во сколько раз расширяем каждый шаг
    max_expansions: int = 60

    def bracket(self, f: Callable[[float], float], u0: float) -> Tuple[float, float, float]:
        span = self.initial_span
        uL, uM, uR = u0 - span, u0, u0 + span
        fL, fM, fR = f(uL), f(uM), f(uR)

        # Если центр уже меньше краёв — ок. Иначе расширяем симметрично.
        expansions = 0
        while not (fM < fL and fM < fR):
            span *= self.expand_factor
            uL, uR = u0 - span, u0 + span
            fL, fR = f(uL), f(uR)
            expansions += 1
            if expansions > self.max_expansions:
                raise RuntimeError("Bracketing failed: too many expansions")
        return uL, uM, uR


@dataclass
class BrentMinimizer(UnimodalMinimizer):
    """Метод Брента (комбинация параболич. интерполяции и золотого сечения)."""
    max_iters: int = 100
    golden_ratio: float = (3 - math.sqrt(5)) / 2  # 1-1/phi ~ 0.381966...

    def minimize(self, f: Callable[[float], float], uL: float, uR: float,
                 tol_u: float, tol_f: float) -> Tuple[float, float]:
        # Инициализация по Бренту
        a, b = (uL, uR) if uL < uR else (uR, uL)
        x = w = v = a + self.golden_ratio * (b - a)
        fx = fw = fv = f(x)
        d = e = 0.0

        for _ in range(self.max_iters):
            m = 0.5 * (a + b)
            tol1 = math.sqrt(1e-16) * abs(x) + tol_u  # машинная точность + tol_u
            tol2 = 2 * tol1
            # Критерий остановки по u и по f
            if abs(x - m) <= tol2 - 0.5 * (b - a):
                return x, fx

            p = q = r = 0.0
            if abs(e) > tol1:
                # Параболич. интерполяция по трём точкам (x, w, v)
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0:
                    p = -p
                q = abs(q)
                # Проверка на допустимость шага
                if (abs(p) < abs(0.5 * q * e)) and (p > q * (a - x)) and (p < q * (b - x)):
                    d = p / q
                    u = x + d
                    # u не слишком близко к границам
                    if (u - a) < tol2 or (b - u) < tol2:
                        d = -tol1 if x < m else tol1
                else:
                    # Золотое сечение
                    e = (a - x) if x >= m else (b - x)
                    d = self.golden_ratio * e
            else:
                # Золотое сечение
                e = (a - x) if x >= m else (b - x)
                d = self.golden_ratio * e

            u = x + d if abs(d) >= tol1 else x + (tol1 if d > 0 else -tol1)
            fu = f(u)

            # Обновления по Бренту
            if fu <= fx or abs(fu - fx) <= tol_f:
                if u < x:
                    b = x
                else:
                    a = x
                v, fv = w, fw
                w, fw = x, fx
                x, fx = u, fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if (fu <= fw) or (w == x):
                    v, fv = w, fw
                    w, fw = u, fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v, fv = u, fu

        return x, fx  # по достижении max_iters возвращаем лучшее найденное


@dataclass
class ThresholdBoundary(BoundaryFinder):
    """Экспоненциальный поиск + бисекция до e(u) > threshold."""
    growth: float = 2.0
    init_step: float = 1.0
    max_expansions: int = 60
    bisection_iters: int = 80

    def _bisect(self, f: Callable[[float], float], ua: float, ub: float,
                threshold: float, tol_u: float, side: int) -> float:
        # side = -1 (левая), +1 (правая). Монотонность достаточна локально от u*.
        fa = f(ua) - threshold
        fb = f(ub) - threshold
        if not (fa * fb <= 0):
            raise RuntimeError("Bisection: threshold not bracketed")
        a, b = ua, ub
        for _ in range(self.bisection_iters):
            m = 0.5 * (a + b)
            fm = f(m) - threshold
            if fm == 0 or abs(b - a) <= tol_u:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    def find_left(self, f: Callable[[float], float], u_star: float, f_star: float,
                  threshold: float, tol_u: float) -> float:
        step = self.init_step
        uprev, fprev = u_star, f_star
        for _ in range(self.max_expansions):
            u = u_star - step
            fu = f(u)
            if fu > threshold:
                # Бранкет: [u, uprev] — на uprev ещё внутри порога
                return self._bisect(f, u, uprev, threshold, tol_u, side=-1)
            uprev, fprev = u, fu
            step *= self.growth
        raise RuntimeError("Left boundary not found within expansions")

    def find_right(self, f: Callable[[float], float], u_star: float, f_star: float,
                   threshold: float, tol_u: float) -> float:
        step = self.init_step
        uprev, fprev = u_star, f_star
        for _ in range(self.max_expansions):
            u = u_star + step
            fu = f(u)
            if fu > threshold:
                # Бранкет: [uprev, u]
                return self._bisect(f, uprev, u, threshold, tol_u, side=+1)
            uprev, fprev = u, fu
            step *= self.growth
        raise RuntimeError("Right boundary not found within expansions")


# ---------- Фасад ----------

@dataclass
class FlatMinimumIntervalFinder:
    """Фасад: минимум и интервал «плоского дна» по порогу e*+Δ."""
    error_fn: ErrorFunction
    transform: DomainTransform
    bracketer: Bracketer
    minimizer: UnimodalMinimizer
    boundaries: BoundaryFinder

    def find(self,
             x0: float,
             rel_threshold: Optional[float] = 0.02,
             abs_threshold: Optional[float] = None,
             tol_x: float = 1e-6,
             tol_f: float = 0.0
             ) -> Tuple[float, float, float, float]:
        """
        Возвращает: (a, x_star, b, e_star)
          a,b — границы по порогу e* + Δ (Δ по abs_threshold или rel_threshold*e*).
        """
        if x0 <= 0:
            raise ValueError("x0 must be > 0")

        # Обёртка f(u) = e(x(u))
        def f(u: float) -> float:
            x = self.transform.to_x(u)
            return self.error_fn.evaluate(x)

        u0 = self.transform.to_u(x0)
        uL, uM, uR = self.bracketer.bracket(f, u0)
        u_star, e_star = self.minimizer.minimize(f, uL, uR, tol_u=self._tol_u(tol_x), tol_f=tol_f)
        delta = self._threshold(e_star, rel_threshold, abs_threshold)

        u_left = self.boundaries.find_left(f, u_star, e_star, e_star + delta, tol_u=self._tol_u(tol_x))
        u_right = self.boundaries.find_right(f, u_star, e_star, e_star + delta, tol_u=self._tol_u(tol_x))

        a = self.transform.to_x(u_left)
        x_star = self.transform.to_x(u_star)
        b = self.transform.to_x(u_right)
        return a, x_star, b, e_star

    def _threshold(self, e_star: float, rel: Optional[float], abs_: Optional[float]) -> float:
        if abs_ is not None:
            return abs_
        if rel is not None:
            return rel * e_star
        raise ValueError("Either rel_threshold or abs_threshold must be provided")

    def _tol_u(self, tol_x: float) -> float:
        """Перевод точности по x в точность по u с учётом трансформации."""
        # Для лог-трансформации du ~ dx/x. Здесь используем безопасный малый tol_u.
        # Пользователь может явно задать tol_x поменьше/побольше.
        return tol_x if isinstance(self.transform, IdentityTransform) else max(1e-12, tol_x)


# ---------- Пример использования ----------

class DemoParabola(ErrorFunction):
    """
    Пример e(x) с единичным минимумом при x*=3:
    e(x) = (log(x) - log(3))^2 + 1
    Удобен, чтобы увидеть преимущество лог-шкалы.
    """

    def evaluate(self, x: float) -> float:
        return (math.log(x) - math.log(3.0)) ** 2 + 1.0


def build_default_finder(error_fn: ErrorFunction, log_scale: bool = True) -> FlatMinimumIntervalFinder:
    return FlatMinimumIntervalFinder(
        error_fn=error_fn,  # <-- внедрите свою реализацию ErrorFunction
        transform=LogTransform() if log_scale else IdentityTransform(),
        bracketer=GeometricBracketer(initial_span=0.5, expand_factor=2.0, max_expansions=60),
        minimizer=BrentMinimizer(max_iters=200),
        boundaries=ThresholdBoundary(growth=2.0, init_step=0.25, max_expansions=60, bisection_iters=80),
    )


if __name__ == "__main__":
    error_fn = DemoParabola()
    x = np.linspace(1, 5, 100)
    y = [error_fn.evaluate(v) for v in x]
    plt.plot(x, y)
    plt.show()

    finder = build_default_finder(error_fn, log_scale=True)
    a, x_star, b, e_star = finder.find(
        x0=1,  # старт >0
        rel_threshold=0.02,  # Δ = 2% от e*
        abs_threshold=None,  # или задайте абсолютный допуск Δ
        tol_x=1e-8,  # точность по x
        tol_f=0.0  # допуск по значению (можно оставить 0)
    )
    print(f"a={a:.6f}, x*={x_star:.6f}, b={b:.6f}, e*={e_star:.9f}")
