import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional
from scipy.optimize import minimize_scalar, brentq


# ===== Обёртка для подсчёта вызовов =====
class CallCounter:
    def __init__(self, func: Callable[[float], float]):
        self.func = func
        self.calls = 0

    def __call__(self, x: float) -> float:
        self.calls += 1
        return self.func(x)

    def reset(self):
        self.calls = 0


# ===== Стратегии =====

class MinStrategy(ABC):
    @abstractmethod
    def find_min(self, func: Callable[[float], float], start: float, stop: float) -> Tuple[float, float]:
        pass


class GridSearchMin(MinStrategy):
    def __init__(self, n_points: int = 200, log_scale: bool = False):
        self.n_points = n_points
        self.log_scale = log_scale

    def find_min(self, func, start, stop):
        x_vals = (np.logspace(np.log10(start), np.log10(stop), self.n_points)
                  if self.log_scale else
                  np.linspace(start, stop, self.n_points))

        y_vals = np.array([func(x) for x in x_vals])
        i_min = np.argmin(y_vals)

        return x_vals[i_min], y_vals[i_min]


class BrentMin(MinStrategy):
    def find_min(self, func, start, stop):
        res = minimize_scalar(func, bounds=(start, stop), method="bounded")
        return res.x, res.fun


class BoundStrategy(ABC):
    @abstractmethod
    def find_bounds(self, func: Callable[[float], float],
                    x_min: float, y_min: float,
                    start: float, stop: float,
                    eps: float) -> Tuple[Optional[float], Optional[float]]:
        pass


class GridBounds(BoundStrategy):
    def __init__(self, n_points: int = 200, log_scale: bool = False):
        self.n_points = n_points
        self.log_scale = log_scale

    def find_bounds(self, func, x_min, y_min, start, stop, eps):
        x_vals = (np.logspace(np.log10(start), np.log10(stop), self.n_points)
                  if self.log_scale else
                  np.linspace(start, stop, self.n_points))

        y_vals = np.array([func(x) for x in x_vals])
        mask = y_vals <= y_min + eps

        if not np.any(mask):
            return None, None

        where = np.where(mask)[0]
        return x_vals[where[0]], x_vals[where[-1]]


class RootBounds(BoundStrategy):
    def find_bounds(self, func, x_min, y_min, start, stop, eps):
        target = y_min + eps
        def g(x): return func(x) - target

        left = right = None
        try:
            left = brentq(g, start, x_min)
        except ValueError:
            pass
        try:
            right = brentq(g, x_min, stop)
        except ValueError:
            pass

        return left, right


# ===== Контекст =====
class ErrorIntervalFinder:
    def __init__(self, min_strategy: MinStrategy, bound_strategy: BoundStrategy):
        self.min_strategy = min_strategy
        self.bound_strategy = bound_strategy

    def find_interval(self, func: Callable[[float], float],
                      start_x: float, stop_x: float,
                      delta: Optional[float] = None,
                      eps: Optional[float] = None):

        assert (delta is None) ^ (eps is None), "Задай либо delta, либо eps"

        func_counter = CallCounter(func)

        # минимум
        x_min, y_min = self.min_strategy.find_min(func_counter, start_x, stop_x)

        if eps is None:
            eps = y_min * delta

        # границы
        x_left, x_right = self.bound_strategy.find_bounds(
            func_counter, x_min, y_min, start_x, stop_x, eps
        )

        return (x_left, x_right), (x_min, y_min), func_counter.calls


def main():
    def f(x):
        return (x - 50) ** 2 + 10  # простая квадратичная ошибка

    start_x, stop_x = 1e-3, 300

    # Brent + RootBounds
    finder1 = ErrorIntervalFinder(BrentMin(), RootBounds())
    interval1, optimum1, calls1 = finder1.find_interval(f, start_x, stop_x, delta=0.05)

    # Grid + GridBounds
    finder2 = ErrorIntervalFinder(GridSearchMin(n_points=500), GridBounds(n_points=500))
    interval2, optimum2, calls2 = finder2.find_interval(f, start_x, stop_x, delta=0.05)

    print("Brent + RootBounds")
    print("  Интервал:", interval1, "Оптимум:", optimum1, "Вызовов:", calls1)

    print("Grid + GridBounds")
    print("  Интервал:", interval2, "Оптимум:", optimum2, "Вызовов:", calls2)



if __name__ == '__main__':
    main()