import math
from _bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence, Literal

from dto import Point


def sqm_to_hectares(area_sqm: float) -> float:
    """
    Перевод площади из квадратных метров в гектары.
    :param area_sqm: площадь в м²
    :return: площадь в гектарах
    """
    return area_sqm / 10_000.0


def dist(a: Point, b: Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    return math.hypot(dx, dy)


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def polyline_lengths(pts: list[Point]) -> list[float]:
    """Список накопленных сумм длин вдоль ломаной, s[i] — расстояние от начала до pts[i]."""
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + dist(pts[i - 1], pts[i]))
    return s


def point_on_path(pts: list[Point], distance: float) -> Point:
    """
    Точка на ломаной на расстоянии `distance` от начала.
    Правила:
      - distance == 0 -> первая точка
      - 0 < distance < length -> линейная интерполяция на соответствующем сегменте
      - distance == length -> последняя точка
      - иначе -> ValueError
    Используются dist и polyline_lengths.
    """
    if not pts:
        raise ValueError("Список точек пуст")
    if distance < 0:
        raise ValueError("Дистанция не может быть отрицательной")

    s = polyline_lengths(pts)
    length = s[-1]

    if distance > length:
        raise ValueError(f"Дистанция {distance} больше длины траектории {length}")

    if distance == 0:
        return pts[0]

    if distance == length:
        return pts[-1]

    # Найти индекс правой границы отрезка: s[i-1] <= distance < s[i]
    i = bisect_right(s, distance)
    # Из-за нулевых сегментов s может содержать равные значения — пролистываем вперёд
    while i < len(s) and s[i] == s[i - 1]:
        i += 1
    if i >= len(pts):
        # теоретически недостижимо при проверках выше, но оставим страховку
        raise RuntimeError("Не найден валидный сегмент (все оставшиеся сегменты нулевой длины)")

    p1, p2 = pts[i - 1], pts[i]
    seg_len = s[i] - s[i - 1]  # > 0 после пролистывания
    t = (distance - s[i - 1]) / seg_len

    x_ = p1.x + t * (p2.x - p1.x)
    y_ = p1.y + t * (p2.y - p1.y)
    return Point(x_, y_)


def _is_strict_increasing(x: Sequence[float]) -> bool:
    return all(x[i] < x[i + 1] for i in range(len(x) - 1))


def _is_uniform(x: Sequence[float], rel_tol: float = 1e-9) -> bool:
    if len(x) < 3:
        return True
    h0 = x[1] - x[0]
    if h0 <= 0:
        return False
    for i in range(1, len(x) - 1):
        hi = x[i + 1] - x[i]
        if hi <= 0:
            return False
        # относительное отклонение шага
        if abs(hi - h0) > rel_tol * max(abs(h0), abs(hi), 1.0):
            return False
    return True


def integrate_samples(
        t: Sequence[float],
        y: Sequence[float],
        *,
        method: Literal["auto", "simpson", "trapezoid"] = "auto",  # "auto" | "simpson" | "trapezoid"
        uniform_tol: float = 1e-9  # допуск равномерности шага
) -> float:
    r"""
    Численный интеграл по точкам (t[i], y[i]).
    Предполагается, что t возрастают.

    method="auto": если шаг равномерен (с допуском) и число точек нечётное (число интервалов чётное) — Симпсон,
                   иначе — трапеция.
    method="simpson": требует равномерный шаг и нечётное число точек.
    method="trapezoid": работает на неравномерной сетке.

    Возвращает оценку интеграла \int_{t0}^{tN} y(t) dt.
    """
    n = len(t)
    if n != len(y):
        raise ValueError("t и y должны быть одинаковой длины")
    if n < 2:
        return 0.0
    if not _is_strict_increasing(t):
        raise ValueError("t должны строго возрастать")

    if method not in {"auto", "simpson", "trapezoid"}:
        raise ValueError("method должен быть 'auto', 'simpson' или 'trapezoid'")

    # выбор метода
    if method == "auto":
        uniform = _is_uniform(t, rel_tol=uniform_tol)
        if uniform and (n % 2 == 1):  # для Симпсона нужно нечётное число узлов
            method_use = "simpson"
        else:
            method_use = "trapezoid"
    else:
        method_use = method

    if method_use == "trapezoid":
        s = 0.0
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            s += 0.5 * (y[i] + y[i + 1]) * dt
        return s

    # Симпсон (равномерная сетка, n узлов, n-1 интервалов чётно)
    if not _is_uniform(t, rel_tol=uniform_tol):
        raise ValueError("Симпсон требует (почти) равномерный шаг")
    if n % 2 == 0:
        raise ValueError("Для Симпсона нужно нечётное число точек (чётное число интервалов)")

    h = (t[-1] - t[0]) / (n - 1)
    s_odd = sum(y[i] for i in range(1, n - 1, 2))
    s_even = sum(y[i] for i in range(2, n - 1, 2))
    return (h / 3.0) * (y[0] + y[-1] + 4.0 * s_odd + 2.0 * s_even)


def simulate_time_param(
        wp: list[Point],
        speed_list: list[float],
        *,
        min_speed: float = 1e-9
) -> list[float]:
    """
    Восстанавливает времена в узлах полилинии при линейной интерполяции скорости по дистанции.
    Возвращает t_list той же длины, что wp/speed_list; t_list[0] = 0.

    Математика (на отрезке s∈[s0,s1], v(s) линейна):
      v(s) = v0 + (v1 - v0) * (s - s0) / Δs
      Δt = ∫_{s0}^{s1} ds / v(s) =
           - при v0==v1:      Δs / v0
           - при v0!=v1:      (Δs / (v1 - v0)) * ln(v1 / v0)

    Параметры:
      - min_speed: нижняя граница для скоростей (защита от деления на 0 и ln(0)).

    Ограничения/проверки:
      - длина speed_list должна совпадать с wp;
      - дистанции по polyline_lengths(wp) неубывающие;
      - скорости строго > 0 (принудительно зажимаются не ниже min_speed).
    """
    if len(wp) == 0:
        return []

    if len(speed_list) != len(wp):
        raise ValueError("speed_list и wp должны быть одинаковой длины")

    # Кумулятивные расстояния по полилинии (предполагается, что функция дана)
    s_list = polyline_lengths(wp)  # длина = len(wp)
    if any(s_list[i + 1] < s_list[i] for i in range(len(s_list) - 1)):
        raise ValueError("polyline_lengths(wp) должна быть неубывающей")

    # Времена в узлах
    t_list: list[float] = [0.0]

    for i in range(len(s_list) - 1):
        s0, s1 = s_list[i], s_list[i + 1]
        ds = s1 - s0
        if ds < 0:
            raise ValueError("Отрицательный приращение дистанции (полилиния некорректна)")

        v0 = max(float(speed_list[i]), min_speed)
        v1 = max(float(speed_list[i + 1]), min_speed)

        if ds == 0.0:
            dt_seg = 0.0
        else:
            dv = v1 - v0
            # Численно устойчивый переход к пределу при |dv| << v
            if abs(dv) <= 1e-12 * max(v0, v1):
                dt_seg = ds / v0
            else:
                ratio = v1 / v0
                if ratio <= 0.0:
                    raise ValueError("Скорости должны быть > 0 для интегрирования времени")
                dt_seg = (ds / dv) * math.log(ratio)

            if dt_seg < 0:
                # При линейной v(s) и v0,v1>0 этого быть не должно
                raise ValueError("Получено отрицательное время на отрезке — проверьте входные данные")

        t_list.append(t_list[-1] + dt_seg)

    return t_list


@dataclass(frozen=True)
class Polygon:
    vertices: list[Point]

    def __post_init__(self):
        # Если полигон задан с продублированной замыкающей точной (совпадающей с начальной) отбросим её
        if len(self.vertices) > 1 and self.vertices[0] == self.vertices[-1]:
            object.__setattr__(self, "vertices", self.vertices[:-1])

    @property
    def area(self) -> float:
        verts = self.vertices
        n = len(verts)
        if n < 3:
            return 0.0

        s = 0.0
        for i in range(n):
            x1, y1 = verts[i]

            x2, y2 = verts[(i + 1) % n]

            s += x1 * y2 - x2 * y1

        return abs(s) * 0.5
