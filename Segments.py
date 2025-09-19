import math
from dataclasses import dataclass
from typing import Protocol, Optional

from dto import BoundPoints, Point
from utils import _circle_center_from_bounds


class Segment(Protocol):
    length: float
    type: str

    def speed_at(self, s_along: float) -> float: ...

    def duration(self) -> float: ...

    def distance_at_time(self, tau: float) -> float: ...

    def speed_at_time(self, tau: float) -> float: ...
    # Новое: точка вдоль сегмента по локальной длине s_along
    def point_at(self, s_along: float) -> Point: ...


@dataclass
class StraightSegment(Segment):
    type: str
    length: float
    bounds: BoundPoints
    v_in: float
    v_peak: float
    v_out: float
    d_acc: float
    d_cruise: float
    d_dec: float
    a_max: float  # ускорение (>=0)
    d_max: float  # замедление (>=0)

    # --- кинематика (без изменений по сути) ---
    def speed_at(self, s_along: float) -> float:
        s = max(0.0, min(self.length, s_along))
        if s <= self.d_acc:
            return math.sqrt(max(0.0, self.v_in ** 2 + 2.0 * self.a_max * s))
        elif s <= self.d_acc + self.d_cruise:
            return self.v_peak
        else:
            s_dec = s - (self.d_acc + self.d_cruise)
            v_sq = max(0.0, self.v_peak ** 2 - 2.0 * self.d_max * s_dec)
            return max(self.v_out, math.sqrt(v_sq))

    def times(self, eps=1e-12):
        t_acc = (self.v_peak - self.v_in) / max(self.a_max, eps) if self.v_peak > self.v_in else 0.0
        t_dec = (self.v_peak - self.v_out) / max(self.d_max, eps) if self.v_peak > self.v_out else 0.0
        t_cruise = self.d_cruise / self.v_peak if self.v_peak > eps else 0.0
        return t_acc, t_dec, t_cruise

    def duration(self, eps=1e-12) -> float:
        return sum(self.times(eps))

    def distance_at_time(self, tau: float, eps=1e-12) -> float:
        """
        Точная s(tau) по фазам:
          1) разгон:   s = v_in * t + 0.5 * a * t^2
          2) крейсер:  s = d_acc + v_peak * (t - t_acc)
          3) тормож.:  s = d_acc + d_cruise + (v_peak * t_d - 0.5 * d * t_d^2)
        """
        t_acc, t_dec, t_cruise = self.times(eps)

        if tau <= 0.0:
            return 0.0
        total_time = t_acc + t_cruise + t_dec
        if tau >= total_time:
            return self.length

        if tau <= t_acc:
            # разгон
            return self.v_in * tau + 0.5 * self.a_max * tau * tau

        if tau <= t_acc + t_cruise:
            # крейсер
            return self.d_acc + self.v_peak * (tau - t_acc)

        # торможение
        t_d = tau - (t_acc + t_cruise)
        return self.d_acc + self.d_cruise + (self.v_peak * t_d - 0.5 * self.d_max * t_d * t_d)

    def speed_at_time(self, tau: float, eps=1e-12) -> float:
        """Опционально: точная v(tau) по фазам."""
        t_acc, t_dec, t_cruise = self.times(eps)

        if tau <= 0.0:
            return self.v_in
        total_time = t_acc + t_cruise + t_dec

        if tau >= total_time:
            return self.v_out

        if tau <= t_acc:
            return self.v_in + self.a_max * tau

        if tau <= t_acc + t_cruise:
            return self.v_peak
        t_d = tau - (t_acc + t_cruise)
        return max(self.v_out, self.v_peak - self.d_max * t_d)

    # --- геометрия: точка на отрезке ---
    def point_at(self, s_along: float) -> Point:
        s = max(0.0, min(self.length, s_along))
        if self.length <= 0.0:
            return self.bounds.start
        u = s / self.length
        x = self.bounds.start.x + u * (self.bounds.end.x - self.bounds.start.x)
        y = self.bounds.start.y + u * (self.bounds.end.y - self.bounds.start.y)
        return Point(x, y)


@dataclass
class TurnSegment(Segment):
    type: str
    length: float            # длина дуги, м; 0 для поворота на месте
    v_const: float           # постоянная скорость на дуге; 0 при повороте на месте
    phi_deg: float
    radius: Optional[float]
    yaw_rate: float          # для поворота на месте

    # --- ДОБАВЛЕНО: опциональная геометрия для вычисления точки ---
    # Вариант A: заданы границы дуги (старт/финиш)
    bounds: Optional[BoundPoints] = None
    # Вариант B: задан центр и стартовый угол
    center: Optional[Point] = None
    start_angle_rad: Optional[float] = None  # atan2(y - cy, x - cx) в начале дуги
    # Направление поворота: True = CW (по часовой), False = CCW (против), None = неизвестно
    cw: Optional[bool] = None
    # Для поворота "на месте": точка-пивот (если не задана, берём bounds.start)
    pivot: Optional[Point] = None

    # --- кинематика (как у вас) ---
    def speed_at(self, s_along: float) -> float:
        return self.v_const

    def duration(self) -> float:
        eps = 1e-12
        if self.length > 0.0 and self.v_const > eps:
            return self.length / self.v_const
        # поворот на месте
        return self.phi_deg / max(self.yaw_rate, eps)

    def distance_at_time(self, tau: float) -> float:
        if self.length <= 0.0 or self.v_const <= 0.0:
            return 0.0
        if tau <= 0.0:
            return 0.0
        T = self.length / self.v_const
        if tau >= T:
            return self.length
        return self.v_const * tau

    def speed_at_time(self, tau: float) -> float:
        return self.v_const

    # --- геометрия: точка на дуге или на месте ---
    def point_at(self, s_along: float) -> Point:
        s = max(0.0, min(self.length, s_along))
        # Поворот на месте: стоим в одной точке (pivot или начало)
        if self.length <= 0.0:
            if self.pivot is not None:
                return self.pivot
            if self.bounds is not None:
                return self.bounds.start
            raise ValueError("TurnSegment.point_at: для поворота на месте задайте pivot или bounds.start")

        # Дуга с центром и стартовым углом
        if self.center is not None and self.start_angle_rad is not None:
            if self.radius is None or self.radius <= 0.0:
                raise ValueError("TurnSegment.point_at: требуется положительный radius при заданном center/start_angle_rad")
            R = self.radius
            # знак приращения угла
            sign = -1.0 if (self.cw is True) else +1.0  # по часовой — уменьшаем угол
            dtheta = (s / R) * sign
            theta = self.start_angle_rad + dtheta
            cx, cy = self.center
            return Point(cx + R * math.cos(theta), cy + R * math.sin(theta))

        # Дуга по двум точкам и радиусу
        if self.bounds is not None and self.radius is not None and self.radius > 0.0 and self.cw is not None:
            p0 = self.bounds.start
            p1 = self.bounds.end
            cx, cy, R, theta0 = _circle_center_from_bounds(p0, p1, self.radius, self.cw)
            # Контроль: если задан phi_deg, длина должна быть R*|phi|
            # (не останавливаем, но можем сигнализировать разработчику при сильном расхождении)
            # Движение по дуге
            sign = -1.0 if self.cw else +1.0
            dtheta = (s / R) * sign
            theta = theta0 + dtheta
            return Point(cx + R * math.cos(theta), cy + R * math.sin(theta))

        raise ValueError(
            "TurnSegment.point_at: недостаточно геометрии. "
            "Задайте (center, start_angle_rad, radius, cw) ИЛИ (bounds, radius, cw)."
        )
