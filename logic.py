"""
# Формируем список ключевых точек на маршруте через которые дрон должен пролететь
# 1. Согласно заданным параметрам аппарата (максимальная скорость, ускорение/торможение, прохождение поворотов)
# и с учётом набора точек через которые дрон должен пролететь
# выполняем расчёт горизонтальной скорости дрона в точках на маршруте

# 2. Зная траекторию и предсказанную скорость движения аппарата по ней (v_motion(t))
# и учитывая заданные параметры и ограничения:

# минимальная/максимальная скорость работы насоса (и настройка поведения при нарушении границ - например отключение или работа на границах возможного),
# объём бака V_tank
# общая длина L
# сколько всего литров должны вылить V_total

# Перед полётом (или в ходе полёта) выполняем вычисления (грубо говоря - переводим в задачу движение по прямой с переменной скоростью и необходимостью вылить заданный объём жидкостью с учётом ограничений):
# 1) проверяем не выходим ли мы в ходе полёта за ограничения насоса (считаем скорость насоса по формуле v_pump(t) = V_total * v_motion(t) / L)
# 2) (если есть, объём бака не позволит нам за раз вылить требуемый суммарный объём) определяем точки опустошения бака (согласно вычисленным v_pump(t))
"""

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, NamedTuple, Literal
from typing import Sequence

import matplotlib.pyplot as plt


# ---------------------------- Геометрия/утилиты ----------------------------

def sqm_to_hectares(area_sqm: float) -> float:
    """
    Перевод площади из квадратных метров в гектары.
    :param area_sqm: площадь в м²
    :return: площадь в гектарах
    """
    return area_sqm / 10_000.0


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


def dist(a: Point, b: Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    return math.hypot(dx, dy)


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def polyline_lengths(pts: list[Point]) -> list[float]:
    """Накопленная длина вдоль ломаной, s[i] — расстояние от начала до pts[i]."""
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


@dataclass(frozen=True)
class Polygon:
    vertices: list[Point]
    
    def __post_init__(self):
        if len(self.vertices) >= 2 and self.vertices[0] == self.vertices[-1]:
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


# Преобразование Гео координат в декартовы на пласкости


class GeoPointToXY:
    R = 6378137.0  # м, WGS84 (экваториальный радиус, сферическое приближение)
    
    @classmethod
    def geo_to_xy_eq(cls, center: GeoPoint, geo_point: GeoPoint) -> Point:
        lon0_deg, lat0_deg = center.lon, center.lat
        
        lon0, lat0 = math.radians(lon0_deg), math.radians(lat0_deg)
        
        lon, lat = math.radians(geo_point.lon), math.radians(geo_point.lat)
        
        x = cls.R * (lon - lon0) * math.cos(lat0)
        y = cls.R * (lat - lat0)
        
        return Point(x=x, y=y)
    
    @classmethod
    def geo_to_xy_aeqd(cls, center: GeoPoint, geo_point: GeoPoint) -> Point:
        from pyproj import CRS, Transformer
        
        lon0_deg, lat0_deg = center.lon, center.lat
        
        espg_to_wgs84_code = 4326
        wgs84 = CRS.from_epsg(espg_to_wgs84_code)
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0_deg} +lon_0={lon0_deg} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
        tr = Transformer.from_crs(wgs84, aeqd, always_xy=True)
        
        x, y = tr.transform(geo_point.lon, geo_point.lat)
        
        return Point(x=float(x), y=float(y))
    
    @staticmethod
    def utm_zone_from_lon(lon: float) -> int:
        return int((lon + 180) // 6) + 1
    
    @classmethod
    def geo_to_xy_utm(cls, center: GeoPoint, geo_point: GeoPoint) -> Point:
        from pyproj import CRS, Transformer  # требуются: pip install pyproj
        
        lon0, lat0 = center.lon, center.lat
        
        zone = cls.utm_zone_from_lon(lon0)
        epsg = (32600 if lat0 >= 0.0 else 32700) + zone  # 326xx — север, 327xx — юг
        wgs84 = CRS.from_epsg(4326)
        utm = CRS.from_epsg(epsg)
        tr = Transformer.from_crs(wgs84, utm, always_xy=True)
        x, y = tr.transform(geo_point.lon, geo_point.lat)
        return Point(x=float(x), y=float(y))


# ---------------------------- Контроллер насоса ----------------------------

class BoundaryAction(Enum):
    CLAMP = "clamp"  # прижимать к границе (мин/макс)
    ZERO = "zero"  # установить 0 л/с (экв. "выключить")


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


class PumpController:
    def __init__(self, constraints: PumpConstraints):
        self.c = constraints
    
    def _apply_low(self, x: float) -> float:
        m = self.c.low_mode
        if m is BoundaryAction.CLAMP:
            return max(x, self.c.q_min)
        if m is BoundaryAction.ZERO:
            return 0.0
        return x
    
    def _apply_high(self, x: float) -> float:
        m = self.c.high_mode
        if m is BoundaryAction.CLAMP:
            return min(x, self.c.q_max)
        if m is BoundaryAction.ZERO:
            return 0.0
        return x
    
    def compute_flow_series(self, t, v_motion, length: float, volume_total: float, eps=1e-9) -> PumpPlan:
        """
        :param t: дискретные моменты времени
        :param v_motion: моментальная скорость в соответствующие моменты времени
        :param length: Общая дистанция (путь)
        :param volume_total: Общий объём который необходимо вылить
        :return:
        """
        assert len(t) == len(v_motion)
        
        if length <= 0 or volume_total <= 0:
            return PumpPlan(t=t, q=[0.0] * len(t), empty_events=[])
        
        # Сперва определим требуемую скорость насоса на всём маршруте
        q_req = [volume_total * max(vm, 0.0) / max(length, eps) for vm in v_motion]
        
        q = []
        for x in q_req:
            # Важно делаем ОТКЛЮЧЕНИЕ насоса при требуемой около нулевой скорости
            if x <= eps:
                q.append(0.0)  # нулевая скорость — насос 0
                continue
            
            if x < self.c.q_min:
                q.append(self._apply_low(x))
            
            elif x > self.c.q_max:
                q.append(self._apply_high(x))
            
            else:
                q.append(x)
        
        # интеграция и определение временных моментов опустошения бака
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
    def total_dispensed(plan: PumpPlan) -> float:
        """Интегрируем q(t) по времени — сколько реально выльется."""
        V = 0.0
        for i in range(1, len(plan.t)):
            dt = max(plan.t[i] - plan.t[i - 1], 0.0)
            V += plan.q[i - 1] * dt
        return V


# Визуализации:

@dataclass(frozen=True)
class PolylineSampler:
    waypoints: list[Point]
    s_nodes: list[float]  # накопленные длины для waypoints
    
    @classmethod
    def from_waypoints(cls, waypoints: list[Point]) -> "PolylineSampler":
        return cls(waypoints=waypoints, s_nodes=polyline_lengths(waypoints))
    
    @property
    def total_length(self) -> float:
        return self.s_nodes[-1]
    
    def position_at_s(self, s: float) -> Point:
        """
        Интерполяция точки на ломаной по дуговой длине s.
        s должен быть в диапазоне [0, total_length].
        """
        if s <= 0.0:
            return self.waypoints[0]
        if s >= self.total_length:
            return self.waypoints[-1]
        
        # индекс правого узла сегмента: s_nodes[i-1] <= s < s_nodes[i]
        i = bisect.bisect_right(self.s_nodes, s)
        # защитный случай (на границе)
        i = min(max(i, 1), len(self.waypoints) - 1)
        
        s0, s1 = self.s_nodes[i - 1], self.s_nodes[i]
        (x0, y0), (x1, y1) = self.waypoints[i - 1], self.waypoints[i]
        seg_len = s1 - s0
        # На случай нулевой длины сегмента (совпадающие точки)
        if seg_len == 0.0:
            return Point(x0, y0)
        t = (s - s0) / seg_len
        return Point(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    
    def sample_points(self, s_list: list[float]) -> list[Point]:
        return [self.position_at_s(s) for s in s_list]
    
    @classmethod
    def plot_trajectory_with_samples(cls,
                                     waypoints: list[Point],
                                     s_list: list[float],
                                     v_list: list[float],
                                     s_nodes: list[float] | None = None,
                                     figsize: tuple[int, int] = (12, 8),  # Увеличил картинку
                                     grid_step: float = 20.0,  # Шаг сетки
                                     ):
        if len(s_list) != len(v_list):
            raise ValueError("Длины списков s_list и v_list должны совпадать.")
        
        sampler = (
            cls(waypoints=waypoints, s_nodes=s_nodes)
            if s_nodes is not None
            else cls.from_waypoints(waypoints)
        )
        
        pts = sampler.sample_points(s_list)
        xs, ys = zip(*waypoints)
        xs_s, ys_s = zip(*pts) if pts else ([], [])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ломаная
        ax.plot(xs, ys, "--", linewidth=1.5)
        
        # Точки по s_list с раскраской по v_list
        sc = ax.scatter(xs_s, ys_s, c=v_list, s=40)
        
        # Colorbar меньшего размера
        cbar = fig.colorbar(sc, ax=ax, fraction=0.1, pad=0.02)
        cbar.set_label("Скорость, м/с")
        
        # Сетка равномерная
        ax.set_xticks(range(0, int(max(xs)) + 1, int(grid_step)))
        ax.set_yticks(range(0, int(max(ys)) + 1, int(grid_step)))
        ax.grid(True)
        
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x, м")
        ax.set_ylabel("y, м")
        ax.set_title("Скорость по траектории")
        plt.tight_layout()
        plt.show()


# --------- Конфигурация/ограничения движения ---------

@dataclass(frozen=True)
class MotionConstraints:
    v_max: float  # м/с
    a_max: float  # м/с^2 (разгон)
    d_max: float  # м/с^2 (торможение, положительное число)
    yaw_rate: float = 90.0  # град/с, если поворот на месте
    turn_radius: Optional[float] = None  # м; None => поворот на месте
    turn_speed: Optional[float] = None  # м/с для дуги; если None — из a_lat_max
    a_lat_max: float = 1.5  # м/с^2 допустимое поперечное ускорение
    angle_eps_deg: float = 3.0  # порог "почти прямая"
    start_speed: float = 0.0  # м/с на старте
    end_speed: float = 0.0  # м/с в конце


# --------- Стратегия допустимой скорости на повороте ---------

class TurnPolicy(Protocol):
    def corner_speed(self, phi_rad: float, motion: MotionConstraints) -> Optional[float]:
        """Возвращает допустимую скорость на повороте; None => ограничения нет (почти прямая)."""


class CosAngleScaledTurnPolicy:
    """
    Эвристика: v_turn = v_base * cos(phi/2), где
      v_base = min(turn_speed (или sqrt(a_lat * R)), v_max).
    При phi→0 ограничение исчезает, при 180° → 0.
    """
    
    def corner_speed(self, phi_rad: float, motion: MotionConstraints) -> Optional[float]:
        phi_deg = math.degrees(phi_rad)
        if phi_deg <= motion.angle_eps_deg:
            return None
        if motion.turn_radius is None:
            return 0.0
        v_base = (
            motion.turn_speed
            if motion.turn_speed is not None
            else math.sqrt(max(motion.a_lat_max, 1e-9) * motion.turn_radius)
        )
        v_base = min(v_base, motion.v_max)
        scale = math.cos(0.5 * phi_rad)  # 1..0
        return max(0.0, v_base * scale)


class PureLateralLimitTurnPolicy:
    """
    Чисто физический предел по боковому ускорению: v_turn = sqrt(a_lat * R),
    без зависимости от угла (кроме «почти прямая» и «на месте»).
    """
    
    def corner_speed(self, phi_rad: float, motion: MotionConstraints) -> Optional[float]:
        phi_deg = math.degrees(phi_rad)
        if phi_deg <= motion.angle_eps_deg:
            return None
        if motion.turn_radius is None:
            return 0.0
        v_base = (
            motion.turn_speed
            if motion.turn_speed is not None
            else math.sqrt(max(motion.a_lat_max, 1e-9) * motion.turn_radius)
        )
        return min(v_base, motion.v_max)


# --------- Сегменты пути ---------
import math
import bisect
from dataclasses import dataclass
from typing import Protocol, Optional


# ---------- Протокол сегмента ----------

class Segment(Protocol):
    length: float
    type: str
    
    def speed_at(self, s_along: float) -> float: ...
    
    def duration(self) -> float: ...
    
    def distance_at_time(self, tau: float) -> float: ...
    
    def speed_at_time(self, tau: float) -> float: ...


# ---------- Реализации сегментов ----------

@dataclass
class StraightSegment:
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


@dataclass
class TurnSegment:
    type: str
    length: float  # длина дуги, м; 0 для поворота на месте
    v_const: float  # постоянная скорость на дуге; 0 при повороте на месте
    phi_deg: float
    radius: Optional[float]
    yaw_rate: float  # для поворота на месте
    
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


# ---------- Профиль скорости ----------
def _frange_inclusive(t0: float, t1: float, step: float) -> list[float]:
    """Создать возрастающий список [t0, ..., t1] с шагом <= step, гарантируя t1 в конце."""
    if step <= 0:
        return [t0, t1] if t1 > t0 else [t0]
    out = []
    k = 0
    cur = t0
    while cur + 1e-12 < t1:
        out.append(cur)
        k += 1
        cur = t0 + k * step
    out.append(t1)
    return out


class SpeedProfile:
    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        
        # Кумулятивные длины
        self._cum: list[float] = [0.0]
        s = 0.0
        for seg in segments:
            s += seg.length
            self._cum.append(s)
        
        # Кумулятивные времена
        self._cum_time: list[float] = [0.0]
        t = 0.0
        for seg in segments:
            t += seg.duration()
            self._cum_time.append(t)
    
    @property
    def all_segments(self) -> list[Segment]:
        return list(self._segments)
    
    @property
    def total_distance(self) -> float:
        return self._cum[-1]
    
    @property
    def total_duration(self) -> float:
        return self._cum_time[-1]
    
    # --- Поиск по дистанции/времени ---
    
    def find_segment_with_bounds(self, s_along_path: float):
        s = s_along_path
        if s < 0 or s > self.total_distance:
            raise ValueError(f"segment at distance {s} out of range")
        
        for i, seg in enumerate(self._segments):
            start = self._cum[i]
            end = self._cum[i + 1]
            if s <= end or i == len(self._segments) - 1:
                return {'seg': seg, 'start': start, 'end': end, 'i': i}
        
        raise RuntimeError(f"segment at distance {s} not found")
    
    def find_segment_by_time(self, t: float):
        if t < 0 or t > self.total_duration:
            raise ValueError(f"time {t} out of range")
        
        for i, seg in enumerate(self._segments):
            t_start = self._cum_time[i]
            t_end = self._cum_time[i + 1]
            if t <= t_end or i == len(self._segments) - 1:
                return {'seg': seg, 't_start': t_start, 't_end': t_end, 'i': i}
        
        raise RuntimeError("segment at time not found")
    
    # --- Запросы ---
    
    def seg_at_distance(self, s_along_path: float) -> Segment:
        return self.find_segment_with_bounds(s_along_path)['seg']
    
    def speed_at_distance(self, s_along_path: float) -> float:
        res = self.find_segment_with_bounds(s_along_path)
        seg = res['seg']
        start = res['start']
        return seg.speed_at(s_along_path - start)
    
    def distance_at_time(self, t: float) -> float:
        """
        Точная пройденная дистанция к моменту времени t.
        Без дискретизации; использует duration()/distance_at_time() сегментов.
        """
        if t <= 0.0:
            return 0.0
        if t >= self.total_duration:
            return self.total_distance
        
        res = self.find_segment_by_time(t)
        seg = res['seg']
        t_start = res['t_start']
        i = res['i']
        
        tau = t - t_start  # локальное время в сегменте
        s_local = seg.distance_at_time(tau)
        # защита от накопл. ошибок
        s_local = max(0.0, min(s_local, seg.length))
        return self._cum[i] + s_local
    
    def speed_at_time(self, t: float) -> float:
        """
        Точная скорость в момент времени t.
        Без дискретизации; использует piecewise формулы сегментов.
        """
        if t <= 0.0:
            # скорость на старте первого сегмента во времени t=0
            first = self._segments[0]
            return first.speed_at_time(0.0) if hasattr(first, "speed_at_time") else first.speed_at(0.0)
        
        if t >= self.total_duration:
            last = self._segments[-1]
            # скорость в конце последнего сегмента
            tau_last = last.duration()
            return last.speed_at_time(tau_last) if hasattr(last, "speed_at_time") else last.speed_at(last.length)
        
        res = self.find_segment_by_time(t)
        seg = res['seg']
        t_start = res['t_start']
        tau = t - t_start
        return seg.speed_at_time(tau) if hasattr(seg, "speed_at_time") else seg.speed_at(seg.distance_at_time(tau))
    
    def simulate_time_param(self, dt: float = 0.05) -> tuple[list[float], list[float], list[float]]:
        """
        Дискретизация (t, s(t), v(t)) по всему профилю.
        Время сегмента берётся из seg.duration().
        Внутри сегмента: сетка по ~dt + точные границы фаз (для StraightSegment).
        Возвращает (t_list, s_list, v_list) одинаковой длины.
        """
        if not self._segments:
            return [0.0], [0.0], [0.0]
        
        eps = 1e-9
        
        # Начальные точки
        t_list: list[float] = [0.0]
        s_list: list[float] = [0.0]
        # Начальная скорость из первого сегмента
        first_seg = self._segments[0]
        v0 = first_seg.speed_at_time(0.0) if hasattr(first_seg, "speed_at_time") else first_seg.speed_at(0.0)
        v_list: list[float] = [v0]
        
        t_offset = 0.0
        s_offset = 0.0
        
        for seg in self._segments:
            T = seg.duration()
            if T < 0:
                raise ValueError("segment duration() < 0")
            
            # Глобальные базовые точки времени для сегмента
            t_start = t_offset
            t_end = t_offset + T
            global_ts = self._frange_inclusive_global(t_start, t_end, dt)
            
            # Фазовые точки для StraightSegment: t_acc, t_acc + t_cruise (в глобальном времени)
            if isinstance(seg, StraightSegment):
                # вычисляем времена фаз
                t_acc, t_dec, t_cruise = seg.times(eps=1e-12)
                # контроль согласованности
                if abs(T - (t_acc + t_cruise + t_dec)) > 1e-9:
                    raise ValueError("Несогласованные параметры StraightSegment: сумма фаз не равна duration()")
                
                phase_ts = []
                if t_acc > 0.0:
                    phase_ts.append(t_start + t_acc)
                if t_cruise > 0.0:
                    phase_ts.append(t_start + t_acc + t_cruise)
                
                # слить и удалить дубликаты с eps
                global_ts = self._unique_sorted_eps(
                    global_ts + [x for x in phase_ts if t_start + eps < x < t_end - eps],
                    eps=eps)
            
            # Вычислить s,v в этих ГЛОБАЛЬНЫХ моментах времени
            for tg in global_ts:
                # пропуск точек-дубликатов (стык сегментов и др.)
                if abs(tg - t_list[-1]) <= eps:
                    continue
                
                tau = tg - t_offset  # локальное время внутри сегмента
                s_local = seg.distance_at_time(tau)
                v_local = seg.speed_at_time(tau) if hasattr(seg, "speed_at_time") else seg.speed_at(s_local)
                
                # защита от погрешностей
                s_local = min(max(0.0, s_local), seg.length)
                if abs(tg - t_end) <= eps:
                    s_local = seg.length
                
                s_global = s_offset + s_local
                
                t_list.append(tg)
                s_list.append(s_global)
                v_list.append(v_local)
            
            # переход к следующему сегменту
            t_offset = t_end
            s_offset += seg.length
        
        return t_list, s_list, v_list
    
    @staticmethod
    def _unique_sorted_eps(values: list[float], eps: float = 1e-9) -> list[float]:
        """Отсортировать и удалить точки, совпадающие с точностью eps."""
        if not values:
            return []
        values = sorted(values)
        out = [values[0]]
        for v in values[1:]:
            if abs(v - out[-1]) > eps:
                out.append(v)
        return out
    
    @staticmethod
    def _frange_inclusive_global(t0: float, t1: float, dt: float) -> list[float]:
        """
        Вернуть точки [t0, ..., t1]; шаг ~ dt.
        Без накопления ошибок: последняя точка принудительно t1.
        """
        if t1 < t0:
            return [t0]
        if dt <= 0:
            return [t0, t1] if t1 > t0 else [t0]
        span = t1 - t0
        # сколько шагов dt укладывается ДО последней точки (последняя — t1 отдельно)
        n = int(math.floor(span / dt))
        ts = [t0 + k * dt for k in range(n)]  # t0 .. t0+(n-1)dt
        ts.append(t1)  # конец точно t1
        return ts


# --------- Предиктор/строитель ---------

class SpeedPredictor:
    def __init__(self, motion: MotionConstraints, policy: Optional[TurnPolicy] = None) -> None:
        self.motion = motion
        self.policy = policy if policy is not None else CosAngleScaledTurnPolicy()
    
    @staticmethod
    def _heading(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        return (math.degrees(math.atan2(dy, dx))) % 360.0
    
    @staticmethod
    def _angle_diff_deg(h1: float, h2: float) -> float:
        d = ((h2 - h1 + 180.0) % 360.0) - 180.0
        return abs(d)
    
    def _straight_segment(self, L: float, v_in: float, v_out_req: Optional[float],
                          bound_points: BoundPoints) -> StraightSegment:
        m = self.motion
        a, d, vmax = m.a_max, m.d_max, m.v_max
        
        if v_out_req is None:
            # НЕТ жёсткого ограничения на выходе => чистый разгон в пределах длины L
            # v_peak = min(vmax, sqrt(v_in^2 + 2*a*L))
            v_peak_free = math.sqrt(max(0.0, v_in * v_in + 2.0 * a * L))
            v_peak = min(vmax, v_peak_free)
            v_out = v_peak
            # Разбивка по фазам: торможения нет
            d_acc = max(0.0, (v_peak * v_peak - v_in * v_in) / (2.0 * max(a, 1e-9)))
            d_dec = 0.0
            d_cruise = max(0.0, L - d_acc)
        else:
            # Есть требуемая скорость на выходе (ограничение поворотом/финишем)
            v_out = max(0.0, float(v_out_req))
            # Трапеция/треугольник: разгон до v_peak и торможение до v_out внутри L
            num = 2.0 * a * d * L + d * v_in * v_in + a * v_out * v_out
            den = a + d
            v_peak_limit = math.sqrt(max(0.0, num / max(den, 1e-9)))
            v_peak = min(vmax, v_peak_limit)
            
            d_acc = max(0.0, (v_peak * v_peak - v_in * v_in) / (2.0 * max(a, 1e-9)))
            d_dec = max(0.0, (v_peak * v_peak - v_out * v_out) / (2.0 * max(d, 1e-9)))
            d_cruise = max(0.0, L - d_acc - d_dec)
        
        return StraightSegment(
            type="straight",
            length=L,
            v_in=v_in,
            v_peak=v_peak,
            v_out=v_out,
            d_acc=d_acc,
            d_cruise=d_cruise,
            d_dec=d_dec,
            a_max=a,
            d_max=d,
            bounds=bound_points,
        )
    
    def build_profile(self, waypoints: list[tuple[float, float]]) -> SpeedProfile:
        m = self.motion
        n = len(waypoints)
        if n < 2:
            return SpeedProfile([])
        
        # прямые: длины и курсы
        seg_len: list[float] = []
        seg_head: list[float] = []
        for i in range(n - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            seg_len.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            seg_head.append(self._heading(p1, p2))
        
        # допустимая скорость на узлах (концах прямых), может быть None
        v_node: list[Optional[float]] = [None] * (n - 1)
        for i in range(n - 2):
            phi_deg = self._angle_diff_deg(seg_head[i], seg_head[i + 1])
            phi_rad = math.radians(phi_deg)
            v_node[i] = self.policy.corner_speed(phi_rad, m)
        
        segments: list[Segment] = []
        v_in_prev = max(0.0, m.start_speed)
        
        for i in range(n - 1):
            L = seg_len[i]
            
            # целевая скорость на выходе из прямого сегмента
            v_out_req = v_node[i]
            if i == (n - 2):
                # последний прямой — ещё и конечное условие
                if v_out_req is None:
                    v_out_req = m.end_speed
                else:
                    v_out_req = min(v_out_req, m.end_speed)
            
            # Граничные точки сегмента
            p1 = Point(*waypoints[i])
            p2 = Point(*waypoints[i + 1])
            
            # прямой сегмент
            straight = self._straight_segment(L, v_in_prev, v_out_req, BoundPoints(p1, p2))
            segments.append(straight)
            
            if i >= (n - 2):
                continue
            
            # добавляем поворот, если не последний сегмент
            phi_deg = self._angle_diff_deg(seg_head[i], seg_head[i + 1])
            if phi_deg <= m.angle_eps_deg:
                # почти прямая: поворотного сегмента нет
                v_in_prev = straight.v_out
                continue
            
            if m.turn_radius is None:
                # поворот на месте: скорость к нулю уже обеспечена на прямом
                segments.append(TurnSegment(
                    type="turn",
                    length=0.0,
                    v_const=0.0,
                    phi_deg=phi_deg,
                    radius=None,
                    yaw_rate=m.yaw_rate
                ))
                v_in_prev = 0.0
            else:
                phi_rad = math.radians(phi_deg)
                arc_length = m.turn_radius * phi_rad
                v_turn_lim = self.policy.corner_speed(phi_rad, m)
                
                # policy гарантирует не-None при phi > eps; проверим на всякий случай
                v_turn_lim = m.v_max if v_turn_lim is None else v_turn_lim
                v_turn = min(v_turn_lim, straight.v_out)
                segments.append(TurnSegment(
                    type="turn",
                    length=arc_length,
                    v_const=v_turn,
                    phi_deg=phi_deg,
                    radius=m.turn_radius,
                    yaw_rate=m.yaw_rate
                ))
                v_in_prev = v_turn
        
        return SpeedProfile(segments)


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


# ---------------------------- main: пример использования ----------------------------

def main():
    waypoints = [(0, 0), (50, 0), (50, 50), (100, 50)]
    
    motion = MotionConstraints(
        v_max=20.0,
        a_max=4.0,
        d_max=5.0,
        yaw_rate=90.0,
        turn_radius=5.0,  # дуговой поворот (None => на месте)
        a_lat_max=2.0,
        angle_eps_deg=3.0,
        start_speed=0.0,
        end_speed=0.0
    )
    
    predictor = SpeedPredictor(motion)
    profile = predictor.build_profile(waypoints)
    
    # доступ к данным
    segs = profile.all_segments
    S = profile.total_distance
    v_mid = profile.speed_at_distance(S * 0.5)


if __name__ == '__main__':
    main()
