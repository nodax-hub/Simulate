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

import bisect
import math
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional, Protocol, Literal

import matplotlib.pyplot as plt
from typing_extensions import NamedTuple


# ---------------------------- Геометрия/утилиты ----------------------------

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
      - 0 < distance < L -> линейная интерполяция на соответствующем сегменте
      - distance == L -> последняя точка
      - иначе -> ValueError
    Используются dist и polyline_lengths.
    """
    if not pts:
        raise ValueError("Список точек пуст")
    if distance < 0:
        raise ValueError("Дистанция не может быть отрицательной")
    
    s = polyline_lengths(pts)
    L = s[-1]
    
    if distance > L:
        raise ValueError(f"Дистанция {distance} больше длины траектории {L}")
    
    if distance == 0:
        return pts[0]
    
    if distance == L:
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


def curvature_radius(p_prev: Point, p: Point, p_next: Point) -> Optional[float]:
    """
    Приближённый радиус окружности через три точки.
    Если точки почти на одной прямой — вернём None (бесконечный радиус).
    """
    (x1, y1), (x2, y2), (x3, y3) = p_prev, p, p_next
    # Формулы через пересечение серединных перпендикуляров
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    e = ((x1 ** 2 - x2 ** 2) + (y1 ** 2 - y2 ** 2)) / 2.0
    f = ((x1 ** 2 - x3 ** 2) + (y1 ** 2 - y3 ** 2)) / 2.0
    det = a * d - b * c
    if abs(det) < 1e-9:
        return None
    cx = (d * e - b * f) / det
    cy = (-c * e + a * f) / det
    r = math.hypot(cx - x1, cy - y1)
    return r if r > 1e-6 else None


# Преобразование Гео координат в декартовы на пласкости


class GeoPointToXY:
    @classmethod
    def geo_to_xy(cls, points: list[GeoPoint], mode: Literal['EQ', 'AEQD', 'UTM'] = 'EQ') -> list[Point]:
        converters = {
            'EQ': cls.make_equirect_xy_converter,
            'AEQD': cls.make_aeqd_xy_converter,
            'UTM': cls.make_utm_xy_converter,
        }
        
        return cls.geo_to_xy_points(points, converters[mode](points))
    
    @staticmethod
    def geo_to_xy_points(points: Iterable[GeoPoint],
                         to_xy: Callable[[float, float], tuple[float, float]]) -> list[Point]:
        out: list[Point] = []
        for p in points:
            x, y = to_xy(p.lon, p.lat)
            out.append(Point(x, y))
        return out
    
    # ========== 1) Локальная equirectangular (малые области) ==========
    # x ≈ R cos(lat0) (lon - lon0), y ≈ R (lat - lat0)
    # Углы в радианах. Быстро и просто, приемлемо в пределах ~10–20 км от опорной точки.
    @staticmethod
    def make_equirect_xy_converter(points: list[GeoPoint]) -> Callable[[float, float], tuple[float, float]]:
        if not points:
            raise ValueError("Нужна хотя бы одна точка для опорного центра.")
        R = 6378137.0  # м, WGS84 (экваториальный радиус, сферическое приближение)
        lon0_deg, lat0_deg = points[0].lon, points[0].lat
        lon0, lat0 = math.radians(lon0_deg), math.radians(lat0_deg)
        
        def to_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
            lon, lat = math.radians(lon_deg), math.radians(lat_deg)
            x = R * (lon - lon0) * math.cos(lat0)
            y = R * (lat - lat0)
            return x, y
        
        return to_xy
    
    # ========== 2) AEQD — азимутальная равноудалённая (локальная «метрика») ==========
    # Лучше, чем (1) для радиальных расстояний вокруг центра. Нужен pyproj.
    @staticmethod
    def make_aeqd_xy_converter(points: list[GeoPoint]) -> Callable[[float, float], tuple[float, float]]:
        if not points:
            raise ValueError("Нужна хотя бы одна точка для центра проекции.")
        from pyproj import CRS, Transformer  # требуются: pip install pyproj
        
        lon0, lat0 = points[0].lon, points[0].lat
        wgs84 = CRS.from_epsg(4326)
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        tr = Transformer.from_crs(wgs84, aeqd, always_xy=True)
        
        def to_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
            x, y = tr.transform(lon_deg, lat_deg)
            return float(x), float(y)
        
        return to_xy
    
    # ========== 3) UTM — зональная поперечно-меркаторская ==========
    # Подходит для работы в пределах одной UTM-зоны. Нужен pyproj.
    @staticmethod
    def make_utm_xy_converter(points: list[GeoPoint]) -> Callable[[float, float], tuple[float, float]]:
        if not points:
            raise ValueError("Нужна хотя бы одна точка для определения зоны.")
        from pyproj import CRS, Transformer  # требуются: pip install pyproj
        
        lon0, lat0 = points[0].lon, points[0].lat
        
        def utm_zone_from_lon(lon: float) -> int:
            return int((lon + 180) // 6) + 1
        
        zone = utm_zone_from_lon(lon0)
        epsg = (32600 if lat0 >= 0.0 else 32700) + zone  # 326xx — север, 327xx — юг
        wgs84 = CRS.from_epsg(4326)
        utm = CRS.from_epsg(epsg)
        tr = Transformer.from_crs(wgs84, utm, always_xy=True)
        
        def to_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
            x, y = tr.transform(lon_deg, lat_deg)
            return float(x), float(y)
        
        return to_xy


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
    
    def compute_flow_series(self, t, v_motion, L: float, V_total: float, eps = 1e-9) -> PumpPlan:
        """
        :param t: дискретные моменты времени
        :param v_motion: моментальная скорость в соответствующие моменты времени
        :param L: Общая дистанция (путь)
        :param V_total: Общий объём который необходимо вылить
        :return:
        """
        assert len(t) == len(v_motion)
        
        if L <= 0 or V_total <= 0:
            return PumpPlan(t=t, q=[0.0] * len(t), empty_events=[])
        
        # Сперва определим требуемую скорость насоса на всём маршруте
        q_req = [V_total * max(vm, 0.0) / max(L, eps) for vm in v_motion]
        
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

class Segment(Protocol):
    length: float
    type: str
    
    def speed_at(self, s_along: float) -> float: ...
    
    def duration(self) -> float: ...


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
    a_max: float
    d_max: float
    
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
    
    def duration(self) -> float:
        # Точное интегрирование по трем фазам
        t_acc = (self.v_peak - self.v_in) / max(self.a_max, 1e-9) if self.v_peak > self.v_in else 0.0
        t_dec = (self.v_peak - self.v_out) / max(self.d_max, 1e-9) if self.v_peak > self.v_out else 0.0
        t_cruise = self.d_cruise / self.v_peak if self.v_peak > 1e-9 else 0.0
        return t_acc + t_cruise + t_dec


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
        if self.length > 0.0 and self.v_const > 1e-9:
            return self.length / self.v_const
        # поворот на месте
        return self.phi_deg / max(self.yaw_rate, 1e-9)


# --------- Профиль скорости ---------

class SpeedProfile:
    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        # префиксные длины для быстрого поиска по дистанции
        self._cum: list[float] = [0.0]
        s = 0.0
        for i, seg in enumerate(segments):
            s += seg.length
            self._cum.append(s)
    
    @property
    def all_segments(self) -> list[Segment]:
        return list(self._segments)
    
    @property
    def total_distance(self) -> float:
        return self._cum[-1]
    
    def find_segment_with_bounds(self, s_along_path):
        s = s_along_path
        if s < 0 or s > self.total_distance:
            raise ValueError(f"segment at distance {s} out of range")
        
        #  TODO: бинпоиск можно, но линейный тоже ок для малых списков
        for i, seg in enumerate(self._segments):
            start = self._cum[i]
            end = self._cum[i + 1]
            if s <= end or i == len(self._segments) - 1:
                return {'seg': seg, 'start': start, 'end': end}
        
        raise RuntimeError(f"segment at distance {s} not found")
    
    def seg_at_distance(self, s_along_path: float) -> Segment:
        return self.find_segment_with_bounds(s_along_path)['seg']
    
    def speed_at_distance(self, s_along_path: float) -> float:
        """Скорость в точке пути на расстоянии s_along_path (м) от начала."""
        res = self.find_segment_with_bounds(s_along_path)
        
        seg = res['seg']
        start = res['start']
        end = res['end']
        
        return seg.speed_at(s_along_path - start)
    
    def speed_at_time(self, t, sim_dt=0.01):
        # t_list — возрастающий список времён; v_list — скорости в те же моменты
        t_list, s_list, v_list = self.simulate_time_param(dt=sim_dt)
        
        # Границы: экстраполяции не делаем — берём крайние значения
        if t <= t_list[0]:
            return v_list[0]
        if t >= t_list[-1]:
            return v_list[-1]
        
        # Позиция вставки слева
        i = bisect.bisect_left(t_list, t)
        
        # Точное попадание в узел сетки
        if t_list[i] == t:
            return v_list[i]
        
        # Линейная интерполяция между i-1 и i
        t0, t1 = t_list[i - 1], t_list[i]
        v0, v1 = v_list[i - 1], v_list[i]
        
        dt = t1 - t0
        if dt == 0:
            # На случай дубликатов времени: возвращаем левое значение
            return v0
        
        alpha = (t - t0) / dt
        return v0 + (v1 - v0) * alpha
    
    def simulate_time_param(self, dt: float = 0.05) -> tuple[list[float], list[float], list[float]]:
        """
        Интегрируем движение вдоль s при заданном v(s), получаем t-кривую.
        Возвращает (t_list, s_list, v_list) дискретно по времени.
        Предполагаем мгновенную адаптацию к v(s) (квазистационарно).
        """
        if not self._cum:
            return [], [], []
        t, s_pos = 0.0, 0.0
        t_list, s_list, v_list = [0.0], [0.0], [self.speed_at_distance(0.0)]
        
        L = self.total_distance
        while s_pos < L:
            v = max(self.speed_at_distance(s_pos), 1e-6)
            
            ds = v * dt
            s_pos = min(L, s_pos + ds)
            
            t += dt
            
            t_list.append(t)
            s_list.append(s_pos)
            v_list.append(self.speed_at_distance(s_pos))
        
        return t_list, s_list, v_list


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
