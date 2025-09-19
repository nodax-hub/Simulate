import math
from dataclasses import dataclass
from typing import Optional, Protocol

from Segments import StraightSegment, Segment, TurnSegment
from SpeedPofile import SpeedProfile
from dto import Point, BoundPoints


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


class TurnPolicy(Protocol):
    def corner_speed(self, phi_rad: float, motion: MotionConstraints) -> Optional[float]:
        """Возвращает допустимую скорость на повороте; None => ограничения нет (почти прямая)."""


class CosAngleScaledTurnPolicy(TurnPolicy):
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


class PureLateralLimitTurnPolicy(TurnPolicy):
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


class SpeedPredictor:
    def __init__(self, motion: MotionConstraints, policy: Optional[TurnPolicy] = None) -> None:
        self.motion = motion
        self.policy = policy if policy is not None else CosAngleScaledTurnPolicy()

    @staticmethod
    def _heading(a: Point, b: Point) -> float:
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

    def build_profile(self, waypoints: list[Point]) -> SpeedProfile:
        """
        Строит профиль с геометрией:
        - если дуга с радиусом R помещается в узле → режем соседние прямые на R*tan(φ/2),
          задаём TurnSegment(bounds=..., radius=R, cw=...);
        - иначе поворот на месте в узле (pivot=сам waypoint), а предыдущая прямая заканчивается v=0.
        Только SpeedPredictor.py меняется. Segments/SpeedPofile не трогаем.
        """
        m = self.motion
        n = len(waypoints)
        if n < 2:
            return SpeedProfile([])

        # длины и курсы прямых
        seg_len: list[float] = []
        seg_head: list[float] = []
        for i in range(n - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            seg_len.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            seg_head.append(self._heading(p1, p2))

        # ограничение скорости в узлах по policy
        v_node: list[Optional[float]] = [None] * (n - 1)
        for i in range(n - 2):
            phi_deg = self._angle_diff_deg(seg_head[i], seg_head[i + 1])
            phi_rad = math.radians(phi_deg)
            v_node[i] = self.policy.corner_speed(phi_rad, m)

        # можно ли поставить дугу в узле i (между сегментами i и i+1)
        def can_place_arc(i: int) -> tuple[bool, float, float]:
            if m.turn_radius is None or m.turn_radius <= 0.0:
                return (False, 0.0, 0.0)
            phi_deg = self._angle_diff_deg(seg_head[i], seg_head[i + 1])
            if phi_deg <= m.angle_eps_deg:
                return (False, 0.0, math.radians(phi_deg))
            R = m.turn_radius
            phi_rad = math.radians(phi_deg)
            L_tan = R * math.tan(0.5 * phi_rad)
            ok = (seg_len[i] > 2.0 * L_tan + 1e-9) and (seg_len[i + 1] > 2.0 * L_tan + 1e-9)
            return (ok, L_tan, phi_rad)

        # предварительно посчитаем, где будут дуги, и их "срезы"
        cut_start = [0.0] * (n - 1)  # сколько убрать в начале сегмента i
        cut_end = [0.0] * (n - 1)  # сколько убрать в конце сегмента i
        place_arc = [False] * (n - 1)
        arc_phi_rad = [0.0] * (n - 1)

        if m.turn_radius and m.turn_radius > 0.0:
            for i in range(n - 2):
                ok, L_tan, phi_rad = can_place_arc(i)
                if ok:
                    place_arc[i] = True
                    arc_phi_rad[i] = phi_rad
                    cut_end[i] += L_tan
                    cut_start[i + 1] += L_tan

        # подготовим укороченные границы прямых
        straight_bounds: list[BoundPoints] = []
        straight_eff_len: list[float] = []
        for i in range(n - 1):
            A = Point(*waypoints[i])
            B = Point(*waypoints[i + 1])
            L = seg_len[i]
            s0 = cut_start[i]
            s1 = cut_end[i]
            # если вдруг две дуги «съели» сегмент — ограничим нулём (редко, но безопасно)
            L_eff = max(0.0, L - s0 - s1)
            A2 = self._point_on_line(A, B, s0)  # начало после среза
            B2 = self._point_on_line(A, B, L - s1)  # конец до среза
            straight_bounds.append(BoundPoints(A2, B2))
            straight_eff_len.append(L_eff)

        segments: list[Segment] = []
        v_in_prev = max(0.0, m.start_speed)

        # знак поворота (True — по часовой)
        def is_cw(h1: float, h2: float) -> bool:
            d = ((h2 - h1 + 180.0) % 360.0) - 180.0
            return d < 0.0

        for i in range(n - 1):
            is_last = (i == n - 2)

            # целевая скорость на выходе с прямой i
            v_out_req: Optional[float] = None
            if not is_last:
                phi_deg = self._angle_diff_deg(seg_head[i], seg_head[i + 1])
                if phi_deg > m.angle_eps_deg:
                    if place_arc[i]:
                        v_turn_lim = v_node[i]
                        v_out_req = m.v_max if v_turn_lim is None else min(v_turn_lim, m.v_max)
                    else:
                        v_out_req = 0.0  # будет поворот на месте → тормозим до нуля к концу прямой

            if is_last:
                v_out_req = m.end_speed if v_out_req is None else min(v_out_req, m.end_speed)

            # строим прямую с укороченными границами
            straight = self._straight_segment(
                straight_eff_len[i],
                v_in_prev,
                v_out_req,
                straight_bounds[i],  # ВАЖНО: передаём новые BoundPoints
            )
            segments.append(straight)

            if is_last:
                break

            # добавляем поворот
            phi_deg = self._angle_diff_deg(seg_head[i], seg_head[i + 1])
            if phi_deg <= m.angle_eps_deg:
                v_in_prev = straight.v_out
                continue

            if not place_arc[i]:
                # поворот на месте — задаём pivot/bounds, чтобы point_at() не падал
                pivot_pt = Point(*waypoints[i + 1])
                segments.append(TurnSegment(
                    type="turn",
                    length=0.0,
                    v_const=0.0,
                    phi_deg=phi_deg,
                    radius=None,
                    yaw_rate=m.yaw_rate,
                    pivot=pivot_pt,
                    bounds=BoundPoints(pivot_pt, pivot_pt)  # безопасно для point_at()
                ))
                v_in_prev = 0.0
                continue

            # дуга радиуса R: границы — конец текущей укороченной прямой и начало следующей
            R = float(m.turn_radius)  # гарантировано >0, т.к. place_arc[i] == True
            phi_rad = arc_phi_rad[i]
            arc_len = R * phi_rad
            arc_start = straight_bounds[i].end
            arc_end = straight_bounds[i + 1].start
            cw_flag = is_cw(seg_head[i], seg_head[i + 1])

            # скорость на дуге: не больше corner_speed и выходной со straight
            v_turn_lim = v_node[i]
            v_turn_lim = m.v_max if v_turn_lim is None else v_turn_lim
            v_turn = min(v_turn_lim, straight.v_out)

            segments.append(TurnSegment(
                type="turn",
                length=arc_len,
                v_const=v_turn,
                phi_deg=phi_deg,
                radius=R,
                yaw_rate=m.yaw_rate,
                bounds=BoundPoints(arc_start, arc_end),
                cw=cw_flag
            ))
            v_in_prev = v_turn

        return SpeedProfile(segments)

    @staticmethod
    def _unit_vec(a: Point, b: Point) -> tuple[float, float]:
        dx, dy = b.x - a.x, b.y - a.y
        L = math.hypot(dx, dy)
        if L <= 1e-12:
            return 0.0, 0.0
        return dx / L, dy / L

    @classmethod
    def _point_on_line(cls, a: Point, b: Point, dist_from_a: float) -> Point:
        ux, uy = cls._unit_vec(a, b)
        return Point(a.x + ux * dist_from_a, a.y + uy * dist_from_a)
