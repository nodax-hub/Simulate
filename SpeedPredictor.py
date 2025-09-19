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
