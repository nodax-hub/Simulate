import math

from Segments import Segment, StraightSegment
from dto import Point


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

    # --- НОВОЕ: точка по дистанции/времени ---
    def point_at_distance(self, s_along_path: float) -> Point:
        res = self.find_segment_with_bounds(s_along_path)
        seg: Segment = res['seg']
        s_local = s_along_path - res['start']
        return seg.point_at(s_local)

    def point_at_time(self, t: float) -> Point:
        s = self.distance_at_time(t)
        return self.point_at_distance(s)

    # --- симуляция без изменений ---
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
            t = seg.duration()
            if t < 0:
                raise ValueError("segment duration() < 0")

            # Глобальные базовые точки времени для сегмента
            t_start = t_offset
            t_end = t_offset + t
            global_ts = self._frange_inclusive_global(t_start, t_end, dt)

            # Фазовые точки для StraightSegment: t_acc, t_acc + t_cruise (в глобальном времени)
            if isinstance(seg, StraightSegment):
                # вычисляем времена фаз
                t_acc, t_dec, t_cruise = seg.times(eps=1e-12)
                # контроль согласованности
                if abs(t - (t_acc + t_cruise + t_dec)) > 1e-9:
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
