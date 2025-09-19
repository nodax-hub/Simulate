import csv
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import numpy as np

from dto import Point, GeoPoint
from GeoTransformer import GeoTransformer
from utils import sqm_to_hectares, dist, polyline_lengths, Polygon


def parse_key_points_from_qgc_plan(plan: dict,
                                   include_home: bool = True,
                                   include_takeoff: bool = True,
                                   include_polygon: bool = True,
                                   include_transects: bool = True,
                                   deduplicate: bool = True) -> tuple[GeoPoint, ...]:
    """
    Извлекает ключевые точки из QGC .plan:
      - plannedHomePosition
      - SimpleItem command=22 (взлёт) → params[4]=lat, params[5]=lon
      - ComplexItem 'survey' → polygon [[lat, lon], ...], старт с entryLocation (1-based)
      - ComplexItem 'survey' → TransectStyleComplexItem.VisualTransectPoints [[lat, lon], ...]
      - SimpleItem с coordinate (если есть)
    Возвращает GeoPointDTO(lon, lat, speed=0.0).
    """
    mission = plan.get("mission", {})
    items = mission.get("items", []) or []

    pts: list[GeoPoint] = []

    def add(lat: Optional[float], lon: Optional[float]):
        if lat is None or lon is None:
            return
        pts.append(GeoPoint(lon=float(lon), lat=float(lat)))

    # 1) Home
    if include_home:
        php = mission.get("plannedHomePosition")
        if isinstance(php, list) and len(php) >= 2:
            add(float(php[0]), float(php[1]))

    # 2) Takeoff (command 22)
    if include_takeoff:
        for it in items:
            if it.get("type") == "SimpleItem" and it.get("command") == 22:
                params = it.get("params", [])
                if isinstance(params, list) and len(params) >= 7:
                    add(float(params[4]), float(params[5]))
                break

    # 3,4) Survey polygon + VisualTransectPoints
    for it in items:
        if it.get("type") == "ComplexItem" and it.get("complexItemType") == "survey":
            if include_polygon:
                poly = it.get("polygon") or []
                verts: list[tuple[float, float]] = []
                for p in poly:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        verts.append((float(p[0]), float(p[1])))  # [lat, lon]
                if verts:
                    entry = it.get("entryLocation")
                    start = (entry - 1) if isinstance(entry, int) and 1 <= entry <= len(verts) else 0
                    ordered = verts[start:] + verts[:start]
                    for lat, lon in ordered:
                        add(lat, lon)
            if include_transects:
                tsc = it.get("TransectStyleComplexItem", {})
                vtp = tsc.get("VisualTransectPoints") or []
                for p in vtp:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        add(float(p[0]), float(p[1]))  # [lat, lon]
            break

    # 5) Прочие SimpleItem с coordinate
    for it in items:
        if it.get("type") == "SimpleItem":
            coord = it.get("coordinate")
            if isinstance(coord, list) and len(coord) >= 2:
                add(float(coord[0]), float(coord[1]))

    # 6) Дедупликация (по lat/lon)
    if deduplicate and pts:
        seen = set()
        uniq: list[GeoPoint] = []
        for p in pts:
            key = (round(p.lat, 8), round(p.lon, 8))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        pts = uniq

    return tuple(pts)


# ====== Парсер CSV с ';' и десятичной ',' ======
def _to_float(s: str) -> Optional[float]:
    """Безопасное превращение строки с десятичной ',' в float. Пустые -> None."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # заменяем десятичную запятую на точку
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return None


def parse_points_csv(
        path: Path,
        *,
        lat_col: int = 20,  # c20
        lon_col: int = 21,  # c21
        speed_col: int = 1  # c1 (текущая горизонтальная скорость)
) -> list[dict]:
    """
    Читает CSV и возвращает список PointDTO:
    - разделитель ';'
    - десятичная запятая ','
    - координаты берём из c20/c21 (lon/lat), скорость из c1
    - проекция: локальная AEQD вокруг первой валидной точки
    """
    # Сначала собираем геоточки
    geo_points = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            # пропускаем пустые/короткие строки
            if not row or len(row) <= max(lon_col, lat_col, speed_col):
                continue
            lon = _to_float(row[lon_col])
            lat = _to_float(row[lat_col])
            speed = _to_float(row[speed_col])
            if lon is None or lat is None or speed is None:
                continue
            geo_points.append(dict(lon=lon, lat=lat, speed=speed))

    if not geo_points:
        # честно сообщаем о проблеме входных данных
        raise ValueError("Не найдено ни одной валидной строки c lon/lat/speed в заданных столбцах.")

    return geo_points


@dataclass(frozen=True)
class Plan:
    center_geo: GeoPoint
    waypoints_geo: tuple[GeoPoint, ...]
    polygon_points: tuple[GeoPoint, ...]
    raw_data: dict

    geo_transformer = GeoTransformer()

    def calculate_total_volume(self, norma: float) -> float:
        """
        Посчитает необходимый суммарный объём согласно норме внесения (л/Га)
        :param norma: норма внесения (л/Га)
        :return: суммарный объём в литрах
        """
        return self.total_area_in_ga * norma

    def to_xy(self, geo_point: GeoPoint) -> Point:
        return self.geo_transformer.geo_to_xy(center=self.center_geo, geo_point=geo_point)

    def to_geo(self, point: Point) -> GeoPoint:
        return self.geo_transformer.xy_to_geo(center=self.center_geo, point=point)

    def to_xy_points(self, geo_points: list[GeoPoint]) -> list[Point]:
        return [self.to_xy(geo_point=gp) for gp in geo_points]

    def to_geo_points(self, points: list[Point]) -> list[GeoPoint]:
        return [self.to_geo(point=p) for p in points]

    @functools.cached_property
    def waypoints_xy(self) -> list[Point]:
        return [self.to_xy(gp) for gp in self.waypoints_geo]

    @functools.cached_property
    def polygon(self) -> Polygon:
        return Polygon([self.to_xy(gp) for gp in self.polygon_points])

    @functools.cached_property
    def total_area(self) -> float:
        """Вернёт площадь полигона в квадратных метрах"""
        return self.polygon.area

    @functools.cached_property
    def total_distance(self) -> float:
        """Вернёт общую длину маршрута"""
        return polyline_lengths(self.waypoints_xy)[-1]

    @functools.cached_property
    def total_area_in_ga(self) -> float:
        """Вернёт площадь полигона в Гектарах"""
        return sqm_to_hectares(self.polygon.area)

    @classmethod
    def get_plan_from_log_file(cls, log_file_path: Path) -> Self:
        import json

        with open(log_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

            [center_geo] = parse_key_points_from_qgc_plan(
                raw_data['mission']['mission_plan'],
                include_home=True,
                include_takeoff=False,
                include_polygon=False,
                include_transects=False,
                deduplicate=False,
            )

            polygon_plan_points_geo = parse_key_points_from_qgc_plan(
                raw_data['mission']['mission_plan'],
                include_home=False,
                include_takeoff=False,
                include_polygon=True,
                include_transects=False,
                deduplicate=False,
            )

            waypoints_geo = parse_key_points_from_qgc_plan(
                raw_data['mission']['mission_plan'],
                include_home=False,
                include_takeoff=False,
                include_polygon=False,
                include_transects=True,
                deduplicate=False,
            )

        return cls(center_geo=center_geo,
                   waypoints_geo=waypoints_geo,
                   polygon_points=polygon_plan_points_geo,
                   raw_data=raw_data)


class Logs:
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path

        self.plan: Plan = Plan.get_plan_from_log_file(self.log_file_path)
        self._parse_flights()

    def get_waypoints_and_speed_list(self, index_fly=0):

        actual_points = self.flight_points[index_fly]['points']
        speeds = self.flight_points[index_fly]['speed']

        # TODO: хорошо бы более корректно определять точки входа и выхода с поля в случае когда у нас > 1 полёта
        # Находим точки начала и окончания на плане (нормально работает только на одном пролёте всего поля)
        start = np.argmin([dist(self.plan.waypoints_xy[0], p) for p in actual_points])
        end = np.argmin([dist(self.plan.waypoints_xy[-1], p) for p in actual_points]) + 1

        waypoints = actual_points[start:end]
        speed_list = speeds[start:end]

        return waypoints, speed_list

    def _parse_flights(self, vehicle_id=1):
        from matplotlib import pyplot as plt
        from plots import scatter_with_color_and_profile

        flights = self.plan.raw_data['flights']
        self.flight_points = []

        for f in flights:
            geo_points: list[GeoPoint] = []
            speed_list = []

            for s in f['samples']:
                if s['VehicleID'] != vehicle_id:
                    continue

                telemetry = s['telemetry']
                p = telemetry['coordinate']
                geo_points.append(GeoPoint(lat=p['lat'], lon=p['lon']))
                speed_list.append(telemetry['speed']['groundSpeed'])

            # переводим гео координаты в xy
            points: list[Point] = self.plan.to_xy_points(geo_points=geo_points)

            scatter_with_color_and_profile(points, speed_list)

            plt.show()
            self.flight_points.append({'points': points, 'speed': speed_list})

        if len(self.flight_points) > 1:
            print('[Warning] Корректная обработка полётов в количестве более 1 ещё не реализована')
