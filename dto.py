from typing import NamedTuple, Optional


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
