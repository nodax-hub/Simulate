import math
from abc import ABC, abstractmethod

from dto import Point, GeoPoint


class GeoProjectionStrategy(ABC):
    @abstractmethod
    def geo_to_xy(self, center: GeoPoint, geo_point: GeoPoint) -> Point:
        pass

    @abstractmethod
    def xy_to_geo(self, center: GeoPoint, point: Point) -> GeoPoint:
        pass


class EquidistantProjection(GeoProjectionStrategy):
    R = 6378137.0  # м, WGS84

    def geo_to_xy(self, center: GeoPoint, geo_point: GeoPoint) -> Point:
        lon0, lat0 = math.radians(center.lon), math.radians(center.lat)
        lon, lat = math.radians(geo_point.lon), math.radians(geo_point.lat)

        x = self.R * (lon - lon0) * math.cos(lat0)
        y = self.R * (lat - lat0)
        return Point(x=x, y=y)

    def xy_to_geo(self, center: GeoPoint, point: Point) -> GeoPoint:
        lon0, lat0 = math.radians(center.lon), math.radians(center.lat)

        cos_lat0 = math.cos(lat0)
        eps = 1e-12
        if abs(cos_lat0) < eps:
            raise ValueError("cos(lat0) ≈ 0 — неустойчиво на широтах, близких к ±90°.")

        lon = point.x / (self.R * cos_lat0) + lon0
        lat = point.y / self.R + lat0
        return GeoPoint(lon=math.degrees(lon), lat=math.degrees(lat))


class AzimuthalEquidistantProjection(GeoProjectionStrategy):
    def geo_to_xy(self, center: GeoPoint, geo_point: GeoPoint) -> Point:
        from pyproj import CRS, Transformer
        wgs84 = CRS.from_epsg(4326)
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center.lat} +lon_0={center.lon} +x_0=0 +y_0=0 "
            f"+datum=WGS84 +units=m +no_defs"
        )
        tr = Transformer.from_crs(wgs84, aeqd, always_xy=True)
        x, y = tr.transform(geo_point.lon, geo_point.lat)
        return Point(x=float(x), y=float(y))

    def xy_to_geo(self, center: GeoPoint, point: Point) -> GeoPoint:
        from pyproj import CRS, Transformer
        wgs84 = CRS.from_epsg(4326)
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={center.lat} +lon_0={center.lon} +x_0=0 +y_0=0 "
            f"+datum=WGS84 +units=m +no_defs"
        )
        tr = Transformer.from_crs(aeqd, wgs84, always_xy=True)
        lon, lat = tr.transform(point.x, point.y)
        return GeoPoint(lon=float(lon), lat=float(lat))


class UTMProjection(GeoProjectionStrategy):
    @staticmethod
    def utm_zone_from_lon(lon: float) -> int:
        return int((lon + 180) // 6) + 1

    def geo_to_xy(self, center: GeoPoint, geo_point: GeoPoint) -> Point:
        from pyproj import CRS, Transformer
        zone = self.utm_zone_from_lon(center.lon)
        epsg = (32600 if center.lat >= 0.0 else 32700) + zone
        wgs84 = CRS.from_epsg(4326)
        utm = CRS.from_epsg(epsg)
        tr = Transformer.from_crs(wgs84, utm, always_xy=True)
        x, y = tr.transform(geo_point.lon, geo_point.lat)
        return Point(x=float(x), y=float(y))

    def xy_to_geo(self, center: GeoPoint, point: Point) -> GeoPoint:
        from pyproj import CRS, Transformer
        zone = self.utm_zone_from_lon(center.lon)
        epsg = (32600 if center.lat >= 0.0 else 32700) + zone
        wgs84 = CRS.from_epsg(4326)
        utm = CRS.from_epsg(epsg)
        tr = Transformer.from_crs(utm, wgs84, always_xy=True)
        lon, lat = tr.transform(point.x, point.y)
        return GeoPoint(lon=float(lon), lat=float(lat))


class GeoTransformer:
    def __init__(self, strategy: GeoProjectionStrategy = EquidistantProjection()):
        self._strategy = strategy

    def set_strategy(self, strategy: GeoProjectionStrategy):
        self._strategy = strategy

    def geo_to_xy(self, center: GeoPoint, geo_point: GeoPoint) -> Point:
        return self._strategy.geo_to_xy(center, geo_point)

    def xy_to_geo(self, center: GeoPoint, point: Point) -> GeoPoint:
        return self._strategy.xy_to_geo(center, point)
