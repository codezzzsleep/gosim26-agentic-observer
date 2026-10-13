"""Sky geometry, vectorised with numpy. Same public formulas as the participant guide (sections 2-3):
sidereal time (1), alt/az (2)-(4), Sun (5)-(7), Moon (8)-(14), airmass (15), gnomonic fibre plane (16).

Angles are degrees at the interface; azimuth 0 = north, 90 = east. Horizontal unit vectors use
x = north, y = east, z = up.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0
D2R = math.pi / 180.0
R2D = 180.0 / math.pi


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def format_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unix(value: str) -> float:
    return parse_utc(value).timestamp()


def format_unix(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def days_j2000(unix_seconds: float) -> float:
    return unix_seconds / 86400.0 + 2440587.5 - 2451545.0


def lst_deg(unix_seconds: float, longitude_deg: float) -> float:
    return (280.46061837 + 360.98564736629 * days_j2000(unix_seconds) + longitude_deg) % 360.0


def normalized_airmass(alt_deg):
    """Formula (15); works on arrays. Altitudes at or below 0 give +inf."""
    alt = np.asarray(alt_deg, dtype=float)
    zenith = 90.0 - np.maximum(alt, 0.01)
    raw = 1.0 / (np.cos(zenith * D2R) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    return raw * (1.0 + 0.50572 * 96.07995 ** -1.6364)


def sun_radec(unix_seconds: float) -> tuple[float, float]:
    d = days_j2000(unix_seconds)
    mean_longitude = (280.460 + 0.9856474 * d) % 360.0
    g = math.radians((357.528 + 0.9856003 * d) % 360.0)
    lon = math.radians((mean_longitude + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360.0)
    eps = math.radians(23.439 - 0.0000004 * d)
    return (math.degrees(math.atan2(math.cos(eps) * math.sin(lon), math.cos(lon))) % 360.0,
            math.degrees(math.asin(math.sin(eps) * math.sin(lon))))


def moon_radec(unix_seconds: float) -> tuple[float, float]:
    d = days_j2000(unix_seconds)
    mean_longitude = math.radians((218.316 + 13.176396 * d) % 360.0)
    anomaly = math.radians((134.963 + 13.064993 * d) % 360.0)
    arg_latitude = math.radians((93.272 + 13.229350 * d) % 360.0)
    lon = mean_longitude + math.radians(6.289) * math.sin(anomaly)
    lat = math.radians(5.128) * math.sin(arg_latitude)
    eps = math.radians(23.439 - 0.0000004 * d)
    x = math.cos(lon) * math.cos(lat)
    y = math.sin(lon) * math.cos(lat) * math.cos(eps) - math.sin(lat) * math.sin(eps)
    z = math.sin(lon) * math.cos(lat) * math.sin(eps) + math.sin(lat) * math.cos(eps)
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(z))


def radec_unit(ra_deg, dec_deg) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=float) * D2R
    dec = np.asarray(dec_deg, dtype=float) * D2R
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


class Site:
    """Fixed observatory: converts equatorial unit vectors to horizontal ones with one 3x3 matrix per instant."""

    def __init__(self, latitude_deg: float, longitude_deg: float):
        self.lat = float(latitude_deg)
        self.lon = float(longitude_deg)
        self.sin_lat = math.sin(self.lat * D2R)
        self.cos_lat = math.cos(self.lat * D2R)

    def lst(self, unix_seconds: float) -> float:
        return lst_deg(unix_seconds, self.lon)

    def rotation(self, lst_degrees: float) -> np.ndarray:
        """R with  horizontal(north, east, up) = R @ equatorial(x, y, z)."""
        c, s = math.cos(lst_degrees * D2R), math.sin(lst_degrees * D2R)
        sl, cl = self.sin_lat, self.cos_lat
        # north = sin(dec) cos(lat) - cos(dec) cos(H) sin(lat);  east = -cos(dec) sin(H);  up = sin(lat) sin(dec) + cos(lat) cos(dec) cos(H)
        # with cos(dec) cos(H) = c * x + s * y  and  cos(dec) sin(H) = s * x - c * y
        return np.array([[-sl * c, -sl * s, cl],
                         [-s, c, 0.0],
                         [cl * c, cl * s, sl]])

    def horizontal(self, equatorial: np.ndarray, lst_degrees: float) -> np.ndarray:
        return equatorial @ self.rotation(lst_degrees).T

    def altaz_of(self, ra_deg: float, dec_deg: float, lst_degrees: float) -> tuple[float, float]:
        v = self.horizontal(radec_unit(ra_deg, dec_deg), lst_degrees)
        return math.degrees(math.asin(max(-1.0, min(1.0, float(v[2]))))), math.degrees(math.atan2(v[1], v[0])) % 360.0

    def max_hour_angle(self, dec_deg, alt_limit_deg: float) -> np.ndarray:
        """Largest |hour angle| (deg) with altitude >= limit: 0 = never up, 180 = always up."""
        dec = np.asarray(dec_deg, dtype=float) * D2R
        denom = self.cos_lat * np.cos(dec)
        value = (math.sin(alt_limit_deg * D2R) - self.sin_lat * np.sin(dec)) / np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))


def altaz_unit(alt_deg, az_deg) -> np.ndarray:
    alt = np.asarray(alt_deg, dtype=float) * D2R
    az = np.asarray(az_deg, dtype=float) * D2R
    return np.stack([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)], axis=-1)


def unit_to_altaz(v: np.ndarray):
    alt = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    return alt, az


def frames(alt_deg, az_deg):
    """Centre, up (+alt) and right (+az) unit vectors of pointings; each (..., 3)."""
    alt = np.asarray(alt_deg, dtype=float) * D2R
    az = np.asarray(az_deg, dtype=float) * D2R
    ca, sa, cz, sz = np.cos(alt), np.sin(alt), np.cos(az), np.sin(az)
    centre = np.stack([ca * cz, ca * sz, sa], axis=-1)
    up = np.stack([-sa * cz, -sa * sz, ca], axis=-1)
    right = np.stack([-sz, cz, np.zeros_like(sz)], axis=-1)
    return centre, up, right


def project(targets_h: np.ndarray, alt_deg, az_deg):
    """Gnomonic offsets (d_alt, d_az) in degrees of horizontal unit vectors (M, 3) around K pointings.

    Returns two (K, M) arrays; offsets behind the tangent plane are +inf."""
    centre, up, right = frames(np.atleast_1d(alt_deg), np.atleast_1d(az_deg))
    depth = centre @ targets_h.T
    safe = np.where(depth > 1e-9, depth, np.nan)
    d_alt = (up @ targets_h.T) / safe * R2D
    d_az = (right @ targets_h.T) / safe * R2D
    return np.nan_to_num(d_alt, nan=np.inf), np.nan_to_num(d_az, nan=np.inf)


class Moon:
    """Moon and Sun at one instant; lunar_factor follows formula (14) with the card's lunar_model."""

    def __init__(self, unix_seconds: float, site: Site, lunar_model: dict):
        ra, dec = moon_radec(unix_seconds)
        sun_ra, sun_dec = sun_radec(unix_seconds)
        self.vec = radec_unit(ra, dec)
        sun = radec_unit(sun_ra, sun_dec)
        self.illumination = (1.0 - float(np.clip(self.vec @ sun, -1.0, 1.0))) / 2.0
        self.alt = float(np.degrees(np.arcsin(np.clip(site.horizontal(self.vec, site.lst(unix_seconds))[2], -1.0, 1.0))))
        self.max_penalty = float(lunar_model["maximum_penalty"])
        self.alt_exp = float(lunar_model["altitude_exponent"])
        self.decay = float(lunar_model["angular_decay_scale_deg"])
        self.strength = (self.max_penalty * self.illumination * math.sin(self.alt * D2R) ** self.alt_exp) if self.alt > 0.0 else 0.0

    def lunar_factor(self, equatorial: np.ndarray) -> np.ndarray:
        if self.strength <= 0.0:
            return np.ones(equatorial.shape[:-1])
        separation = np.degrees(np.arccos(np.clip(equatorial @ self.vec, -1.0, 1.0)))
        return np.clip(1.0 - self.strength * np.exp(-separation / self.decay), 0.0, 1.0)
