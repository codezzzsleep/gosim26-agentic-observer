"""Directional hazards from the bulletins: terrain obstruction, rocket launches, directional weather.

A notice only names an event kind and one of eight compass directions. The participant guide says what
that means: the label is the centre of a hidden azimuth sector with an altitude ceiling. This module turns
the labels into an expected share of the clear-sky completion rate for any (alt, az), and sharpens the
terrain sectors with the agent's own zero-score hits.
"""
from __future__ import annotations

import os

import numpy as np


def _env(name: str, default):
    return type(default)(os.environ.get(f"AO_{name}", default))


DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
CLOSING = {"rain", "storm"}                     # close the dome when they cover the whole sky
BLOCKING = {"rocket_launch"}                    # zero score inside the sector while active
WEATHER_FACTOR = {"overcast": 0.40, "haze": 0.55, "cold_snap": 0.92, "rain": 0.0, "storm": 0.0}
# sector geometry priors (label = sector centre, to the nearest 45 deg)
TERRAIN_AZ = (24.0, 58.0)       # surely inside below this distance from the label, surely outside beyond
TERRAIN_ALT = (36.0, 51.0)
ROCKET_AZ = (22.0, 64.0)
ROCKET_ALT = (32.0, 62.0)
WEATHER_AZ = (30.0, 84.0)
WEATHER_ALT = (42.0, 81.0)
# --- learning a blocked terrain sector from the agent's own zero-score evidence ---
TERRAIN_SPAN = _env("TERRAIN_SPAN", 1)          # 1: blocked points span a sector; 0: only the point's own patch (v1)
TERRAIN_EPISODES = _env("TERRAIN_EPISODES", 2)  # separate occasions needed before a whole sector is called blocked
TERRAIN_EPISODE_GAP = _env("TERRAIN_EPISODE_GAP", 6.0)   # hours of quiet after which the next evidence is a new occasion


def _episodes(times: np.ndarray, gap_hours: float) -> int:
    """How many separate occasions a set of evidence times stands for.

    One bad-weather spell puts many points in `blocked` within an hour; terrain blocks the same
    direction night after night. Counting the groups separated by a quiet spell separates the two."""
    if len(times) == 0:
        return 0
    ordered = np.sort(times)
    return 1 + int(np.count_nonzero(np.diff(ordered) > gap_hours * 3600.0))


def az_distance(az, centre: float):
    return np.abs((np.asarray(az, dtype=float) - centre + 180.0) % 360.0 - 180.0)


def _ramp(value, sure: float, never: float):
    """1 below `sure`, 0 above `never`, linear in between."""
    return np.clip((never - np.asarray(value, dtype=float)) / (never - sure), 0.0, 1.0)


class Hazards:
    def __init__(self, log=lambda text: None):
        self.log = log
        self.terrain: list[str] = []
        self.notices: set = set()                 # (kind, direction) in the latest bulletin
        self.during: set = set()                  # union of notices since the last decision
        self.extra_avoid: set = set()             # directions an adviser asked to avoid tonight
        self.blocked: list[tuple[float, float, float]] = []    # (az, alt, time) of zero-score hits beside positive ones
        self.clear: list[tuple[float, float]] = []      # (az, alt) of positive hits inside a suspect zone
        self.probe_avoid: list[tuple[float, float]] = []   # (az, alt) of an all-zero exposure being checked

    # --- messages ---------------------------------------------------------------------------------------

    def on_bulletins(self, bulletins: list, latest) -> None:
        self.during = set()
        for bulletin in bulletins:
            for notice in bulletin.get("notices", []):
                kind, direction = notice.get("event_kind", ""), notice.get("direction", "")
                if kind == "terrain_obstruction":
                    if direction in DIRECTION_AZ and direction not in self.terrain:
                        self.terrain.append(direction)
                else:
                    self.during.add((kind, direction))
        self.notices = {(n.get("event_kind", ""), n.get("direction", "")) for n in (latest or {}).get("notices", [])
                        if n.get("event_kind") != "terrain_obstruction"}
        self.during |= self.notices

    def site_closed(self) -> bool:
        return any(kind in CLOSING and direction == "ALL" for kind, direction in self.notices)

    # --- expected share of the clear-sky rate -----------------------------------------------------------

    def terrain_risk(self, alt, az):
        """Probability that (alt, az) is behind the terrain."""
        alt = np.asarray(alt, dtype=float)
        risk = np.zeros(alt.shape)
        for direction in self.terrain:
            dist = az_distance(az, DIRECTION_AZ[direction])
            risk = np.maximum(risk, _ramp(dist, *TERRAIN_AZ) * _ramp(alt, *TERRAIN_ALT))
        if self.blocked:
            b = np.array(self.blocked[-400:])            # (az, alt, t)
            # zero-score evidence. A terrain sector is one azimuth interval with one altitude ceiling, so the
            # blocked points near one announced direction span it: everything between them, up to the highest
            # of them, is blocked too. A single occasion can be local weather rather than terrain, so the
            # sector is only spanned once the same direction has been hit on separate occasions.
            used = np.zeros(len(b), dtype=bool)
            if TERRAIN_SPAN:
                for direction in self.terrain:
                    centre = DIRECTION_AZ[direction]
                    rel = (b[:, 0] - centre + 180.0) % 360.0 - 180.0
                    mine = np.abs(rel) <= TERRAIN_AZ[1] + 10.0
                    if not mine.any():
                        continue
                    if _episodes(b[mine, 2], TERRAIN_EPISODE_GAP) < TERRAIN_EPISODES:
                        continue        # one occasion can be local weather: keep the point's own patch only
                    used |= mine
                    lo, hi, ceiling = rel[mine].min() - 5.0, rel[mine].max() + 5.0, b[mine, 1].max() + 1.0
                    rel_q = (np.asarray(az, dtype=float) - centre + 180.0) % 360.0 - 180.0
                    risk = np.where((rel_q >= lo) & (rel_q <= hi) & (alt <= ceiling), 1.0, risk)
            for b_az, b_alt, _ in b[~used][-60:]:
                near = (az_distance(az, b_az) <= 6.0) & (alt <= b_alt + 1.0)
                risk = np.where(near, 1.0, risk)
        if self.clear:
            c = np.array(self.clear[-200:])
            for c_az, c_alt in c:
                near = (az_distance(az, c_az) <= 4.0) & (alt >= c_alt - 0.5)
                risk = np.where(near, np.minimum(risk, 0.05), risk)
        return risk

    def factor(self, alt, az, notices=None):
        """Expected share of the clear-sky completion rate at (alt, az) under the given notices."""
        alt = np.asarray(alt, dtype=float)
        factor = 1.0 - self.terrain_risk(alt, az)
        for kind, direction in sorted(self.notices if notices is None else notices):
            if direction not in DIRECTION_AZ:
                continue
            dist = az_distance(az, DIRECTION_AZ[direction])
            if kind in BLOCKING:
                inside = _ramp(dist, *ROCKET_AZ) * _ramp(alt, *ROCKET_ALT)
                factor = factor * (1.0 - inside)
            elif kind in WEATHER_FACTOR:
                inside = _ramp(dist, *WEATHER_AZ) * _ramp(alt, *WEATHER_ALT)
                factor = factor * (1.0 - inside * (1.0 - WEATHER_FACTOR[kind]))
        for p_az, p_alt in self.probe_avoid:
            factor = np.where((az_distance(az, p_az) <= 25.0) & (alt <= p_alt + 8.0), 0.0, factor)
        for direction in sorted(self.extra_avoid):
            if direction in DIRECTION_AZ:
                inside = _ramp(az_distance(az, DIRECTION_AZ[direction]), *WEATHER_AZ) * _ramp(alt, *WEATHER_ALT)
                factor = factor * (1.0 - 0.4 * inside)
        return factor

    def clean(self, alt, az):
        """True where no directional notice seen since the last decision could reach (alt, az)."""
        alt = np.asarray(alt, dtype=float)
        ok = np.ones(alt.shape, dtype=bool)
        for kind, direction in sorted(self.during):
            if direction not in DIRECTION_AZ or kind == "earthquake":
                continue
            limits = (ROCKET_AZ[1], ROCKET_ALT[1]) if kind in BLOCKING else (WEATHER_AZ[1], WEATHER_ALT[1])
            ok &= ~((az_distance(az, DIRECTION_AZ[direction]) <= limits[0] + 3.0) & (alt <= limits[1] + 2.0))
        for direction in self.terrain:
            ok &= ~((az_distance(az, DIRECTION_AZ[direction]) <= TERRAIN_AZ[1]) & (alt <= TERRAIN_ALT[1]))
        return ok

    # --- learning ---------------------------------------------------------------------------------------

    def on_zero_hit(self, alt: float, az: float, now: float = 0.0) -> None:
        """A hit that scored zero while others in the same exposure scored: something blocks this direction."""
        if any(kind in BLOCKING and d in DIRECTION_AZ for kind, d in self.during):
            return    # a launch explains it; that is temporary
        self.blocked.append((float(az), float(alt), float(now)))

    def block_field(self, points, now: float = 0.0) -> None:
        """A field that scored nothing while a probe elsewhere scored: the whole field is behind something."""
        for az, alt in points:
            self.blocked.append((float(az), float(alt), float(now)))

    def on_positive_hit(self, alt: float, az: float) -> None:
        if self.terrain and float(self.terrain_risk(np.array([alt]), np.array([az]))[0]) > 0.05:
            self.clear.append((float(az), float(alt)))
