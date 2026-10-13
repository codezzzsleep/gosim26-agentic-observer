"""Directional hazards from the bulletins: terrain obstruction, rocket launches, directional weather.

A notice only names an event kind and one of eight compass directions. The participant guide says what
that means: the label is the centre of a hidden azimuth sector with an altitude ceiling. This module turns
the labels into an expected share of the clear-sky completion rate for any (alt, az), and sharpens the
terrain sectors with the agent's own zero-score hits.
"""
from __future__ import annotations

import numpy as np

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
        self.blocked: list[tuple[float, float]] = []    # (az, alt) of zero-score hits beside positive ones
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
        if self.blocked and len(self.terrain):
            b = np.array(self.blocked)
            # zero-score evidence: anything near a blocked azimuth and not higher than it is blocked
            for b_az, b_alt in b[-60:]:
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

    def on_zero_hit(self, alt: float, az: float) -> None:
        """A hit that scored zero while others in the same exposure scored: something blocks this direction."""
        if any(kind in BLOCKING and d in DIRECTION_AZ for kind, d in self.during):
            return    # a launch explains it; that is temporary
        self.blocked.append((float(az), float(alt)))

    def on_positive_hit(self, alt: float, az: float) -> None:
        if self.terrain and float(self.terrain_risk(np.array([alt]), np.array([az]))[0]) > 0.05:
            self.clear.append((float(az), float(alt)))
