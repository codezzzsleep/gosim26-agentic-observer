import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import sys, random, math
sys.path.insert(0, str(_Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else WORK / "agent")); sys.path.insert(0, str(REPO))      # optional argument: agent folder
import numpy as np
from datetime import datetime, timezone, timedelta
import skymath as sm
from challenge import v4_fiber_map as fm
from challenge.v4_scorer import lunar_quality_factor
from challenge.tile_geometry_simulator import _normalized_airmass
lat, lon = -24.6157, -70.3976
site = sm.Site(lat, lon)
rng = random.Random(1)
grid = fm.FiberGrid(0.4, 0.0, 16)
cfg = {"lunar_model": {"angular_decay_scale_deg": 35.0, "altitude_exponent": 1.0, "maximum_penalty": 0.75}}
worst = [0, 0, 0, 0, 0]
mism = 0; tot = 0
for _ in range(300):
    t = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.uniform(0, 90 * 86400))
    u = t.timestamp()
    ras = np.array([rng.uniform(0, 360) for _ in range(40)]); decs = np.array([rng.uniform(-85, 30) for _ in range(40)])
    eq = sm.radec_unit(ras, decs)
    h = site.horizontal(eq, site.lst(u))
    alt, az = sm.unit_to_altaz(h)
    moon = sm.Moon(u, site, cfg["lunar_model"]); lf = moon.lunar_factor(eq)
    c_alt, c_az = rng.uniform(30, 89.9), rng.uniform(0, 360)
    da, dz = sm.project(h, c_alt, c_az)
    for k in range(40):
        ea, ez = fm.radec_to_altaz(ras[k], decs[k], t, lat, lon)
        worst[0] = max(worst[0], abs(ea - alt[k])); worst[1] = max(worst[1], abs(((ez - az[k] + 180) % 360) - 180) * math.cos(math.radians(ea)))
        worst[2] = max(worst[2], abs(lunar_quality_factor(ras[k], decs[k], t, {"latitude_deg": lat, "longitude_deg": lon}, cfg) - lf[k]))
        if ea > 1: worst[3] = max(worst[3], abs(_normalized_airmass(ea) - float(sm.normalized_airmass(alt[k]))))
        off = fm.FiberGrid.tangent_offsets(ea, ez, c_alt, c_az)
        if off is not None and abs(off[0]) < 5 and abs(off[1]) < 5:
            worst[4] = max(worst[4], abs(off[0] - da[0, k]), abs(off[1] - dz[0, k]))
print("max errors: alt %.2e az %.2e lunar %.2e airmass %.2e tangent %.2e" % tuple(worst))
