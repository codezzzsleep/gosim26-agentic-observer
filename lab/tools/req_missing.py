import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the handoff folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # clone of gosimfoundation/hackathon-survey26

import csv, sys, collections, math
sys.path.insert(0, str(WORK / "agent"))
import numpy as np, skymath as sm
run, card = sys.argv[1], sys.argv[2]
T = {r["target_id"]: r for r in csv.DictReader(open(f"{card}/public/targets.csv"))}
dec = {d["observe_index"]: d for d in csv.DictReader(open(f"{run}/decisions.csv")) if d["action"] == "observe"}
att = collections.defaultdict(list)
for o in csv.DictReader(open(f"{run}/observations.csv")):
    if o["valid"] == "true": att[o["target_id"]].append((dec[o["observe_index"]]["start_utc"][5:16], int(dec[o["observe_index"]]["duration_seconds"]), float(o["factor"]), float(o["quality"])))
nights = list(csv.DictReader(open(f"{card}/public/v4_night_calendar.csv")))
site = sm.Site(-24.6157, -70.3976)
miss = [r for t, r in T.items() if r["required"] == "true" and max([a[2] for a in att.get(t, [])], default=0) < 0.5]
print(f"{run}: required missing {len(miss)}")
for r in sorted(miss, key=lambda r: float(r["feature_flux"])):
    ra, dec_ = float(r["ra_deg"]), float(r["dec_deg"]); hmax = float(site.max_hour_angle(dec_, 30.1))
    vis = []
    for n in nights:
        st, en = sm.unix(n["observing_start_utc"]), sm.unix(n["observing_end_utc"]); span = (en - st) * sm.SIDEREAL_DEG_PER_SECOND
        rise = (ra - hmax - site.lst(st)) % 360.0
        ov = max(0.0, min(span, rise + 2 * hmax) - rise) + max(0.0, min(span, rise - 360.0 + 2 * hmax))
        vis.append(ov / sm.SIDEREAL_DEG_PER_SECOND / 3600.0)
    f = float(r["feature_flux"]); tr_alt = 90 - abs(dec_ + 24.6157)
    print(f"  {r['target_id']} {r['target_class']:4s} flux {f:.3f} need(0.5,Q=1) {225/f:5.0f}s transit alt {tr_alt:.0f} vis h/night first {vis[0]:.1f} mid {vis[len(vis)//2]:.1f} last {vis[-1]:.1f} nights>=1h {sum(v>=1 for v in vis)} attempts {att.get(r['target_id'], [])[:4]}")
