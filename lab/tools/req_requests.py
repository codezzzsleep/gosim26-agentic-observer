import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the handoff folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # clone of gosimfoundation/hackathon-survey26

import csv, sys, json, collections
sys.path.insert(0, str(WORK / "agent"))
import numpy as np, skymath as sm
run, card = sys.argv[1], sys.argv[2]
T = {r["target_id"]: r for r in csv.DictReader(open(f"{card}/public/targets.csv"))}
dec = {d["observe_index"]: d for d in csv.DictReader(open(f"{run}/decisions.csv")) if d["action"] == "observe"}
obs = collections.defaultdict(list)
for o in csv.DictReader(open(f"{run}/observations.csv")):
    d = dec[o["observe_index"]]; obs[o["target_id"]].append((d["start_utc"], d["end_utc"], int(d["duration_seconds"]), float(o["factor"]), float(o["quality"])))
site = sm.Site(-24.6157, -70.3976)
truth = {r["timestamp_utc"]: r for r in csv.DictReader(open(f"{card}/truth/v4_weather_truth.csv"))}
nights = list(csv.DictReader(open(f"{card}/public/v4_night_calendar.csv")))
for line in open(f"{card}/truth/v4_observation_requests.jsonl"):
    r = json.loads(line); iss, dl = r["issued_at_utc"], r["deadline_utc"]
    done = 0; rows = []
    for t in r["target_ids"]:
        inw = [(a[0][5:16], a[2], round(a[3], 2)) for a in obs.get(t, []) if a[0] >= iss and a[1] <= dl]
        ok = any(a[2] >= 0.5 for a in inw); done += ok
        # visible hours within window nights
        vis = 0.0
        for n in nights:
            st, en = n["observing_start_utc"], n["observing_end_utc"]
            if en <= iss or st >= dl: continue
            u = sm.unix(max(st, iss)); e = sm.unix(min(en, dl))
            while u < e:
                alt, az = site.altaz_of(float(T[t]["ra_deg"]), float(T[t]["dec_deg"]), site.lst(u))
                tr = truth.get(sm.format_unix(u - u % 900))
                if alt >= 30.2 and tr and tr["is_observable"] == "true": vis += 0.25
                u += 900
        rows.append(f"    {t} flux {float(T[t]['feature_flux']):.2f} open-visible {vis:.1f}h in-window obs {inw} {'OK' if ok else '--'}")
    print(f"{r['request_id']} {iss[5:16]} -> {dl[5:16]} need {r['minimum_completed']} done {done} {'COMPLETED' if done >= r['minimum_completed'] else 'MISSED'}")
    if done < r["minimum_completed"]: print("\n".join(rows))
