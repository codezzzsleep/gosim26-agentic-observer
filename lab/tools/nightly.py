#!/usr/bin/env python3
"""Per-night yield of a run: exposures, fill, new score, true mean quality."""
import csv, sys, collections
run, card = sys.argv[1], sys.argv[2]
dec = {int(d["observe_index"]): d for d in csv.DictReader(open(f"{run}/decisions.csv")) if d["action"] == "observe"}
T = {r["target_id"]: r for r in csv.DictReader(open(f"{card}/public/targets.csv"))}
best = collections.defaultdict(float); rows = collections.OrderedDict()
slots = collections.defaultdict(lambda: [0, 0, 0.0])
for s in csv.DictReader(open(f"{card}/truth/v4_weather_truth.csv")):
    n = s["night_id"]; slots[n][0] += 1
    if s["is_observable"] == "true":
        slots[n][1] += 1; slots[n][2] += float(s["instrument_efficiency"]) * float(s["transparency"]) * float(s["sky_quality"]) / float(s["seeing_arcsec"])
night_of = {}
for s in csv.DictReader(open(f"{card}/truth/v4_slots.csv")): night_of[s["timestamp_utc"][:13]] = s["night_id"]
agg = collections.OrderedDict()
obs_by = collections.defaultdict(list)
for o in csv.DictReader(open(f"{run}/observations.csv")): obs_by[int(o["observe_index"])].append(o)
for k in sorted(dec):
    d = dec[k]; n = night_of.get(d["start_utc"][:13], "?")
    a = agg.setdefault(n, {"n": 0, "T": 0, "fib": 0, "gain": 0.0, "sat": 0, "hits": 0, "redo": 0})
    a["n"] += 1; a["T"] += int(d["duration_seconds"])
    for o in obs_by.get(k, []):
        if o["valid"] != "true": continue
        s = float(o["score"]); t = o["target_id"]; a["hits"] += 1; a["fib"] += int(d["duration_seconds"])
        if best[t] > 0: a["redo"] += 1
        if s > best[t]: a["gain"] += s - best[t]; best[t] = s
        a["sat"] += float(o["factor"]) >= 0.999
print("night      open  A_true  exp  meanT  fill  redo%  sat%   gain  gain/h   cum")
cum = 0
for n, a in agg.items():
    so = slots[n]; cum += a["gain"]
    print(f"{n} {so[1]:3d}/{so[0]:2d} {so[2]/max(1,so[1]):6.2f} {a['n']:4d} {a['T']/max(1,a['n']):6.0f} {a['fib']/max(1,a['T'])/16:5.2f} {a['redo']/max(1,a['hits']):5.2f} {a['sat']/max(1,a['hits']):5.2f} {a['gain']:6.0f} {a['gain']/max(1,a['T'])*3600:6.1f} {cum:6.0f}")
