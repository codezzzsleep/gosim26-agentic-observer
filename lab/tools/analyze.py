#!/usr/bin/env python3
"""Loss breakdown of one run directory against its card."""
import csv, json, sys, collections
run, card = sys.argv[1], sys.argv[2]
T = {r["target_id"]: r for r in csv.DictReader(open(f"{card}/public/targets.csv"))}
obs = list(csv.DictReader(open(f"{run}/observations.csv")))
dec = [d for d in csv.DictReader(open(f"{run}/decisions.csv"))]
dur = {int(d["observe_index"]): int(d["duration_seconds"]) for d in dec if d["action"] == "observe"}
nobs = sum(1 for d in dec if d["action"] == "observe")
tot_t = sum(dur.values())
slots = list(csv.DictReader(open(f"{card}/truth/v4_weather_truth.csv")))
open_s = sum(900 for s in slots if s["is_observable"] == "true")
best = {}; bestf = {}; fib_sec = 0; used = collections.Counter()
per = collections.defaultdict(list)
for o in obs:
    if o["valid"] != "true": continue
    t = o["target_id"]; s = float(o["score"]); f = float(o["factor"])
    per[t].append((int(o["observe_index"]), f, s, float(o["prog_mult"]), float(o["quality"])))
    fib_sec += dur[int(o["observe_index"])]
sumw = sum(float(r["science_weight"]) for r in T.values())
got = 0; lost_unobs = 0; lost_partial = 0; lost_mult = 0; wasted = 0; over = 0; useful = 0
mults = collections.Counter(); nsat = 0
for t, r in T.items():
    w = float(r["science_weight"]); fl = float(r["feature_flux"])
    if t not in per:
        lost_unobs += w * 1.2; continue
    es = per[t]; b = max(es, key=lambda e: e[2])
    got += b[2]
    lost_partial += w * (1 - b[1]) * 1.2
    lost_mult += w * b[1] * (1.2 - b[3])
    mults[b[3]] += 1; nsat += b[1] >= 0.999
    for e in es:
        d = dur[e[0]]
        if e is not b: wasted += d
        else:
            # over-exposure: time beyond saturation
            if e[1] >= 0.999 and e[4] > 0:
                need = 450.0 / (fl * e[4]); over += max(0, d - need); useful += min(d, need)
            else: useful += d
print(f"total night open seconds {open_s}  exposure seconds {tot_t} ({tot_t/open_s:.1%})  observes {nobs} meanT {tot_t/max(1,nobs):.0f}")
print(f"fibre-seconds: capacity {16*open_s/1e6:.2f}M  assigned-hit {fib_sec/1e6:.2f}M ({fib_sec/(16*open_s):.1%})  useful {useful/1e6:.2f}M  superseded {wasted/1e6:.2f}M  over-saturation {over/1e6:.2f}M")
print(f"max {sumw*1.2:.0f}  got {got:.0f}  lost: unobserved {lost_unobs:.0f}  partial {lost_partial:.0f}  multiplier {lost_mult:.0f}")
print("targets", len(T), "observed", len(per), "saturated", nsat, "best mult", dict(sorted(mults.items())))
rep = json.load(open(f"{run}/score_report.json")); print("components", rep["components"], "total", rep["total"])
