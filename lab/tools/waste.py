#!/usr/bin/env python3
"""Where did the telescope time of a run go? (needs a run made with --keep)

    python3 tools/waste.py runs/TAG/CARD cards/SET/CARD

Fibre time is split into: empty fibres, exposures of targets that were later bettered or already better
(superseded), time beyond what the target needed to reach full completion (overexposed), and the useful rest.
Also: how much of the open sky time was used, how the best exposures split over the program multipliers, and
what each target class reached.
"""
import collections, csv, json, sys
from datetime import datetime

run, card = sys.argv[1], sys.argv[2]
fiber = json.load(open(f"{card}/config/v4_fiber_config.json"))
score = json.load(open(f"{card}/config/v4_score_config.json"))
n_fib = int(fiber["field"]["n_fibers"])
f0t0 = float(score["flux_zero_point"]) * float(score["exposure_zero_point_seconds"])
T = {r["target_id"]: r for r in csv.DictReader(open(f"{card}/public/targets.csv"))}
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

open_seconds = sum(int(s["duration_seconds"]) for s in csv.DictReader(open(f"{card}/truth/v4_weather_truth.csv")) if s["is_observable"] == "true")
dec = {}
waits = 0.0
for d in csv.DictReader(open(f"{run}/decisions.csv")):
    if d["action"] == "observe" and d["observe_index"] != "":
        dec[d["observe_index"]] = (int(d["duration_seconds"]), int(d["assigned_count"]), int(d["hit_count"]), ts(d["end_utc"]) - ts(d["start_utc"]))
obs = [o for o in csv.DictReader(open(f"{run}/observations.csv")) if o["valid"] == "true"]
best = {}
for o in obs:
    s = float(o["score"])
    if o["target_id"] not in best or s > float(best[o["target_id"]]["score"]):
        best[o["target_id"]] = o

exp_seconds = sum(v[3] for v in dec.values())
fib_total = exp_seconds * n_fib
fib_assigned = sum(v[3] * v[2] for v in dec.values())
useful = over = superseded = zero = 0.0
for o in obs:
    dur = dec[o["observe_index"]][3]
    if float(o["score"]) <= 0:
        zero += dur
        continue
    if best[o["target_id"]] is not o:
        superseded += dur
        continue
    flux = float(T[o["target_id"]]["feature_flux"])
    raw = flux * dur * float(o["quality"]) / f0t0
    need = dur / raw if raw > 1.0 else dur
    useful += need
    over += dur - need
print(f"card {card.rstrip('/').split('/')[-1]}: {n_fib} fibres, {len(T)} targets, open sky {open_seconds / 3600:.0f} h, exposed {exp_seconds / 3600:.0f} h "
      f"({100 * exp_seconds / max(open_seconds, 1):.0f}% of open), {len(dec)} exposures, mean {exp_seconds / max(len(dec), 1):.0f} s")
print(f"fibre time {fib_total / 3600:.0f} fibre-hours:")
for label, value in (("useful", useful), ("overexposed (beyond full completion)", over), ("superseded (a better exposure exists)", superseded),
                     ("scored nothing (closed dome / blocked)", zero), ("empty fibres", fib_total - fib_assigned)):
    print(f"  {100 * value / max(fib_total, 1):5.1f}%  {label}")
print(f"  mean fibres with a target: {fib_assigned / max(exp_seconds, 1):.1f} of {n_fib}")

mult = collections.Counter(); wsum = 0.0; got = 0.0; possible = 0.0
agg = collections.defaultdict(collections.Counter)
req_missing = 0
for t, r in T.items():
    w = float(r["science_weight"]); c = r["target_class"]; a = agg[c]
    a["n"] += 1; a["max"] += w * 1.2
    possible += w * 1.2
    b = best.get(t)
    f = float(b["factor"]) if b else 0.0
    if r["required"] == "true" and f < 0.5:
        req_missing += 1
    if not b or float(b["score"]) <= 0:
        a["none"] += 1
        continue
    got += float(b["score"]); a["score"] += float(b["score"]); a["factor"] += f
    a["full"] += f >= 0.97; a["part"] += f < 0.97
    mult[b["prog_mult"][:4]] += w * f; wsum += w * f
print(f"target scores {got:.0f} of {possible:.0f} possible ({100 * got / possible:.1f}%); must-observe missing {req_missing}")
print("best exposures by program multiplier (share of weighted completion): " + ", ".join(f"x{k} {100 * v / max(wsum, 1e-9):.0f}%" for k, v in sorted(mult.items(), reverse=True)))
print("class      n   full  partial   none  score/max  median flux")
for c, a in sorted(agg.items(), key=lambda kv: -kv[1]["max"]):
    fl = sorted(float(r["feature_flux"]) for r in T.values() if r["target_class"] == c)
    print(f"{c:5s} {a['n']:6d}  {a['full'] / a['n']:5.2f}  {a['part'] / a['n']:7.2f}  {a['none'] / a['n']:5.2f}  {a['score'] / a['max']:9.2f}  {fl[len(fl) // 2]:10.2f}")
