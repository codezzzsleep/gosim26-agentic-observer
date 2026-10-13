import csv, sys, collections, datetime
run, card = sys.argv[1], sys.argv[2]
truth = {r["timestamp_utc"]: r["is_observable"] for r in csv.DictReader(open(f"{card}/truth/v4_weather_truth.csv"))}
dec = {int(d["observe_index"]): d for d in csv.DictReader(open(f"{run}/decisions.csv")) if d["action"] == "observe"}
def open_frac(d):
    t = datetime.datetime.fromisoformat(d["start_utc"].replace("Z", "+00:00")); e = datetime.datetime.fromisoformat(d["end_utc"].replace("Z", "+00:00"))
    tot = op = 0; c = t.replace(minute=t.minute // 15 * 15, second=0)
    while c < e:
        ov = (min(e, c + datetime.timedelta(minutes=15)) - max(t, c)).total_seconds(); tot += ov
        if truth.get(c.strftime("%Y-%m-%dT%H:%M:%SZ")) == "true": op += ov
        c += datetime.timedelta(minutes=15)
    return op / max(1, tot)
zero_open = zero_closed = 0; fs_zero_open = fs_closed = 0; n = 0; fs = 0
exp_closed = 0; t_closed = 0
by_obs = collections.defaultdict(list)
for o in csv.DictReader(open(f"{run}/observations.csv")): by_obs[int(o["observe_index"])].append(float(o["score"]))
for k, d in dec.items():
    f = open_frac(d); T = int(d["duration_seconds"]); sc = by_obs.get(k, [])
    n += len(sc); fs += len(sc) * T
    z = sum(1 for x in sc if x == 0)
    if f < 0.02: zero_closed += z; fs_closed += z * T; exp_closed += 1; t_closed += T
    else: zero_open += z; fs_zero_open += z * T
    if f < 0.98 and f >= 0.02: fs_closed += len(sc) * T * (1 - f)
miss = sum(int(d["assigned_count"]) - int(d["hit_count"]) for d in dec.values())
print(f"{run}: hits {n}, zero-score while dome open {zero_open} ({fs_zero_open/fs:.2%} of fibre-time), exposures fully in closed dome {exp_closed} ({t_closed}s), fibre-time lost to closures {fs_closed/fs:.2%}, misses {miss}")
