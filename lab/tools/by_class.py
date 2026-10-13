import csv, sys, collections
run, card = sys.argv[1], sys.argv[2]
T = {r["target_id"]: r for r in csv.DictReader(open(f"{card}/public/targets.csv"))}
best = collections.defaultdict(lambda: [0.0, 0.0]); nobs = collections.Counter(); dur = collections.defaultdict(float)
dec = {d["observe_index"]: int(d["duration_seconds"]) for d in csv.DictReader(open(f"{run}/decisions.csv")) if d["action"] == "observe"}
for o in csv.DictReader(open(f"{run}/observations.csv")):
    if o["valid"] != "true": continue
    b = best[o["target_id"]]; b[0] = max(b[0], float(o["factor"])); b[1] = max(b[1], float(o["score"])); nobs[o["target_id"]] += 1; dur[o["target_id"]] += dec[o["observe_index"]]
agg = collections.defaultdict(lambda: collections.Counter())
for t, r in T.items():
    c = r["target_class"]; f, s = best[t]; a = agg[c]; w = float(r["science_weight"])
    a["n"] += 1; a["sat"] += f >= 0.97; a["zero"] += f == 0; a["partial"] += 0 < f < 0.97; a["factor"] += f; a["score"] += s; a["max"] += w * 1.2
    a["nobs"] += nobs[t]; a["dur"] += dur[t]; a["need"] += 450 / float(r["feature_flux"])
print("class     n   sat%  partial%  none%  mean factor  score/max  obs/target  fibre-h spent  (need@Q1 h)")
for c, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
    print(f"{c:5s} {a['n']:5d}  {a['sat']/a['n']:5.2f}  {a['partial']/a['n']:7.2f}  {a['zero']/a['n']:5.2f}  {a['factor']/a['n']:10.2f}  {a['score']/a['max']:9.2f}  {a['nobs']/a['n']:9.2f}  {a['dur']/3600:11.0f}  {a['need']/3600:10.0f}")
