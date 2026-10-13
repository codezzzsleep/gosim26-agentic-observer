import csv, sys
card = sys.argv[1]; repair = len(sys.argv) > 2
T = list(csv.DictReader(open(f"{card}/public/targets.csv")))
ev = list(csv.DictReader(open(f"{card}/truth/v4_events.csv")))
faults = [(e["actual_start_utc"], float(e["instrument_efficiency_multiplier"])) for e in ev if e["event_type"] == "instrument_fault"]
cap = 0.0; open_s = 0
for s in csv.DictReader(open(f"{card}/truth/v4_weather_truth.csv")):
    if s["is_observable"] != "true": continue
    eff = float(s["instrument_efficiency"])
    if repair:
        for st, mu in faults:
            if s["timestamp_utc"] >= st: eff = min(1.0, eff / mu)
    q = eff * float(s["transparency"]) * float(s["sky_quality"]) / float(s["seeing_arcsec"]) / 0.68
    cap += 900 * 16 * q; open_s += 900
for geo in (1.0, 0.9, 0.8, 0.7, 0.6):
    c = cap * geo; val = 0.0
    for r in sorted(T, key=lambda r: -float(r["science_weight"]) * float(r["feature_flux"])):
        need = 450 / float(r["feature_flux"]); w = float(r["science_weight"]) * 1.2
        if c >= need: c -= need; val += w
        else: val += w * c / need; break
    print(f"geo/airmass/fill factor {geo:.1f}: fluid bound {val:.0f}")
print("open hours", open_s / 3600, "mean Q", cap / (open_s * 16))
