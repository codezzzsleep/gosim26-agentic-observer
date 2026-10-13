#!/usr/bin/env python3
"""Replay the agent's sky tracker over a finished run (any agent's decisions) and compare with the truth.

    python3 tools/replay_tracker.py runs/v0/d101 d101 [--from 420 --to 460] [--agent-dir agent] [--quiet]

Uses the agent's own ingest() code path: the hits of each recorded exposure are fed back as last_result.
"""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the handoff folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # clone of gosimfoundation/hackathon-survey26

import argparse, csv, json, math, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("card"); ap.add_argument("--from", dest="lo", type=int, default=0)
ap.add_argument("--to", dest="hi", type=int, default=10**9); ap.add_argument("--agent-dir", default=str(WORK / "agent"))
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--policy", action="store_true", help="also run the report policy; stop at the first correct report")
a = ap.parse_args()
sys.path.insert(0, str(REPO)); sys.path.insert(0, a.agent_dir)
import numpy as np
import skymath as sm
from challenge.v4_workflow import V4Workflow
import agent as agent_module
agent_module.log = lambda text: None
card = [p for p in list((WORK / "cards").glob(f"*/{a.card}")) + [REPO / "examples/_local/cards" / a.card] if p.is_dir()][0]
init = V4Workflow(card).initialize_payload(900.0, 1.0)
ag = agent_module.Agent(init); s = ag.survey; tr = ag.tracker
run = Path(a.run)
dec = [d for d in csv.DictReader(open(run / "decisions.csv"))]
obs = {}
for o in csv.DictReader(open(run / "observations.csv")):
    obs.setdefault(int(o["observe_index"]), []).append(o)
bulletins = [json.loads(l) for l in open(card / "public/v4_bulletins.jsonl")]
btimes = [sm.unix(b["issued_at_utc"]) for b in bulletins]
truth = {r["timestamp_utc"]: r for r in csv.DictReader(open(card / "truth/v4_weather_truth.csv"))}
events = list(csv.DictReader(open(card / "truth/v4_events.csv")))
faults = [[sm.unix(e["actual_start_utc"]), float(e["instrument_efficiency_multiplier"]), False] for e in events if e["event_type"] == "instrument_fault"]
def active_fault(t):
    live = [f for f in faults if f[0] <= t]
    return live[-1] if live and not live[-1][2] else None
def true_state(start, end):
    t = math.floor(start / 900) * 900; wsum = esum = 0.0; open_s = 0.0
    while t < end:
        r = truth.get(sm.format_unix(t)); ov = min(end, t + 900) - max(start, t)
        if r and r["is_observable"] == "true":
            eff = float(r["instrument_efficiency"])
            for f in faults:
                if f[2] and f[0] <= t: eff = min(1.0, eff / f[1])
            wsum += ov * float(r["transparency"]) * float(r["sky_quality"]) / float(r["seeing_arcsec"]); esum += ov * eff; open_s += ov
        t += 900
    return wsum / (end - start), (esum / open_s if open_s else float("nan")), open_s / (end - start)
cursor = 0; last_result = None; rows_out = []; pending_meta = None
policy_reports = []
for d in dec + [None]:
    now = sm.unix(d["start_utc"]) if d else sm.unix(dec[-1]["end_utc"])
    new = []
    while cursor < len(bulletins) and btimes[cursor] <= now:
        new.append(bulletins[cursor]); cursor += 1
    payload = {"now_utc": sm.format_unix(now), "new_messages": new, "latest_bulletin": bulletins[cursor - 1] if cursor else None, "last_result": last_result}
    before = len(tr.fault_history)
    ag.ingest(payload)
    night = s.current_night(now)
    if night: ag.night_seen = night[0]
    if pending_meta is not None and len(tr.fault_history) > before:
        k, st, en, prog, n = pending_meta
        tw, te, to = true_state(st, en)
        pf = tr.fault_history[-1][1]; fo = active_fault(st) is not None
        rows_out.append((k, sm.format_unix(st), pf, fo))
        if a.lo <= k <= a.hi and not a.quiet:
            print(f"{k:4d} {sm.format_unix(st)[5:16]} T={int(en - st):4d} {prog:6s} n={n:2d} est W {tr.w_mean():.3f} eta {tr.eta_mean():.3f} pF {pf:.3f} | true W {tw:.3f} eta {te:.3f} open {to:.2f} fault {int(fo)} {sorted(ag.hazards.during)}")
    pending_meta = None
    if d is None: break
    if a.policy and night and not ag.hazards.site_closed():
        rep = ag._maybe_report((now - s.start) / 3600.0, now)
        if rep is not None:
            f = active_fault(now); correct = f is not None
            policy_reports.append((sm.format_unix(now), correct, sm.format_unix(f[0]) if f else None, round((now - f[0]) / 3600.0, 2) if f else None))
            ag._on_report_result({"correct": correct}, (now - s.start) / 3600.0)
            if correct:
                break
    if d["action"] == "report":
        f = active_fault(now); correct = f is not None
        if f: f[2] = True
        last_result = {"action": "report", "correct": correct, "repaired": correct, "score_delta": 0.0}
    elif d["action"] == "observe":
        k = int(d["observe_index"]); rows = obs.get(k, [])
        start = now; end = sm.unix(d["end_utc"])
        idx = np.array([s.index_of[r["target_id"]] for r in rows], dtype=int)
        if len(idx):
            h0 = s.site.horizontal(s.eq[idx], s.site.lst(start)); alt0, az0 = sm.unit_to_altaz(h0)
            ag.planner.pending = {"start": start, "duration": float(d["duration_seconds"]), "program": d["program"], "cmd": (float(d["alt_deg"]), float(d["az_deg"])),
                                  "idx": idx, "fibres": np.array([int(r["fiber_id"]) for r in rows]), "h0": h0, "alt0": alt0, "az0": az0,
                                  "dirf": ag.hazards.factor(alt0, az0), "raw": np.ones(len(idx)), "level": 1.0, "night_end": end}
            pending_meta = (k, start, end, d["program"], len(idx))
        last_result = {"action": "observe", "observe_index": k, "hits": [{"target_id": r["target_id"], "score": float(r["score"])} for r in rows]}
    else:
        last_result = {"action": "wait"}
fa = [x for x in rows_out if x[2] > 0.5 and not x[3]]
print(f"exposures used {len(rows_out)}; false-alarm exposures (pF>0.5, no fault) {len(fa)}: {[(k, t[5:16], round(p, 2)) for k, t, p, _ in fa[:10]]}")
for f in [[sm.unix(e['actual_start_utc'])] for e in events if e['event_type'] == 'instrument_fault']:
    on = [x for x in rows_out if sm.unix(x[1]) >= f[0] and x[3]]
    first = next((x for x in on if x[2] > 0.5), None)
    print("  fault onset", sm.format_unix(f[0])[5:16], "| exposures with fault live", len(on), "| first pF>0.5:", (first[1][5:16] + f" (#{on.index(first) + 1} after onset)") if first else None)

if a.policy:
    print("  policy reports:", policy_reports)
