#!/usr/bin/env python3
"""Run an agent in-process against a card with the official engine (fast iteration, profiling).

    python3 tools/inproc.py --agent-dir agent --card d101 [--max-decisions N] [--profile] [--out runs/x]
"""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import argparse, cProfile, json, os, pstats, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--agent-dir", default=str(WORK / "agent")); ap.add_argument("--card", required=True)
ap.add_argument("--max-decisions", type=int, default=10**9); ap.add_argument("--profile", action="store_true")
ap.add_argument("--out"); ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(a.agent_dir).resolve()))
hits = [p for p in list((WORK / "cards").glob(f"*/{a.card}")) + [REPO / "examples/_local/cards" / a.card, Path(a.card)] if p.is_dir()]
card = hits[0]
from challenge.v4_workflow import V4Workflow
from challenge.v4_runner import run_scenario
import agent as agent_module
if a.quiet:
    agent_module.log = lambda text: None
wf = V4Workflow(card)
init = wf.initialize_payload(900.0, 1.0)
out = Path(a.out) if a.out else WORK / "runs" / "inproc" / card.name
out.mkdir(parents=True, exist_ok=True)
state = {"n": 0, "cpu": 0.0}

def factory(_context):
    ag = agent_module.Agent(init)
    state["agent"] = ag
    def decide(snapshot):
        state["n"] += 1
        if state["n"] > a.max_decisions:
            return None
        payload = dict(snapshot)
        payload["wallclock"] = {"remaining_real_cpu_seconds": 640.0 - state["cpu"], "remaining_seconds": 900.0}
        t = time.process_time()
        action = agent_module.validate(ag.respond(payload), ag)
        state["cpu"] += time.process_time() - t
        if action.get("action") == "wait" and "until_utc" not in action and not a.quiet:
            print(f"wait {payload['now_utc'][5:16]} {action.get('duration_seconds')}s: {action.get('reason')}", file=sys.stderr)
        return {k: v for k, v in action.items() if k not in ("reason", "decision_source")}
    return decide

started = time.time()
if a.profile:
    prof = cProfile.Profile(); prof.enable()
report = run_scenario(wf.scenario_path, factory, out)
if a.profile:
    prof.disable(); pstats.Stats(prof).sort_stats(os.environ.get("PROF_SORT", "cumulative")).print_stats(45)
c = report["components"]
print(json.dumps({"card": card.name, "total": report["total"], **c, "required_missing": report["counts"]["required_missing"],
                  "observes": report["counts"]["observe_actions"], "agent_cpu": round(state["cpu"], 1), "elapsed": round(time.time() - started, 1),
                  "termination": report["termination"]}))
