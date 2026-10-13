#!/usr/bin/env python3
"""Is a stored run healthy? Checks every card of a tag for the things that lose a whole card on the platform.

    python3 tools/health.py TAG [--set quick tune ...]        # exit code 1 if any card FAILS

FAIL  no score, not finished normally, an internal error in the agent log, CPU charge above 800 of 900,
      real time above 1500 s of the 1800 s cap
WARN  CPU charge above 680, real time above 900 s (the platform's 2-core containers are slower than this machine),
      a paid false report, more than 5% of the score lost to missing must-observe targets
"""
import argparse, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import RUNS, cards_of, load_summary, log_tail

CPU_FAIL, CPU_WARN, WALL_FAIL, WALL_WARN = 800.0, 680.0, 1500.0, 900.0


def check_card(tag: str, name: str):
    """(fails, warnings, facts) of one stored card run."""
    fails, warns, facts = [], [], {}
    s = load_summary(tag, name)
    if s is None:
        return ["no result"], [], facts
    if "total" not in s:
        return [f"no score: {str(s.get('error'))[-160:]}"], [], facts
    if s.get("termination_reason") != "survey_complete":
        fails.append(f"ended with {s.get('termination_reason')} {s.get('termination_detail') or ''}".strip())
    fc = s.get("fair_clock") or {}
    cpu, wall = float(fc.get("charged_seconds", 0.0)), float(fc.get("run_wall_seconds", s.get("elapsed", 0.0)))
    facts.update(cpu=cpu, wall=wall, total=s["total"])
    if cpu > CPU_FAIL:
        fails.append(f"CPU charge {cpu:.0f}s of 900")
    elif cpu > CPU_WARN:
        warns.append(f"CPU charge {cpu:.0f}s of 900")
    if wall > WALL_FAIL:
        fails.append(f"real time {wall:.0f}s (cap 1800)")
    elif wall > WALL_WARN:
        warns.append(f"real time {wall:.0f}s (cap 1800)")
    log = log_tail(tag, name)
    found = re.search(r"internal_errors=(\d+)", log)
    errors = int(found.group(1)) if found else len(re.findall(r"^error ", log, flags=re.M))
    facts["internal_errors"] = errors
    if errors or "Traceback" in log:
        fails.append(f"{errors or 'some'} internal error(s) in agent.log")
    model = re.findall(r"model stages: (.*)", log)
    if model:
        facts["model"] = model[-1]
    if s.get("report_settlement", 0) < 0 or re.search(r"reports=\d+ \(correct \d+, false ([3-9]|\d\d)", log):
        warns.append(f"report settlement {s.get('report_settlement', 0):.0f}")
    penalty = abs(float(s.get("required_penalty", 0.0)))
    if s["total"] > 0 and penalty > 0.05 * s["total"]:
        warns.append(f"{s.get('required_missing')} must-observe targets missing ({penalty:.0f} points)")
    return fails, warns, facts


def report(tag: str, cards) -> int:
    n_fail = n_warn = n = 0
    worst_cpu = worst_wall = 0.0
    model_lines = set()
    for _, card in cards:
        fails, warns, facts = check_card(tag, card.name)
        if fails == ["no result"]:
            continue
        n += 1
        worst_cpu, worst_wall = max(worst_cpu, facts.get("cpu", 0.0)), max(worst_wall, facts.get("wall", 0.0))
        if "model" in facts:
            model_lines.add(facts["model"].split(" calls=")[0])
        if fails:
            n_fail += 1
            print(f"  FAIL {card.name}: " + "; ".join(fails + warns))
        elif warns:
            n_warn += 1
            print(f"  WARN {card.name}: " + "; ".join(warns))
    verdict = "FAIL" if n_fail else ("ok with warnings" if n_warn else "ok")
    print(f"health {tag}: {verdict} -- {n} cards, {n_fail} failed, {n_warn} with warnings; worst CPU charge {worst_cpu:.0f}s, worst real time {worst_wall:.0f}s"
          + (f"; model: {', '.join(sorted(model_lines))}" if model_lines else ""))
    return 1 if n_fail or n == 0 else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag"); ap.add_argument("--set", nargs="+")
    a = ap.parse_args()
    if a.set:
        cards = cards_of(a.set)
    else:      # every card that has a folder under the tag
        cards = [("", p) for p in sorted((RUNS / a.tag).iterdir()) if p.is_dir()]
    raise SystemExit(report(a.tag, cards))
