#!/usr/bin/env python3
"""One experiment, start to finish: run, check health, compare with a baseline, record the outcome.

    python3 tools/exp.py --tag kappa085 --base base --set quick --env AO_FIXED_EFFORT=1 AO_KAPPA=0.85 --note "higher time price"
    python3 tools/exp.py --tag mychange --base base --set tune --agent-dir ../agent --env AO_FIXED_EFFORT=1

What it does:
  1. if the baseline has no result for some of the cards, runs the baseline on them first (same agent folder and
     environment as recorded in runs/BASE/meta.json);
  2. runs the new variant on the cards (cards already done under the tag are not repeated);
  3. health check of the new run (tools/health.py);
  4. paired comparison with the baseline (tools/compare.py) and its verdict;
  5. appends one line to results/LEDGER.md.

The tag is the experiment's name: one tag = one agent folder + one environment. Use a new tag for every variant.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import WORK, RUNS, cards_of, load_summary
from bench import agent_fingerprint
from compare import compare
from health import report

LEDGER = WORK / "results" / "LEDGER.md"
HEADER = ("| time | tag | base | sets | cards | mean diff | 95% interval | better/worse | verdict | health | agent | env | note |\n"
          "|---|---|---|---|---:|---:|---|---|---|---|---|---|---|\n")


def bench(tag, sets, agent_dir, env, jobs, keep):
    args = [sys.executable, str(WORK / "tools" / "bench.py"), "--tag", tag, "--set", *sets, "--agent-dir", str(agent_dir), "--resume", "--jobs", str(jobs)]
    if env:
        args += ["--env", *env]
    if keep:
        args.append("--keep")
    return subprocess.call(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--set", nargs="+", default=["quick"]); ap.add_argument("--agent-dir", default="../agent")
    ap.add_argument("--env", nargs="*", default=[]); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--note", default=""); ap.add_argument("--keep", action="store_true"); ap.add_argument("--margin", type=float, default=0.3)
    a = ap.parse_args()
    cards = cards_of(a.set)
    agent_dir = (Path(a.agent_dir) if Path(a.agent_dir).is_absolute() else WORK / a.agent_dir).resolve()

    # the tag must keep meaning the same thing
    meta_path = RUNS / a.tag / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        same = meta.get("agent_fingerprint") == agent_fingerprint(agent_dir) and meta.get("env") == dict(e.split("=", 1) for e in a.env)
        done = sum(1 for _, c in cards if (load_summary(a.tag, c.name) or {}).get("total") is not None)
        if not same and done:
            raise SystemExit(f"tag {a.tag} already holds results of a different agent or environment: choose a new tag (or delete runs/{a.tag})")

    # 1. baseline where it is missing
    missing = [c.name for _, c in cards if (load_summary(a.base, c.name) or {}).get("total") is None]
    if missing:
        base_meta_path = RUNS / a.base / "meta.json"
        if not base_meta_path.exists():
            raise SystemExit(f"baseline {a.base} has no results for {len(missing)} of these cards and no meta.json to repeat it from.\n"
                             f"Run it first:  python3 tools/bench.py --tag {a.base} --set {' '.join(a.set)} --agent-dir <folder> --env AO_FIXED_EFFORT=1")
        base_meta = json.loads(base_meta_path.read_text())
        base_dir = Path(base_meta["agent_dir"])
        if not base_dir.is_dir() or agent_fingerprint(base_dir) != base_meta.get("agent_fingerprint"):
            raise SystemExit(f"baseline {a.base} lacks {len(missing)} of these cards, and its agent folder {base_dir} has changed since it ran.\n"
                             f"Keep baselines in folders that do not change (a copy under build/) or run a fresh baseline under a new tag.")
        print(f"== baseline {a.base}: running the {len(missing)} missing cards", flush=True)
        bench(a.base, a.set, base_dir, [f"{k}={v}" for k, v in base_meta.get("env", {}).items()], a.jobs, False)

    # 2. the variant
    print(f"== {a.tag}", flush=True)
    bench(a.tag, a.set, agent_dir, a.env, a.jobs, a.keep)
    # 3. health
    print("== health", flush=True)
    unhealthy = report(a.tag, cards)
    # 4. comparison
    print("== comparison", flush=True)
    result = compare(a.base, a.tag, cards, a.margin)
    # 5. ledger
    LEDGER.parent.mkdir(exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text("# Experiment ledger\n\nOne line per run of tools/exp.py, oldest first. Verdicts: see tools/compare.py.\n\n" + HEADER)
    if "mean_rel" in result:
        line = (f"| {time.strftime('%m-%d %H:%M')} | {a.tag} | {a.base} | {' '.join(a.set)} | {result['cards']} | {result['mean_rel']:+.2f}% | "
                f"[{result['ci_low']:+.2f}%, {result['ci_high']:+.2f}%] | {result['wins']}/{result['losses']} | {result['verdict']} | "
                f"{'FAIL' if unhealthy else 'ok'} | {agent_dir.name} [{agent_fingerprint(agent_dir)}] | {' '.join(a.env)} | {a.note} |\n")
    else:
        line = f"| {time.strftime('%m-%d %H:%M')} | {a.tag} | {a.base} | {' '.join(a.set)} | 0 |  |  |  | NO DATA | {'FAIL' if unhealthy else 'ok'} | {agent_dir.name} | {' '.join(a.env)} | {a.note} |\n"
    with open(LEDGER, "a") as handle:
        handle.write(line)
    print(f"recorded in {LEDGER.relative_to(WORK)}")
    return 1 if unhealthy else 0


if __name__ == "__main__":
    raise SystemExit(main())
