#!/usr/bin/env python3
"""Run an agent over cards with the official local runner (the platform's own engine); store and tabulate scores.

    python3 tools/bench.py --tag NAME --set quick                       # sets, groups, 'L' or card names; several allowed
    python3 tools/bench.py --tag NAME --set real_dev var_dev --agent-dir build/v1_llm --env AO_FIXED_EFFORT=1
    python3 tools/bench.py --tag NAME --set tune --resume               # only the cards that have no result yet
    python3 tools/bench.py --compare pro g1 NAME --set dev              # stored results side by side

Results go to runs/NAME/<card>/ : summary.json, score_report.json and agent.log are kept; the large per-run
files (messages, actions, observations, decisions) are deleted unless --keep is given (the analysis tools need them).
For a statistical comparison of two tags use tools/compare.py; for one whole experiment use tools/exp.py.
"""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import WORK, RUNS, RUNNER, cards_of, card_cost, load_summary

DEFAULT_CMD = f"{shlex.quote(sys.executable)} -u agent.py"      # the interpreter running this tool (the lab's .venv)
HEAVY = ("messages.jsonl", "actions.jsonl", "observations.csv", "decisions.csv", "workflow_result.json")


def agent_fingerprint(agent_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(agent_dir.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def run_one(tag, agent_dir, cmd, set_name, card, env, wallclock, keep):
    out = RUNS / tag / card.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    full_env = dict(os.environ); full_env.update(env)
    started = time.time()
    args = [sys.executable, str(RUNNER), "--inherit-env", "--card", str(card), "--agent", cmd, "--agent-cwd", str(agent_dir),
            "--out", str(out), "--quiet"]
    if wallclock:
        args += ["--wallclock", str(wallclock)]
    proc = subprocess.run(args, capture_output=True, text=True, env=full_env)
    text = proc.stdout
    try:
        summary = json.loads(text[text.rindex("\n{") + 1:]) if "\n{" in text else json.loads(text)
    except Exception:
        summary = {"error": (proc.stderr or text)[-1500:]}
    summary["elapsed"] = round(time.time() - started, 1)
    summary["set"] = set_name
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not keep:
        for name in HEAVY:
            (out / name).unlink(missing_ok=True)
        shutil.rmtree(out / "scratch", ignore_errors=True)
    return set_name, card.name, summary


def row(name, s):
    if "total" not in s:
        return f"{name:8s} ERROR {str(s.get('error'))[-300:]}"
    fc = s.get("fair_clock") or {}
    return (f"{name:8s} total {s['total']:9.1f}  best {s['sum_best_scores']:9.1f}  reqmiss {s['required_missing']:4d}  rep {s['report_settlement']:6.0f}  "
            f"rq {s['observation_request_reward']:5.0f}  unif {s['uniformity_penalty']:6.1f}  obs {s['observe_actions']:5d}  cpu {fc.get('charged_seconds', 0):4.0f}  "
            f"wall {fc.get('run_wall_seconds', s.get('elapsed', 0)):5.0f}s  {s['termination_reason']}")


def compare(tags, cards):
    data = {t: {c.name: load_summary(t, c.name) for _, c in cards} for t in tags}
    print("card     " + "".join(f"{t:>12s}" for t in tags))
    sums = {t: 0.0 for t in tags}; n = 0
    for _, c in cards:
        ok = all(data[t][c.name] and "total" in data[t][c.name] for t in tags)
        if ok:
            n += 1
            for t in tags:
                sums[t] += data[t][c.name]["total"]
        cells = "".join(f"{(data[t][c.name] or {}).get('total', float('nan')):12.1f}" for t in tags)
        print(f"{c.name:8s} {cells}")
    if n:
        print(f"mean({n}) " + "".join(f"{sums[t] / n:12.1f}" for t in tags))
        base = sums[tags[0]]
        if base:
            print("vs first " + "".join(f"{(sums[t] / base - 1) * 100:+11.2f}%" for t in tags))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag"); ap.add_argument("--agent-dir", default="agent"); ap.add_argument("--cmd", default=DEFAULT_CMD)
    ap.add_argument("--set", nargs="+", default=["dev"]); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--env", nargs="*", default=[]); ap.add_argument("--wallclock", type=int)
    ap.add_argument("--resume", action="store_true", help="skip cards that already have a result under this tag")
    ap.add_argument("--keep", action="store_true", help="keep the large per-run files")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    cards = cards_of(a.set)
    if a.compare:
        compare(a.compare, cards)
        return 0
    if not a.tag:
        raise SystemExit("--tag is required")
    agent_dir = Path(a.agent_dir)
    agent_dir = (agent_dir if agent_dir.is_absolute() else WORK / agent_dir).resolve()
    if not (agent_dir / "agent.py").exists() and a.cmd == DEFAULT_CMD:
        raise SystemExit(f"{agent_dir} has no agent.py")
    env = dict(e.split("=", 1) for e in a.env)
    todo = [(s, c) for s, c in cards if not (a.resume and (load_summary(a.tag, c.name) or {}).get("total") is not None)]
    todo.sort(key=lambda item: -card_cost(item[1]))
    (RUNS / a.tag).mkdir(parents=True, exist_ok=True)
    meta = {"tag": a.tag, "agent_dir": str(agent_dir), "agent_fingerprint": agent_fingerprint(agent_dir), "cmd": a.cmd, "env": env,
            "sets": a.set, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": a.jobs}
    (RUNS / a.tag / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.tag}: {len(todo)} cards to run ({len(cards) - len(todo)} already done), {a.jobs} at a time, agent {agent_dir.name} "
          f"[{meta['agent_fingerprint']}] env {env}", flush=True)
    started = time.time()
    with ThreadPoolExecutor(a.jobs) as pool:
        futures = [pool.submit(run_one, a.tag, agent_dir, a.cmd, s, c, env, a.wallclock, a.keep) for s, c in todo]
        for future in as_completed(futures):
            _, name, summary = future.result()
            print(row(name, summary), flush=True)
    # per-set means over everything stored under the tag for the requested cards
    by_set = {}
    for set_name, card in cards:
        s = load_summary(a.tag, card.name)
        if s and "total" in s:
            by_set.setdefault(set_name, []).append(s["total"])
    for set_name, totals in by_set.items():
        print(f"MEAN {set_name:10s} {sum(totals) / len(totals):9.1f} over {len(totals)}")
    everything = [v for totals in by_set.values() for v in totals]
    if everything:
        print(f"MEAN {sum(everything) / len(everything):.1f} over {len(everything)}   ({(time.time() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
