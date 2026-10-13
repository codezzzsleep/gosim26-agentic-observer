#!/usr/bin/env python3
"""Pre-upload gate: package an agent and check everything that can be checked without the platform.

    python3 tools/gate.py                                  # the submitted agent (../agent)
    python3 tools/gate.py --agent build/try1 --name try1   # any other agent folder

Steps (each prints PASS or FAIL; exit code 1 if anything failed):
  1. build      copy the folder to build/NAME, pack build/NAME.zip, read it back with the platform's package code, no secrets
  2. unit       geometry against the official engine, sky tracker, (with model stages) the fault-verdict path
  3. smoke      the four short cards with real A-D catalogues, no model, CPU pacing on: must finish cleanly
  4. model      (with model stages) a local stand-in model in nine modes. In good / notools / texttool / reasoning / busy /
                think the model's answers must really be applied (with and without native tool calling, rate-limited,
                slow); in garbage / error / slow the run must stay healthy on the rules
A passed gate says the package is safe to upload. It says nothing about the score: that is tools/exp.py.
"""
import argparse, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import WORK, cards_of, log_tail
from health import check_card

TOOLS = WORK / "tools"
results = []


def step(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    return ok


def call(args, **kw) -> int:
    return subprocess.call([sys.executable, *map(str, args)], **kw)


def bench(tag, sets, agent_dir, env, jobs) -> None:
    subprocess.run([sys.executable, str(TOOLS / "bench.py"), "--tag", tag, "--set", *sets, "--agent-dir", str(agent_dir),
                    "--jobs", str(jobs), "--env", *env], stdout=subprocess.DEVNULL)


def healthy(tag, sets):
    bad = []
    for _, card in cards_of(sets):
        fails, _, _ = check_card(tag, card.name)
        if fails:
            bad.append(f"{card.name}: {'; '.join(fails)}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="../agent", help="agent folder (default: the submitted agent)")
    ap.add_argument("--name", default="agent", help="build name: build/NAME and build/NAME.zip")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--skip", nargs="*", default=[], help="steps to skip: unit smoke model")
    a = ap.parse_args()
    agent = (WORK / a.agent).resolve() if not Path(a.agent).is_absolute() else Path(a.agent)
    out = WORK / "build" / a.name
    has_model = (agent / "llm.py").exists() and (agent / "supervisor.py").exists()
    started = time.time()

    # 1. build
    if not step("build", call([TOOLS / "package.py", "--agent", agent, "--out", out]) == 0, f"build/{a.name}.zip"):
        return 1

    # 2. unit tests
    if "unit" not in a.skip:
        quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        step("unit: geometry equals the official engine", call([TOOLS / "test_skymath.py", out], **quiet) == 0)
        step("unit: sky tracker", call([TOOLS / "test_tracker.py", out], **quiet) == 0)
        if has_model:
            step("unit: fault verdict path (good, veto, garbage, error)", call([TOOLS / "test_verdict.py", out], **quiet) == 0)

    # 3. smoke: short real-catalogue cards, pacing on, model off
    if "smoke" not in a.skip:
        tag = f"gate_{a.name}_smoke"
        bench(tag, ["smoke"], out, ["OBSERVER_MODEL_DISABLED=1"], a.jobs)
        bad = healthy(tag, ["smoke"])
        step("smoke: four short A-D cards finish cleanly", not bad, "; ".join(bad))

    # 4. model stages against the local stand-in
    if has_model and "model" not in a.skip:
        modes = ["good", "notools", "texttool", "reasoning", "busy", "think", "garbage", "error", "slow"]
        servers, runs = [], []
        for k, mode in enumerate(modes):
            servers.append(subprocess.Popen([sys.executable, str(TOOLS / "mock_llm.py"), "--port", str(8770 + k), "--mode", mode]))
        time.sleep(1.0)
        try:
            for k, mode in enumerate(modes):
                env = ["AO_FIXED_EFFORT=1", "AO_LLM_PREFIX=OPENAI", "OPENAI_API_KEY=mock", f"OPENAI_BASE_URL=http://127.0.0.1:{8770 + k}/v1",
                       "OPENAI_MODEL=mock", "NO_PROXY=127.0.0.1,localhost", "no_proxy=127.0.0.1,localhost", "OBSERVER_MODEL_DISABLED=0"]
                runs.append(subprocess.Popen([sys.executable, str(TOOLS / "bench.py"), "--tag", f"gate_{a.name}_m_{mode}", "--set", "C8",
                                              "--agent-dir", str(out), "--jobs", "1", "--env", *env], stdout=subprocess.DEVNULL))
            for proc in runs:
                proc.wait()
        finally:
            for server in servers:
                server.kill()
        for mode in modes:
            tag = f"gate_{a.name}_m_{mode}"
            fails, _, facts = check_card(tag, "C8")
            log = log_tail(tag, "C8")
            applied = re.findall(r"applied: briefings=(\d+) verdicts=(\d+) \(vetoes \d+\) reviews=(\d+)", log)
            briefings = int(applied[-1][0]) if applied else len(re.findall(r"llm night briefing", log))
            reviews = int(applied[-1][2]) if applied else len(re.findall(r"llm season review", log))
            detail = f"briefings applied {briefings}, reviews applied {reviews}, real time {facts.get('wall', 0):.0f}s"
            if mode in ("good", "notools", "texttool", "reasoning", "busy", "think"):
                what = {"good": "stages called and their answers applied (native tool calls)",
                        "notools": "a service without native tool calling: the review falls back to the text form",
                        "texttool": "a model that writes tool requests as text: understood",
                        "reasoning": "answers found in reasoning_content",
                        "busy": "a rate-limited key (two of three requests refused): retried and applied",
                        "think": "a model that takes 20 s per answer: waited for and applied"}[mode]
                called = re.findall(r"llm season review: .*\((tools \[[^\]]*\]), (\w+) calling", log)
                detail += f", review {called[-1][1]} calling {called[-1][0]}" if called else ""
                expected = {"notools": "text", "texttool": "text"}.get(mode, "native")
                ok = not fails and briefings >= (1 if mode in ("busy", "think") else 3) and reviews >= 1 and (not called or called[-1][1] == expected)
                step(f"model {mode}: {what}", ok, "; ".join(fails) or detail)
            else:
                step(f"model {mode}: run stays healthy on rules", not fails, "; ".join(fails) or detail)

    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"GATE FAILED ({len(failed)} of {len(results)} checks) in {(time.time() - started) / 60:.1f} min: do not upload build/{a.name}.zip")
        return 1
    print(f"GATE PASSED ({len(results)} checks) in {(time.time() - started) / 60:.1f} min: build/{a.name}.zip can be uploaded.")
    if has_model:
        print("After a platform evaluation, look for 'model stages on:' in a card's log: without it the platform did not hand "
              "the key to the program and the run was rules only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
