#!/usr/bin/env python3
"""Try the three model stages once against the real model service, exactly as the agent calls them.

    python3 tools/check_model.py --dir build/v1_llm

The service comes from the environment or from a .env file in that folder (see .env.example there), e.g. for the
organizers' local Kimi relay:

    OPENAI_API_KEY=<your personal API token>
    OPENAI_BASE_URL=<OpenAI-compatible base URL of the service>
    OPENAI_MODEL=kimi-for-coding
    AO_LLM_PREFIX=OPENAI

It makes about five requests (one briefing, one fault verdict, one review with a few tool rounds), prints what
came back and how long it took, and says PASS or FAIL per stage. The key is never printed.
A stage that fails here would silently fall back to the rules during a run -- the run stays healthy, but that
stage would not count as model-driven. Paste the output back if something fails.
"""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import WORK


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="build/v1_llm", help="assembled agent folder with the model stages")
    ap.add_argument("--wait", type=float, default=120.0, help="longest wait per stage, seconds")
    a = ap.parse_args()
    folder = (Path(a.dir) if Path(a.dir).is_absolute() else WORK / a.dir).resolve()
    if not (folder / "llm.py").exists():
        raise SystemExit(f"{folder} has no llm.py: build it first, e.g.  python3 tools/assemble.py --core baselines/v1 --llm --out build/v1_llm")
    sys.path.insert(0, str(folder))
    import llm, advisor
    llm.load_dotenv(str(folder / ".env"))
    os.environ.pop("OBSERVER_MODEL_DISABLED", None)
    notes = []
    client = llm.LLMClient(log=notes.append)
    if not client.enabled:
        print("No model service configured: set NAME_API_KEY (and NAME_BASE_URL, NAME_MODEL) in the environment or in", folder / ".env")
        return 1
    host = client.base.split("//", 1)[-1].split("/", 1)[0]
    print(f"provider {client.provider}, {client.protocol} protocol, model {client.model}, host {host}")
    adv = advisor.Advisor(client, log=notes.append)
    failures = 0

    def raw(call) -> str:
        if call is None:
            return "(not submitted)"
        if call.reply is None:
            return f"error: {call.error}"
        return f"reply text: {call.reply['text'][:400]!r} tool_calls: {call.reply['tool_calls']}"

    # 1. night briefing: a haze sector in tonight's forecast, rain two nights later
    context = {"night": "2026-11-03", "nights_left_including_tonight": 20,
               "forecast_tonight": [{"event_kind": "haze", "direction": "SE"}],
               "forecast_later_nights": [{"event_kind": "rain", "direction": "ALL", "nights": ["2026-11-05"]}],
               "bulletin_now": [{"event_kind": "haze", "direction": "SE"}], "terrain_obstruction": ["W"],
               "sky_level_now": 0.62, "sky_level_typical": 0.68, "sky_level_last_nights": [["2026-11-01", 0.71, 8.2], ["2026-11-02", 0.66, 7.9]],
               "must_observe_pending": 140, "must_observe_pending_needing_good_sky": 22, "open_requests": 0}
    started = time.monotonic()
    adv.start_night(context, {"SE"}, 1700.0)
    call = adv.briefing_call
    briefing = adv.wait_briefing(a.wait)
    seconds = time.monotonic() - started
    ok = briefing is not None
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] night briefing in {seconds:.1f}s: {briefing if ok else raw(call)}")
    if ok and briefing["avoid_directions"] != ["SE"]:
        print("       note: the haze sector SE was not listed in avoid_directions (valid answer, weak reading)")

    # 2. fault verdict: efficiency steps down inside a night with steady weather and no earthquake
    rows = [[f"11-03T0{h}:{m:02d}", 900, 0.74, 0.95, 0.01, []] for h, m in ((1, 0), (1, 20), (1, 40), (2, 0))]
    rows += [[f"11-03T0{h}:{m:02d}", 600, 0.73, eta, p, []] for (h, m), eta, p in (((2, 20), 0.55, 0.35), ((2, 35), 0.53, 0.62), ((2, 50), 0.54, 0.78), ((3, 5), 0.53, 0.85))]
    evidence = {"exposures": {"columns": ["utc", "seconds", "W", "eta", "tracker_fault_probability", "bulletin"], "rows": rows},
                "tracker_fault_probability_now": 0.85, "earthquake_notice_now": False, "free_false_reports_left": 0,
                "false_report_penalty": -150, "correct_report_reward": 100, "nights_left": 20.0, "reports_so_far": {"correct": 0, "false": 2}}
    started = time.monotonic()
    before = len(client.calls)
    verdict = adv.judge_fault(evidence, 1700.0, a.wait)
    seconds = time.monotonic() - started
    ok = verdict is not None
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] fault verdict in {seconds:.1f}s: {verdict if ok else raw(client.calls[before] if len(client.calls) > before else None)}")
    if ok and verdict["verdict"] != "fault":
        print("       note: this table shows a clear fault; the model did not say so (valid answer, weak judgement)")

    # 3. season review with tools
    snapshot = {"night": "2026-11-03", "hold_strength": 0.75,
                "progress": {"nights_done": 16, "nights_left": 20, "targets_completed": 2100, "targets_partly_done": 600, "targets_untouched": 5300,
                             "score_so_far": 2900.0, "useful_time_share": 0.83},
                "sky": [[f"2026-10-{d}", level, 8.0] for d, level in ((28, 0.70), (29, 0.72), (30, 0.55), (31, 0.69))] + [["2026-11-01", 0.71, 8.2], ["2026-11-02", 0.66, 7.9]],
                "required": {"secured": 250, "pending": 140, "pending_needing_good_sky": 22},
                "forecast": [{"event_kind": "rain", "direction": "ALL", "nights": ["2026-11-05"]}]}
    started = time.monotonic()
    before = len(client.calls)
    adv.start_review(snapshot, 1700.0)
    loop = adv.review
    while loop is not None and not loop.done() and time.monotonic() - started < a.wait + 60:
        time.sleep(0.2)
    review = adv.poll_review()
    seconds = time.monotonic() - started
    ok = review is not None
    failures += not ok
    if ok:
        print(f"[PASS] season review in {seconds:.1f}s over {len(client.calls) - before} requests: hold_strength {review['hold_strength']}, "
              f"{review['mode']} tool calling, tools {review['tools']} ({review['reason']})")
        if not review["tools"]:
            print("       note: the model answered without calling a tool (valid, but the tool-calling stage did nothing)")
    else:
        last = client.calls[-1] if len(client.calls) > before else None
        print(f"[FAIL] season review in {seconds:.1f}s: {getattr(loop, 'error', None)}; last request -> {raw(last)}")

    for text in notes:
        print("   log:", text)
    print(f"requests: {len(client.calls)} sent, {client.ok} answered, {client.failed} failed")
    if failures:
        print(f"{failures} of 3 stages FAILED with this service.")
        return 1
    print("All three stages work with this service.")
    slow = [c.seconds for c in client.calls if c.seconds > 10.0]
    if slow:
        print(f"Note: {len(slow)} requests took longer than 10 s (up to {max(slow):.0f} s). In a run the agent waits at most about 10 s at a "
              f"night start; later answers are still applied when they arrive, a few exposures into the night.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
