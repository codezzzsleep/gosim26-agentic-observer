"""The model-driven stages. The numerical planner makes every exposure decision; a language model works where
reading and judgement matter, at three points:

1. night briefing   (natural-language understanding + plan adaptation)
   At the start of each night the model reads tonight's forecast, the bulletin and a short status report and
   sets the night's posture: which announced sectors to keep away from, how cautiously to size exposures,
   and whether to push or hold the must-observe targets.

2. fault adjudication   (data parsing + action decision)
   When the tracker's fault probability sits in the grey zone and a report would cost something (no free
   false report left, or the last one), the model reads the exposure-by-exposure evidence table and gives
   a verdict. It can veto a report or let it through; it cannot spend more than the paid-report budget.

3. season review   (task planning + tool calling)
   Every few nights the model reviews the campaign with tools it calls itself (progress, sky history,
   must-observe status, forecast) and chooses how firmly the planner should hold targets for a complete
   exposure later rather than take partial ones now.

Every answer is validated and clipped to a safe range; no answer, a late answer or a malformed one leaves the
rule-based default in force. Calls run on background threads and never block a decision for longer than the
wall clock allows.
"""
from __future__ import annotations

import json
import threading
import time

from llm import LLMClient, extract_json

DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

BRIEFING_SYSTEM = (
    "You are the night supervisor of a robotic spectroscopic survey telescope. A numerical scheduler picks every "
    "exposure; you set tonight's posture from the forecast, the bulletin and the status report in the user message.\n"
    "Notices name an event kind and a compass sector (N, NE, E, SE, S, SW, W, NW) or ALL (the whole sky). rain and storm "
    "close what they cover; overcast and haze lower quality; cold_snap lowers it slightly; rocket_launch closes a low "
    "sector for a few slots; earthquake lowers instrument efficiency for several nights; terrain_obstruction is permanent. "
    "A forecast names nights, not hours.\n"
    "Decide:\n"
    "- avoid_directions: sectors (not ALL) that tonight's forecast or the bulletin names for rain, storm, overcast, haze "
    "or rocket_launch. Only list sectors a notice names.\n"
    "- caution: \"high\" when tonight's sky is likely to change during exposures (all-sky weather forecast for tonight, "
    "or the sky level tonight far below typical), \"low\" when nothing is forecast and the sky is at or above typical, "
    "else \"normal\".\n"
    "- required_push: must-observe targets cost 50 points each if they never reach half completion by the end of the "
    "season. \"push\" when the outlook is poor for the coming nights or few nights remain; \"hold\" when tonight is poor "
    "and better nights are expected; else \"normal\".\n"
    "Reply with one JSON object only: {\"avoid_directions\": [..], \"caution\": \"low|normal|high\", "
    "\"required_push\": \"hold|normal|push\", \"reason\": \"<20 words>\"}"
)

FAULT_SYSTEM = (
    "You judge whether a robotic telescope has an unannounced instrument fault. Facts: a fault lowers instrument "
    "efficiency (eta) by a fixed factor between about 0.3 and 0.7, starts at a random moment and stays until it is "
    "reported; a correct report earns points and repairs it at once. An earthquake also lowers eta, is announced in "
    "the bulletin, and its loss fades night by night; a report does not repair it. Weather changes the sky level W, "
    "not eta. A dome closure makes an exposure lose part of its time; the next exposures are normal again.\n"
    "The table lists recent exposures: time, duration, W and eta as estimated from the scores, and the tracker's "
    "fault probability. Signs of a fault: eta steps down inside a night with no new earthquake notice and stays down "
    "for several exposures while W is steady. Signs against: the drop coincides with an earthquake notice; eta "
    "recovers on its own; only one exposure is low.\n"
    "A false report costs the stated penalty. Reply with one JSON object only: "
    "{\"verdict\": \"fault|not_fault|unsure\", \"confidence\": <0..1>, \"reason\": \"<20 words>\"}"
)

REVIEW_SYSTEM = (
    "You review the campaign of a robotic survey telescope every few nights. Only the best single exposure of each "
    "target counts, so a partial exposure is wasted if the target is observed again later; but a target that will "
    "never be revisited is better half done than not done. The scheduler has one knob for this, hold_strength in "
    "[0, 1]: high values hold targets for one complete exposure and avoid partial ones (right while much of the "
    "season remains and most targets can still be completed); low values take whatever gain is available now (right "
    "when little time remains, or the sky is expected to stay poor).\n"
    "Use the tools to look at progress, the sky so far, the must-observe targets and the forecast, then answer. "
    "Call at most four tools. Final reply: one JSON object only: {\"hold_strength\": <0..1>, \"reason\": \"<25 words>\"}"
)

REVIEW_TOOLS = [
    {"name": "get_progress", "description": "Season progress: nights done and left, targets completed / partly done / untouched, score so far, share of telescope time that was useful.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_sky_history", "description": "Median sky level (efficiency x weather; about 0.7 is a normal clear night) and hours observed for each of the last nights.",
     "parameters": {"type": "object", "properties": {"nights": {"type": "integer", "description": "how many recent nights (1-14)"}}, "required": []}},
    {"name": "get_required_status", "description": "Must-observe targets: how many are secured, pending, and pending but only feasible in a better-than-typical sky.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_forecast", "description": "The latest weather forecast notices (event kind, sector, nights).",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]


def _compact(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _tools_as_text(tools: list) -> str:
    """The same tools described in words, for services whose chat endpoint has no native tool calling."""
    lines = ["You have these tools. To call one, reply with one JSON object only: "
             "{\"tool\": \"<name>\", \"arguments\": {...}}. You will get its result in the next message. "
             "When you have what you need, give the final reply described above."]
    for tool in tools:
        params = ", ".join(f"{name} ({spec.get('type', 'any')}): {spec.get('description', '')}"
                           for name, spec in tool["parameters"].get("properties", {}).items()) or "no arguments"
        lines.append(f"- {tool['name']}: {tool['description']} Arguments: {params}.")
    return "\n".join(lines)


class ToolLoop:
    """A short multi-turn tool-calling conversation on a background thread, over a frozen snapshot of the state."""

    def __init__(self, client: LLMClient, system: str, question: str, tools: list, handlers: dict, wall_left: float,
                 max_rounds: int = 5, seconds: float = 150.0):
        self.result = None
        self.calls_made: list[str] = []
        self.error = None
        self.mode = "native"             # "native" tool calling, or "text" when the service or the model needed the plain form
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(client, system, question, tools, handlers, min(wall_left - 60.0, seconds), max_rounds), daemon=True)
        self._thread.start()

    def _run(self, client, system, question, tools, handlers, seconds, max_rounds) -> None:
        """The model asks for tools, the program runs them and returns the results, until the model answers.

        Two ways of asking are understood. Native tool calling (the service's `tools` field) is tried first. If the
        service refuses that request, or the model writes its tool request into the text instead, the conversation
        goes on in a plain-text form of the same protocol, which every chat endpoint can carry."""
        deadline = time.monotonic() + max(0.0, seconds)
        messages = [{"role": "user", "text": question}]
        native = getattr(client, "native_tools", True)       # False once the service has refused a request with tools
        self.mode = "native" if native else "text"

        def run_tool(name, arguments):
            handler = handlers.get(name)
            try:
                output = handler(**(arguments if isinstance(arguments, dict) else {})) if handler else {"error": "unknown tool"}
            except Exception as exc:  # noqa: BLE001 - a bad argument must not end the review
                output = {"error": type(exc).__name__}
            self.calls_made.append(str(name))
            return _compact(output)

        try:
            for _ in range(max_rounds + 1):
                left = deadline - time.monotonic()
                call = client.submit("season_review", messages, system if native else system + "\n\n" + _tools_as_text(tools),
                                     left + 45.0, tools=tools if native else None, seconds=left)
                if call is None:
                    self.error = "not submitted"
                    break
                call.wait(left)
                reply = client.collect(call)
                if reply is None:
                    if native and str(call.error).startswith("HTTP 4") and call.error not in ("HTTP 401", "HTTP 403", "HTTP 408", "HTTP 429"):
                        # the service does not take this request with tools: start again in the text form, and remember
                        native, self.mode = False, "text"
                        client.native_tools = False
                        messages = [{"role": "user", "text": question}]
                        self.calls_made.clear()
                        continue
                    self.error = call.error or "timeout"
                    break
                if reply["tool_calls"]:
                    messages.append({"role": "assistant", "text": reply["text"], "tool_calls": reply["tool_calls"]})
                    for tool_call in reply["tool_calls"][:4]:
                        messages.append({"role": "tool", "id": tool_call["id"], "text": run_tool(tool_call["name"], tool_call["arguments"])})
                    continue
                answer = extract_json(reply["text"])
                if isinstance(answer, dict) and isinstance(answer.get("tool"), str) and len(self.calls_made) < 2 * max_rounds:
                    # a tool request written as text
                    if native:
                        native, self.mode = False, "text"
                    messages.append({"role": "assistant", "text": reply["text"]})
                    messages.append({"role": "user", "text": f"Result of {answer['tool']}: " + run_tool(answer["tool"], answer.get("arguments"))})
                    continue
                self.result = answer
                if self.result is None:
                    self.error = "no JSON in the final reply"
                break
            else:
                self.error = "too many rounds"
        except Exception as exc:  # noqa: BLE001
            self.error = type(exc).__name__
        self._done.set()

    def done(self) -> bool:
        return self._done.is_set()


class Advisor:
    def __init__(self, client: LLMClient, log=lambda text: None):
        self.client = client
        self.log = log
        self.enabled = client.enabled
        self.briefing_call = None
        self.briefing_named: set = set()
        self.review = None
        self.stats = {"briefings": 0, "verdicts": 0, "reviews": 0}

    # --- 1. night briefing ------------------------------------------------------------------------------

    def start_night(self, context: dict, named_directions: set, wall_left: float, seconds: float = 90.0) -> None:
        """Submit tonight's briefing request (answer picked up with wait_briefing); it gives up after `seconds`."""
        self.briefing_named = set(named_directions)
        self.briefing_call = self.client.submit("night_briefing", [{"role": "user", "text": _compact(context)}],
                                                BRIEFING_SYSTEM, wall_left, seconds=seconds)

    def wait_briefing(self, seconds: float):
        if self.briefing_call is None:
            return None
        self.client.wait(self.briefing_call, seconds)
        return self.poll_briefing()

    def poll_briefing(self):
        """A validated briefing once, when the answer has arrived; otherwise None."""
        call = self.briefing_call
        if call is None or not call.done():
            return None
        self.briefing_call = None
        reply = self.client.collect(call)
        answer = extract_json(reply["text"]) if reply else None
        if not isinstance(answer, dict):
            return None
        avoid = answer.get("avoid_directions", [])
        if not isinstance(avoid, list):
            avoid = []
        # the model may only close sky that a notice names
        avoid = sorted({str(d).upper() for d in avoid if str(d).upper() in DIRECTIONS and str(d).upper() in self.briefing_named})
        caution = str(answer.get("caution", "normal")).lower()
        push = str(answer.get("required_push", "normal")).lower()
        if caution not in ("low", "normal", "high") or push not in ("hold", "normal", "push"):
            return None
        self.stats["briefings"] += 1
        return {"avoid_directions": avoid, "caution": caution, "required_push": push, "reason": str(answer.get("reason", ""))[:120]}

    # --- 2. fault adjudication --------------------------------------------------------------------------

    def judge_fault(self, evidence: dict, wall_left: float, wait_seconds: float):
        """{"verdict": "fault"|"not_fault"|"unsure", "confidence": float} or None when no usable answer came in time."""
        call = self.client.submit("fault_verdict", [{"role": "user", "text": _compact(evidence)}], FAULT_SYSTEM, wall_left, seconds=wait_seconds)
        reply = self.client.wait(call, wait_seconds)
        answer = extract_json(reply["text"]) if reply else None
        if not isinstance(answer, dict):
            return None
        verdict = str(answer.get("verdict", "")).lower()
        try:
            confidence = float(answer.get("confidence", 0.5))
        except (TypeError, ValueError):
            return None
        if verdict not in ("fault", "not_fault", "unsure") or not 0.0 <= confidence <= 1.0:
            return None
        self.stats["verdicts"] += 1
        return {"verdict": verdict, "confidence": confidence, "reason": str(answer.get("reason", ""))[:120]}

    # --- 3. season review with tools --------------------------------------------------------------------

    def start_review(self, snapshot: dict, wall_left: float, seconds: float = 150.0) -> None:
        """snapshot: {"progress": {...}, "sky": [[night, level, hours], ...], "required": {...}, "forecast": [...]}"""
        if not self.enabled or (self.review is not None and not self.review.done()):
            return
        frozen = json.loads(json.dumps(snapshot))

        def sky(nights: int = 7, **_):
            try:
                n = max(1, min(14, int(nights)))
            except (TypeError, ValueError):
                n = 7
            return {"columns": ["night", "median_level", "hours_observed"], "rows": frozen.get("sky", [])[-n:]}

        handlers = {"get_progress": lambda **_: frozen.get("progress", {}),
                    "get_sky_history": sky,
                    "get_required_status": lambda **_: frozen.get("required", {}),
                    "get_forecast": lambda **_: frozen.get("forecast", [])}
        question = _compact({"task": "Review the campaign and set hold_strength.", "night": frozen.get("night"),
                             "current_hold_strength": frozen.get("hold_strength")})
        self.review = ToolLoop(self.client, REVIEW_SYSTEM, question, REVIEW_TOOLS, handlers, wall_left, seconds=seconds)

    def wait_review(self, seconds: float):
        """Wait for the review that was just started; the validated answer, or None if it is not there in time
        (a late one is still returned by poll_review afterwards)."""
        review = self.review
        if review is None:
            return None
        review._done.wait(max(0.0, seconds))
        return self.poll_review()

    def poll_review(self):
        """{"hold_strength": float, "tools": [...]} once, when a valid review has arrived."""
        review = self.review
        if review is None or not review.done():
            return None
        self.review = None
        answer = review.result
        if not isinstance(answer, dict):
            if review.error:
                self.log(f"llm: season review gave no answer ({review.error}); defaults stay")
            return None
        try:
            hold = float(answer.get("hold_strength"))
        except (TypeError, ValueError):
            return None
        if not 0.0 <= hold <= 1.0:
            return None
        self.stats["reviews"] += 1
        return {"hold_strength": hold, "tools": list(review.calls_made), "mode": review.mode, "reason": str(answer.get("reason", ""))[:140]}
