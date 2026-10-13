#!/usr/bin/env python3
"""A local stand-in for an OpenAI-compatible chat endpoint, to exercise the model stages without a real key.

    python3 mock_llm.py --port 8765 [--mode good|slow|garbage|error|flaky|veto|notools|texttool|reasoning|busy|think]

good       sensible answers; the season review uses native tool calls
slow       the same after 40 s           garbage    text that is not JSON        error   HTTP 429
flaky      a random mix of the above     veto       the fault verdict is always "not_fault"
notools    HTTP 400 for any request that carries a `tools` field (a service without native tool calling)
texttool   accepts `tools` but writes its tool requests into the text
reasoning  an empty content with the answer in `reasoning_content`, after 3 s (some reasoning models)
busy       HTTP 429 for two requests out of three, the rest answered after 2 s (a shared, rate-limited key)
think      good answers after 20 s (a reasoning model's usual pace)
"""
import argparse, json, random, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--mode", default="good")
ap.add_argument("--log", default="")
args = ap.parse_args()
rng = random.Random(1)
count = {"n": 0}


def answer(body):
    system = body["messages"][0]["content"]
    last = body["messages"][-1]
    if "night supervisor" in system:
        ctx = json.loads(last["content"])
        named = [n["direction"] for n in ctx.get("forecast_tonight", []) + ctx.get("bulletin_now", []) if n["direction"] != "ALL" and n["event_kind"] in ("rain", "storm", "overcast", "haze", "rocket_launch")]
        allsky = any(n["direction"] == "ALL" and n["event_kind"] in ("rain", "storm", "overcast", "haze") for n in ctx.get("forecast_tonight", []))
        return {"content": json.dumps({"avoid_directions": sorted(set(named)), "caution": "high" if allsky else "normal",
                                       "required_push": "hold" if allsky else "normal", "reason": "mock"})}
    if "instrument fault" in system:
        ev = json.loads(last["content"])
        p = ev.get("tracker_fault_probability_now", 0)
        if args.mode == "veto":
            return {"content": json.dumps({"verdict": "not_fault", "confidence": 0.9, "reason": "mock veto"})}
        return {"content": "Verdict follows.\n```json\n" + json.dumps({"verdict": "fault" if p > 0.5 else "unsure", "confidence": 0.7, "reason": "mock"}) + "\n```"}
    if "review the campaign" in system:
        tool_msgs = [m for m in body["messages"] if m.get("role") == "tool"]
        text_results = [m for m in body["messages"] if m.get("role") == "user" and str(m.get("content", "")).startswith("Result of ")]
        as_text = "You have these tools" in system or args.mode == "texttool"
        if as_text:
            if len(text_results) < 2:
                name = ["get_progress", "get_sky_history"][len(text_results)]
                return {"content": "I will look first.\n" + json.dumps({"tool": name, "arguments": {"nights": 5} if name == "get_sky_history" else {}})}
            return {"content": json.dumps({"hold_strength": 0.6, "reason": "mock review (text tools)"})}
        if len(tool_msgs) < 2:
            name = ["get_progress", "get_sky_history"][len(tool_msgs)]
            return {"content": "", "tool_calls": [{"id": f"call_{len(tool_msgs)}", "type": "function",
                                                   "function": {"name": name, "arguments": json.dumps({"nights": 5} if name == "get_sky_history" else {})}}]}
        return {"content": json.dumps({"hold_strength": 0.6, "reason": "mock review"})}
    return {"content": "{}"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        count["n"] += 1
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        mode = args.mode
        if mode == "flaky":
            mode = rng.choice(["good", "good", "slow", "garbage", "error"])
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(json.dumps({"n": count["n"], "mode": mode, "system": body["messages"][0]["content"][:40], "user": str(body["messages"][-1].get("content"))[:300]}) + "\n")
        if mode == "slow":
            time.sleep(40)
        if mode == "error":
            self.send_response(429); self.end_headers(); self.wfile.write(b'{"error":"rate limit"}'); return
        if mode == "notools" and body.get("tools"):
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"tools are not supported"}'); return
        if mode == "reasoning":
            time.sleep(3)
        if mode == "busy":
            if rng.random() < 0.67:
                self.send_response(429); self.end_headers(); self.wfile.write(b'{"error":"too many requests"}'); return
            time.sleep(2)
        if mode == "think":
            time.sleep(20)
        if self.path.endswith("/v1/messages"):
            # Anthropic wire format -> the neutral form answer() understands -> back
            msgs = [{"role": "system", "content": body.get("system", "")}]
            for m in body["messages"]:
                if isinstance(m["content"], list):
                    for block in m["content"]:
                        if block.get("type") == "tool_result":
                            msgs.append({"role": "tool", "content": block.get("content")})
                        elif block.get("type") == "text":
                            msgs.append({"role": m["role"], "content": block.get("text")})
                else:
                    msgs.append({"role": m["role"], "content": m["content"]})
            out = answer({"messages": msgs}) if mode != "garbage" else {"content": "not json at all"}
            content = ([{"type": "text", "text": out["content"]}] if out.get("content") else [])
            for c in out.get("tool_calls", []):
                content.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": json.loads(c["function"]["arguments"])})
            data = json.dumps({"content": content, "stop_reason": "end_turn"}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data); return
        message = {"role": "assistant", **(answer(body) if mode != "garbage" else {"content": "I think the weather is nice <not json"})}
        if mode == "reasoning" and not message.get("tool_calls"):
            message = {"role": "assistant", "content": "", "reasoning_content": "Let me think. " + message["content"]}
        data = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
