"""Minimal chat client for the model stages (standard library only).

Two wire protocols. The provider comes from the environment the platform (or a local .env) provides, as a
group of variables with a common prefix: NAME_API_KEY, NAME_BASE_URL, NAME_MODEL.

    AO_LLM_PREFIX=NAME          use exactly that group
    otherwise, in this order    KIMI, OPENAI, DEEPSEEK, MOONSHOT, then any other NAME_API_KEY that comes with a
                                NAME_BASE_URL, then ANTHROPIC (Anthropic wire protocol)

OPENAI and KIMI share OPENAI_BASE_URL / OPENAI_MODEL when their own are missing (default: Kimi Coding Plan).
When several services are configured the first is used; if two consultations in a row get no answer from it,
the client moves on to the next one (and round again), so one service going away does not silence the stages.

Calls run on background threads: the survey never blocks on the network longer than the caller allows, and
every failure (timeout, rate limit, malformed reply) simply leaves the rule-based decision in place.
The key is never logged.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

DEFAULT_OPENAI_BASE = "https://api.kimi.com/coding/v1"
DEFAULT_OPENAI_MODEL = "kimi-for-coding"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
# prefix -> (default base URL, default model) of OpenAI-compatible services, in order of preference
KNOWN_PROVIDERS = (("KIMI", DEFAULT_OPENAI_BASE, DEFAULT_OPENAI_MODEL),
                   ("OPENAI", DEFAULT_OPENAI_BASE, DEFAULT_OPENAI_MODEL),
                   ("DEEPSEEK", "https://api.deepseek.com", "deepseek-chat"),
                   ("MOONSHOT", "https://api.moonshot.cn/v1", "moonshot-v1-8k"))
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _max_tokens() -> int:
    """Room for the answer. Reasoning models spend part of it thinking, so it is generous; AO_LLM_MAX_TOKENS overrides."""
    try:
        return max(256, int(os.environ.get("AO_LLM_MAX_TOKENS", "3000")))
    except ValueError:
        return 3000


def _text_of(message: dict) -> str:
    """The text of a chat message: plain content, a list of content parts, or (when a reasoning model returned
    nothing else) its reasoning text, where the JSON answer sometimes ends up."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else ""


def load_dotenv(path: str) -> None:
    """Fill missing environment variables from a local .env (local runs only; never packed)."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip().strip('"').strip("'")
                if name and value and not os.environ.get(name):
                    os.environ[name] = value
    except OSError:
        pass


def _get(name: str) -> str:
    return os.environ.get(name, "").strip()


def configured_providers() -> list:
    """Every model service found in the environment, in order of preference:
    [(prefix, protocol, key, base URL, model)]. Empty when there is none."""
    prefix = _get("AO_LLM_PREFIX").upper()
    known = {name: (base, model) for name, base, model in KNOWN_PROVIDERS}
    candidates = [prefix] if prefix else [name for name, _, _ in KNOWN_PROVIDERS]
    if not prefix:
        others = sorted(name[:-len("_API_KEY")] for name in os.environ
                        if name.endswith("_API_KEY") and name[:-len("_API_KEY")] + "_BASE_URL" in os.environ)
        candidates += [name for name in others if name not in candidates and name != "ANTHROPIC"] + ["ANTHROPIC"]
    found, seen = [], set()
    for name in candidates:
        key = _get(name + "_API_KEY")
        if not key:
            continue
        base, model = _get(name + "_BASE_URL"), _get(name + "_MODEL")
        if name == "ANTHROPIC" or "anthropic" in base:
            entry = (name, "anthropic", key, (base or DEFAULT_ANTHROPIC_BASE).rstrip("/"), model or DEFAULT_ANTHROPIC_MODEL)
        else:
            if name in ("OPENAI", "KIMI"):
                base, model = base or _get("OPENAI_BASE_URL"), model or _get("OPENAI_MODEL")
            default_base, default_model = known.get(name, (DEFAULT_OPENAI_BASE, DEFAULT_OPENAI_MODEL))
            entry = (name, "openai", key, (base or default_base).rstrip("/"), model or default_model)
        if entry[2:] not in seen:              # the same key, address and model under two names is one service
            seen.add(entry[2:])
            found.append(entry)
    return found


def choose_provider():
    """(prefix, protocol, key, base URL, model) of the preferred model service; key is "" when there is none."""
    found = configured_providers()
    return found[0] if found else ("", "openai", "", "", "")


def model_disabled() -> bool:
    """The platform sets OBSERVER_MODEL_DISABLED=1 for an evaluation started without a model."""
    return _get("OBSERVER_MODEL_DISABLED") == "1" or _get("AO_NO_MODEL") == "1"


def extract_json(text: str):
    """The first JSON object in a model reply (models like to wrap it in prose or a code fence)."""
    if not isinstance(text, str):
        return None
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    blob = match.group(0)
    for end in range(len(blob), 1, -1):          # trim trailing prose after the object
        if blob[end - 1] != "}":
            continue
        try:
            parsed = json.loads(blob[:end])
        except ValueError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


class Call:
    """One request on a background thread."""

    def __init__(self, client: "LLMClient", tag: str, messages: list, system: str, tools, timeout: float):
        self.tag = tag
        self.reply = None            # {"text": str, "tool_calls": [{"id", "name", "arguments": dict}]}
        self.error = None
        self.seconds = 0.0
        self.attempts = 0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(client, messages, system, tools, timeout), daemon=True)
        self._thread.start()

    def _run(self, client, messages, system, tools, timeout) -> None:
        try:
                    """Try until an answer comes or `timeout` real seconds are over. "Too many requests" and network trouble
                    are waited out with growing pauses (several cards may share one key); a refused request is not repeated."""
                    started = time.monotonic()
                    pause = 2.0
                    while True:
                        left = timeout - (time.monotonic() - started)
                        if left < 3.0:
                            self.error = self.error or "timeout"
                            break
                        self.attempts += 1
                        try:
                            self.reply = client._request(messages, system, tools, left)
                            if time.monotonic() - started >= timeout:
                                self.reply = None
                                self.error = "timeout"
                            else:
                                self.error = None
                            break
                        except urllib.error.HTTPError as exc:
                            self.error = f"HTTP {exc.code}"
                            if exc.code in (400, 401, 403, 404, 422):     # not worth retrying
                                break
                        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError, http.client.HTTPException) as exc:
                            self.error = type(exc).__name__
                        if self.attempts >= client.max_retries and not str(self.error).startswith(("HTTP 429", "HTTP 5")):
                            break
                        wait = min(pause * (0.6 + 0.8 * client.rng.random()), max(0.0, timeout - (time.monotonic() - started) - 3.0))
                        if wait <= 0.0:
                            break
                        time.sleep(wait)
                        pause = min(pause * 1.8, 15.0)
        except Exception as exc:  # nothing may leak the call slot
            self.error = self.error or type(exc).__name__
        finally:
            self.seconds = time.monotonic() - started
            self.finished_at = time.monotonic()
            self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, seconds: float) -> bool:
        return self._done.wait(max(0.0, seconds))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None                          # never forward API credentials to another origin


class LLMClient:
    def __init__(self, log=lambda text: None, call_timeout: float = 100.0, max_calls: int = 200, max_retries: int = 3,
                 max_in_flight: int = 1):
        self.log = self._safe_log(log)
        self.providers = configured_providers()
        self.provider_index = 0
        self.provider, self.protocol, self.key, self.base, self.model = self.providers[0] if self.providers else ("", "openai", "", "", "")
        self.enabled = bool(self.key) and not model_disabled()
        self.call_timeout = call_timeout
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.max_in_flight = max_in_flight
        self.calls: list[Call] = []
        self.ok = 0
        self.failed = 0
        self.consecutive_failures = 0
        self.wait_seconds = 0.0          # wall time the survey spent waiting for answers
        self.native_tools = True         # set to False when the service refuses requests that carry tools
        self.held_back = 0
        self.rng = random.Random(time.time_ns() ^ os.getpid())       # only for the pauses between retries

    def _safe_log(self, logger):
        """Wrap the logger so the API key is never written to a log, wherever it appears."""
        def safe(text: str) -> None:
            text = str(text)
            if self.key:
                text = text.replace(self.key, "[redacted]")
            logger(" ".join(text.splitlines()))
        return safe

    # --- wire formats -----------------------------------------------------------------------------------

    def _post(self, url: str, body: dict, headers: dict, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json", **headers})
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            chunks, size = [], 0
            limit = 512 * 1024
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("model response expired")
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(left)
                read = getattr(response, "read1", response.read)
                chunk = read(min(16384, limit + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > limit:
                    raise ValueError("model response too large")
            if time.monotonic() >= deadline:
                raise TimeoutError("model response expired")
            data = json.loads(b"".join(chunks).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("model response must be an object")
            return data

    def _request(self, messages: list, system: str, tools, timeout: float) -> dict:
        """messages: [{"role": "user"|"assistant"|"tool", ...}] in a neutral form (see Advisor)."""
        if self.protocol == "anthropic":
            return self._request_anthropic(messages, system, tools, timeout)
        return self._request_openai(messages, system, tools, timeout)

    def _request_openai(self, messages, system, tools, timeout) -> dict:
        wire = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                item = {"role": "assistant", "content": m.get("text") or ""}
                if m.get("tool_calls"):
                    item["tool_calls"] = [{"id": c["id"], "type": "function",
                                           "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                          for c in m["tool_calls"]]
                wire.append(item)
            elif m["role"] == "tool":
                wire.append({"role": "tool", "tool_call_id": m["id"], "content": m["text"]})
            else:
                wire.append({"role": "user", "content": m["text"]})
        body = {"model": self.model, "messages": wire, "max_tokens": _max_tokens()}      # no temperature: some models refuse it
        if system.startswith("DUTY_SCHEDULE_V1") and self.provider == "MINIMAX":
            # A narrow extraction call, separately validated; no chain-of-thought budget needed.
            body.update(response_format={"type": "json_object"}, thinking={"type": "disabled"})
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                             "parameters": t["parameters"]}} for t in tools]
        data = self._post(self.base + "/chat/completions", body, {"Authorization": "Bearer " + self.key}, timeout)
        message = data["choices"][0]["message"]
        calls = []
        for c in message.get("tool_calls") or []:
            try:
                arguments = json.loads(c["function"].get("arguments") or "{}")
            except ValueError:
                arguments = {}
            calls.append({"id": str(c.get("id", "")), "name": str(c["function"]["name"]),
                          "arguments": arguments if isinstance(arguments, dict) else {}})
        return {"text": "" if calls and not message.get("content") else _text_of(message), "tool_calls": calls}

    def _request_anthropic(self, messages, system, tools, timeout) -> dict:
        wire = []
        for m in messages:
            if m["role"] == "assistant":
                content = ([{"type": "text", "text": m["text"]}] if m.get("text") else [])
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                            for c in m.get("tool_calls") or []]
                wire.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["text"]}
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            else:
                wire.append({"role": "user", "content": m["text"]})
        body = {"model": self.model, "system": system, "messages": wire, "max_tokens": _max_tokens()}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        data = self._post(self.base + "/v1/messages", body,
                          {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, timeout)
        text, calls = "", []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append({"id": str(block.get("id", "")), "name": str(block.get("name", "")),
                              "arguments": block.get("input") if isinstance(block.get("input"), dict) else {}})
        return {"text": text, "tool_calls": calls}

    # --- calls ------------------------------------------------------------------------------------------

    def in_flight(self) -> int:
        return sum(1 for call in self.calls if not call.done())

    def submit(self, tag: str, messages: list, system: str, wall_left: float, tools=None, seconds: float = None):
        """Start a call in the background; None when the model is off or the run's limits say no.
        `seconds`: how long the caller is prepared to wait (retries included)."""
        if not self.enabled:
            return None
        timeout = min(self.call_timeout, wall_left - 45.0, seconds if seconds is not None else self.call_timeout)
        if len(self.calls) >= self.max_calls or timeout < 6.0 or self.in_flight() >= self.max_in_flight:
            return None
        if self.consecutive_failures >= 4:
            self.held_back += 1
            if self.held_back % 3:
                return None             # the service looks down: only try every third time
        call = Call(self, tag, messages, system, tools, timeout)
        self.calls.append(call)
        return call

    def collect(self, call):
        """The reply of a finished call (None while running or after a failure). Counts and logs once."""
        if call is None or not call.done():
            return None
        if not getattr(call, "_logged", False):
            call._logged = True
            if call.reply is not None:
                self.ok += 1
                self.consecutive_failures = 0
            else:
                self.failed += 1
                self.consecutive_failures += 1
                self.log(f"llm: {call.tag} failed ({call.error}) after {call.seconds:.1f}s; rules decide")
                if self.consecutive_failures >= 2 and len(self.providers) > 1:
                    self.provider_index = (self.provider_index + 1) % len(self.providers)
                    self.provider, self.protocol, self.key, self.base, self.model = self.providers[self.provider_index]
                    self.consecutive_failures, self.held_back, self.native_tools = 0, 0, True
                    self.log(f"llm: no answers from the last service; switching to provider {self.provider}, model {self.model}")
        return call.reply

    def wait(self, call, seconds: float):
        if call is None:
            return None
        started = time.monotonic()
        call.wait(seconds)
        self.wait_seconds += time.monotonic() - started
        return self.collect(call)
