#!/usr/bin/env python3
"""Agentic Observer agent for participant-agent-protocol-v4.

One JSON object per line on stdin, one per line on stdout, logs on stderr. See README.md.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

# third-party packages are installed next to the code by the build step (see observer.project.json)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import skymath as sm  # noqa: E402
from hazards import Hazards  # noqa: E402
from planner import Planner  # noqa: E402
from pointing import PointingModel  # noqa: E402
from state import Survey  # noqa: E402
from tracker import SkyTracker  # noqa: E402
from duty import DutyLog  # noqa: E402

try:    # the model stages are optional: without supervisor.py the agent is purely rule-based
    from supervisor import Supervisor  # noqa: E402
except ImportError:
    Supervisor = None

PROTOCOL = "participant-agent-protocol-v4"


def _env(name: str, default):
    return type(default)(os.environ.get(f"AO_{name}", default))


P_REPORT_FREE = _env("P_REPORT_FREE", 0.5)      # report with a free probe left when P(fault) stays above this
P_REPORT_LAST_FREE = _env("P_REPORT_LAST_FREE", 0.55)
P_REPORT_PAID = _env("P_REPORT_PAID", 0.55)
REPORT_PERSIST = _env("REPORT_PERSIST", 2)      # consecutive informative exposures above the threshold
REPORT_PERSIST_PAID = _env("REPORT_PERSIST_PAID", 3)
REPORT_COOLDOWN_HOURS = _env("REPORT_COOLDOWN_HOURS", 1.5)
REPORT_FRESH_HOURS = _env("REPORT_FRESH_HOURS", 20.0)   # the latest evidence may be from the end of last night
P_REPORT_SURE = _env("P_REPORT_SURE", 0.97)
P_DIAGNOSE = _env("P_DIAGNOSE", 0.12)            # above this, keep exposures short until the question is settled
DIAGNOSE_SECONDS = _env("DIAGNOSE_SECONDS", 600.0)
DIAGNOSE_EXPOSURES = _env("DIAGNOSE_EXPOSURES", 8)
PROBE_ANY = _env("PROBE_ANY", 1)                # 1: a dome probe may use targets that cannot gain score any more
OFFSET_UNKNOWN_MARGIN = _env("OFFSET_UNKNOWN_MARGIN", 0.0)    # extra edge margin (deg) while the pointing offset is unknown
OFFSET_RADIUS_SHARE = _env("OFFSET_RADIUS_SHARE", 0.7)        # share of the offset uncertainty kept as margin (the rest is explored)
FIRST_LOOK_SECONDS = _env("FIRST_LOOK_SECONDS", 0.0)   # cap of the first exposure of a night or of a new all-sky regime (0 = off)
CLOSED_PROBE_SECONDS = _env("CLOSED_PROBE_SECONDS", 90.0)
NIGHT_YIELD = _env("NIGHT_YIELD", 150.0)        # typical score of one night (for the value of a repair)
MAX_PAID_FALSE = _env("MAX_PAID_FALSE", 2)
REPORT_RECOVERY_HOURS = _env("REPORT_RECOVERY_HOURS", 6.0)  # bounded recovery probes, never a season-long lockout
CPU_SHARE = _env("CPU_SHARE", 0.70)             # share of the CPU budget the planner may use
VERBOSE = _env("VERBOSE", 1)
FIXED_EFFORT = _env("FIXED_EFFORT", 0.0)        # development: pin the search effort (0 = adaptive)
EFFORT_LEVELS = (1.0, 0.7, 0.5, 0.35, 0.22, 0.12, 0.05)
WALL_GUARD = _env("WALL_GUARD", 1)               # 1: watch the real-time cap and slow the run's real-time use before it is hit
WALL_SHARE = _env("WALL_SHARE", 0.85)            # share of the real time left that the rest of the survey may be projected to need
WALL_DEBUG = _env("WALL_DEBUG", 0)               # 1: log every projection (development)
WALL_MIN_SAMPLES = _env("WALL_MIN_SAMPLES", 400)  # decisions measured before the projection is trusted
# what to give up, mildest first: (ceiling of the search effort, shortest ordinary exposure in seconds)
WALL_LADDER = ((1.0, 0), (0.7, 0), (0.5, 0), (0.35, 0), (0.35, 360), (0.35, 540), (0.22, 780), (0.22, 1080),
               (0.22, 1500), (0.12, 1800), (0.12, 2400), (0.05, 3600))


_LOGFILE = open(os.environ["AO_LOGFILE"], "a") if os.environ.get("AO_LOGFILE") else None


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)
    if _LOGFILE is not None:
        print(text, file=_LOGFILE, flush=True)


class Agent:
    def __init__(self, init: dict):
        started = time.process_time()
        self.survey = Survey(init)
        self.tracker = SkyTracker(init["scoring"], log=log)
        self.hazards = Hazards(log=log)
        self.planner = Planner(self.survey, self.tracker, self.hazards, log=log)
        self.pointing = PointingModel(self.survey.side, self.survey.pitch, log=log)
        self.planner.margin_extra = OFFSET_UNKNOWN_MARGIN * OFFSET_RADIUS_SHARE
        self.limits = init.get("limits", {})
        self.night_seen = -1
        self.last_obs_time = None          # end of the last exposure used by the tracker
        self.closed_suspect = False        # the last exposure scored zero everywhere
        self.closed_probe_slot = -1
        self.suspect_field: list = []
        self.was_probe = False
        self.diagnose_count = 0
        self.after_closure = False
        self.level_unknown = True          # no measurement of the sky level yet under the current conditions
        self.allsky_seen = frozenset()
        self.free_left = self.survey.free_allowance
        self.reports = 0
        self.correct_reports = 0
        self.false_reports = 0
        self.paid_false = 0
        self.consecutive_reports = 0
        self.last_report_hours = -1e9
        self.forecast = []
        self.cpu_plan = 0.0
        self.cpu_ema = 0.0
        self.decisions = 0
        self.step_ema = 900.0
        self.last_now = None
        self.internal_errors = 0           # decisions that raised and were replaced by a wait
        # real-time guard: what the engine and the agent cost in real seconds, and what the rest will need
        self.t_done = None                 # monotonic clock when the previous answer was ready
        self.prev_day = False              # the previous answer was the wait through the day
        self.entries = 0                   # fibres assigned so far (the engine's ledger grows with them)
        self.growth_ema = -1.0             # fibres assigned per decision, recently (waits count as zero)
        self.gap_ema = 0.0
        self.fit_night = [0.0] * 5         # least squares of the engine's time per decision against the ledger size
        self.fit_day = [0.0] * 5           # the same for the wait through the day
        self.turn_ema = 0.0                # own real seconds per decision at night (model waits excluded)
        self.step_free = 900.0             # simulated seconds per decision while no minimum duration is imposed
        self.free_steps = 0
        self.cpu_left_real = 1e9
        self.rung = 0                      # position on WALL_LADDER
        self.effort_cap = 1.0
        self.wall_need = 0.0               # projected real seconds for the rest of the survey (model waits excluded)
        self.wall_checked = -1
        self.model_wait = 0.0              # real seconds spent waiting for model answers (no CPU)
        self.supervisor = Supervisor(self, log=log) if Supervisor is not None else None
        client = self.supervisor.client if self.supervisor is not None else None
        self.duty = DutyLog(self, init, client=client, log=client.log if client is not None else log)
        log(f"agent: {self.survey.n} targets ({int(self.survey.required.sum())} required), {len(self.survey.nights)} nights, "
            f"{self.survey.n_fibers} fibres; init {time.process_time() - started:.2f}s cpu")

    # --- decision ---------------------------------------------------------------------------------------

    def respond(self, payload: dict) -> dict:
        received = time.monotonic()
        self._note_gap(received)
        waited = self.model_wait
        cpu0 = time.process_time()
        try:
            action = self._respond(payload)
        finally:
            self.t_done = time.monotonic()
            self.prev_day = False
        cpu = time.process_time() - cpu0
        if action.get("action") == "observe":
            self.cpu_ema = cpu if self.cpu_ema == 0.0 else 0.9 * self.cpu_ema + 0.1 * cpu
            self.entries += len(action["assignments"])
        if self.survey.current_night(self.last_now or 0.0) is not None:
            added = float(len(action["assignments"])) if action.get("action") == "observe" else 0.0
            self.growth_ema = added if self.growth_ema < 0.0 else 0.995 * self.growth_ema + 0.005 * added
            own = max(0.0, self.t_done - received - (self.model_wait - waited))
            self.turn_ema = own if self.turn_ema == 0.0 else 0.99 * self.turn_ema + 0.01 * own
        self.prev_day = action.get("action") == "wait" and "until_utc" in action
        return action

    # --- real-time guard --------------------------------------------------------------------------------

    def _note_gap(self, received: float) -> None:
        """Real time between the previous answer and this request: the engine's own work. It grows with the
        ledger (the engine walks all scored fibres at every step), so it is fitted against the ledger size."""
        if self.t_done is None:
            return
        gap = max(0.0, received - self.t_done)
        if self.prev_day:
            fit = self.fit_day
        else:
            fit = self.fit_night
            if self.gap_ema > 0.0:
                gap = min(gap, 5.0 * self.gap_ema + 0.2)       # a calibration pause of the engine is not a trend
            self.gap_ema = gap if self.gap_ema == 0.0 else 0.98 * self.gap_ema + 0.02 * gap
        x = self.entries * 1e-4
        fit[0] += 1.0; fit[1] += x; fit[2] += x * x; fit[3] += gap; fit[4] += x * gap

    @staticmethod
    def _line(fit) -> tuple:
        """(seconds at an empty ledger, seconds per 10^4 ledger entries), neither negative."""
        n, sx, sxx, sy, sxy = fit
        if n < 1:
            return 0.0, 0.0
        den = n * sxx - sx * sx
        if n < 3 or den <= 1e-12:
            return sy / n, 0.0
        slope = (n * sxy - sx * sy) / den
        if slope <= 0.0:
            return sy / n, 0.0
        base = (sy - slope * sx) / n
        if base < 0.0:
            return 0.0, sxy / max(sxx, 1e-12)
        return base, slope

    def _wall_context(self, now: float) -> dict:
        """What the projection needs, measured now."""
        s = self.survey
        k0, k1 = self._line(self.fit_night)
        d0, d1 = self._line(self.fit_day)
        # the ledger cannot grow for ever: every new entry improves a target that is still open. How many
        # entries an open target takes is read off the season so far (targets are often exposed more than once)
        still_open = float((s.cur < s.top_mult - 0.004).sum())
        per_target = self.entries / max(1.0, float((s.tries > 0).sum()))
        room = (max(1.5, 1.2 * per_target) * still_open + 200.0) * 1e-4
        return {"k0": k0, "k1": k1, "d0": d0, "d1": d1, "ledger": self.entries * 1e-4, "room": room,
                "growth": max(0.0, self.growth_ema) * 1e-4, "seconds": s.seconds_left_in_season(now),
                "nights": sum(1 for _, end, _ in s.nights if end > now)}

    def _wall_projection(self, c: dict, rung: int) -> float:
        """Real seconds the rest of the survey would take on this rung of the ladder (model waits excluded)."""
        cap, d_min = WALL_LADDER[rung]
        d_min = min(float(d_min), float(self.survey.max_exposure))
        step = max(300.0, self.step_free, 1.15 * d_min)
        # measured on two 40-night cards the projection ran between 0.6 and 1.3 of what the rest really took
        # (steps get shorter late in a season). Too early costs little, too late costs the season's end: lean early
        decisions = c["seconds"] / step * 1.25 + 20.0
        # ledger size summed over the decisions still to come: grows as lately until the open targets run out
        until_full = c["room"] / max(c["growth"], 1e-12)
        if decisions <= until_full:
            area = c["ledger"] * decisions + 0.5 * c["growth"] * decisions * decisions
        else:
            area = c["ledger"] * decisions + 0.5 * c["growth"] * until_full * until_full + c["room"] * (decisions - until_full)
        engine = decisions * c["k0"] + c["k1"] * area + c["nights"] * (c["d0"] + c["d1"] * area / decisions)
        effort = max(self.planner.effort, 1e-3)
        own = self.turn_ema * (0.25 + 0.75 * min(cap, effort) / effort)
        # the agent's own time is bounded by its CPU budget whatever the projection says (the pace keeps it inside)
        return engine + min(decisions * own, self.cpu_left_real + 0.01 * decisions)

    def _wall_guard(self, payload: dict, now: float) -> None:
        """Will the run fit under the real-time cap? Engine time is outside the agent's CPU budget but inside
        the cap, and it grows through the season. When the projection does not fit: first search less, then
        make fewer, longer exposures; when even that does not fit, plan for the season to end early."""
        wall = payload.get("wallclock") or {}
        if not WALL_GUARD or "wall_remaining_seconds" not in wall:
            return
        left = float(wall["wall_remaining_seconds"])
        self.cpu_left_real = float(wall.get("remaining_real_cpu_seconds", wall.get("remaining_seconds", 1e9)))
        supervisor = self.supervisor
        model = max(0.0, float(getattr(supervisor, "budget_left", 0.0))) if supervisor is not None and getattr(supervisor, "enabled", False) else 0.0
        allowed = WALL_SHARE * left - model
        rung = self.rung
        c = self._wall_context(now)
        self.wall_need = self._wall_projection(c, rung)
        if WALL_DEBUG:
            log(f"real time check: left {left:.1f}s need {self.wall_need:.1f}s step {rung} k0 {c['k0'] * 1000:.1f}ms k1 {c['k1'] * 1000:.2f}ms "
                f"d0 {c['d0'] * 1000:.0f}ms d1 {c['d1'] * 1000:.1f}ms ledger {self.entries} growth {c['growth'] * 1e4:.2f} room {c['room'] * 1e4:.0f} "
                f"turn {self.turn_ema * 1000:.0f}ms step_free {self.step_free:.0f}s samples {self.fit_night[0]:.0f}")
        if self.fit_night[0] < WALL_MIN_SAMPLES:
            return
        top = len(WALL_LADDER) - 1
        while rung < top and self._wall_projection(c, rung) > allowed:
            rung += 1
        if rung == self.rung and rung > 0 and self._wall_projection(c, rung - 1) < 0.75 * allowed:
            rung -= 1
        need = self._wall_projection(c, rung)
        scale = 1.0 if need <= allowed else max(0.05, allowed / max(need, 1e-6))
        if rung != self.rung or abs(scale - self.planner.season_scale) > 0.05:
            cap, d_min = WALL_LADDER[rung]
            log(f"real time: {left:.0f}s left before the cap, the rest of the survey needs about {need:.0f}s (+{model:.0f}s model budget); "
                f"engine {c['k0'] * 1000:.0f} ms/decision + {c['k1'] * 1000:.1f} ms per 10^4 ledger entries ({self.entries} so far, "
                f"room for {c['room'] * 1e4:.0f} more); "
                f"step {rung}: effort <= {cap:.2f}, exposures >= {min(d_min, int(self.survey.max_exposure))}s, season share {scale:.2f}")
        if scale != self.planner.season_scale:
            self.planner.plan_age = 10 ** 9
        self.rung = rung
        self.effort_cap, self.planner.min_duration = WALL_LADDER[rung][0], float(min(WALL_LADDER[rung][1], self.survey.max_exposure))
        self.planner.season_scale = scale
        self.wall_need = need

    def ingest(self, payload: dict):
        """Digest everything the request brings: messages, the result of the previous action, the bulletin now in force."""
        s = self.survey
        now = sm.unix(payload["now_utc"])
        hours = (now - s.start) / 3600.0
        messages = payload.get("new_messages") or []
        bulletins = [m for m in messages if m.get("record_type") == "bulletin"]
        previous_notices = set(self.hazards.notices)
        self.hazards.on_bulletins(bulletins, payload.get("latest_bulletin"))
        night = s.current_night(now)
        night_index = night[0] if night else self.night_seen
        last = payload.get("last_result") or {}
        if last.get("action") == "report":
            self._on_report_result(last, hours)
        elif last.get("action") == "observe":
            self._on_observe_result(last, now, hours, previous_notices, night_index)
        for m in messages:
            kind = m.get("record_type")
            if kind == "forecast":
                self.forecast = m.get("notices", [])
            elif kind == "state_resync":
                self._on_resync(m)      # after the last exposure is in the ledger: it may be part of the loss
        # the bulletin of the slot that starts now may change the outlook
        self.tracker.advance(night_index, self.hazards.notices, 0.0)
        return now, hours, night

    def _respond(self, payload: dict) -> dict:
        s = self.survey
        self.decisions += 1
        now, hours, night = self.ingest(payload)
        self.planner.set_requests(payload.get("active_requests") or [], now)
        self._pace(payload, now)
        self.duty.ingest(payload, now)

        if night is None:
            nxt = s.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": sm.format_unix(nxt), "reason": "daytime"}
        night_index, night_start, night_end = night
        if night_index != self.night_seen:
            self.night_seen = night_index
            self.closed_suspect = False
            self.suspect_field = []
            self.hazards.probe_avoid = []
            self.planner.plan_age = 10 ** 9      # new night: refresh the season plan
            self.level_unknown = True
            if self.supervisor is not None:
                self.supervisor.night_start(payload, now, night_index)
        elif self.supervisor is not None:
            self.supervisor.poll()
        if night_end - now < 30.0:
            nxt = s.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "survey over"}
            return {"action": "wait", "until_utc": sm.format_unix(nxt), "reason": "night ending"}
        if self.hazards.site_closed():
            return self._wait_slot(now, night_start, night_end, "bulletin: dome closed by weather")
        scheduled = self.duty.action(now)
        if scheduled is not None:
            return scheduled
        report = self._maybe_report(hours, now, payload)
        if report is not None:
            return report
        max_duration = 1e9
        if self.closed_suspect:
            # the last exposure scored nothing anywhere: an unannounced dome closure. Closures start and end on
            # slot boundaries, so test each new slot with one short exposure and otherwise wait for the next slot.
            into_slot = (now - night_start) % s.slot_seconds
            if into_slot > 45.0 and self.closed_probe_slot == int((now - night_start) // s.slot_seconds):
                return self._wait_slot(now, night_start, night_end, "dome seems closed: wait for the next slot")
            self.closed_probe_slot = int((now - night_start) // s.slot_seconds)
            max_duration = CLOSED_PROBE_SECONDS
        elif self.tracker.p_fault() > P_DIAGNOSE and self.free_left + MAX_PAID_FALSE - self.paid_false > 0:
            # a fault is suspected but not established: a few short exposures settle it quickly, and little is lost
            self.diagnose_count += 1
            if self.diagnose_count <= DIAGNOSE_EXPOSURES:
                max_duration = DIAGNOSE_SECONDS
        else:
            self.diagnose_count = 0
        allsky = frozenset(kind for kind, direction in self.hazards.notices if direction == "ALL")
        if allsky != self.allsky_seen:
            self.allsky_seen = allsky
            self.level_unknown = True
        if self.level_unknown and FIRST_LOOK_SECONDS > 0:
            # new night or new all-sky weather: the sky level may have moved; look before committing to a long exposure
            max_duration = min(max_duration, FIRST_LOOK_SECONDS)
        max_duration = self.duty.max_duration(now, max_duration)
        if max_duration < s.min_exposure:
            return {"action": "wait", "until_utc": sm.format_unix(now + max_duration),
                    "reason": "wait for published operating boundary"}
        action = self.planner.plan(now, night_end, max_duration, probe=self.closed_suspect and bool(PROBE_ANY))
        self.was_probe = self.closed_suspect and action is not None
        if action is None:
            return self._wait_slot(now, night_start, night_end, "nothing useful is up")
        return action

    def _wait_slot(self, now: float, night_start: float, night_end: float, reason: str) -> dict:
        slot = self.survey.slot_seconds
        to_boundary = slot - ((now - night_start) % slot)
        if self.planner.min_duration > slot:
            # real time is short (see _wall_guard): every wait is an engine step too, so wait through several slots
            to_boundary += slot * (int(self.planner.min_duration // slot) - 1)
        seconds = int(max(60, min(3600, round(to_boundary))))
        return {"action": "wait", "duration_seconds": seconds, "reason": reason}

    # --- pace -------------------------------------------------------------------------------------------

    def _pace(self, payload: dict, now: float) -> None:
        """Search effort from the CPU budget left and the decisions still to come.

        The effort moves in fixed steps and only when the measured cost clearly leaves the budget, so that
        runs on the same machine make the same decisions (timing noise must not change the plan)."""
        wall = payload.get("wallclock") or {}
        cpu_left = float(wall.get("remaining_real_cpu_seconds", wall.get("remaining_seconds", 1e9)))
        if self.last_now is not None and self.survey.current_night(now) is not None:
            step = now - self.last_now
            if 0 < step <= 3600:
                self.step_ema = 0.98 * self.step_ema + 0.02 * step
                if self.planner.min_duration <= 0:
                    # a slow average: single nights differ a lot (closed dome: long waits; bright fields: short exposures)
                    self.free_steps += 1
                    self.step_free += (step - self.step_free) * max(0.002, 1.0 / self.free_steps)
        self.last_now = now
        if self.decisions - self.wall_checked >= 50:
            self.wall_checked = self.decisions
            try:
                self._wall_guard(payload, now)
            except Exception as exc:  # noqa: BLE001 - the guard is a precaution: it must never cost a decision
                if self.wall_checked < 10 ** 9:
                    log(f"real-time guard switched off after an error: {type(exc).__name__}: {exc}")
                self.wall_checked = 10 ** 12
        if FIXED_EFFORT > 0:
            self.planner.effort = FIXED_EFFORT
            return
        decisions_left = max(1.0, self.survey.seconds_left_in_season(now) / max(700.0, self.step_ema)) * 1.1 + 20.0
        budget = CPU_SHARE * cpu_left / decisions_left
        levels = EFFORT_LEVELS
        k = min(range(len(levels)), key=lambda n: abs(levels[n] - self.planner.effort))
        if self.cpu_ema > 1.15 * budget and k + 1 < len(levels):
            k += 1
            self.cpu_ema *= levels[k] / levels[k - 1]      # expected cost at the new level, until measured
        elif self.cpu_ema < 0.45 * budget * levels[k] / levels[max(0, k - 1)] and k > 0 and self.decisions % 25 == 0:
            k -= 1
            self.cpu_ema *= levels[k] / levels[k + 1]
        while levels[k] > self.effort_cap + 1e-9 and k + 1 < len(levels):
            k += 1                                   # the real-time guard's ceiling
        if cpu_left < 15.0:
            k = len(levels) - 1
        if levels[k] != self.planner.effort:
            log(f"pace: effort {levels[k]:.2f} (cpu/decision {self.cpu_ema * 1000:.0f} ms, budget {budget * 1000:.0f} ms, cpu left {cpu_left:.0f}s, "
                f"~{decisions_left:.0f} decisions left)")
            self.planner.effort = levels[k]

    # --- results ----------------------------------------------------------------------------------------

    def _on_observe_result(self, result: dict, now: float, hours: float, previous_notices: set, night_index: int) -> None:
        s = self.survey
        p = self.planner.pending
        self.planner.pending = None
        if p is None:
            return
        hits = {h["target_id"]: float(h["score"]) for h in result.get("hits", [])}
        idx = p["idx"]
        start = p["start"]
        end = min(start + p["duration"], p["night_end"], s.end)
        T = max(1.0, end - start)
        score = np.array([hits.get(s.ids[int(i)], -1.0) for i in idx])
        hit = score >= 0.0
        s.misses[idx[~hit]] += 1
        self._offset_evidence(p, hit)
        positive = score > 0.0
        ratio = np.where(positive, score / s.weight[idx], 0.0)
        # exact sky model of each target over the exposure actually taken
        pieces = max(1, int(math.ceil(T / 120.0)))
        mids = start + (np.arange(pieces) + 0.5) * (T / pieces)
        eq = s.eq[idx]
        m = np.zeros(len(idx))
        for t in mids:
            h = s.site.horizontal(eq, s.site.lst(float(t)))
            alt = np.degrees(np.arcsin(np.clip(h[:, 2], -1.0, 1.0)))
            m += sm.Moon(float(t), s.site, s.lunar_model).lunar_factor(eq) / (s.q0 * sm.normalized_airmass(alt) ** s.beta)
        m /= pieces
        a = s.flux[idx] * T * m / s.f0t0
        h_end = s.site.horizontal(eq, s.site.lst(end))
        alt_end, az_end = sm.unit_to_altaz(h_end)
        # the hazards seen at any time during the exposure decide which hits are clean evidence
        clean = self.hazards.clean(p["alt0"], p["az0"]) & self.hazards.clean(alt_end, az_end) & (p["dirf"] >= 0.999)
        any_positive = bool(positive.any())
        if hit.any() and not any_positive:
            # nothing scored: the dome is closed, or the whole field sits behind terrain / a launch sector.
            # The next exposure is a short probe somewhere else; its result tells which.
            if not self.closed_suspect:
                risky = self.hazards.terrain_risk(p["alt0"], p["az0"]) > 0.0
                self.suspect_field = [(float(z), float(h)) for z, h in zip(p["az0"][risky], p["alt0"][risky])]
                self.hazards.probe_avoid = list(self.suspect_field)
            self.closed_suspect = True
        elif hit.any():
            if self.closed_suspect:
                self.after_closure = True
                if self.suspect_field and self.was_probe and not any(k in ("rocket_launch",) for k, _ in self.hazards.during):
                    # the probe scored at once, so the dome was open: the earlier field was blocked
                    self.hazards.block_field(self.suspect_field, now)
                    log(f"hazards: field at az {self.suspect_field[0][0]:.0f} alt {self.suspect_field[0][1]:.0f} is blocked ({len(self.suspect_field)} points)")
            self.closed_suspect = False
            self.suspect_field = []
            self.hazards.probe_avoid = []
        self.was_probe = False
        for k in np.flatnonzero(hit & ~positive):
            if any_positive:
                self.hazards.on_zero_hit(float(p["alt0"][k]), float(p["az0"][k]), now)
        if positive.any():
            self.hazards.on_positive_hits(np.minimum(p["alt0"], alt_end)[positive], p["az0"][positive])
        # tracker: advance to this exposure, then update with its clean hits. An exposure during which an
        # all-sky notice or an earthquake notice came or went mixes two regimes: it only advances the clock.
        slots = T / s.slot_seconds
        regime = lambda notices: {kind for kind, direction in notices if direction == "ALL" or kind == "earthquake"}
        mixed = regime(previous_notices) != regime(self.hazards.during) or regime(previous_notices) != regime(self.hazards.notices)
        self.tracker.advance(night_index, previous_notices, slots)
        use = positive & clean
        summary = {}
        if use.any() and not mixed:
            summary = self.tracker.observe(a[use], m[use], ratio[use], p["program"], hours, after_closure=self.after_closure)
            self.after_closure = False
            self.level_unknown = False
            if not self.hazards.site_closed():
                self.planner.note_level(summary["level"])
        if self.supervisor is not None:
            self.supervisor.note_exposure(start, T, summary, self.hazards.during)
        # completion factor of each hit: undo the program multiplier
        declared = s.multipliers[p["program"]]
        factor = np.zeros(len(idx))
        if any_positive:
            pd, pb, pk = self.tracker.band_probabilities(m)
            p_match = {"DARK": pd, "BRIGHT": pb, "BACKUP": pk}[p["program"]]
            matched = (ratio > s.mismatch * (1.0 + 1e-6)) | (p_match >= 0.25)
            factor = np.where(positive, np.minimum(1.0, np.where(matched, ratio / declared, ratio / s.mismatch)), 0.0)
        observe_index = int(result.get("observe_index", s.observes))
        s.observes = observe_index + 1
        if hit.any():
            s.record(observe_index, start, end, idx[hit], factor[hit], ratio[hit])
        if VERBOSE and summary:
            pred = np.minimum(1.0, p["raw"][positive])
            got = factor[positive]
            log(f"obs {observe_index} {sm.format_unix(start)[5:16]} T={int(T)} {p['program']} hit {int(hit.sum())}/{len(idx)} "
                f"pred/got {pred.mean():.2f}/{got.mean():.2f} level {summary['level']:.3f} W {summary['w']:.3f} eta {summary['eta']:.3f} "
                f"pF {summary['p_fault']:.3f}")

    def _on_resync(self, message: dict) -> None:
        s = self.survey
        window = message.get("invalidated_window") or {}
        first = int(window.get("action_index_start", 0))
        last = int(window.get("action_index_end_exclusive", 0))
        count = s.invalidate(first, last)
        # the engine's best scores are authoritative for what is still held
        best = {row["target_id"]: float(row["best_score"]) for row in message.get("best_scores", [])}
        cur = np.zeros(s.n)
        for target_id, value in best.items():
            i = s.index_of.get(target_id)
            if i is not None:
                cur[i] = value / s.weight[i]
        drift = float(np.abs(cur - s.cur).max()) if s.n else 0.0
        s.cur = cur
        s.factor = np.where(cur > 0, np.minimum(s.factor, 1.0), 0.0)
        log(f"state_resync: observes [{first}, {last}) lost ({count} of ours); {len(best)} targets keep a score; max drift {drift:.4f}")

    # --- pointing offset (hard-mode cards): see pointing.py --------------------------------------------

    def _offset_evidence(self, p: dict, hit: np.ndarray) -> None:
        planner, pointing = self.planner, self.pointing
        was_zero = float(np.abs(pointing.offset).max()) == 0.0
        if pointing.add(p["cmd"], p["h0"], p["fibres"], hit) and was_zero:
            self.survey.misses[:] = 0          # the misses were the offset, not the targets
        planner.offset = pointing.offset.copy()
        planner.margin_extra = pointing.margin(0.0, OFFSET_UNKNOWN_MARGIN) * OFFSET_RADIUS_SHARE

    # --- instrument faults ------------------------------------------------------------------------------

    def _maybe_report(self, hours: float, now: float, payload: dict = None):
        """Report when the expected value is positive: P(fault) x (reward + what a repair is worth for the rest
        of the season) against (1 - P) x the cost of being wrong. A report costs no time and answers at once;
        waiting one more exposure for evidence costs almost nothing, so the probability must also persist."""
        tr = self.tracker
        if self.consecutive_reports >= 1 or hours - self.last_report_hours < REPORT_COOLDOWN_HOURS:
            return None
        history = tr.fault_history
        if not history:
            return None
        free = self.free_left > 0
        # The lifetime error count is diagnostic, not an irreversible ban on repairing
        # future faults. After the ordinary risk budget is spent, require much stronger
        # evidence and a long cooldown; cap the evidence window as the season grows.
        persist = REPORT_PERSIST if free else REPORT_PERSIST_PAID + min(self.paid_false, MAX_PAID_FALSE)
        if free and history[-1][1] >= P_REPORT_SURE:
            persist = 1                      # overwhelming evidence and nothing to lose: do not wait
        recent = history[-persist:]
        if len(recent) < persist or hours - recent[-1][0] > REPORT_FRESH_HOURS:
            return None                      # a fault persists across nights, but the evidence must be fresh
        p = min(prob for _, prob in recent)
        if p < (P_REPORT_FREE if self.free_left >= 2 else P_REPORT_LAST_FREE if free else P_REPORT_PAID):
            return None
        if not free:
            if self.paid_false >= MAX_PAID_FALSE:
                if p < P_REPORT_SURE or hours - self.last_report_hours < REPORT_RECOVERY_HOURS:
                    return None
            nights_left = self.survey.seconds_left_in_season(now) / 28800.0
            loss_share = 0.6 * max(0.0, 1.0 - tr.eta_mean() / 0.95)          # yield lost while the fault stays
            benefit = self.survey.report_reward + NIGHT_YIELD * nights_left * loss_share
            if p * benefit < (1.0 - p) * self.survey.report_penalty * 1.3:
                return None
        if self.supervisor is not None and payload is not None and not self.supervisor.allow_report(payload, now, hours, p, self.free_left):
            return None
        self.last_report_hours = hours
        self.reports += 1
        self.consecutive_reports += 1
        log(f"report at {sm.format_unix(now)}: P(fault) {p:.3f}, eta {tr.eta_mean():.2f}, free probes left {max(0, self.free_left)}")
        return {"action": "report", "reason": f"instrument efficiency below what weather explains (P={p:.2f})"}

    def _on_report_result(self, result: dict, hours: float) -> None:
        correct = bool(result.get("correct"))
        self.duty.on_report(correct, self.survey.start + hours * 3600.)
        self.tracker.on_report(correct)
        self.diagnose_count = 0
        self.consecutive_reports = 0
        if correct:
            self.correct_reports += 1
            self.free_left = self.survey.free_allowance
            log(f"report correct: fault repaired (delta {result.get('score_delta')})")
        else:
            self.false_reports += 1
            if self.free_left <= 0:
                self.paid_false += 1
            self.free_left -= 1
            log(f"report false (delta {result.get('score_delta')}); free probes left {max(0, self.free_left)}")


def validate(action: dict, agent: Agent) -> dict:
    """Last line of defence: never send an action the engine would reject."""
    s = agent.survey
    kind = action.get("action")
    if kind == "observe":
        pointing = action["pointing"]
        alt, az = float(pointing["alt_deg"]), float(pointing["az_deg"])
        if not (math.isfinite(alt) and math.isfinite(az) and 0.0 <= alt <= 90.0 and 0.0 <= az < 360.0):
            raise ValueError("pointing out of range")
        duration = int(action["duration_seconds"])
        if not s.min_exposure <= duration <= s.max_exposure:
            raise ValueError("duration out of range")
        seen = set()
        for fibre, target in action["assignments"].items():
            if not 0 <= int(fibre) < s.n_fibers or target not in s.index_of or target in seen:
                raise ValueError("bad assignment")
            seen.add(target)
        if action["program"] not in s.multipliers:
            raise ValueError("bad program")
    elif kind == "wait":
        if "until_utc" not in action:
            action["duration_seconds"] = int(max(s.min_exposure, min(s.max_exposure, int(action["duration_seconds"]))))
    return action


def main() -> int:
    agent = None
    errors = []          # tracebacks of decisions that failed (each cost one wait instead of the run)
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        kind = message.get("message_type")
        if kind == "initialize":
            agent = Agent(message["payload"])
        elif kind == "decision_request":
            try:
                action = validate(agent.respond(message["payload"]), agent)
            except Exception as exc:  # noqa: BLE001 - never crash a run: lose one slot instead
                import traceback
                errors.append(f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}")
                del errors[3:-1]        # keep the first three and the latest
                log(f"error {errors[-1]}")
                if agent is not None:
                    agent.planner.pending = None
                    agent.internal_errors += 1
                action = {"action": "wait", "duration_seconds": 900, "reason": "internal error"}
            print(json.dumps({"protocol_version": PROTOCOL, "message_type": "decision_response",
                              "decision_sequence": message["decision_sequence"], **action}, separators=(",", ":")), flush=True)
        elif kind == "finish":
            payload = message.get("payload", {})
            if agent is not None:
                # the run log keeps only its tail: repeat what a reader must not miss
                for text in errors[:4]:
                    log(f"error (repeated at the end) {text}")
                supervisor = getattr(agent, "supervisor", None)
                if supervisor is not None and hasattr(supervisor, "stats_line"):
                    log(supervisor.stats_line())
                log(agent.duty.stats_line())
                log(f"finished: {payload.get('termination_reason')} observes={agent.survey.observes} reports={agent.reports} "
                    f"(correct {agent.correct_reports}, false {agent.false_reports}) plans={agent.planner.stats['plans']} "
                    f"candidates={agent.planner.stats['candidates']} repriced={agent.planner.stats.get('repriced', 0)} "
                    f"effort={agent.planner.effort:.2f} model_wait={agent.model_wait:.0f}s time_step={agent.rung} internal_errors={agent.internal_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
