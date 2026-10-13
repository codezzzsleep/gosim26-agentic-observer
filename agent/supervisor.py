"""Glue between the agent's state and the model stages (advisor.py).

The supervisor gathers what the model should read, asks, waits for the answer, and applies validated answers
inside fixed, safe ranges:

    night briefing     -> sectors to keep away from tonight, exposure caution, push/hold of must-observe targets
    fault adjudication -> veto or go for a report that would cost something
    season review      -> how firmly the planner holds targets for one complete exposure (tool-calling loop)

Timing. A reasoning model takes 15-30 s per answer, and in that real time the simulated survey would run through
several nights. So a consultation is a pause: the survey waits for the answer (waiting costs no CPU budget) and
then goes on with it. That makes real time the resource to manage:

  * the run has a hard real-time cap; the supervisor measures how much real time a decision takes and never
    spends on the model what the rest of the survey still needs (plus a reserve);
  * a card may spend at most MODEL_WALL_BUDGET seconds waiting for the model in all;
  * consultations are therefore few and placed where they matter: reviews at fixed points of the season,
    briefings on nights with news spread over the season, a verdict when a report would cost points;
  * one request at a time, and patience with "too many requests" (several cards share one key).

Without a model (no key, or OBSERVER_MODEL_DISABLED=1) nothing here runs and the rule defaults stand.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np

import skymath as sm
from advisor import Advisor
from llm import LLMClient


def _env(name: str, default):
    return type(default)(os.environ.get(f"AO_{name}", default))


MODEL_WALL_BUDGET = _env("MODEL_WALL_BUDGET", 300.0)     # real seconds one card may spend waiting for the model
WALL_RESERVE = _env("WALL_RESERVE", 150.0)               # real seconds always kept back from the hard cap
BRIEF_WAIT_SECONDS = _env("BRIEF_WAIT_SECONDS", 70.0)    # longest wait for one briefing
VERDICT_WAIT_SECONDS = _env("VERDICT_WAIT_SECONDS", 70.0)
REVIEW_WAIT_SECONDS = _env("REVIEW_WAIT_SECONDS", 100.0)   # a review is two or more requests
REVIEW_POINTS = (0.12, 0.35, 0.60, 0.82)                 # shares of the season at which the campaign is reviewed
BRIEF_SLOTS = _env("BRIEF_SLOTS", 6)                     # briefings per season: the first night with news in each part
QUIET_KINDS = ("cold_snap", "terrain_obstruction")       # notices that need no reading: a night with only these is routine
CAUTION_SCALE = {"low": 1.0, "normal": 1.0, "high": 0.94}  # tonight's planning level x this
REQUIRED_SCALE = {"hold": 0.6, "normal": 1.0, "push": 1.6}  # tonight's must-observe ramp x this
HOLD_RANGE = _env("HOLD_RANGE", 0.10)                       # the review may move the hold strength by +- this
ASK_BELOW = _env("ASK_BELOW", 0.92)                         # fault probability below which a costly report is put to the model
VETO_CONFIDENCE = _env("VETO_CONFIDENCE", 0.6)
VETO_HOURS = _env("VETO_HOURS", 3.0)


class Supervisor:
    def __init__(self, agent, log=lambda text: None):
        self.agent = agent
        self.client = LLMClient(log=log)
        self.log = self.client.log
        self.advisor = Advisor(self.client, log=self.log)
        self.enabled = self.client.enabled
        self.sky_nights: list = []            # [night date, median level, hours observed]
        self.tonight_levels: list = []
        self.tonight_seconds = 0.0
        self.tonight_date = None
        n_nights = max(1, len(agent.survey.nights))
        # cards evaluated side by side share one key: a small random shift keeps them from all asking at once
        self.rng = random.Random(time.time_ns() ^ os.getpid())
        shift = self.rng.uniform(-0.03, 0.03)
        self.review_nights = sorted({min(n_nights - 1, max(1, int(round((p + shift) * n_nights)))) for p in REVIEW_POINTS})
        slots = max(1, int(BRIEF_SLOTS))
        self.brief_from = [min(n_nights - 1, max(0, int(round((k + 0.25 + shift) * n_nights / slots)))) for k in range(slots)]
        self.brief_done = 0                    # briefing slots used or expired
        self.budget_left = float(MODEL_WALL_BUDGET)
        self.latency = 25.0                    # typical real seconds of one answer (measured as the run goes)
        self.decision_wall = 0.0               # real seconds one decision takes without model waits (measured)
        self.last_tick = None
        self.waited_since_tick = 0.0
        self.skipped = 0                       # consultations not started for lack of real time
        self.veto_until_hours = -1e9
        self.exposures: list = []             # recent exposures for the fault table
        self.applied = {"briefing": 0, "verdict": 0, "veto": 0, "review": 0}      # model answers that took effect
        if self.enabled:
            log(f"model stages on: provider {self.client.provider}, {self.client.protocol} protocol, model {self.client.model}")
        else:
            log("model stages off (no key or OBSERVER_MODEL_DISABLED=1): rules only")

    def stats_line(self) -> str:
        """One line for the end of the run log: did the model stages really work?"""
        if not self.enabled:
            return "model stages: off (rules only)"
        c, a = self.client, self.applied
        return (f"model stages: provider {c.provider} model {c.model} calls={len(c.calls)} ok={c.ok} failed={c.failed} "
                f"waited={MODEL_WALL_BUDGET - self.budget_left:.0f}s of {MODEL_WALL_BUDGET:.0f}s skipped={self.skipped} "
                f"latency~{self.latency:.0f}s applied: briefings={a['briefing']} verdicts={a['verdict']} (vetoes {a['veto']}) reviews={a['review']}")

    # --- wall clock --------------------------------------------------------------------------------------

    def _wall_left(self, payload: dict) -> float:
        wall = payload.get("wallclock") or {}
        return float(wall.get("wall_remaining_seconds", wall.get("remaining_seconds", 1e9)))

    def _tick(self) -> None:
        """Called once per decision: measures the real time a decision takes when nobody waits for the model."""
        clock = time.monotonic()
        if self.last_tick is not None:
            own = max(0.0, clock - self.last_tick - self.waited_since_tick)
            self.decision_wall = own if self.decision_wall == 0.0 else 0.95 * self.decision_wall + 0.05 * min(own, 10.0 * self.decision_wall + 1.0)
        self.last_tick = clock
        self.waited_since_tick = 0.0

    def _allowance(self, payload: dict, now: float, ceiling: float) -> float:
        """Real seconds that may be spent waiting for the model now: within the card's model budget, and never
        from the time the rest of the survey needs before the hard cap."""
        agent = self.agent
        decisions_left = agent.survey.seconds_left_in_season(now) / max(500.0, agent.step_ema)
        per_decision = self.decision_wall if self.decision_wall > 0 else 2.0 * max(agent.cpu_ema, 0.05) + 0.08
        needed = 1.3 * decisions_left * per_decision
        # the core's own projection also knows that the engine gets slower as the season's ledger grows
        needed = max(needed, 1.1 * float(getattr(agent, "wall_need", 0.0) or 0.0))
        spare = self._wall_left(payload) - needed - WALL_RESERVE
        return max(0.0, min(ceiling, self.budget_left, 0.6 * spare))

    def _spent(self, started: float, answered: bool) -> None:
        waited = time.monotonic() - started
        self.budget_left -= waited
        self.waited_since_tick += waited
        self.agent.model_wait += waited
        if answered:
            self.latency = 0.7 * self.latency + 0.3 * waited

    def _worth_asking(self, allowance: float, requests: int = 1) -> bool:
        """Ask only when the answer is likely to arrive within the allowance."""
        if allowance >= min(0.9 * self.latency * requests, 20.0 * requests):
            return True
        self.skipped += 1
        return False

    # --- records -----------------------------------------------------------------------------------------

    def note_exposure(self, start: float, duration: float, summary: dict, notices: set) -> None:
        if summary:
            self.tonight_levels.append(float(summary["level"]))
            self.exposures.append([sm.format_unix(start)[5:16], int(duration), round(summary["w"], 3), round(summary["eta"], 3),
                                   round(summary["p_fault"], 3), sorted(f"{k} {d}" for k, d in notices)])
            del self.exposures[:-40]
        self.tonight_seconds += duration

    def _close_night(self) -> None:
        if self.tonight_date is not None and self.tonight_levels:
            self.sky_nights.append([self.tonight_date, round(float(np.median(self.tonight_levels)), 3), round(self.tonight_seconds / 3600.0, 1)])
        self.tonight_levels, self.tonight_seconds = [], 0.0

    # --- 1. night briefing ---------------------------------------------------------------------------------

    def night_start(self, payload: dict, now: float, night_index: int) -> None:
        """Reset tonight's posture to the rule defaults; on the nights chosen for it, review the campaign or get a
        briefing for tonight, and wait for the answer."""
        agent, planner, hazards = self.agent, self.agent.planner, self.agent.hazards
        self._close_night()
        night_date = agent.survey.nights[night_index][2]
        self.tonight_date = night_date
        hazards.extra_avoid = set()
        planner.level_scale = 1.0
        planner.req_scale = 1.0
        if not self.enabled:
            return
        self._tick()
        self._take_late_review()
        if night_index % 10 == 9:
            self.log(self.stats_line())          # the run log keeps only its tail: say now and then how the stages are doing

        # --- season review at its points of the season ---
        if self.review_nights and night_index >= self.review_nights[0]:
            self.review_nights.pop(0)
            allowance = self._allowance(payload, now, REVIEW_WAIT_SECONDS)
            if self._worth_asking(allowance, requests=2):
                started = time.monotonic()
                self.advisor.start_review(self._review_snapshot(night_date, night_index), self._wall_left(payload), allowance)
                review = self.advisor.wait_review(allowance)
                self._spent(started, review is not None)
                self._apply_review(review)
            return                                # one consultation a night

        # --- night briefing: the first night with news in each part of the season ---
        if self.brief_done >= len(self.brief_from) or night_index < self.brief_from[self.brief_done]:
            return
        following = self.brief_from[self.brief_done + 1] if self.brief_done + 1 < len(self.brief_from) else len(agent.survey.nights)
        tonight = [n for n in agent.forecast if night_date in n.get("nights", [])]
        later = [n for n in agent.forecast if any(d > night_date for d in n.get("nights", []))]
        bulletin = sorted(hazards.notices)
        news = any(kind not in QUIET_KINDS for kind in [n.get("event_kind") for n in tonight] + [k for k, _ in bulletin])
        if not news and self.applied["briefing"] > 0:
            if night_index >= following - 1:
                self.brief_done += 1              # this part of the season had nothing to read
            return                                # otherwise wait for a night that has
        self.brief_done += 1
        named = {n.get("direction") for n in tonight} | {d for _, d in bulletin}
        s = agent.survey
        pending = s.required & (s.factor < s.required_threshold)
        context = {
            "night": night_date,
            "nights_left_including_tonight": len(s.nights) - night_index,
            "forecast_tonight": [{"event_kind": n.get("event_kind"), "direction": n.get("direction")} for n in tonight],
            "forecast_later_nights": [{"event_kind": n.get("event_kind"), "direction": n.get("direction"),
                                       "nights": [d for d in n.get("nights", []) if d > night_date]} for n in later],
            "bulletin_now": [{"event_kind": k, "direction": d} for k, d in bulletin],
            "terrain_obstruction": list(hazards.terrain),
            "sky_level_now": round(agent.tracker.level(), 3),
            "sky_level_typical": round(planner.level_typical, 3),
            "sky_level_last_nights": self.sky_nights[-3:],
            "must_observe_pending": int(pending.sum()),
            "must_observe_pending_needing_good_sky": int((pending & (planner.req_hard > 0.5)).sum()),
            "open_requests": int((planner.request_bonus > 0).sum() > 0),
        }
        allowance = self._allowance(payload, now, BRIEF_WAIT_SECONDS)
        if not self._worth_asking(allowance):
            return
        started = time.monotonic()
        self.advisor.start_night(context, named, self._wall_left(payload), allowance)
        briefing = self.advisor.wait_briefing(allowance)
        self._spent(started, briefing is not None)
        self._apply_briefing(briefing)

    def poll(self) -> None:
        """Once per decision inside a night: keep the clock, and take a review that came after its wait ended."""
        if not self.enabled:
            return
        self._tick()
        self._take_late_review()

    def _take_late_review(self) -> None:
        # a review is about the season, not about one night: an answer that arrives late is still good
        self._apply_review(self.advisor.poll_review())

    def _apply_review(self, review) -> None:
        if review is None:
            return
        planner = self.agent.planner
        hold = planner.base_hold + HOLD_RANGE * (2.0 * review["hold_strength"] - 1.0)
        planner.hold = float(min(0.95, max(0.3, hold)))
        planner.plan_age = 10 ** 9          # rebuild the season plan with it
        self.applied["review"] += 1
        self.log(f"llm season review: hold_strength {review['hold_strength']:.2f} -> hold {planner.hold:.2f} "
                 f"(tools {review['tools']}, {review.get('mode', 'native')} calling; {review['reason']})")

    def _apply_briefing(self, briefing) -> None:
        if briefing is None:
            return
        agent = self.agent
        agent.hazards.extra_avoid = set(briefing["avoid_directions"])
        agent.planner.level_scale = CAUTION_SCALE[briefing["caution"]]
        agent.planner.req_scale = REQUIRED_SCALE[briefing["required_push"]]
        self.applied["briefing"] += 1
        self.log(f"llm night briefing {self.tonight_date}: avoid {briefing['avoid_directions']}, caution {briefing['caution']}, "
                 f"required {briefing['required_push']} ({briefing['reason']})")

    # --- 2. fault adjudication ----------------------------------------------------------------------------

    def allow_report(self, payload: dict, now: float, hours: float, p_fault: float, free_left: int) -> bool:
        """The rules want to report. With a free false report to spare, or overwhelming evidence, go ahead;
        otherwise the model reads the evidence and may veto."""
        if not self.enabled:
            return True
        if p_fault >= 0.97:  # fresh overwhelming evidence supersedes an older model veto
            return True
        if hours < self.veto_until_hours:
            return False
        if free_left >= 2 or p_fault >= ASK_BELOW:
            return True
        agent = self.agent
        s = agent.survey
        evidence = {
            "exposures": {"columns": ["utc", "seconds", "W", "eta", "tracker_fault_probability", "bulletin"], "rows": self.exposures[-24:]},
            "tracker_fault_probability_now": round(p_fault, 3),
            "earthquake_notice_now": bool(agent.tracker.quake_active),
            "free_false_reports_left": max(0, free_left),
            "false_report_penalty": s.report_penalty,
            "correct_report_reward": s.report_reward,
            "nights_left": round(s.seconds_left_in_season(now) / 28800.0, 1),
            "reports_so_far": {"correct": agent.correct_reports, "false": agent.false_reports},
        }
        allowance = self._allowance(payload, now, VERDICT_WAIT_SECONDS)
        if not self._worth_asking(allowance):
            return True                          # no real time to ask: the rule's decision stands
        started = time.monotonic()
        verdict = self.advisor.judge_fault(evidence, self._wall_left(payload), allowance)
        self._spent(started, verdict is not None)
        if verdict is None:
            return True                          # no answer in time: the rule's decision stands
        self.log(f"llm fault verdict: {verdict['verdict']} ({verdict['confidence']:.2f}; {verdict['reason']})")
        self.applied["verdict"] += 1
        if verdict["verdict"] == "not_fault" and verdict["confidence"] >= VETO_CONFIDENCE:
            self.applied["veto"] += 1
            self.veto_until_hours = hours + VETO_HOURS
            return False
        return True

    # --- 3. season review ----------------------------------------------------------------------------------

    def _review_snapshot(self, night_date: str, night_index: int) -> dict:
        agent, planner = self.agent, self.agent.planner
        s = agent.survey
        done = s.cur >= s.top_mult - 0.02
        partly = (s.cur > 0) & ~done
        pending = s.required & (s.factor < s.required_threshold)
        return {
            "night": night_date,
            "hold_strength": 0.5 + (planner.hold - planner.base_hold) / (2.0 * HOLD_RANGE),
            "progress": {"nights_done": night_index, "nights_left": len(s.nights) - night_index,
                         "targets_total": s.n, "targets_completed": int(done.sum()), "targets_partly_done": int(partly.sum()),
                         "targets_untouched": int((s.cur <= 0).sum()),
                         "score_from_targets_so_far": round(float((s.cur * s.weight).sum()), 1),
                         "targets_the_plan_expects_to_complete_later": int((planner.plan_pi > 0.5 * planner.hold).sum()),
                         "typical_sky_level": round(planner.level_typical, 3)},
            "sky": self.sky_nights[-14:],
            "required": {"total": int(s.required.sum()), "secured": int((s.required & ~pending).sum()), "pending": int(pending.sum()),
                         "pending_needing_better_than_typical_sky": int((pending & (planner.req_hard > 0.5)).sum())},
            "forecast": agent.forecast,
        }
