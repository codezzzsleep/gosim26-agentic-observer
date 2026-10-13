"""Exposure planner: pointing, fibre assignment, duration and program chosen together.

One decision
------------
1. Every target that is up gets, for each candidate duration T, a planning gain

       weight * max(0, reach(T) * E[multiplier] - best so far)
       + required bonus * P(reach >= 0.5)       (targets still missing their required threshold)
       + request bonus  * P(reach >= threshold) (targets of an open observation request)

   reach(T) = min(1, flux * level * integral of the public sky model over the exposure / (f0 * T0)); `level`
   is the tracker's estimate of efficiency x weather.
2. Candidate field centres come from anchors (the best single targets and the densest patches of gain): for
   each anchor, a grid of shifted centres that keeps the anchor in the field. A centre is evaluated exactly
   as the engine does it: gnomonic projection, row/column cell, one target per fibre. Each fibre takes the
   target with the largest gain in its cell.
3. The winner maximises  sum of gains - time price * T ; the best few centres are then refined on a finer
   grid of shifts. The time price follows the recently available gain rate.
4. The program is the one with the largest expected score for the chosen fibres.

Everything is vectorised with numpy; one decision evaluates tens of thousands of (centre, duration) pairs.
"""
from __future__ import annotations

import math
import os

import numpy as np

import skymath as sm
from hazards import Hazards
from state import Survey
from tracker import SkyTracker


def _env(name: str, default):
    return type(default)(os.environ.get(f"AO_{name}", default))


DURATIONS = np.array([90, 120, 180, 240, 300, 360, 450, 540, 660, 780, 900, 1080, 1260, 1500, 1800, 2100, 2400, 2700,
                      3000, 3300, 3600], dtype=float)
NODES = np.array([0.0, 300.0, 900.0, 1800.0, 2700.0, 3600.0])
KAPPA = _env("KAPPA", 0.6)                    # time price as a share of the recent best gain rate
RATE_EMA = _env("RATE_EMA", 0.03)
N_VALUE_ANCHORS = _env("N_VALUE_ANCHORS", 20)
N_DENSE_ANCHORS = _env("N_DENSE_ANCHORS", 12)
ANCHOR_SEP_DEG = _env("ANCHOR_SEP_DEG", 0.9)
COARSE_STEP = _env("COARSE_STEP", 6)           # coarse shift step = pitch / COARSE_STEP
FINE_STEP = _env("FINE_STEP", 18)              # fine shift step = pitch / FINE_STEP
REFINE_TOP = _env("REFINE_TOP", 10)
COARSE_T_STRIDE = _env("COARSE_T_STRIDE", 1)
EDGE_MARGIN = _env("EDGE_MARGIN", 0.02)        # keep targets this far (deg) inside their fibre cell
ALT_MARGIN = _env("ALT_MARGIN", 0.12)          # keep targets this far above the altitude limit
REQUIRED_BONUS = _env("REQUIRED_BONUS", 1.0)    # planning value of securing a required target, in units of the card's penalty
REQUIRED_SIGMA = _env("REQUIRED_SIGMA", 0.16)  # log spread of a predicted factor
REQUIRED_AIM = _env("REQUIRED_AIM", 1.0)
REQUEST_MULT = _env("REQUEST_MULT", 2.0)           # planning value of a request = this x its reward, split over the targets still needed
REQUEST_MARGIN = _env("REQUEST_MARGIN", 1.3)       # aim this far above the threshold when judging feasibility
REQUEST_SPARE = _env("REQUEST_SPARE", 1)           # extra targets (beyond those needed) that also carry the bonus
REQUEST_MAX_SECONDS = _env("REQUEST_MAX_SECONDS", 9000.0)   # most telescope time worth spending on a 100-point request
LEVEL_SAFETY = _env("LEVEL_SAFETY", 0.97)      # plan as if the sky were this much worse than estimated
# --- season plan (see _season_plan) ---
SEASON_PLAN = _env("SEASON_PLAN", 1)
PLAN_LEVEL_PRIOR = _env("PLAN_LEVEL_PRIOR", 0.68)   # typical efficiency x weather of a season, before any is observed
PLAN_LEVEL_N0 = _env("PLAN_LEVEL_N0", 250.0)     # weight of that prior, in exposures
PLAN_UTIL_ADAPT = _env("PLAN_UTIL_ADAPT", 0)       # 1: learn the utilisation from the season so far
PLAN_UTIL_NIGHTS = _env("PLAN_UTIL_NIGHTS", 5.0) # weight of the prior utilisation, in nights
PLAN_UTIL_DECAY = _env("PLAN_UTIL_DECAY", 0.85)  # later nights deliver a little less than earlier ones (thinned-out fields)
PLAN_UTIL = _env("PLAN_UTIL", 0.55)            # share of the remaining fibre-time expected to end up useful
PLAN_OPEN = _env("PLAN_OPEN", 0.90)            # share of the remaining night time with the dome open
PLAN_PI = _env("PLAN_PI", 0.75)                # confidence that a planned target really gets completed later
PLAN_SOFT = _env("PLAN_SOFT", 0.35)            # width (log density) of the planned / filler transition
PLAN_M_SHARE = _env("PLAN_M_SHARE", 0.90)      # typical sky model of a target as a share of its transit value
PLAN_DONE_FROM = _env("PLAN_DONE_FROM", 0.78)   # (variant 2) share of the top score from which an exposure counts as completing
PLAN_DONE_FULL = _env("PLAN_DONE_FULL", 0.95)
PLAN_EVERY = _env("PLAN_EVERY", 25)            # decisions between plan updates
PRICE_MODE = _env("PRICE_MODE", "ema")         # "ema": time price follows the recent gain rate; "plan": from the season plan
PRICE_UTIL = _env("PRICE_UTIL", 0.55)
MAX_VISIBLE = _env("MAX_VISIBLE", 9000)          # most visible targets given the full evaluation in one decision
T_POLISH = _env("T_POLISH", 30.0)              # step (s) of the final duration search around the grid winner; 0 = off
# --- required targets: the penalty is settled at the end of the season, so the bonus of securing one now is
# the penalty x the risk that it would not be secured later. That risk is small while many nights remain,
# and grows with the season and as the target's own remaining nights run out. ---
REQ_EPS0 = _env("REQ_EPS0", 0.08)              # share of the bonus at the start of the season
REQ_FULL_AT = _env("REQ_FULL_AT", 0.65)        # season progress at which the full bonus applies
REQ_RAMP_POWER = _env("REQ_RAMP_POWER", 2.0)
REQ_LAST_NIGHTS = _env("REQ_LAST_NIGHTS", 2.5) # full bonus when a target has about this many usable nights left
REQ_HARD_FROM = _env("REQ_HARD_FROM", 0.45)    # needed level / typical level above which a required target is "hard" ...
REQ_HARD_FULL = _env("REQ_HARD_FULL", 0.75)    # ... and at which it gets the full bonus whenever it is feasible
REQ_COND_POWER = _env("REQ_COND_POWER", 2.0)   # easy required targets wait for a normal sky: bonus x (level now / typical) ** power
DONE_EPS = 0.004


def _phi(x):
    """Standard normal CDF (vectorised, good to ~1e-7)."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi) * poly
    return np.where(x >= 0, cdf, 1.0 - cdf)


class Planner:
    def __init__(self, survey: Survey, tracker: SkyTracker, hazards: Hazards, log=lambda text: None):
        self.s = survey
        self.tracker = tracker
        self.hazards = hazards
        self.log = log
        s = survey
        self.hmax = s.site.max_hour_angle(s.dec, s.min_alt + ALT_MARGIN)
        self.rate_ref = 0.0
        self.offset = np.zeros(2)               # learned pointing offset (alt, az): actual = command + offset
        self.margin_extra = 0.0                 # extra edge margin while the offset is uncertain
        self.request_bonus = np.zeros(s.n)
        self.request_threshold = np.full(s.n, 0.5)
        self.request_deadline = np.full(s.n, np.inf)
        self.requests_dropped: set = set()
        self.level_scale = 1.0                  # tonight's caution (set by the night briefing; 1 = none)
        self.req_scale = 1.0                    # tonight's push or hold of must-observe targets (night briefing)
        self.effort = 1.0                       # search effort (pace control)
        self.pending = None
        self.last_centre_eq = None
        self.stats = {"plans": 0, "candidates": 0}
        # season plan
        transit_alt = np.clip(90.0 - np.abs(s.dec - s.site.lat), 1.0, 90.0)
        self.m_typical = PLAN_M_SHARE / (s.q0 * sm.normalized_airmass(transit_alt) ** s.beta)
        self.level_typical = PLAN_LEVEL_PRIOR
        self.level_samples: list[float] = []
        self.future_cost = np.zeros(s.n)       # c_j = time price x fibre-seconds a later completion would take
        self.plan_pi = np.zeros(s.n)           # confidence that the target is completed later if left alone
        self.plan_lambda = 0.0
        self.base_hold = PLAN_PI                # how firmly planned targets are held for one complete exposure
        self.hold = PLAN_PI                     # ... as adjusted by the season review (model stage), within a fixed range
        self.util = PLAN_UTIL
        self.util_measured = 0.0
        self.plan_age = 10 ** 9
        self.plan_night = -1
        # seconds each target spends above the altitude limit in each night (public geometry)
        rate = sm.SIDEREAL_DEG_PER_SECOND
        self.night_visible = np.zeros((s.n, len(s.nights)), dtype=np.float32)
        for k, (start, end, _) in enumerate(s.nights):
            span = (end - start) * rate
            rise = (s.ra - self.hmax - s.site.lst(start)) % 360.0
            overlap = np.maximum(0.0, np.minimum(span, rise + 2 * self.hmax) - rise) + np.maximum(0.0, np.minimum(span, rise - 360.0 + 2 * self.hmax))
            self.night_visible[:, k] = np.where(self.hmax >= 180.0, end - start, overlap / rate)
        self.req_ramp = REQ_EPS0
        self.req_last = np.zeros(s.n)
        self.req_hard = np.zeros(s.n)
        self.nights_left = np.full(s.n, len(s.nights))

    # --- requests ---------------------------------------------------------------------------------------

    def _visible_seconds(self, idx: np.ndarray, t0: float, t1: float) -> np.ndarray:
        """Seconds each target in idx spends above the altitude limit during night time inside [t0, t1]."""
        s = self.s
        rate = sm.SIDEREAL_DEG_PER_SECOND
        total = np.zeros(len(idx))
        hmax = self.hmax[idx]
        for start, end, _ in s.nights:
            a, b = max(start, t0), min(end, t1)
            if b <= a:
                continue
            span = (b - a) * rate
            rise = (s.ra[idx] - hmax - s.site.lst(a)) % 360.0
            overlap = np.maximum(0.0, np.minimum(span, rise + 2 * hmax) - rise) + np.maximum(0.0, np.minimum(span, rise - 360.0 + 2 * hmax))
            total += np.where(hmax >= 180.0, b - a, overlap / rate)
        return total

    def set_requests(self, requests: list, now: float) -> None:
        """Turn open observation requests into per-target bonuses.

        A request pays only when enough of its targets reach the threshold inside its window, so it is worth
        chasing only if that is feasible; then the cheapest targets that complete it carry the reward."""
        s = self.s
        self.request_bonus[:] = 0.0
        self.request_threshold[:] = 0.5
        self.request_deadline[:] = np.inf
        level = max(self.tracker.level(), 0.7 * self.level_typical)
        for request in sorted(requests, key=lambda r: str(r.get("request_id", ""))):
            remaining = int(request.get("remaining_count", request["minimum_completed"]))
            if remaining <= 0:
                continue
            deadline = sm.unix(request["deadline_utc"])
            if deadline <= now:
                continue
            completed = set(request.get("completed_target_ids", []))
            idx = np.array([s.index_of[t] for t in request["target_ids"] if t not in completed and t in s.index_of], dtype=int)
            if len(idx) < remaining:
                continue
            threshold = float(request["completion_factor_threshold"])
            reward = float(request["completion_reward"])
            # exposure each target needs for the threshold (with a margin), and the time it is up before the deadline
            need = threshold * REQUEST_MARGIN * s.f0t0 / (s.flux[idx] * level * self.m_typical[idx])
            up = self._visible_seconds(idx, now, deadline)
            feasible = (need <= 0.9 * s.max_exposure) & (up >= 1.15 * need)
            if int(feasible.sum()) < remaining:
                key = str(request.get("request_id"))
                if key not in self.requests_dropped:
                    self.requests_dropped.add(key)
                    self.log(f"request {key}: only {int(feasible.sum())} of the {remaining} targets still needed are feasible; not chased")
                continue
            order = np.argsort(np.where(feasible, need, np.inf), kind="stable")
            chosen = order[:min(len(order), remaining + REQUEST_SPARE)]
            chosen = chosen[feasible[chosen]]
            if float(np.sort(need[chosen])[:remaining].sum()) > REQUEST_MAX_SECONDS * reward / 100.0:
                continue      # the reward does not pay for the telescope time
            unit = REQUEST_MULT * reward / remaining
            for k in chosen:
                i = int(idx[k])
                if self.request_bonus[i] > 0:
                    self.request_threshold[i] = max(self.request_threshold[i], threshold)
                else:
                    self.request_threshold[i] = threshold
                self.request_bonus[i] += unit
                self.request_deadline[i] = min(self.request_deadline[i], deadline)

    # --- season plan ------------------------------------------------------------------------------------

    def note_level(self, level: float) -> None:
        """Feed the tracker's level after each informative exposure: its running median is the typical sky."""
        self.level_samples.append(float(level))
        if len(self.level_samples) > 600:
            del self.level_samples[:200]
        # what the rest of the season will offer is unknown (weather spells, earthquakes, faults): the season's
        # own median counts more as it accumulates, a prior carries the start
        n = len(self.level_samples)
        if n >= 4:
            self.level_typical = (n * float(np.median(self.level_samples)) + PLAN_LEVEL_N0 * PLAN_LEVEL_PRIOR) / (n + PLAN_LEVEL_N0)

    def _season_plan(self, now: float) -> None:
        """Which targets will the rest of the season complete, and what is a fibre-second worth?

        Fluid picture: the remaining nights offer K useful fibre-seconds. Completing target j later takes
        t_j fibre-seconds under typical sky and is worth V_j = weight * (top multiplier - best so far).
        Filling K in order of density V_j / t_j gives the planned set and the density of the marginal
        target: the price of a fibre-second. For a planned target the value of an exposure now is not V_j
        (it would be collected anyway) but what it saves later: price x t_j when it completes the target,
        nothing when it leaves it to be redone - only the best exposure of a target counts."""
        s = self.s
        rate = s.flux * self.level_typical * self.m_typical / s.f0t0          # completion per second, typical sky
        t_full = 1.0 / np.maximum(rate, 1e-9)
        reach_max = np.minimum(1.0, rate * s.max_exposure)
        t = np.minimum(t_full, float(s.max_exposure))
        value = s.weight * np.maximum(0.0, s.top_mult * reach_max - s.cur)
        value = np.where(self.hmax > 0.0, value, 0.0)
        density = value / t
        # how much of the telescope's fibre-time ends up as completed work, in the plan's own units: measured
        # on the season so far (closures, bad spells, empty fibres and over-exposure all included), blended
        # with a prior that carries the first nights
        elapsed = sum(max(0.0, min(end, now) - start) for start, end, _ in s.nights if start < now) * s.n_fibers
        delivered = float((t * np.minimum(1.0, s.factor) ** 2).sum())      # partial exposures count for little
        prior_weight = PLAN_UTIL_NIGHTS * 28800.0 * s.n_fibers
        self.util_measured = delivered / elapsed if elapsed > 0 else 0.0
        if PLAN_UTIL_ADAPT:
            util = (delivered * PLAN_UTIL_DECAY + prior_weight * PLAN_UTIL) / (elapsed + prior_weight)
        else:
            util = PLAN_UTIL * PLAN_OPEN
        self.util = util
        capacity = s.seconds_left_in_season(now) * s.n_fibers * util
        order = np.argsort(-density)
        cum = np.cumsum(t[order])
        k = int(np.searchsorted(cum, capacity))
        positive = density[order] > 0
        n_pos = int(positive.sum())
        if n_pos == 0:
            lam = 0.0
        elif k >= n_pos:
            lam = float(density[order][n_pos - 1]) * 0.5      # time is not binding: every target fits
        else:
            lam = float(density[order][k])
        self.plan_lambda = lam
        self.future_cost = lam * t
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where((density > 0) & (lam > 0), np.log(np.maximum(density, 1e-300) / max(lam, 1e-300)) / PLAN_SOFT, -50.0)
        self.plan_pi = self.hold / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        # nights on which each target can still be observed long enough to matter
        night = s.current_night(now)
        k0 = night[0] if night else int(np.searchsorted([start for start, _, _ in s.nights], now))
        need = np.clip(0.6 * t, 600.0, 3000.0)
        usable = self.night_visible[:, k0:] >= need[:, None]
        self.nights_left = usable.sum(axis=1)
        self.plan_pi = self.plan_pi * np.clip((self.nights_left - 1.0) / 3.0, 0.0, 1.0)
        total = max(1.0, s.nights[-1][1] - s.nights[0][0])
        progress = min(1.0, max(0.0, (now - s.nights[0][0]) / total))
        self.req_ramp = REQ_EPS0 + (1.0 - REQ_EPS0) * min(1.0, progress / REQ_FULL_AT) ** REQ_RAMP_POWER
        self.req_last = np.minimum(1.0, (REQ_LAST_NIGHTS / np.maximum(self.nights_left, 0.5)) ** 1.5)
        # faint required targets reach the threshold only in a sky better than typical, even with the longest
        # exposure: for them a good night is the scarce thing, so the full bonus applies whenever one comes
        level_needed = s.required_threshold * 1.1 / np.maximum(s.flux * s.max_exposure * self.m_typical / s.f0t0, 1e-9)
        self.req_hard = np.clip((level_needed / self.level_typical - REQ_HARD_FROM) / (REQ_HARD_FULL - REQ_HARD_FROM), 0.0, 1.0)
        self.plan_age = 0
        self.log(f"season plan: {min(k, n_pos)} of {n_pos} open targets fit; fibre-second price {lam * 1000:.3f}e-3, "
                 f"typical level {self.level_typical:.3f}, utilisation {self.util:.2f} (measured {self.util_measured:.2f})")

    # --- planning ---------------------------------------------------------------------------------------

    def plan(self, now: float, night_end: float, max_duration: float = 1e9):
        s = self.s
        seconds_left = min(night_end, s.end) - now
        if seconds_left < 30.0:
            return None
        self.plan_age += 1
        if self.plan_age >= PLAN_EVERY:
            self._season_plan(now)
        cap = min(seconds_left, max(max_duration, float(s.min_exposure)), float(s.max_exposure))
        durations = DURATIONS[(DURATIONS <= cap) & (DURATIONS >= s.min_exposure)]
        if len(durations) == 0 or (cap < s.max_exposure and cap - durations[-1] > 20.0):
            # the night (or a probe) ends before the next grid duration: use what is left
            durations = np.append(durations, max(float(s.min_exposure), math.floor(cap)))
        nT = len(durations)

        # --- who is up, and for how long ---
        lst0 = s.site.lst(now)
        H0 = (lst0 - s.ra + 180.0) % 360.0 - 180.0
        up = np.where(self.hmax >= 180.0, 1e9, (self.hmax - H0) / sm.SIDEREAL_DEG_PER_SECOND)
        want = (s.cur < s.top_mult - DONE_EPS) | (s.required & (s.factor < s.required_threshold)) | (self.request_bonus > 0)
        vis = np.flatnonzero((np.abs(H0) <= self.hmax) & (up >= min(durations[0], 240.0)) & want)
        if len(vis) == 0:
            return None
        limit = int(MAX_VISIBLE * (1.0 if self.effort >= 0.3 else 0.35))
        if len(vis) > limit:
            # a very large catalogue or a nearly spent CPU budget: keep the targets with the most value per second
            sin_alt = s.site.sin_lat * s.eq[vis, 2] + s.site.cos_lat * (s.eq[vis, 0] * math.cos(math.radians(lst0)) + s.eq[vis, 1] * math.sin(math.radians(lst0)))
            proxy = s.weight[vis] * np.maximum(0.0, s.top_mult - s.cur[vis]) * np.minimum(s.flux[vis], 2.0) * np.maximum(sin_alt, 0.0)
            proxy = proxy + np.where(s.required[vis] & (s.factor[vis] < s.required_threshold), 10.0, 0.0) + self.request_bonus[vis]
            # whole sky patches, richest first: a thinned-out field cannot fill its fibres
            patch = (np.floor(s.ra[vis] * np.cos(np.radians(s.dec[vis])) / 6.0).astype(np.int64) * 1000
                     + np.floor((s.dec[vis] + 90.0) / 6.0).astype(np.int64))
            uniq, inv = np.unique(patch, return_inverse=True)
            richness = np.bincount(inv, weights=proxy, minlength=len(uniq))
            counts = np.bincount(inv, minlength=len(uniq))
            order = np.argsort(-richness, kind="stable")
            keep_patches = order[:max(1, int(np.searchsorted(np.cumsum(counts[order]), limit)) + 1)]
            keep = np.isin(inv, keep_patches) | (self.request_bonus[vis] > 0)
            vis = vis[keep]
        eq = s.eq[vis]
        up_v = up[vis]

        # --- sky model integrated over each duration ---
        nodes = NODES[NODES <= durations[-1] + 1.0]
        if nodes[-1] < durations[-1]:
            nodes = np.append(nodes, durations[-1])
        m_nodes = np.empty((len(vis), len(nodes)))
        h0 = None
        for k, t in enumerate(nodes):
            h = s.site.horizontal(eq, s.site.lst(now + t))
            if k == 0:
                h0 = h
            alt = np.degrees(np.arcsin(np.clip(h[:, 2], -1.0, 1.0)))
            moon = sm.Moon(now + t, s.site, s.lunar_model)
            m_nodes[:, k] = moon.lunar_factor(eq) / (s.q0 * sm.normalized_airmass(alt) ** s.beta)
        integral = self._integral(nodes, m_nodes, durations)                     # (vis, nT)
        alt0, az0 = sm.unit_to_altaz(h0)
        dirf = self.hazards.factor(alt0, az0)
        level = self.tracker.level() * LEVEL_SAFETY * self.level_scale
        ctx = {"vis": vis, "level": level, "dirf": dirf, "up": up_v, "now": now}
        gain, raw, reach, bands = self._gain(ctx, np.arange(len(vis)), integral, durations)
        p_dark, p_bright, p_backup = bands
        weight = s.weight[vis]
        need = s.required[vis] & (s.factor[vis] < s.required_threshold)
        mm = s.multipliers
        gain[s.misses[vis] >= 3] *= 0.5

        # --- candidates ---
        # the time price follows the gain rate recently on offer, per unit of sky level: when the sky or the
        # instrument degrades, every rate drops with it and so must the price
        lam = KAPPA * self.rate_ref * level
        if PRICE_MODE == "plan" and self.plan_lambda > 0:
            # bid price: what the season plan says a fibre-second is worth, for the fibres an exposure usually fills
            lam = s.n_fibers * PRICE_UTIL * self.plan_lambda * level / max(self.level_typical, 1e-6)
        per_fibre = lam * durations / s.n_fibers
        priority = (gain - per_fibre[None, :]).max(axis=1)
        if not (priority > 0).any():
            self.rate_ref *= 0.85
            lam = KAPPA * self.rate_ref * level
            priority = gain.max(axis=1)
            if not (priority > 0).any():
                return None
        useful = np.flatnonzero(gain.max(axis=1) > 0)
        anchors = self._anchors(priority, eq, useful)
        best = None            # (net, alt, az, T index, total)
        best_rate = 0.0
        tops = []
        cos_local = math.cos(math.radians(s.fov * 1.45))
        margin = EDGE_MARGIN + self.margin_extra + max(0.0, (s.pitch - s.glass) / 2.0)
        half = s.fov / 2.0 - s.pitch * 0.25
        n_coarse = max(3, int(round(2 * half / (s.pitch / COARSE_STEP) * math.sqrt(self.effort))))
        shifts = np.linspace(-half, half, n_coarse)
        su, sv = np.meshgrid(shifts, shifts, indexing="ij")
        su, sv = su.ravel(), sv.ravel()
        # the coarse pass ranks centres on a subset of the durations; the refinement uses them all
        sub = np.unique(np.r_[np.arange(0, nT, COARSE_T_STRIDE), nT - 1])
        gain_sub = np.ascontiguousarray(gain[:, sub])
        dur_sub = durations[sub]
        for a in anchors:
            local = useful[(eq[useful] @ eq[a]) >= cos_local]
            if len(local) == 0:
                continue
            c_alt, c_az = self._shifted_centres(h0[a], su, sv)
            found = self._evaluate(c_alt, c_az, h0[local], gain_sub[local], margin)
            if found is None:
                continue
            kidx, total = found
            net = total - lam * dur_sub[None, :]
            rate = (total / dur_sub[None, :]).max()
            best_rate = max(best_rate, float(rate))
            flat = np.argsort(net.max(axis=1))[::-1][:3]
            for f in flat:
                tops.append((float(net[f].max()), float(c_alt[kidx[f]]), float(c_az[kidx[f]]), a))
        if not tops:
            return None
        tops.sort(reverse=True)
        self.stats["plans"] += 1
        # --- refine the best coarse centres on a fine grid of shifts ---
        fine = s.pitch / FINE_STEP
        span = int(math.ceil((s.pitch / COARSE_STEP) / fine / 2.0)) + 1
        fs = np.arange(-span, span + 1) * fine
        fu, fv = np.meshgrid(fs, fs, indexing="ij")
        fu, fv = fu.ravel(), fv.ravel()
        seen = []
        for net0, alt_c, az_c, a in tops[:max(2, int(REFINE_TOP * self.effort))]:
            if any(abs(alt_c - x) < 1e-6 and abs(az_c - y) < 1e-6 for x, y in seen):
                continue
            seen.append((alt_c, az_c))
            centre_h = sm.altaz_unit(alt_c, az_c)
            c_alt, c_az = self._shifted_centres(centre_h, fu, fv)
            local = useful[(h0[useful] @ centre_h) >= math.cos(math.radians(s.fov * 0.95))]
            if len(local) == 0:
                continue
            found = self._evaluate(c_alt, c_az, h0[local], gain[local], margin)
            if found is None:
                continue
            kidx, total = found
            net = total - lam * durations[None, :]
            best_rate = max(best_rate, float((total / durations[None, :]).max()))
            f, t = np.unravel_index(int(np.argmax(net)), net.shape)
            if best is None or net[f, t] > best[0]:
                best = (float(net[f, t]), float(c_alt[kidx[f]]), float(c_az[kidx[f]]), int(t), float(total[f, t]))
        norm_rate = best_rate / max(level, 1e-6)
        self.rate_ref = norm_rate if self.rate_ref <= 0 else (1 - RATE_EMA) * self.rate_ref + RATE_EMA * norm_rate
        if best is None or best[4] <= 0:
            return None
        _, c_alt, c_az, t_idx, _ = best
        T = float(durations[t_idx])
        # --- fibres of the winning centre ---
        local = useful[(h0[useful] @ sm.altaz_unit(c_alt, c_az)) >= math.cos(math.radians(s.fov * 0.95))]
        cell, ok = self._cells(np.array([c_alt]), np.array([c_az]), h0[local], margin)
        cell, ok = cell[0], ok[0]
        pick = {}
        for j in np.flatnonzero(ok):
            g = gain[local[j], t_idx]
            if g > 0 and (cell[j] not in pick or g > pick[cell[j]][0]):
                pick[int(cell[j])] = (float(g), int(local[j]))
        if not pick:
            return None
        fibres = sorted(pick)
        chosen = np.array([pick[f][1] for f in fibres])        # indices into vis
        raw_c, reach_c, bands_c = raw[chosen, t_idx], reach[chosen, t_idx], [b[chosen, t_idx] for b in (p_dark, p_bright, p_backup)]
        if T_POLISH and nT > 1:
            # the duration grid is coarse: for the fibres chosen, look for the best duration between its neighbours
            lo = durations[max(0, t_idx - 1)]
            hi = durations[min(nT - 1, t_idx + 1)]
            fine = np.unique(np.r_[np.arange(lo, hi + 1e-6, T_POLISH), T])
            fine = fine[(fine >= s.min_exposure) & (fine <= cap)]
            g_f, raw_f, reach_f, bands_f = self._gain(ctx, chosen, self._integral(nodes, m_nodes[chosen], fine), fine)
            net_f = g_f.sum(axis=0) - lam * fine
            k_best = int(np.argmax(net_f))
            if net_f[k_best] > net_f[int(np.searchsorted(fine, T))] + 1e-9:
                T = float(fine[k_best])
                raw_c, reach_c, bands_c = raw_f[:, k_best], reach_f[:, k_best], [b[:, k_best] for b in bands_f]
        # --- program: largest expected score over the chosen fibres ---
        w_reach = weight[chosen] * reach_c
        w_reach = w_reach + np.where(need[chosen], 0.02 * s.required_penalty, 0.0)
        scores = {"DARK": float((w_reach * (s.mismatch + (mm["DARK"] - s.mismatch) * bands_c[0])).sum()),
                  "BRIGHT": float((w_reach * (s.mismatch + (mm["BRIGHT"] - s.mismatch) * bands_c[1])).sum()),
                  "BACKUP": float((w_reach * (s.mismatch + (mm["BACKUP"] - s.mismatch) * bands_c[2])).sum())}
        program = max(scores, key=lambda p: (scores[p], p))
        cmd_alt = round(min(90.0, max(0.0, c_alt - float(self.offset[0]))), 4)
        cmd_az = round((c_az - float(self.offset[1])) % 360.0, 4)
        if cmd_az >= 360.0:
            cmd_az = 0.0
        self.pending = {"start": now, "duration": T, "program": program, "cmd": (cmd_alt, cmd_az),
                        "idx": vis[chosen], "fibres": np.array(fibres), "h0": h0[chosen], "alt0": alt0[chosen],
                        "az0": az0[chosen], "dirf": dirf[chosen], "raw": raw_c, "level": level,
                        "night_end": night_end}
        self.last_centre_eq = None
        return {"action": "observe", "pointing": {"alt_deg": cmd_alt, "az_deg": cmd_az},
                "assignments": {str(f): s.ids[int(vis[c])] for f, c in zip(fibres, chosen)},
                "duration_seconds": int(T), "program": program}

    # --- value of an exposure for each target ---------------------------------------------------------

    @staticmethod
    def _integral(nodes: np.ndarray, m_nodes: np.ndarray, durations: np.ndarray) -> np.ndarray:
        """Integral over [0, T] of the piecewise-linear sky model, for every row and every T in durations."""
        seg = 0.5 * (m_nodes[:, 1:] + m_nodes[:, :-1]) * np.diff(nodes)[None, :]
        cum = np.concatenate([np.zeros((m_nodes.shape[0], 1)), np.cumsum(seg, axis=1)], axis=1)
        k_hi = np.clip(np.searchsorted(nodes, durations, side="left"), 1, len(nodes) - 1)
        k_lo = k_hi - 1
        frac = (durations - nodes[k_lo]) / (nodes[k_hi] - nodes[k_lo])
        m_at_T = m_nodes[:, k_lo] + (m_nodes[:, k_hi] - m_nodes[:, k_lo]) * frac[None, :]
        return cum[:, k_lo] + 0.5 * (m_nodes[:, k_lo] + m_at_T) * (durations - nodes[k_lo])[None, :]

    def _gain(self, ctx: dict, rows: np.ndarray, integral: np.ndarray, durations: np.ndarray):
        """Planning gain (rows x durations) of the visible targets `rows`, plus the uncapped factor, the reach and
        the band probabilities. `integral` holds the sky-model integrals of exactly those rows."""
        s = self.s
        idx = ctx["vis"][rows]
        level, now = ctx["level"], ctx["now"]
        flux, weight = s.flux[idx], s.weight[idx]
        raw = flux[:, None] * level * integral * ctx["dirf"][rows][:, None] / s.f0t0          # uncapped factor
        raw = np.where(durations[None, :] <= ctx["up"][rows][:, None], raw, 0.0)
        reach = np.minimum(1.0, raw)
        # expected program multiplier: the best single declaration for each target
        mbar = integral / durations[None, :]
        p_dark, p_bright, p_backup = self.tracker.band_probabilities(mbar)
        mm = s.multipliers
        exp_mult = np.maximum.reduce([s.mismatch + (mm["DARK"] - s.mismatch) * p_dark,
                                      s.mismatch + (mm["BRIGHT"] - s.mismatch) * p_bright,
                                      s.mismatch + (mm["BACKUP"] - s.mismatch) * p_backup])
        cur = s.cur[idx]
        new = np.maximum(reach * exp_mult, cur[:, None])
        gain = weight[:, None] * (new - cur[:, None])
        if SEASON_PLAN == 1:
            # value of the target's future, before and after this exposure (see _season_plan)
            pi = self.plan_pi[idx]
            cost = self.future_cost[idx]
            before = pi * np.maximum(0.0, weight * (s.top_mult - cur) - cost)
            after = pi[:, None] * np.maximum(0.0, weight[:, None] * (s.top_mult - new) - cost[:, None])
            gain = gain + after - before[:, None]
        elif SEASON_PLAN == 2:
            # simpler variant: an exposure that (nearly) completes a planned target counts in full, one that
            # leaves it to be redone counts for little
            pi = self.plan_pi[idx][:, None]
            done = np.clip((new / s.top_mult - PLAN_DONE_FROM) / (PLAN_DONE_FULL - PLAN_DONE_FROM), 0.0, 1.0)
            gain = gain * ((1.0 - pi) + pi * done)
        # required targets
        need = s.required[idx] & (s.factor[idx] < s.required_threshold)
        if need.any():
            z = np.log(np.maximum(raw[need], 1e-9) / (s.required_threshold * REQUIRED_AIM)) / REQUIRED_SIGMA
            condition = min(1.0, max(0.2, level / max(self.level_typical, 1e-6))) ** REQ_COND_POWER
            eps = np.maximum(self.req_last[idx][need], np.maximum(min(1.0, self.req_ramp * self.req_scale) * condition, self.req_hard[idx][need]))
            gain[need] += REQUIRED_BONUS * s.required_penalty * eps[:, None] * _phi(z)
        # observation requests
        rb = self.request_bonus[idx]
        asked = rb > 0
        if asked.any():
            z = np.log(np.maximum(raw[asked], 1e-9) / self.request_threshold[idx][asked][:, None]) / REQUIRED_SIGMA
            in_time = (now + durations[None, :]) <= self.request_deadline[idx][asked][:, None]
            gain[asked] += rb[asked][:, None] * _phi(z) * in_time
        return gain, raw, reach, (p_dark, p_bright, p_backup)

    # --- geometry helpers -------------------------------------------------------------------------------

    def _anchors(self, priority: np.ndarray, eq: np.ndarray, useful: np.ndarray) -> list[int]:
        """Indices (into the visible set) of anchor targets: best single targets, then densest patches."""
        s = self.s
        n_value = max(2, int(round(N_VALUE_ANCHORS * self.effort)))
        n_dense = max(1, int(round(N_DENSE_ANCHORS * self.effort)))
        order = useful[np.argsort(priority[useful])[::-1]]
        cos_sep = math.cos(math.radians(ANCHOR_SEP_DEG))
        anchors: list[int] = []
        for j in order[:400]:
            if priority[j] <= 0 and anchors:
                break
            if all(float(eq[j] @ eq[a]) < cos_sep for a in anchors):
                anchors.append(int(j))
                if len(anchors) >= n_value:
                    break
        # densest patches of positive priority on a coarse sky grid
        pos = useful[priority[useful] > 0]
        if len(pos):
            vis_idx = np.arange(len(priority))
            ra = np.degrees(np.arctan2(eq[pos, 1], eq[pos, 0]))
            dec = np.degrees(np.arcsin(np.clip(eq[pos, 2], -1.0, 1.0)))
            bin_deg = s.fov * 0.9
            key = (np.floor(ra * np.cos(np.radians(dec)) / bin_deg).astype(np.int64) * 100003
                   + np.floor((dec + 90.0) / bin_deg).astype(np.int64))
            uniq, inv = np.unique(key, return_inverse=True)
            sums = np.bincount(inv, weights=priority[pos], minlength=len(uniq))
            cos_sep2 = math.cos(math.radians(s.fov * 0.6))
            for b in np.argsort(sums)[::-1][:n_dense * 2]:
                members = pos[inv == b]
                j = int(members[np.argmax(priority[members])])
                if all(float(eq[j] @ eq[a]) < cos_sep2 for a in anchors):
                    anchors.append(j)
                    if len(anchors) >= n_value + n_dense:
                        break
        return anchors

    def _shifted_centres(self, centre_h: np.ndarray, du: np.ndarray, dv: np.ndarray):
        """Pointings whose tangent-plane offsets from `centre_h` are (du up, dv right) degrees; rounded as sent."""
        alt, az = sm.unit_to_altaz(centre_h)
        _, up, right = sm.frames(alt, az)
        vec = centre_h[None, :] + np.radians(du)[:, None] * up[None, :] + np.radians(dv)[:, None] * right[None, :]
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        c_alt, c_az = sm.unit_to_altaz(vec)
        c_alt = np.round(np.clip(c_alt, 0.0, 89.5), 4)
        c_az = np.round(c_az, 4) % 360.0
        return c_alt, c_az

    def _cells(self, c_alt: np.ndarray, c_az: np.ndarray, targets_h: np.ndarray, margin):
        """Fibre cell of each target for each centre (K, M) and whether it is safely inside its cell.

        Same arithmetic as the engine: gnomonic offsets along +alt and +az, cell = floor(offset / pitch + side / 2)."""
        s = self.s
        centre, up, right = sm.frames(c_alt, c_az)
        t = targets_h.T
        depth = centre @ t
        scale = sm.R2D / s.pitch
        u = (up @ t) / depth * scale + s.side / 2.0
        v = (right @ t) / depth * scale + s.side / 2.0
        row = np.floor(u)
        col = np.floor(v)
        m = margin / s.pitch
        fu = u - row
        fv = v - col
        ok = ((depth > 0.5) & (u >= m) & (u < s.side - m) & (v >= m) & (v < s.side - m)
              & (fu >= m) & (fu <= 1.0 - m) & (fv >= m) & (fv <= 1.0 - m))
        cell = (np.where(ok, row, 0.0) * s.side + np.where(ok, col, 0.0)).astype(np.int64)
        return cell, ok

    def _evaluate(self, c_alt, c_az, targets_h, gain, margin):
        """Sum over fibres of the best gain in each cell, for every centre and duration."""
        s = self.s
        cell, ok = self._cells(c_alt, c_az, targets_h, margin)
        kk, jj = np.nonzero(ok)
        if len(kk) == 0:
            return None
        self.stats["candidates"] += len(c_alt)
        key = kk * s.n_fibers + cell[kk, jj]
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        cellmax = np.maximum.reduceat(gain[jj[order]], starts, axis=0)
        gk = key_s[starts] // s.n_fibers
        kstarts = np.flatnonzero(np.r_[True, gk[1:] != gk[:-1]])
        total = np.add.reduceat(cellmax, kstarts, axis=0)
        return gk[kstarts], total
