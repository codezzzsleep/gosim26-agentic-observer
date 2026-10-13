"""Sky and instrument tracker: a grid Bayes filter over (weather level W, instrument efficiency eta, fault flag F).

What the public score formula lets an agent infer from its own results
---------------------------------------------------------------------
For one exposure and one hit target i (participant guide, formulas 17-20):

    score_i / weight_i = min(1, a_i * eta * W) * multiplier(program, band(W * m_i))

    a_i = flux_i * T * m_i / (f0 * T0)         known
    m_i = mean over the exposure of lunar_i / (q0 * airmass_i ** beta)     known (public ephemeris)
    W   = mean of transparency * sky_quality / seeing over the exposure (0 while the dome is closed)
    eta = instrument efficiency (jitter x earthquake x instrument fault)

Weather moves W, which enters both the completion factor and the program band. The instrument only enters
the completion factor. So a set of hits constrains the product eta * W tightly (unsaturated targets) and W
alone loosely (the multiplier shows on which side of the band thresholds W * m_i sits). Over time W drifts
slowly inside a night, announced all-sky weather moves it at known moments, and eta is constant apart from
three kinds of steps: an earthquake (announced, fades night by night), an instrument fault (never announced,
permanent until reported) and the repair after a correct report.

The filter keeps a probability table P[F, W, eta] and applies exactly those transitions; P(F = 1) is the
probability that an unreported instrument fault is active. Standard library + numpy.
"""
from __future__ import annotations

import math

import numpy as np

ETA_NORMAL = 0.95            # mean of the efficiency jitter without earthquake or fault
LOGW_MIN, LOGW_MAX, DX = math.log(0.004), math.log(2.6), 0.03
LOGE_MIN, LOGE_MAX, DY = math.log(0.04), math.log(1.0), 0.03
SIGMA_OBS = 0.04             # relative model error of one hit
OUTLIER = 0.03               # share of hits allowed to disagree (unannounced local trouble)
SIGMA_JITTER = 0.03          # exposure-to-exposure efficiency jitter
SIGMA_SLOT = 0.04            # weather drift per 15-minute slot inside a night
SIGMA_NIGHT = 0.17           # weather change from one night to the next
FAULT_HAZARD = 1.0 / 1500.0  # per slot of observing time
QUAKE_AGAIN_HAZARD = 1.0 / 1500.0
FAULT_MU = (0.22, 0.78)      # plausible range of a fault's efficiency multiplier
P_CLOSURE = 0.04             # an exposure partly lost to an unannounced dome closure
# announced all-sky weather: expected log shift of W and its spread while the notice is active
ALLSKY_SHIFT = {"overcast": (-0.85, 0.35), "haze": (-0.55, 0.30), "cold_snap": (-0.07, 0.06)}


def _gauss_kernel(sigma: float, step: float) -> np.ndarray:
    half = max(1, int(math.ceil(4.0 * sigma / step)))
    x = np.arange(-half, half + 1) * step
    k = np.exp(-0.5 * (x / max(sigma, 1e-9)) ** 2)
    return k / k.sum()


def _conv_axis(table: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve along one axis; every line keeps its total mass (what would leave the grid is spread back)."""
    half = len(kernel) // 2
    moved = np.moveaxis(table, axis, -1)
    pad = np.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(half, half)], mode="constant")
    out = np.zeros_like(moved)
    n = moved.shape[-1]
    for k, weight in enumerate(kernel):
        out += weight * pad[..., k:k + n]
    before = moved.sum(axis=-1, keepdims=True)
    after = out.sum(axis=-1, keepdims=True)
    out *= np.where(after > 0, before / np.maximum(after, 1e-300), 1.0)
    return np.moveaxis(out, -1, axis)


class SkyTracker:
    def __init__(self, scoring: dict, log=lambda text: None):
        self.log = log
        self.bands = (float(scoring["program"]["bands"]["DARK"]), float(scoring["program"]["bands"]["BRIGHT"]))
        self.mult = {name: float(value) for name, value in scoring["program"]["multipliers"].items()}
        self.mismatch = float(scoring["program"]["mismatch_multiplier"])
        self.xw = np.arange(LOGW_MIN, LOGW_MAX + 1e-9, DX)
        self.ye = np.arange(LOGE_MIN, LOGE_MAX + 1e-9, DY)
        self.nw, self.ne = len(self.xw), len(self.ye)
        self.W = np.exp(self.xw)
        self.E = np.exp(self.ye)
        self.normal_eta = np.exp(-0.5 * ((self.ye - math.log(ETA_NORMAL)) / 0.025) ** 2)
        self.normal_eta /= self.normal_eta.sum()
        prior_w = np.exp(-0.5 * ((self.xw - math.log(0.80)) / 0.30) ** 2)
        prior_w /= prior_w.sum()
        self.stationary_w = prior_w
        self.P = np.zeros((2, self.nw, self.ne))
        self.P[0] = np.outer(prior_w, self.normal_eta)
        self.jitter_kernel = _gauss_kernel(SIGMA_JITTER, DY)
        # fault onset: eta -> eta * mu, mu uniform in FAULT_MU (density in log mu proportional to mu)
        lo, hi = (int(round(math.log(v) / DY)) for v in FAULT_MU)
        self.fault_shifts = np.arange(lo, hi + 1)
        weights = np.exp(self.fault_shifts * DY)
        self.fault_weights = weights / weights.sum()
        # earthquake onset: eta -> eta * (1 - d), d uniform in (0.03, 0.95)
        d = np.linspace(0.03, 0.95, 60)
        self.quake_shifts = np.round(np.log(1.0 - d) / DY).astype(int)
        self.quake_active = False
        self.quake_nights = 0            # nights since the earthquake notice appeared
        self.allsky: set = set()         # all-sky weather kinds active in the current bulletin
        self.last_time = None
        self.night_index = -1
        self.closed_recently = False
        self.updates = 0
        self.fault_history: list = []    # (hours, P(fault)) after each informative exposure
        self.last_eta_obs = None

    # --- summaries ------------------------------------------------------------------------------------

    def p_fault(self) -> float:
        return float(self.P[1].sum())

    def level(self) -> float:
        """E[eta * W]: the factor in front of every target's completion rate."""
        return float(((self.P[0] + self.P[1]) * np.outer(self.W, self.E)).sum())

    def w_marginal(self) -> np.ndarray:
        return (self.P[0] + self.P[1]).sum(axis=1)

    def eta_mean(self) -> float:
        return float(((self.P[0] + self.P[1]).sum(axis=0) * self.E).sum())

    def w_mean(self) -> float:
        return float((self.w_marginal() * self.W).sum())

    def band_probabilities(self, m: np.ndarray):
        """P(DARK), P(BRIGHT), P(BACKUP) for targets with sky model m (array)."""
        cdf = np.concatenate([[0.0], np.cumsum(self.w_marginal())])   # P(W < W_k)
        def below(threshold):
            x = (np.log(np.maximum(threshold / np.maximum(m, 1e-9), 1e-12)) - LOGW_MIN) / DX + 0.5
            return np.interp(np.clip(x, 0.0, self.nw), np.arange(self.nw + 1), cdf)
        p_below_dark = below(self.bands[0])
        p_below_bright = below(self.bands[1])
        return 1.0 - p_below_dark, p_below_dark - p_below_bright, p_below_bright

    def quantile_level(self, q: float) -> float:
        """Quantile of eta * W (for cautious planning)."""
        joint = (self.P[0] + self.P[1])
        values = (self.xw[:, None] + self.ye[None, :]).ravel()
        order = np.argsort(values)
        cum = np.cumsum(joint.ravel()[order])
        return float(math.exp(values[order][min(len(cum) - 1, int(np.searchsorted(cum, q * cum[-1])))]))

    # --- transitions ------------------------------------------------------------------------------------

    def _shift_eta(self, table: np.ndarray, shifts: np.ndarray, weights: np.ndarray) -> np.ndarray:
        out = np.zeros_like(table)
        n = self.ne
        for shift, weight in zip(shifts, weights):
            if shift == 0:
                out += weight * table
            elif shift < 0:
                k = -int(shift)
                if k >= n:
                    out[:, 0] += weight * table.sum(axis=1)
                else:
                    out[:, :n - k] += weight * table[:, k:]
                    out[:, 0] += weight * table[:, :k].sum(axis=1)
            else:
                k = int(shift)
                if k >= n:
                    out[:, n - 1] += weight * table.sum(axis=1)
                else:
                    out[:, k:] += weight * table[:, :n - k]
                    out[:, n - 1] += weight * table[:, n - k:].sum(axis=1)
        return out

    def _shift_w(self, shift_log: float, sigma: float) -> None:
        steps = int(round(shift_log / DX))
        for f in (0, 1):
            t = self.P[f]
            if steps < 0:
                k = -steps
                moved = np.zeros_like(t); moved[:self.nw - k] = t[k:]; moved[0] += t[:k].sum(axis=0)
            elif steps > 0:
                k = steps
                moved = np.zeros_like(t); moved[k:] = t[:self.nw - k]; moved[self.nw - 1] += t[self.nw - k:].sum(axis=0)
            else:
                moved = t
            self.P[f] = _conv_axis(moved, _gauss_kernel(sigma, DX), 0)

    def advance(self, night_index: int, notices: set, slots: float) -> None:
        """Move the belief forward: `slots` 15-minute slots of observing time have passed (0 = only apply the
        bulletin now in force), possibly into a new night, under the given bulletin notices."""
        new_night = night_index != self.night_index and self.night_index >= 0
        first = self.night_index < 0
        self.night_index = night_index
        quake_now = any(kind == "earthquake" for kind, _ in notices)
        allsky_now = {kind for kind, direction in notices if direction == "ALL" and kind in ALLSKY_SHIFT}
        if not first:
            if new_night:
                for f in (0, 1):
                    blurred = _conv_axis(self.P[f], _gauss_kernel(SIGMA_NIGHT, DX), 0)
                    mass_eta = blurred.sum(axis=0)
                    self.P[f] = 0.85 * blurred + 0.15 * np.outer(self.stationary_w, mass_eta)
            elif slots > 0:
                sigma = min(0.13, SIGMA_SLOT * math.sqrt(max(0.25, slots)))
                for f in (0, 1):
                    self.P[f] = _conv_axis(self.P[f], _gauss_kernel(sigma, DX), 0)
            for kind in sorted(allsky_now - self.allsky):
                shift, spread = ALLSKY_SHIFT[kind]
                self._shift_w(shift, spread)
            for kind in sorted(self.allsky - allsky_now):
                shift, spread = ALLSKY_SHIFT[kind]
                self._shift_w(-shift, spread)
        self.allsky = allsky_now
        # --- earthquake: announced; efficiency drops at once, then recovers night by night ---
        uniform = np.full(len(self.quake_shifts), 1.0 / len(self.quake_shifts))
        if quake_now and not self.quake_active:
            for f in (0, 1):
                self.P[f] = self._shift_eta(self.P[f], self.quake_shifts, uniform)
            self.quake_nights = 0
            self.log("tracker: earthquake notice, efficiency will be re-learned")
        elif quake_now and new_night:
            self.quake_nights += 1
            self._quake_recovery()
        elif self.quake_active and not quake_now:
            # the notice is gone: the earthquake's effect is negligible again
            self.P[0] = np.outer(self.P[0].sum(axis=1), self.normal_eta)
            self._clip_fault_range()
        self.quake_active = quake_now
        # --- hazards per slot of elapsed observing time ---
        if slots > 0:
            p_fault = 1.0 - (1.0 - FAULT_HAZARD) ** slots
            moving = p_fault * self.P[0]
            self.P[0] -= moving
            self.P[1] += self._shift_eta(moving, self.fault_shifts, self.fault_weights)
            if quake_now:
                p_again = 1.0 - (1.0 - QUAKE_AGAIN_HAZARD) ** slots
                for f in (0, 1):
                    moving = p_again * self.P[f]
                    self.P[f] += self._shift_eta(moving, self.quake_shifts, uniform) - moving
            self._clip_fault_range()
        self._normalise()

    def _quake_recovery(self) -> None:
        """New night with the earthquake notice still up: degradation d -> d * rho, rho in [0.55, 0.9]."""
        rhos = np.linspace(0.55, 0.90, 8)
        for f in (0, 1):
            table = self.P[f]
            out = np.zeros_like(table)
            base = ETA_NORMAL if f == 0 else None
            if f == 0:
                d = np.clip(1.0 - self.E / base, 0.0, 0.99)
                for rho in rhos:
                    new_eta = base * (1.0 - d * rho)
                    idx = np.clip(np.round((np.log(new_eta) - LOGE_MIN) / DY).astype(int), 0, self.ne - 1)
                    np.add.at(out, (slice(None), idx), table / len(rhos))
            else:
                # fault x earthquake: the split is unknown; allow a recovery of up to 60%
                ups = np.arange(0, int(round(math.log(1.6) / DY)) + 1)
                out = self._shift_eta(table, ups, np.full(len(ups), 1.0 / len(ups)))
            self.P[f] = out
        self._clip_fault_range()

    def _clip_fault_range(self) -> None:
        """A fault keeps eta at or below ETA_NORMAL * max(mu)."""
        limit = math.log(ETA_NORMAL * FAULT_MU[1]) + 0.5 * DY
        over = self.ye > limit
        if over.any():
            cap = int(np.argmax(over)) - 1
            self.P[1][:, cap] += self.P[1][:, over].sum(axis=1)
            self.P[1][:, over] = 0.0

    def _normalise(self) -> None:
        total = self.P.sum()
        if not np.isfinite(total) or total <= 0:
            self.P[:] = 0.0
            self.P[0] = np.outer(self.stationary_w, self.normal_eta)
        else:
            self.P /= total

    # --- observations -----------------------------------------------------------------------------------

    def observe(self, a: np.ndarray, m: np.ndarray, ratio: np.ndarray, program: str, hours: float,
                after_closure: bool = False) -> dict:
        """Update with the hits of one exposure.

        a, m: per-hit constants (see module docstring); ratio = score / weight (> 0); program: declared.
        Returns a small summary for logs."""
        if len(a) == 0:
            return {}
        declared = self.mult[program]
        # predicted log ratio on the grid: g depends on x + y, the multiplier on x
        Wm = self.W[:, None] * m[None, :]                                    # (nw, hits)
        band = np.where(Wm >= self.bands[0], 0, np.where(Wm >= self.bands[1], 1, 2))
        declared_band = {"DARK": 0, "BRIGHT": 1, "BACKUP": 2}[program]
        log_mult = np.where(band == declared_band, math.log(declared), math.log(self.mismatch))   # (nw, hits)
        log_obs = np.log(ratio)
        loglik = np.zeros((self.nw, self.ne))
        ew = self.W[:, None] * self.E[None, :]                               # (nw, ne)
        floor = OUTLIER * 0.15
        for k in range(len(a)):
            log_g = np.minimum(0.0, np.log(a[k] * ew))
            delta = (log_obs[k] - log_g - log_mult[:, k][:, None]) / SIGMA_OBS
            loglik += np.log((1.0 - OUTLIER) * np.exp(-0.5 * delta * delta) + floor)
        loglik -= loglik.max()
        like = np.exp(loglik)
        # exposure-to-exposure efficiency jitter: blur along eta
        like = _conv_axis(like, self.jitter_kernel, 1)
        # part of the exposure lost to a closure: effective W = W * phi, phi uniform in (0, 1)
        weighted = like * self.W[:, None]
        closure = np.cumsum(weighted, axis=0) * DX / self.W[:, None]
        p_closure = 0.5 if after_closure else P_CLOSURE
        like = (1.0 - p_closure) * like + p_closure * closure
        self.P *= like[None, :, :]
        self._normalise()
        self.updates += 1
        p = self.p_fault()
        self.fault_history.append((hours, p))
        if len(self.fault_history) > 400:
            del self.fault_history[:100]
        return {"p_fault": p, "level": self.level(), "w": self.w_mean(), "eta": self.eta_mean()}

    # --- reports ----------------------------------------------------------------------------------------

    def on_report(self, correct: bool) -> None:
        mass_w = self.w_marginal()
        if correct:
            # repaired: efficiency returns to what the earthquake (if any) allows; W as believed under the fault
            w_fault = self.P[1].sum(axis=1)
            mass_w = w_fault / w_fault.sum() if w_fault.sum() > 0 else mass_w
            if self.quake_active:
                broad = np.where(self.E >= 0.2, 1.0, 0.1)
                self.P[0] = np.outer(mass_w, broad / broad.sum())
            else:
                self.P[0] = np.outer(mass_w, self.normal_eta)
        elif self.P[0].sum() < 1e-12:
            # the fault hypothesis is dead and nothing else was left: start again from the prior
            self.P[0] = np.outer(self.stationary_w, self.normal_eta)
        self.P[1][:] = 0.0
        self._normalise()
        self.fault_history.clear()

