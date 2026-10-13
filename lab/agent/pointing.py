"""Hidden pointing offset (Hard-mode cards): actual centre = commanded centre + a fixed (d_alt, d_az).

The engine never reveals the offset; it only tells which assigned targets were hit. Every assigned target is
therefore a yes/no measurement: for a candidate offset, would the target have landed on its fibre? The
estimator keeps the evidence, scores a coarse grid of offsets as exposures come in, and periodically scores a
fine grid around the best coarse cell. The set of offsets that explain the evidence best gives the estimate
(its centre) and how well it is known (its radius), which the planner turns into an edge margin.
"""
from __future__ import annotations

import numpy as np

import skymath as sm

COARSE_STEP = 0.01
COARSE_HALF = 0.16          # deg; the guide's example offsets are about 0.05 deg, cards may differ
FINE_STEP = 0.002
FINE_HALF = 0.014
MISS_COST = 4.0             # log-likelihood cost of one unexplained outcome
KEEP = 400                  # exposures of evidence kept


class PointingModel:
    def __init__(self, side: int, pitch: float, log=lambda text: None):
        self.side, self.pitch, self.log = side, pitch, log
        g = np.arange(-COARSE_HALF, COARSE_HALF + 1e-9, COARSE_STEP)
        da, dz = np.meshgrid(g, g, indexing="ij")
        self.da, self.dz = da.ravel(), dz.ravel()
        self.ll = np.zeros(self.da.size)
        self.zero = int(np.argmin(np.abs(self.da) + np.abs(self.dz)))
        self.evidence: list = []
        self.n = 0
        self.misses = 0
        self._fine_centre = None        # the fine grid's centre the cached rows belong to
        self._fine_rows: list = []      # per kept exposure: unexplained outcomes for each fine offset
        self._fine_sum = None
        self.offset = np.zeros(2)       # (alt, az) estimate, degrees
        self.radius = None              # how far (deg on the sky) equally good offsets lie from the estimate; None = unknown

    def _consistency(self, cmd, h0, fibres, hit, da, dz) -> np.ndarray:
        """For each candidate offset: how many assigned targets' outcomes it fails to explain."""
        c_alt = np.clip(cmd[0] + da, 0.0, 90.0)
        c_az = (cmd[1] + dz) % 360.0
        d_alt, d_az = sm.project(h0, c_alt, c_az)
        u = d_alt / self.pitch + self.side / 2.0
        v = d_az / self.pitch + self.side / 2.0
        inside = (u >= 0) & (u < self.side) & (v >= 0) & (v < self.side)
        cell = (np.floor(np.where(inside, u, 0)) * self.side + np.floor(np.where(inside, v, 0))).astype(int)
        on_fibre = inside & (cell == fibres[None, :])
        return (on_fibre != hit[None, :]).sum(axis=1)

    def add(self, cmd, h0: np.ndarray, fibres: np.ndarray, hit: np.ndarray) -> bool:
        """Record one exposure. Returns True when the estimate changed noticeably."""
        self.evidence.append((cmd, h0, fibres, hit))
        dropped = max(0, len(self.evidence) - KEEP)
        del self.evidence[:-KEEP]
        if dropped and self._fine_rows:
            # the fine-grid sums follow the evidence window (see _estimate)
            for row in self._fine_rows[:dropped]:
                self._fine_sum -= row
            del self._fine_rows[:dropped]
        self.ll -= MISS_COST * self._consistency(cmd, h0, fibres, hit, self.da, self.dz)
        self.n += 1
        self.misses += int((~hit).sum())
        due = self.n <= 12 or self.n % 8 == 0 or (not hit.all() and self.n % 2 == 0)
        return self._estimate() if due else False

    def _estimate(self) -> bool:
        best = float(self.ll.max())
        if self.ll[self.zero] >= best - 0.5 and self.misses < 3:
            centre = np.zeros(2)
        else:
            top = self.ll >= best - 0.5
            centre = np.array([self.da[top].mean(), self.dz[top].mean()])
            centre = np.round(centre / COARSE_STEP) * COARSE_STEP
        g = np.arange(-FINE_HALF, FINE_HALF + 1e-9, FINE_STEP)
        fa, fz = np.meshgrid(g, g, indexing="ij")
        fa, fz = centre[0] + fa.ravel(), centre[1] + fz.ravel()
        # Counts of unexplained outcomes are whole numbers, so their sum can be kept from one estimate to the
        # next (same fine grid) and extended by the new exposures only: the result is exactly the full sum.
        key = (float(centre[0]), float(centre[1]))
        if self._fine_centre != key or len(self._fine_rows) > len(self.evidence):
            self._fine_centre, self._fine_rows, self._fine_sum = key, [], np.zeros(fa.size, dtype=np.int64)
        for cmd, h0, fibres, hit in self.evidence[len(self._fine_rows):]:
            row = self._consistency(cmd, h0, fibres, hit, fa, fz).astype(np.int64)
            self._fine_rows.append(row)
            self._fine_sum += row
        ll = np.zeros(fa.size) - MISS_COST * self._fine_sum
        mean_cos = 0.0
        for cmd, h0, fibres, hit in self.evidence:
            mean_cos += np.cos(np.radians(cmd[0]))
        mean_cos /= max(1, len(self.evidence))
        top = ll >= ll.max() - 0.5
        estimate = np.array([fa[top].mean(), fz[top].mean()])
        # radius of the equally good set, as a distance on the sky (an azimuth offset shrinks with cos(alt))
        radius = float(np.hypot(fa[top] - estimate[0], (fz[top] - estimate[1]) * mean_cos).max())
        edge = bool(top[[0, -1]].any()) or abs(fa[top] - centre[0]).max() >= FINE_HALF - 1e-9 or abs(fz[top] - centre[1]).max() >= FINE_HALF - 1e-9
        if edge:
            radius = max(radius, FINE_HALF)        # the fine window does not bound the set: not well known yet
        changed = bool(np.abs(estimate - self.offset).max() > 0.004)
        if changed:
            self.log(f"pointing offset: alt {estimate[0]:+.3f} az {estimate[1]:+.3f} deg (radius {radius:.3f}, {self.n} exposures, {self.misses} misses)")
        self.offset = estimate
        self.radius = radius
        return changed

    def margin(self, base: float, unknown: float) -> float:
        """Edge margin (deg) to plan with: the base margin plus what the offset uncertainty requires."""
        if self.radius is None:
            return base + unknown
        return base + min(unknown, self.radius)
