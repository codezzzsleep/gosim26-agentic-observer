"""Survey state: the public catalogue as arrays, what has been achieved so far, and a replayable ledger.

Only public information lives here: the initialize payload, our own actions and the engine's feedback.
"""
from __future__ import annotations

import numpy as np

from skymath import Site, radec_unit, unix


class Survey:
    def __init__(self, init: dict):
        site = init["site"]
        self.site = Site(site["latitude_deg"], site["longitude_deg"])
        self.min_alt = float(site["minimum_altitude_deg"])
        survey = init["survey"]
        self.start = unix(survey["start_utc"])
        self.end = unix(survey["end_utc"])
        self.slot_seconds = int(survey["slot_seconds"])
        self.nights = [(unix(n["observing_start_utc"]), unix(n["observing_end_utc"]), n.get("night_date", ""))
                       for n in survey["nights"]]
        instrument = init["instrument"]
        self.n_fibers = int(instrument["n_fibers"])
        self.side = int(instrument["grid_side"])
        self.pitch = float(instrument["pitch_deg"])
        self.glass = float(instrument.get("glass_side_deg", self.pitch))
        self.fov = float(instrument["fov_side_deg"])
        self.min_exposure = int(instrument["exposure"]["min_duration_seconds"])
        self.max_exposure = int(instrument["exposure"]["max_duration_seconds"])
        score = init["scoring"]
        self.scoring = score
        self.f0t0 = float(score["flux_zero_point"]) * float(score["exposure_zero_point_seconds"])
        self.q0 = float(score["q0"])
        self.beta = float(score["airmass_exponent"])
        self.lunar_model = score["lunar_model"]
        self.multipliers = {k: float(v) for k, v in score["program"]["multipliers"].items()}
        self.mismatch = float(score["program"]["mismatch_multiplier"])
        self.top_mult = max(self.multipliers.values())
        self.required_penalty = float(score["required"]["penalty_per_missing"])
        self.required_threshold = float(score["required"]["observed_factor_threshold"])
        self.uniformity = score.get("uniformity", {})
        reporting = score.get("reporting", {})
        self.report_reward = float(reporting.get("correct_reward", 0.0))
        self.report_penalty = abs(float(reporting.get("false_penalty", 0.0)))
        self.free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.max_consecutive_reports = int(reporting.get("max_consecutive_reports", 32))

        columns = init["targets"]["columns"]
        col = {name: columns.index(name) for name in columns}
        rows = init["targets"]["rows"]
        self.n = len(rows)
        self.ids = [row[col["target_id"]] for row in rows]
        self.index_of = {target_id: i for i, target_id in enumerate(self.ids)}
        self.ra = np.array([float(row[col["ra_deg"]]) for row in rows])
        self.dec = np.array([float(row[col["dec_deg"]]) for row in rows])
        self.flux = np.maximum(np.array([float(row[col["feature_flux"]]) for row in rows]), 1e-6)
        self.weight = np.array([float(row[col["science_weight"]]) for row in rows])
        self.required = np.array([bool(row[col["required"]]) for row in rows])
        self.eq = radec_unit(self.ra, self.dec)

        # achievements
        self.cur = np.zeros(self.n)        # best score / weight so far (includes the program multiplier)
        self.factor = np.zeros(self.n)     # best completion factor so far (estimated; decides required/uniformity/requests)
        self.misses = np.zeros(self.n, dtype=int)   # assigned but not hit
        self.tries = np.zeros(self.n, dtype=int)    # exposures that hit the target
        self.ledger: list[dict] = []       # one entry per observe action, replayable after data loss
        self.observes = 0

    # --- nights -----------------------------------------------------------------------------------------

    def current_night(self, now: float):
        for k, (start, end, _) in enumerate(self.nights):
            if start <= now < end:
                return k, start, end
        return None

    def next_night_start(self, now: float):
        for start, _, _ in self.nights:
            if start > now:
                return start
        return None

    def seconds_left_in_season(self, now: float) -> float:
        return float(sum(max(0.0, end - max(start, now)) for start, end, _ in self.nights if end > now))

    # --- ledger -----------------------------------------------------------------------------------------

    def record(self, observe_index: int, start: float, end: float, idx: np.ndarray, factor: np.ndarray, ratio: np.ndarray) -> None:
        self.ledger.append({"index": int(observe_index), "start": start, "end": end, "idx": idx.copy(),
                            "factor": factor.copy(), "ratio": ratio.copy(), "valid": True})
        np.maximum.at(self.cur, idx, ratio)
        np.maximum.at(self.factor, idx, factor)
        np.add.at(self.tries, idx, 1)

    def replay(self) -> None:
        self.cur[:] = 0.0
        self.factor[:] = 0.0
        for entry in self.ledger:
            if entry["valid"]:
                np.maximum.at(self.cur, entry["idx"], entry["ratio"])
                np.maximum.at(self.factor, entry["idx"], entry["factor"])

    def invalidate(self, first: int, last_exclusive: int) -> int:
        count = 0
        for entry in self.ledger:
            if entry["valid"] and first <= entry["index"] < last_exclusive:
                entry["valid"] = False
                count += 1
        self.replay()
        return count

    def window_factors(self, issued: float, deadline: float, idx: np.ndarray) -> np.ndarray:
        """Best factor of each target in idx from exposures wholly inside [issued, deadline]."""
        best = {int(i): 0.0 for i in idx}
        for entry in self.ledger:
            if entry["valid"] and entry["start"] >= issued and entry["end"] <= deadline:
                for i, f in zip(entry["idx"], entry["factor"]):
                    if int(i) in best and f > best[int(i)]:
                        best[int(i)] = float(f)
        return np.array([best[int(i)] for i in idx])
