#!/usr/bin/env python3
"""Build local, scorable v4 cards with the organizers' public generators (for local validation only).

Three kinds of card, all complete (config/ public/ truth/) and playable by the official local runner:

    # practice-like: small catalogue made here, simulated weather (tiers 1-4 mirror the local cards L1-L4)
    python3 tools/gen_card.py --seed 101 --tier 2 --out cards/dev/d101

    # varied: site, season length, catalogue size, fibre grid and score zero points drawn from the range the
    # competition cards A-D span -- what the hidden cards E-H may look like
    python3 tools/gen_card.py --kind var --seed 501 --out cards/var_dev/v501

    # real: the published inputs of an official card (targets, footprint, calendar, fibres, score rules) with
    # simulated weather, events and requests -- the official cards' own weather is not published
    python3 tools/gen_card.py --kind real --src cards/official/a --seed 1 --tier 2 --out cards/real_dev/A1
    python3 tools/gen_card.py --kind real --src cards/official/d --seed 1 --tier 3 --stress data_loss --nights 8 --out cards/smoke/D8

The weather, the events and the requests of every card built here are simulated: they follow the public
simulator's rules, not the hidden weather of any official card.
"""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import argparse, csv, json, random, shutil, sys, tempfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(REPO))
SITE = {"name": "Paranal, Chile (virtual)", "latitude_deg": -24.6157, "longitude_deg": -70.3976, "utc_offset_hours": -4.0}

# Event counts for a season of about 36 nights; longer seasons scale them.
TIERS = {
    1: dict(rainy=2, cloudy=3, smoggy=1, cold_wave=1, tornado=0, rocket=1, quake=1, terrain=[1, 2], stress=False, fault=[0.4, 0.7]),
    2: dict(rainy=2, cloudy=3, smoggy=2, cold_wave=1, tornado=0, rocket=2, quake=1, terrain=[1, 2], stress=False, fault=[0.4, 0.7]),
    3: dict(rainy=3, cloudy=3, smoggy=2, cold_wave=2, tornado=0, rocket=1, quake=2, terrain=[1, 3], stress=True, fault=[0.3, 0.5]),
    4: dict(rainy=3, cloudy=4, smoggy=2, cold_wave=2, tornado=1, rocket=2, quake=2, terrain=[1, 3], stress=True, fault=[0.3, 0.5]),
}
EVENT_KEYS = ("rainy", "cloudy", "smoggy", "cold_wave", "tornado", "rocket", "quake")
CLASS_MODELS = {"ELG": {"flux_median": 0.55, "flux_log_sigma": 0.40, "science_weight": 1.0},
                "BGS": {"flux_median": 1.25, "flux_log_sigma": 0.30, "science_weight": 0.45},
                "LRG": {"flux_median": 0.35, "flux_log_sigma": 0.35, "science_weight": 1.0},
                "QSO": {"flux_median": 0.18, "flux_log_sigma": 0.50, "science_weight": 1.7},
                "Star": {"flux_median": 4.0, "flux_log_sigma": 0.25, "science_weight": 0.3}}
# Sites for the varied cards: the four of cards A-D and a few more at other latitudes.
VAR_SITES = [
    {"name": "Paranal, Chile (virtual)", "latitude_deg": -24.6157, "longitude_deg": -70.3976, "utc_offset_hours": -4.0},
    {"name": "Cape Town, South Africa (virtual)", "latitude_deg": -33.9249, "longitude_deg": 18.4241, "utc_offset_hours": 2.0},
    {"name": "Mauna Kea, Hawaii (virtual)", "latitude_deg": 19.8207, "longitude_deg": -155.4681, "utc_offset_hours": -10.0},
    {"name": "Nemo Observatory, South Pacific (fictional)", "latitude_deg": -45.0, "longitude_deg": -135.0, "utc_offset_hours": -9.0},
    {"name": "La Palma, Canary Islands (virtual)", "latitude_deg": 28.76, "longitude_deg": -17.89, "utc_offset_hours": 0.0},
    {"name": "Siding Spring, Australia (virtual)", "latitude_deg": -31.27, "longitude_deg": 149.07, "utc_offset_hours": 10.0},
    {"name": "Xinglong, China (virtual)", "latitude_deg": 40.39, "longitude_deg": 117.58, "utc_offset_hours": 8.0},
    {"name": "Equator Station (fictional)", "latitude_deg": 2.0, "longitude_deg": 78.0, "utc_offset_hours": 5.0},
]
STRESS_MODES = ("none", "data_loss", "hard")     # hard = data loss and a hidden pointing offset


def scaled_counts(tier: int, n_nights: int, rng=None) -> dict:
    """Event counts of a tier for a season of n_nights; with rng, each count also varies by about +-30%."""
    t = dict(TIERS[tier])
    scale = n_nights / 36.0
    for key in EVENT_KEYS:
        value = t[key] * scale * (rng.uniform(0.7, 1.3) if rng is not None else 1.0)
        t[key] = int(round(value))
    return t


CLIMATE = "standard"      # "harsh": a guess at a real site's worse season (more closures, poorer and more variable sky)


def weather_config(seed: int, site: dict, start_date: date, end_date: date, counts: dict, fault_count: int, stress: str) -> dict:
    config = _weather_config(seed, site, start_date, end_date, counts, fault_count, stress)
    if CLIMATE == "harsh":
        q = config["quality"]
        q["seeing_arcsec"].update({"nominal": 1.35, "night_sigma": 0.30, "slot_sigma": 0.12})
        q["transparency"].update({"nominal": 0.78, "night_sigma": 0.14, "slot_sigma": 0.06})
        q["sky_quality"].update({"night_sigma": 0.20, "slot_sigma": 0.07})
        q["night_correlation"], q["slot_correlation"] = 0.70, 0.80
        config["background_closure"].update({"start_probability_per_open_slot": 0.03, "reopen_probability_per_closed_slot": 0.22})
        for name in ("rainy", "cloudy", "smoggy"):
            config["weather_events"]["conditions"][name]["count"] *= 2
    return config


def _weather_config(seed: int, site: dict, start_date: date, end_date: date, counts: dict, fault_count: int, stress: str) -> dict:
    t = counts
    quiet_offset = {"max_abs_alt_rad": 1e-7, "max_abs_az_rad": 1e-7}      # the simulator needs a positive bound
    return {
        "schema_version": "v4-weather-v2", "seed": seed, "seed_derivation": "sha256-v1", "site": site,
        "survey": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "slot_seconds": 900, "sun_altitude_limit_deg": -18.0},
        "quality": {
            "seeing_arcsec": {"nominal": 1.10, "seasonal_amplitude": 0.15, "night_sigma": 0.16, "slot_sigma": 0.07, "minimum": 0.50, "maximum": 4.00},
            "transparency": {"nominal": 0.88, "seasonal_amplitude": 0.07, "night_sigma": 0.06, "slot_sigma": 0.025, "minimum": 0.08, "maximum": 1.00},
            "sky_quality": {"nominal": 1.00, "seasonal_amplitude": 0.12, "night_sigma": 0.10, "slot_sigma": 0.035, "minimum": 0.05, "maximum": 1.50},
            "instrument_efficiency": {"jitter_minimum": 0.90, "jitter_maximum": 1.00, "minimum": 0.10, "maximum": 1.00},
            "night_correlation": 0.82, "slot_correlation": 0.88, "seasonal_phase_day": 210},
        "background_closure": {"start_probability_per_open_slot": 0.008, "reopen_probability_per_closed_slot": 0.32, "seasonal_probability_amplitude": 0.006},
        "weather_events": {
            "sector_width_deg_range": [40.0, 120.0], "sector_altitude_limit_deg_range": [40.0, 80.0],
            "conditions": {
                "rainy": {"count": t["rainy"], "duration_slots": [8, 40], "scope_weights": {"ALL": 1.0}, "force_close": True, "seeing_multiplier": 1.8, "transparency_multiplier": 0.25, "sky_quality_multiplier": 0.35},
                "cloudy": {"count": t["cloudy"], "duration_slots": [8, 32], "scope_weights": {"ALL": 0.35, "HORIZON_SECTOR": 0.65}, "force_close": False, "seeing_multiplier": 1.25, "transparency_multiplier": 0.55, "sky_quality_multiplier": 0.65},
                "smoggy": {"count": t["smoggy"], "duration_slots": [24, 96], "scope_weights": {"ALL": 0.30, "HORIZON_SECTOR": 0.70}, "force_close": False, "seeing_multiplier": 1.15, "transparency_multiplier": 0.68, "sky_quality_multiplier": 0.72},
                "cold_wave": {"count": t["cold_wave"], "duration_slots": [48, 192], "scope_weights": {"ALL": 1.0}, "force_close": False, "seeing_multiplier": 1.05, "transparency_multiplier": 0.96, "sky_quality_multiplier": 0.98},
                "tornado": {"count": t["tornado"], "duration_slots": [4, 12], "scope_weights": {"ALL": 1.0}, "force_close": True, "seeing_multiplier": 2.0, "transparency_multiplier": 0.15, "sky_quality_multiplier": 0.25}}},
        "rocket_launch": {"count": t["rocket"], "duration_slots": [2, 5], "azimuth_sector_width_deg": [40.0, 80.0], "max_altitude_deg": [30.0, 60.0]},
        "earthquake": {"count": t["quake"], "magnitude_range": [4.5, 7.0], "impact_coefficient": 0.75, "reference_magnitude": 6.8,
                       "max_degradation": 0.95, "decay_nights": 4.0, "negligible_degradation": 0.01,
                       "seeing_impact_coefficient": 0.0, "transparency_impact_coefficient": 0.0, "sky_quality_impact_coefficient": 0.0},
        "terrain_obstruction": {"sector_count": t["terrain"], "width_deg_range": [30.0, 70.0], "max_altitude_deg_range": [35.0, 50.0]},
        "instrument_fault": {"count": fault_count, "instrument_efficiency_multiplier_range": t["fault"]},
        "publication": {"forecast_interval_days": 7, "forecast_horizon_days": 7},
        "stress_tests": {"enabled": stress != "none", "data_loss": {"trigger": "after_earthquake", "window_max_fraction": 0.06},
                         "pointing_offset": (quiet_offset if stress == "data_loss" else {"max_abs_alt_rad": 0.002, "max_abs_az_rad": 0.002})},
        "output": {"directory": "."},
    }


def catalog_config(seed: int, site: dict, start_date: date, end_date: date, n_targets: int, area: float, centers: list, weights: list,
                   required_fraction: float, models: dict) -> dict:
    return {
        "schema_version": "v4-catalog-v2", "seed": seed, "seed_derivation": "sha256-v1", "site": site,
        "footprint": {"total_area_deg2": area, "area_tolerance_fraction": 0.03, "n_components": len(weights),
                      "component_area_weights": weights, "component_centers": centers,
                      "component_gap_deg": 2.0, "pole_margin_deg": 5.0, "vertices_per_component": 40,
                      "harmonic_amplitude_ranges": [0.16, 0.10, 0.06]},
        "targets": {"total_count": n_targets, "required_fraction": required_fraction,
                    "class_fractions": {"ELG": 0.34, "BGS": 0.24, "LRG": 0.20, "QSO": 0.14, "Star": 0.08},
                    "clustered_fraction": 0.25, "n_cluster_centers": max(20, n_targets // 200), "cluster_sigma_deg": 0.5,
                    "models": models},
        "observability": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "sun_altitude_limit_deg": -18.0,
                          "minimum_altitude_deg": 30.0, "minimum_window_seconds": 900, "max_position_attempts": 64},
        "output": {"directory": "."},
    }


def lst_at_midseason(start_date: date, n_nights: int) -> float:
    mid = start_date + timedelta(days=n_nights // 2)
    return ((mid - date(2026, 9, 22)).days * 0.9856) % 360.0          # LST at local midnight ~ days since the equinox


def finish(out: Path, work: Path, seed: int, site: dict, fiber: dict, score: dict, stress: str, counts: dict, fault_count: int,
           start_date: date, end_date: date, meta: dict, n_requests: int, reference_calendar: Path = None) -> Path:
    """Simulate weather, events and requests in `work` (which holds targets.csv and footprint.csv) and write the card."""
    from challenge.v4_observation_requests import generate_requests
    from challenge.v4_runner import load_scenario
    from challenge.v4_weather_simulator import generate
    (work / "weather.json").write_text(json.dumps(weather_config(seed, site, start_date, end_date, counts, fault_count, stress)))
    generate(work / "weather.json", work)
    if reference_calendar is not None and reference_calendar.read_text() != (work / "v4_night_calendar.csv").read_text():
        raise SystemExit(f"{out.name}: the simulated night calendar differs from the official card's calendar")
    exposure = fiber["exposure"]
    generate_requests(seed=seed, targets_csv=work / "targets.csv", night_calendar_csv=work / "v4_night_calendar.csv",
                      slots_csv=work / "v4_slots.csv", site=site, minimum_altitude_deg=30.0,
                      output_path=work / "v4_observation_requests.jsonl",
                      settings={"count": n_requests, "targets_per_request": 8, "minimum_completion_fraction": 0.75, "window_nights": 2,
                                "completion_reward": 100.0,
                                "completion_factor_threshold": score["observation_requests"]["completion_factor_threshold"],
                                "minimum_feature_flux": score["observation_requests"]["completion_factor_threshold"] * score["flux_zero_point"] * score["exposure_zero_point_seconds"] / 3600},
                      flux_zero_point=float(score["flux_zero_point"]), exposure_zero_point_seconds=float(score["exposure_zero_point_seconds"]),
                      min_duration_seconds=int(exposure["min_duration_seconds"]), max_duration_seconds=int(exposure["max_duration_seconds"]))
    public = ("targets.csv", "footprint.csv", "v4_night_calendar.csv", "v4_bulletins.jsonl", "v4_forecasts.jsonl")
    truth = ["v4_slots.csv", "v4_weather_truth.csv", "v4_events.csv", "v4_earthquake_effects.csv", "v4_observation_requests.jsonl"]
    if stress != "none":
        truth.append("v4_stress_events.csv")
    if out.exists():
        shutil.rmtree(out)
    for sub in ("config", "public", "truth"):
        (out / sub).mkdir(parents=True)
    for name in public:
        shutil.copyfile(work / name, out / "public" / name)
    for name in truth:
        shutil.copyfile(work / name, out / "truth" / name)
    scenario = {"schema_version": "v4-scenario-v1", "name": out.name,
                "task_card": {"card_id": out.name, "scenario_slug": f"v4-{out.name}", "phase": "local"},
                "site": {**site, "sun_altitude_limit_deg": -18.0}, "minimum_altitude_deg": 30.0,
                "limits": {"global_wallclock_seconds": 900}, "fiber_config": "v4_fiber_config.json", "score_config": "v4_score_config.json",
                "products": {"targets_csv": "../public/targets.csv", "footprint_csv": "../public/footprint.csv",
                             "night_calendar_csv": "../public/v4_night_calendar.csv", "bulletins_jsonl": "../public/v4_bulletins.jsonl",
                             "forecasts_jsonl": "../public/v4_forecasts.jsonl", "slots_csv": "../truth/v4_slots.csv",
                             "weather_truth_csv": "../truth/v4_weather_truth.csv", "events_csv": "../truth/v4_events.csv",
                             "earthquake_effects_csv": "../truth/v4_earthquake_effects.csv",
                             "observation_requests_jsonl": "../truth/v4_observation_requests.jsonl"},
                "stress": ({"enabled": True, "stress_events_csv": "../truth/v4_stress_events.csv"} if stress != "none" else {"enabled": False})}
    dump = lambda p, d: p.write_text(json.dumps(d, indent=2) + "\n")
    dump(out / "config" / "v4_score_config.json", score); dump(out / "config" / "v4_fiber_config.json", fiber); dump(out / "config" / "v4_scenario.json", scenario)
    n_nights = sum(1 for _ in open(out / "public" / "v4_night_calendar.csv")) - 1
    n_targets = sum(1 for _ in open(out / "public" / "targets.csv")) - 1
    (out / "card_meta.json").write_text(json.dumps({"seed": seed, "start": start_date.isoformat(), "nights": n_nights, "targets": n_targets,
                                                     "fibers": fiber["field"]["n_fibers"], "fiber_area_deg2": fiber["field"]["fiber_area_deg2"],
                                                     "latitude_deg": site["latitude_deg"], "stress": stress, "fault_count": fault_count,
                                                     "requests": n_requests, **meta}, indent=1))
    load_scenario(out / "config" / "v4_scenario.json")      # the official loader accepts the card
    return out


def reference_score() -> dict:
    return json.loads((REPO / "challenge/reference/v4/v4_score_config.json").read_text())


def build(seed: int, tier: int, out: Path, nights=None, start=None, targets=None, fault_count=1, required=None, attempt=0):
    """Practice-like card: Paranal, 32-38 nights, 8-10 thousand targets, 4 x 4 fibres (the scale of practice cards)."""
    from challenge.v4_catalog_generator import generate_catalog
    rng = random.Random(seed * 7919 + tier + attempt * 104729)
    t = TIERS[tier]
    start_date = date.fromisoformat(start) if start else date(2026, 10, 1) + timedelta(days=rng.randint(0, 75))
    n_nights = nights or rng.randint(32, 38)
    end_date = start_date + timedelta(days=n_nights)
    n_targets = targets or rng.choice([7800, 8600, 9400, 10000])
    area = n_targets / 5.0
    lst_mid = lst_at_midseason(start_date, n_nights)
    offs = [rng.uniform(-75, -40), rng.uniform(-15, 15), rng.uniform(40, 75)]
    decs = [rng.uniform(-32, -12), rng.uniform(-58, -42), rng.uniform(-28, -8)]
    rng.shuffle(decs)
    centers = [[round((lst_mid + o) % 360.0, 1), round(d, 1)] for o, d in zip(offs, decs)]
    weights = sorted([rng.uniform(0.25, 0.45) for _ in range(3)], reverse=True)
    s = sum(weights); weights = [round(w / s, 3) for w in weights]; weights[0] = round(1 - sum(weights[1:]), 3)
    catalog = catalog_config(seed, SITE, start_date, end_date, n_targets, area, centers, weights,
                             required or rng.choice([0.05, 0.05, 0.06]), CLASS_MODELS)
    fiber = {"schema_version": "v4-fiber-map-v1", "site": SITE, "field": {"fiber_area_deg2": 0.4, "gap_deg": 0.0, "n_fibers": 16},
             "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}}
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "catalog.json").write_text(json.dumps(catalog))
        generate_catalog(work / "catalog.json", work)
        return finish(out, work, seed, SITE, fiber, reference_score(), "hard" if t["stress"] else "none", t, fault_count,
                      start_date, end_date, {"kind": "practice-like", "tier": tier, "centers": centers}, n_requests=2)


def build_var(seed: int, out: Path, attempt=0):
    """Varied card: everything the hidden cards may change, drawn from about the range cards A-D span."""
    from challenge.v4_catalog_generator import generate_catalog
    rng = random.Random(seed * 7919 + 55)
    site = rng.choice(VAR_SITES)
    start_date = date(2026, 10, 1) + timedelta(days=rng.randint(0, 364))
    n_nights = rng.choice([100, 120, 120, 150, 180, 180, 240, 365])
    end_date = start_date + timedelta(days=n_nights)
    n_targets = rng.choice([20000, 30000, 30000, 40000, 50000])
    area = min(8000.0, float(int(n_targets / rng.choice([5.0, 6.25, 7.5]) / 100.0) * 100))
    n_regions = rng.choice([2, 3, 3, 4])
    side = rng.choice([3, 4, 4, 5, 6, 8, 10])
    fiber_area = round(rng.uniform(1.5, 6.5) / (side * side), 3)
    f0, t0 = rng.choice([(0.5, 900), (0.5, 900), (0.8, 1000), (0.6, 1000)])
    models = {name: dict(model) for name, model in CLASS_MODELS.items()}
    for name in sorted(models):
        models[name]["flux_median"] = round(models[name]["flux_median"] * 2.718281828 ** rng.uniform(-0.4, 0.5), 3)
    if rng.random() < 0.25:
        models["QSO"]["flux_median"] = round(models["QSO"]["flux_median"] * 8.0, 3)        # card B has bright quasars
    required_fraction = rng.choice([0.05, 0.05, 0.06])
    tier = rng.choice([1, 2, 3, 4])
    stress = rng.choice(["none", "none", "data_loss", "hard"])
    fault_count = rng.choice([1, 2, 2, 3])
    counts = scaled_counts(tier, n_nights, rng)
    n_requests = max(2, int(round(2 * n_nights / 36.0 * rng.uniform(0.6, 1.2))))
    # regions: spread in right ascension over what the season brings up, at declinations the site sees well.
    # A draw whose regions touch is drawn again; if that keeps happening the footprint is made a little smaller.
    place = random.Random(seed * 7919 + 77 + attempt * 104729)
    area = float(int(area * 0.94 ** max(0, attempt - 7) / 100.0) * 100)
    lat = site["latitude_deg"]
    lst_mid = lst_at_midseason(start_date, n_nights)
    span = min(170.0, 60.0 + n_nights * 0.9856 / 2.0)
    offs = [(-span + 2.0 * span * (k + 0.5) / n_regions) + place.uniform(-8, 8) for k in range(n_regions)]
    decs = [max(-48.0, min(48.0, lat * 0.8 + place.uniform(-18, 18))) for _ in range(n_regions)]
    centers = [[round((lst_mid + o) % 360.0, 1), round(d, 1)] for o, d in zip(offs, decs)]
    weights = sorted([place.uniform(0.25, 0.45) for _ in range(n_regions)], reverse=True)
    s = sum(weights); weights = [round(w / s, 3) for w in weights]; weights[0] = round(1 - sum(weights[1:]), 3)
    catalog = catalog_config(seed, site, start_date, end_date, n_targets, area, centers, weights, required_fraction, models)
    fiber = {"schema_version": "v4-fiber-map-v1", "site": site, "field": {"fiber_area_deg2": fiber_area, "gap_deg": 0.0, "n_fibers": side * side},
             "exposure": {"min_duration_seconds": 60, "max_duration_seconds": 3600}}
    score = reference_score()
    score["flux_zero_point"], score["exposure_zero_point_seconds"] = f0, t0
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "catalog.json").write_text(json.dumps(catalog))
        generate_catalog(work / "catalog.json", work)
        return finish(out, work, seed, site, fiber, score, stress, counts, fault_count, start_date, end_date,
                      {"kind": "varied", "tier": tier, "centers": centers, "area_deg2": area, "f0": f0, "t0": t0}, n_requests=n_requests)


def build_real(src: Path, seed: int, tier: int, out: Path, nights=None, stress="none", fault_count=None):
    """An official card's published inputs with simulated weather, events and requests.

    `nights`: keep only the first N nights of the season (a short card for quick checks)."""
    fiber = json.loads((src / "config" / "v4_fiber_config.json").read_text())
    score = json.loads((src / "config" / "v4_score_config.json").read_text())
    site = dict(fiber["site"])
    calendar = list(csv.DictReader(open(src / "public" / "v4_night_calendar.csv")))
    start_date = date.fromisoformat(calendar[0]["night_date"])
    n_nights = min(nights or len(calendar), len(calendar))
    end_date = date.fromisoformat(calendar[n_nights - 1]["night_date"]) + timedelta(days=1)      # the simulator's end date is exclusive
    rng = random.Random(seed * 7919 + tier + 31)
    counts = scaled_counts(tier, n_nights, rng)
    if nights:      # a short card still needs something to happen
        counts["quake"] = max(1, counts["quake"]); counts["cloudy"] = max(1, counts["cloudy"])
    if fault_count is None:
        fault_count = 1 if n_nights < 60 else rng.choice([1, 2, 2, 3])
    n_requests = max(1 if nights else 2, int(round(2 * n_nights / 36.0 * rng.uniform(0.6, 1.2))))
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        shutil.copyfile(src / "public" / "targets.csv", work / "targets.csv")
        shutil.copyfile(src / "public" / "footprint.csv", work / "footprint.csv")
        return finish(out, work, seed, site, fiber, score, stress, counts, fault_count, start_date, end_date,
                      {"kind": "real catalogue, simulated weather", "source": src.name, "tier": tier, "short": bool(nights)},
                      n_requests=n_requests, reference_calendar=None if nights else src / "public" / "v4_night_calendar.csv")


def build_any(recipe: dict, out: Path) -> Path:
    """Build the card a recipe describes (see cards/SETS.json)."""
    kind = recipe.get("kind", "practice")
    for attempt in range(60):
        try:
            if kind == "practice":
                return build(recipe["seed"], recipe.get("tier", 2), out, recipe.get("nights"), recipe.get("start"), recipe.get("targets"),
                             recipe.get("fault_count", 1), attempt=attempt)
            if kind == "var":
                return build_var(recipe["seed"], out, attempt=attempt)
            if kind == "real":
                return build_real(WORK / "cards" / "official" / recipe["src"], recipe["seed"], recipe.get("tier", 2), out,
                                  recipe.get("nights"), recipe.get("stress", "none"), recipe.get("fault_count"))
            raise SystemExit(f"unknown card kind {kind}")
        except RuntimeError as error:      # the drawn regions overlap or touch a pole: draw again
            if _os.environ.get("AO_CARD_VERBOSE"):
                print(f"{out.name}: attempt {attempt}: {error}", file=sys.stderr)
    raise SystemExit(f"{out.name}: no valid footprint found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("practice", "var", "real"), default="practice")
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--tier", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--nights", type=int); ap.add_argument("--start")
    ap.add_argument("--targets", type=int); ap.add_argument("--fault-count", type=int)
    ap.add_argument("--src", help="real: folder of an official card under cards/official (a, b, c, d, alpha, ...) or a path")
    ap.add_argument("--stress", choices=STRESS_MODES, default="none", help="real: stress events to simulate")
    ap.add_argument("--climate", choices=("standard", "harsh"), default="standard")
    a = ap.parse_args()
    CLIMATE = a.climate
    recipe = {"kind": a.kind, "seed": a.seed, "tier": a.tier, "nights": a.nights, "start": a.start, "targets": a.targets, "stress": a.stress}
    if a.fault_count is not None:
        recipe["fault_count"] = a.fault_count
    if a.kind == "real":
        if not a.src:
            raise SystemExit("--kind real needs --src")
        recipe["src"] = a.src
    print(build_any(recipe, a.out))
