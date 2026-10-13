"""Shared helpers of the lab tools: where things are, which cards a name means, reading stored results."""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import json
import math
import random

CARDS = WORK / "cards"
RUNS = WORK / "runs"
RUNNER = REPO / "examples" / "_local" / "runner" / "run_local.py"
LCARDS = REPO / "examples" / "_local" / "cards"


def load_sets() -> dict:
    return json.loads((CARDS / "SETS.json").read_text())


def cards_of(names) -> list:
    """[(set name, card folder)] for set names, group names, 'L' (official local cards) or single card names.

    Names may be given as a list and/or comma-separated."""
    doc = load_sets()
    out, seen = [], set()

    def add(set_name, path):
        if path.name not in seen and path.is_dir():
            seen.add(path.name)
            out.append((set_name, path))

    flat = [part for name in names for part in str(name).split(",") if part]
    for name in flat:
        if name in doc["groups"] or name in doc["sets"]:
            for set_name in doc["groups"].get(name, [name]):
                for recipe in doc["sets"][set_name]:
                    path = CARDS / set_name / recipe["name"]
                    if not path.is_dir():
                        raise SystemExit(f"card {set_name}/{recipe['name']} is not built: run  python3 tools/make_cards.py")
                    add(set_name, path)
        elif name == "L":
            for card in ("L1", "L2", "L3", "L4"):
                add("L", LCARDS / card)
        else:
            hits = [p for p in sorted(CARDS.glob(f"*/{name}")) + [LCARDS / name, _Path(name)] if (p / "config").is_dir()]
            if not hits:
                raise SystemExit(f"unknown set, group or card: {name}")
            add(hits[0].parent.name, hits[0])
    return out


def card_cost(path) -> float:
    """Rough relative running time of a card (to start the long ones first)."""
    try:
        meta = json.loads((path / "card_meta.json").read_text())
        return meta.get("nights", 36) * (1.0 + meta.get("fibers", 16) / 40.0) * (1.0 + meta.get("targets", 9000) / 60000.0)
    except (OSError, ValueError):
        return 36.0


def load_summary(tag: str, card_name: str):
    path = RUNS / tag / card_name / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def log_tail(tag: str, card_name: str) -> str:
    path = RUNS / tag / card_name / "agent.log"
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def bootstrap_ci(values, seed=12345, rounds=4000):
    """95% interval of the mean of `values`, resampling the cards."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def stdev(values):
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
