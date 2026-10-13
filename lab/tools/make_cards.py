#!/usr/bin/env python3
"""Build the cards listed in cards/SETS.json (only those not built yet) and check them against the recorded checksums.

    python3 tools/make_cards.py                 # everything that is missing
    python3 tools/make_cards.py --set var_dev   # one set or group
    python3 tools/make_cards.py --list          # what exists: set, card, nights, targets, fibres, site latitude, stress
    python3 tools/make_cards.py --force --set smoke
    python3 tools/make_cards.py --record        # (re)write cards/CHECKSUMS.json from the cards on disk

To add cards: add recipes to cards/SETS.json (new seeds), run this script, then --record.
A card is fully determined by its recipe, so everyone who builds it gets the same card.
"""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder

import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(WORK / "tools"))
CARDS = WORK / "cards"


def load_sets() -> dict:
    return json.loads((CARDS / "SETS.json").read_text())


def expand(names, doc) -> list:
    """Set names for a list of set or group names."""
    out = []
    for name in names:
        for part in doc["groups"].get(name, [name]):
            if part not in doc["sets"]:
                raise SystemExit(f"unknown set or group: {part}")
            if part not in out:
                out.append(part)
    return out


def checksum(card: _Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in card.rglob("*") if p.is_file() and p.name != "card_meta.json"):
        digest.update(path.relative_to(card).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def build_one(job):
    set_name, recipe = job
    import gen_card
    out = CARDS / set_name / recipe["name"]
    try:
        gen_card.build_any(recipe, out)
    except (SystemExit, Exception) as error:      # report it, keep building the others
        return f"{set_name}/{recipe['name']}", f"FAILED: {error}"
    return f"{set_name}/{recipe['name']}", checksum(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--list", action="store_true"); ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    doc = load_sets()
    names = expand(a.set, doc) if a.set else list(doc["sets"])
    recorded_path = CARDS / "CHECKSUMS.json"
    recorded = json.loads(recorded_path.read_text()) if recorded_path.exists() else {}

    if a.list:
        print(f"{'set':10s} {'card':6s} {'nights':>6s} {'targets':>7s} {'fibres':>6s} {'fibre deg2':>10s} {'lat':>6s}  stress     kind")
        for set_name in names:
            for recipe in doc["sets"][set_name]:
                meta_path = CARDS / set_name / recipe["name"] / "card_meta.json"
                if not meta_path.exists():
                    print(f"{set_name:10s} {recipe['name']:6s}  (not built)")
                    continue
                m = json.loads(meta_path.read_text())
                print(f"{set_name:10s} {recipe['name']:6s} {m.get('nights', 0):6d} {m.get('targets', 0):7d} {m.get('fibers', 16):6d} "
                      f"{m.get('fiber_area_deg2', 0.4):10.3f} {m.get('latitude_deg', -24.6):6.1f}  "
                      f"{m.get('stress') or ('hard' if m.get('tier', 1) >= 3 else 'none'):10s} {m.get('kind', 'practice-like')}")
        return 0

    if a.record:
        for set_name in names:
            for recipe in doc["sets"][set_name]:
                card = CARDS / set_name / recipe["name"]
                if card.is_dir():
                    recorded[f"{set_name}/{recipe['name']}"] = checksum(card)
        recorded_path.write_text(json.dumps(recorded, indent=1, sort_keys=True))
        print(f"recorded {len(recorded)} checksums")
        return 0

    jobs = [(set_name, recipe) for set_name in names for recipe in doc["sets"][set_name]
            if a.force or not (CARDS / set_name / recipe["name"] / "config" / "v4_scenario.json").exists()]
    built = {}
    if jobs:
        print(f"building {len(jobs)} cards with {a.jobs} processes ...", flush=True)
        with ProcessPoolExecutor(a.jobs) as pool:
            for key, digest in pool.map(build_one, jobs):
                if digest.startswith("FAILED"):
                    print(f"  {key} {digest}", flush=True)
                    continue
                built[key] = digest
                print(f"  built {key}", flush=True)
    # verify everything on disk against the record
    bad = missing = checked = 0
    for set_name in names:
        for recipe in doc["sets"][set_name]:
            key = f"{set_name}/{recipe['name']}"
            card = CARDS / set_name / recipe["name"]
            if not card.is_dir():
                missing += 1
                continue
            digest = built.get(key) or checksum(card)
            if key in recorded:
                checked += 1
                if recorded[key] != digest:
                    bad += 1
                    print(f"  DIFFERENT from the recorded card: {key}")
            else:
                recorded[key] = digest
    if bad:
        print(f"{bad} cards differ from the recorded ones. They are valid cards, but scores on them cannot be compared with scores "
              f"recorded elsewhere. (Different Python version or changed recipe? Run --record to accept them.)")
    else:
        recorded_path.write_text(json.dumps(recorded, indent=1, sort_keys=True))
    total = sum(len(doc["sets"][s]) for s in names)
    print(f"cards: {total - missing} of {total} present, {checked} checked against the record, {bad} different")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
