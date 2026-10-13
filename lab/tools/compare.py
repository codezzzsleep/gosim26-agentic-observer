#!/usr/bin/env python3
"""Compare two stored runs card by card and say whether the difference is real.

    python3 tools/compare.py BASE NEW [--set quick tune ...] [--margin 0.3]

For every card that both tags have: relative difference of the total score, NEW against BASE. The cards are the
sample: the hidden cards are four more draws from the same kind of population, so what matters is the mean
difference over cards and how sure we are of its sign. The 95% interval comes from resampling the cards.

    BETTER        the whole interval is above zero
    WORSE         the whole interval is below zero
    NO DIFFERENCE the interval lies within +-margin percent
    UNCLEAR       anything else: more cards are needed (or the change does different things on different cards)

Also printed: wins and losses, the mean per set, the plain mean of totals (how the platform averages cards), and
where the difference comes from (target scores, must-observe penalty, reports, requests, uniformity).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import RUNS, cards_of, load_summary, bootstrap_ci, mean

PARTS = (("target scores", "sum_best_scores"), ("must-observe penalty", "required_penalty"), ("fault reports", "report_settlement"),
         ("requests", "observation_request_reward"), ("uniformity", "uniformity_penalty"))


def compare(base: str, new: str, cards, margin: float = 0.3, quiet: bool = False) -> dict:
    rows = []
    for set_name, card in cards:
        b, n = load_summary(base, card.name), load_summary(new, card.name)
        if not (b and n and "total" in b and "total" in n):
            continue
        scale = max(abs(b["total"]), 1.0)
        rows.append({"set": set_name, "card": card.name, "base": b["total"], "new": n["total"], "rel": 100.0 * (n["total"] - b["total"]) / scale,
                     "parts": {label: float(n.get(key, 0.0)) - float(b.get(key, 0.0)) for label, key in PARTS}})
    result = {"base": base, "new": new, "cards": len(rows), "missing": len(cards) - len(rows)}
    if not rows:
        result["verdict"] = "NO DATA"
        if not quiet:
            print(f"{base} vs {new}: no card has results under both tags")
        return result
    rel = [r["rel"] for r in rows]
    low, high = bootstrap_ci(rel)
    m = mean(rel)
    wins, losses = sum(1 for v in rel if v > 0.05), sum(1 for v in rel if v < -0.05)
    if len(rows) < 6:
        verdict = "TOO FEW CARDS"
    elif low > 0:
        verdict = "BETTER"
    elif high < 0:
        verdict = "WORSE"
    elif -margin <= low and high <= margin:
        verdict = "NO DIFFERENCE"
    else:
        verdict = "UNCLEAR"
    result.update(mean_rel=m, ci_low=low, ci_high=high, wins=wins, losses=losses, verdict=verdict,
                  mean_base=mean([r["base"] for r in rows]), mean_new=mean([r["new"] for r in rows]),
                  worst=min(rows, key=lambda r: r["rel"])["card"], worst_rel=min(rel))
    if quiet:
        return result
    print(f"{'set':10s} {'card':6s} {base[:12]:>12s} {new[:12]:>12s} {'diff %':>8s}")
    for r in rows:
        print(f"{r['set']:10s} {r['card']:6s} {r['base']:12.1f} {r['new']:12.1f} {r['rel']:+8.2f}")
    print()
    for set_name in dict.fromkeys(r["set"] for r in rows):
        part = [r for r in rows if r["set"] == set_name]
        print(f"  {set_name:10s} {len(part):3d} cards  mean diff {mean([r['rel'] for r in part]):+6.2f}%   "
              f"mean total {mean([r['base'] for r in part]):9.1f} -> {mean([r['new'] for r in part]):9.1f}")
    print(f"  where it comes from (points per card): " + ", ".join(f"{label} {mean([r['parts'][label] for r in rows]):+.1f}" for label, _ in PARTS))
    print(f"  plain mean of totals: {result['mean_base']:.1f} -> {result['mean_new']:.1f} ({100.0 * (result['mean_new'] / result['mean_base'] - 1):+.2f}%)"
          if result["mean_base"] > 0 else "")
    print(f"  worst card: {result['worst']} ({result['worst_rel']:+.2f}%)" + (f"; {result['missing']} cards without results under both tags" if result["missing"] else ""))
    print(f"{new} vs {base}: {verdict}  mean {m:+.2f}% per card, 95% interval [{low:+.2f}%, {high:+.2f}%], "
          f"{wins} better / {losses} worse of {len(rows)} cards")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("new"); ap.add_argument("--set", nargs="+")
    ap.add_argument("--margin", type=float, default=0.3); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.set:
        cards = cards_of(a.set)
    else:
        both = sorted(p.name for p in (RUNS / a.new).iterdir() if p.is_dir() and (RUNS / a.base / p.name).is_dir())
        cards = cards_of(both) if both else []
    out = compare(a.base, a.new, cards, a.margin, quiet=a.json)
    if a.json:
        print(json.dumps(out))
