import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import sys
sys.path.insert(0, str(_Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else WORK / "agent"))      # optional argument: agent folder
import numpy as np, json, math
from tracker import SkyTracker
sc = json.load(open(REPO / "challenge/reference/v4/v4_score_config.json"))
def run(seq, verbose=True):
    tr = SkyTracker(sc)
    rng = np.random.default_rng(0)
    for step, (W, eta, phi, prog, notices) in enumerate(seq):
        n = 14
        flux = rng.lognormal(math.log(0.5), 0.6, n); m = rng.uniform(1.2, 1.45, n); T = 450.0
        a = flux * T * m / 450.0
        Weff = W * phi
        g = np.minimum(1.0, a * eta * Weff)
        band = np.where(Weff * m >= 0.65, 0, np.where(Weff * m >= 0.4, 1, 2))
        mult = np.where(band == {"DARK": 0, "BRIGHT": 1, "BACKUP": 2}[prog], sc["program"]["multipliers"][prog], 1.0)
        ratio = g * mult
        tr.advance(0, notices, T / 900.0)
        out = tr.observe(a, m, ratio, prog, step * 0.125)
        if verbose: print(step, f"true W {W:.2f} eta {eta:.2f} phi {phi:.2f} ->", {k: round(v, 3) for k, v in out.items()})
    return tr
print("closure case")
run([(0.6, 0.95, 1, "DARK", set())] * 6 + [(0.6, 0.95, 0.133, "DARK", set())] + [(0.6, 0.95, 1, "DARK", set())] * 2)
print("fault case")
run([(0.6, 0.95, 1, "DARK", set())] * 6 + [(0.6, 0.95 * 0.55, 1, "DARK", set())] * 4)
print("fault case, high W")
run([(0.85, 0.95, 1, "DARK", set())] * 6 + [(0.85, 0.95 * 0.65, 1, "DARK", set())] * 4)
