#!/usr/bin/env bash
# One-time setup of the lab: Python environment, numpy, all cards, the candidate builds, and a self-check.
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "${SURVEY26_REPO:-hackathon-survey26}/challenge" ]; then
  echo "!! the official repository is missing: put it at lab/hackathon-survey26 (or set SURVEY26_REPO) - see README.md"; exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "numpy==2.2.6"       # the version the platform build installs

echo "== python $(python3 -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')"
echo "== building the cards listed in cards/SETS.json (84 cards; a minute or two)"
python3 tools/make_cards.py || echo "!! some cards are missing or differ from the recorded ones: read the lines above"
echo "== assembling the candidate builds"
python3 tools/assemble.py --core baselines/v1 --llm --out build/v1_llm --zip
python3 tools/assemble.py --core agent --llm --out build/next_llm --zip
echo "== geometry against the official engine"
python3 tools/test_skymath.py | tail -1
echo "== sky tracker"
python3 tools/test_tracker.py > /dev/null && echo ok
echo "== 40 decisions of the agent on card d101 (in-process, official engine; the score of such a short run means nothing)"
python3 tools/inproc.py --agent-dir agent --card d101 --max-decisions 40 --quiet | cut -c1-120
echo "== recorded results: official example (pro) against v1 (g1) on the development cards"
if [ -d runs/pro ] && [ -d runs/g1 ]; then python3 tools/compare.py pro g1 --set dev | tail -1; else echo "(no recorded runs in this copy: run tools/bench.py first)"; fi
echo
echo "Setup done. Activate the environment with:  . .venv/bin/activate"
echo "Then read PROMPT.md (for the AI) or README.md (for a person)."
