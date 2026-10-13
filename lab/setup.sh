#!/usr/bin/env bash
# One-time setup of the lab: Python environment, numpy, the local cards, and a self-check of the submitted agent.
#   bash setup.sh            # all 84 cards
#   bash setup.sh quick      # only the groups / sets named (faster), e.g. quick smoke
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d "${SURVEY26_REPO:-hackathon-survey26}/challenge" ]; then
  echo "!! the official repository is missing: put it at lab/hackathon-survey26 (or set SURVEY26_REPO) - see README.md"; exit 1
fi
for card in a b c d; do
  if [ ! -d "cards/official/$card/public" ]; then
    echo "!! cards/official/$card is missing: unzip the official task cards there - see README.md"; exit 1
  fi
done

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "numpy==2.2.6"       # the version the platform build installs

echo "== python $(python3 -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')"
echo "== building the local cards from cards/SETS.json"
if [ "$#" -gt 0 ]; then
  python3 tools/make_cards.py --set "$@" || echo "!! some cards are missing or differ from the recorded ones: read the lines above"
else
  python3 tools/make_cards.py || echo "!! some cards are missing or differ from the recorded ones: read the lines above"
fi
echo "== geometry of the submitted agent against the official engine"
python3 tools/test_skymath.py | tail -1
echo "== sky tracker"
python3 tools/test_tracker.py > /dev/null && echo ok
if [ -d cards/dev/d101 ]; then
  echo "== 40 decisions of the submitted agent on card d101 (in-process; the score of such a short run means nothing)"
  OBSERVER_MODEL_DISABLED=1 python3 tools/inproc.py --card d101 --max-decisions 40 --quiet | cut -c1-120
fi
echo
echo "Setup done. Activate the environment with:  . .venv/bin/activate   - then see README.md"
