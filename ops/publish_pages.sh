#!/usr/bin/env bash
# Publish only the committed static showcase. No force push or build dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo "Usage: bash ops/publish_pages.sh [--dry-run]" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain -- site)" ]]; then
  echo "Commit changes under site/ before publishing; the deployed page comes from HEAD." >&2
  exit 1
fi
for file in index.html style.css app.js favicon.svg .nojekyll; do
  git cat-file -e "HEAD:site/$file"
done
site_revision=$(git subtree split --prefix=site HEAD)
if [[ "${1:-}" == "--dry-run" ]]; then
  git push --dry-run origin "$site_revision:refs/heads/gh-pages"
else
  git push origin "$site_revision:refs/heads/gh-pages"
fi
