#!/usr/bin/env python3
"""Zip this project for upload as a complete project (observer.project.json at the ZIP root).

    python3 pack_agent.py --out ../observer-agent.zip

Leaves out local-only files: .env, .deps (rebuilt on the platform), caches, logs.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SKIP_DIRS = {".deps", "__pycache__", ".git", "run_output", ".venv"}
SKIP_FILES = {".env"}
SKIP_SUFFIXES = {".pyc", ".log", ".zip"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("../observer-agent.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    names = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (path.is_dir() or any(part in SKIP_DIRS for part in relative.parts) or path.name in SKIP_FILES
                    or path.suffix in SKIP_SUFFIXES or path.resolve() == out):
                continue
            archive.write(path, relative.as_posix())
            names.append(relative.as_posix())
    if "observer.project.json" not in names:
        raise SystemExit("observer.project.json is missing")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(names)} files)")
    for name in names:
        print("  " + name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
