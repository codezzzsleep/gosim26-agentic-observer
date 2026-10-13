#!/usr/bin/env python3
"""Copy an agent folder to build/NAME, pack it as the platform expects and check the archive.

    python3 tools/package.py                         # the submitted agent (../agent) -> build/agent and build/agent.zip
    python3 tools/package.py --agent ../agent --out build/try1

The archive is read back with the platform's own package code (size limits, manifest) and scanned for things
that must not be uploaded (.env files, keys, card or run data).
"""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import argparse, re, shutil, sys, zipfile
from pathlib import Path

SKIP_DIRS = {".deps", "__pycache__", ".git", "run_output", ".venv"}
SKIP_SUFFIXES = {".pyc", ".log", ".zip"}
SECRET = re.compile(r"(s26_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,})")


def copy_agent(agent: Path, out: Path) -> int:
    if not (agent / "agent.py").exists() or not (agent / "observer.project.json").exists():
        raise SystemExit(f"{agent} is not an agent folder (agent.py and observer.project.json are needed)")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(agent, out, ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc", ".env"))
    return sum(1 for path in out.rglob("*") if path.is_file())


def pack(folder: Path, archive: Path) -> None:
    names = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            if path.is_dir() or any(part in SKIP_DIRS for part in rel.parts) or path.name == ".env" or path.suffix in SKIP_SUFFIXES:
                continue
            z.write(path, rel.as_posix())
            names.append(rel.as_posix())
    check(archive)


def check(archive: Path) -> None:
    """Read the archive the way the platform does and look for things that must not be in it."""
    sys.path.insert(0, str(REPO))
    from project_platform.manifest import ProjectManifest
    from project_platform.package import read_project_zip
    data = archive.read_bytes()
    read_project_zip(data)                               # raises on size, count or path problems
    with zipfile.ZipFile(archive) as z:
        listing = z.namelist()
        manifest = ProjectManifest.parse(z.read("observer.project.json"))
        problems = []
        for name in listing:
            base = name.rsplit("/", 1)[-1]
            if base == ".env" or base.endswith((".pem", ".key")):
                problems.append(f"{name}: secrets file")
            if name.startswith(("cards/", "truth/", "runs/")) or "/truth/" in name:
                problems.append(f"{name}: card or run data")
            try:
                text = z.read(name).decode("utf-8", "ignore")
            except Exception:
                continue
            if SECRET.search(text):
                problems.append(f"{name}: looks like it contains a key or token")
            if "AO_FIXED_EFFORT" in text and name == "observer.project.json":
                problems.append("observer.project.json sets AO_FIXED_EFFORT (experiments only)")
    if problems:
        raise SystemExit("NOT uploadable:\n  " + "\n  ".join(problems))
    print(f"{archive}: {len(data) / 1024:.0f} KiB, {len(listing)} files, manifest ok "
          f"(image {getattr(manifest, 'image', '?')}), nothing secret found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="../agent", help="agent folder (default: the submitted agent)")
    ap.add_argument("--out", default="build/agent", help="build folder; the archive is written next to it as .zip")
    ap.add_argument("--check", metavar="ZIP", help="only check an existing archive")
    a = ap.parse_args()
    if a.check:
        check(Path(a.check))
        return 0
    agent = (WORK / a.agent).resolve() if not Path(a.agent).is_absolute() else Path(a.agent)
    out = (WORK / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{out}: {copy_agent(agent, out)} files from {agent}")
    pack(out, out.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
