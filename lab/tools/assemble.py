#!/usr/bin/env python3
"""Put a submission together: a core (the rule-based agent) plus, optionally, the model stages; pack and check it.

    python3 tools/assemble.py --core baselines/v1 --llm --out build/v1_llm        # exact v1 core + model stages
    python3 tools/assemble.py --core agent --llm --out build/next_llm             # working core + model stages
    python3 tools/assemble.py --core agent --out build/next_rules                 # rules only
    python3 tools/assemble.py --core agent --llm --out build/next_llm --zip       # also writes build/next_llm.zip

The core folders (agent/, baselines/v1/) and llm_stages/ are the sources; build/ is always regenerated, so
never edit files under build/. With --zip the archive is read back with the platform's own package code
(size limits, manifest) and scanned for things that must not be uploaded.
"""
import os as _os
from pathlib import Path as _Path
WORK = _Path(_os.environ.get("AO_WORK", _Path(__file__).resolve().parents[1]))        # the lab folder
REPO = _Path(_os.environ.get("SURVEY26_REPO", WORK / "hackathon-survey26"))           # the official repository

import argparse, re, shutil, sys, zipfile
from pathlib import Path

CORE_FILES = ("agent.py", "planner.py", "tracker.py", "hazards.py", "state.py", "skymath.py", "pointing.py",
              "observer.project.json", "requirements.txt", "pack_agent.py")
LLM_FILES = ("llm.py", "advisor.py", "supervisor.py", ".env.example")
SKIP_DIRS = {".deps", "__pycache__", ".git", "run_output", ".venv"}
SKIP_SUFFIXES = {".pyc", ".log", ".zip"}
SECRET = re.compile(r"(s26_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{10,})")
LICENSE = """MIT License

Copyright (c) 2026 the authors of this project

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""


def assemble(core: Path, out: Path, llm: bool) -> list:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = []
    for name in CORE_FILES:
        if (core / name).exists():
            shutil.copyfile(core / name, out / name)
            names.append(name)
    for needed in ("agent.py", "planner.py", "observer.project.json", "requirements.txt"):
        if needed not in names:
            raise SystemExit(f"{core} has no {needed}")
    if llm:
        for name in LLM_FILES:
            shutil.copyfile(WORK / "llm_stages" / name, out / name)
            names.append(name)
    readme = (WORK / "llm_stages" / "README.md") if llm else (core / "README.md")
    if readme.exists():
        shutil.copyfile(readme, out / "README.md")
        names.append("README.md")
    (out / "LICENSE").write_text(LICENSE)
    names.append("LICENSE")
    return names


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
    ap.add_argument("--core", help="folder of the rule-based core, e.g. agent or baselines/v1")
    ap.add_argument("--llm", action="store_true", help="add the model stages from llm_stages/")
    ap.add_argument("--out"); ap.add_argument("--zip", action="store_true")
    ap.add_argument("--check", metavar="ZIP", help="only check an existing archive")
    a = ap.parse_args()
    if a.check:
        check(Path(a.check))
        return 0
    if not a.core or not a.out:
        raise SystemExit("--core and --out are required (or --check ZIP)")
    core = (WORK / a.core) if not Path(a.core).is_absolute() else Path(a.core)
    out = (WORK / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    names = assemble(core, out, a.llm)
    print(f"{out}: {len(names)} files from {core.name}" + (" + llm_stages" if a.llm else ""))
    if a.zip:
        pack(out, out.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
