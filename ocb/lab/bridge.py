"""Locate and import the lab kernel (`rlab`, stdlib-only) without installing it."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_RLAB = Path.home() / ".dev" / "research-lab" / ".claude" / "worktrees" / "rlab"
DEFAULT_PROJECTS = Path.home() / ".dev" / "rlab-projects"


def rlab_path() -> Path:
    return Path(os.environ.get("OCB_RLAB_PATH", DEFAULT_RLAB)).expanduser()


STAMP = ".ocb-commit"  # written by snapshot_lab; a frozen copy must not be asked git about itself


def rlab_version() -> dict:
    """Which lab the bench measured: commit, dirty flag and path.

    A snapshot carries its commit in a stamp file. Asking git inside a copied worktree is wrong:
    `cp -R` copies `.git` as a pointer to the live repository, so git would report whatever the live
    lab has moved on to rather than what actually ran."""
    p = rlab_path()
    out = {"path": str(p), "commit": None, "dirty": None, "pinned": False}
    stamp = p / STAMP
    if stamp.is_file():
        text = stamp.read_text().strip().split()
        out["commit"] = text[0] if text else None
        out["dirty"] = "dirty" in text
        out["pinned"] = True
        return out
    for args, key in ((["rev-parse", "--short", "HEAD"], "commit"), (["status", "--porcelain"], "dirty")):
        try:
            r = subprocess.run(["git", "-C", str(p), *args], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            out[key] = r.stdout.strip() if key == "commit" else bool(r.stdout.strip())
    return out


def snapshot_lab(dest: Path, source: Path | None = None) -> dict:
    """Freeze the lab for measurement: copy it, stamp its commit, drop the worktree pointer.

    `dest` must sit outside ~/.claude (the lab's run sandbox denies reading .claude/jobs, so a
    snapshot there cannot execute and looks exactly like a broken lab). The caller should confirm
    the snapshot passes the lab's own test suite before measuring with it."""
    source = Path(source or DEFAULT_RLAB).expanduser()
    dest = Path(dest).expanduser()
    if ".claude" in dest.parts:
        raise ValueError(f"{dest} is under .claude; the lab's sandbox denies reading it")
    version = {}
    for args, key in ((["rev-parse", "--short", "HEAD"], "commit"), (["status", "--porcelain"], "dirty")):
        r = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, timeout=30)
        version[key] = (r.stdout.strip() if key == "commit" else bool(r.stdout.strip())) if r.returncode == 0 else None
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", "*.pyc"))
    (dest / STAMP).write_text(f"{version['commit']}{' dirty' if version['dirty'] else ''}\n")
    return {"path": str(dest), **version, "pinned": True}


def load_rlab():
    """Import and return the rlab package from the lab checkout."""
    p = rlab_path()
    if not (p / "rlab" / "kernel.py").is_file():
        raise FileNotFoundError(f"rlab not found at {p}; set OCB_RLAB_PATH")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import rlab  # noqa: F401
    return rlab
