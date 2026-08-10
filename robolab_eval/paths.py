"""Define repository, data, configuration, and result paths for robolab_eval.

Mirrors mujoco_eval/paths.py so both harnesses write the same run layout
(<results>/<task>/<MMDD>/<exp>/{success,failure}/{trace,videos}/), and readers written for one
work on the other. The roots differ: the repo mount is read-only inside the container, so data
and results default to the writable pps-mg mount.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import sys

MODULE = pathlib.Path(__file__).resolve().parent
REPO = MODULE.parent

CONFIGS = MODULE / "configs"

# The repo worktree is mounted read-only in the container, so nothing may be written under REPO.
# ROBOLAB_EVAL_{DATA,RESULTS} point at the read-write pps-mg mount; the defaults below are what
# the container sees.
DATA = pathlib.Path(os.environ.get("ROBOLAB_EVAL_DATA", "/workspace/pps-mg/data")).expanduser()
RESULTS = pathlib.Path(
    os.environ.get("ROBOLAB_EVAL_RESULTS", "/workspace/pps-mg/results/robolab")).expanduser()


def ensure_repo_on_path():
    """Add the repository and module directories to sys.path."""
    for p in (str(REPO), str(MODULE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def config(name):
    """Resolve a configuration by path or bare name."""
    p = pathlib.Path(name)
    if p.suffix and p.exists():
        return p
    cand = CONFIGS / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not cand.exists():
        raise FileNotFoundError(f"no config {name!r} in {CONFIGS}")
    return cand


def task_data(task, leaf=None):
    """Per-task artifact directory (rekep_context.json, fk_fit.json), or a file inside it."""
    d = DATA / f"robolab_{task}"
    return d if leaf is None else d / leaf


def run_day():
    """MMDD stamp that groups one run's outputs."""
    return os.environ.get("ROBOLAB_EVAL_DAY") or datetime.datetime.now().strftime("%m%d")


def results_dir(task, exp, day=None):
    d = RESULTS / task / (day or run_day()) / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def episode_path(out_dir, seed, success, kind, suffix):
    """<run>/<success|failure>/<kind>/<seed>_<outcome>.<suffix>."""
    outcome = "success" if success else "failure"
    d = out_dir / outcome / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{seed}_{outcome}.{suffix}"


def pending_path(out_dir, seed, suffix):
    """Where a per-seed file is written before its outcome is known."""
    d = out_dir / "_pending"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{seed}.{suffix}"


def prune_pending(out_dir):
    """Drop _pending once every seed has been filed."""
    try:
        (out_dir / "_pending").rmdir()
    except OSError:
        pass
