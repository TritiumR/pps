"""Define repository, data, configuration, and result paths."""

from __future__ import annotations

import datetime
import os
import pathlib
import sys

MODULE = pathlib.Path(__file__).resolve().parent
REPO = MODULE.parent

CONFIGS = MODULE / "configs"
FK_FITS = MODULE / "bench" / "fk_fits"


def _root(env_var, default):
    return pathlib.Path(os.environ.get(env_var, str(default))).expanduser()


DATA = _root("MUJOCO_EVAL_DATA", REPO / "data")
RESULTS = _root(
    "MUJOCO_EVAL_RESULTS",
    REPO / "results" / "mujoco_eval",
)


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

    cand = CONFIGS / (
        name if name.endswith(".yaml") else f"{name}.yaml"
    )
    if not cand.exists():
        raise FileNotFoundError(f"no config {name!r} in {CONFIGS}")

    return cand


def fk_fit(name):
    """Resolve a stored FK fit by path or name."""
    p = pathlib.Path(name)
    return p if p.suffix and p.exists() else FK_FITS / name


def task_data(task, dataset="demo_224.hdf5"):
    return DATA / f"{task}_d0" / dataset


def run_day():
    """MMDD stamp that groups one run's outputs.

    The launcher exports MUJOCO_EVAL_DAY so every worker it spawns lands in the same day
    directory even when the run crosses midnight.
    """
    return os.environ.get("MUJOCO_EVAL_DAY") or datetime.datetime.now().strftime("%m%d")


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
    """Drop _pending once every seed has been filed; keep it if anything is stranded there."""
    try:
        (out_dir / "_pending").rmdir()
    except OSError:
        pass


# Runs written before the outcome-sorted layout are flat: <run>/<seed>.jsonl. Archived runs under
# old/ keep that shape permanently, so every reader has to handle both.
def find_run(task, exp):
    """Locate a run by name: newest dated directory, then the legacy flat and archived paths."""
    root = RESULTS / task
    days = sorted((d for d in root.glob("[0-9][0-9][0-9][0-9]") if (d / exp).is_dir()),
                  reverse=True)
    for candidate in [d / exp for d in days] + [root / exp, root / "old" / exp]:
        if candidate.is_dir():
            return candidate
    return None


def seed_artifact(run_dir, seed, kind, suffix):
    """Resolve one seed's file under either layout; returns None when it does not exist."""
    for outcome in ("success", "failure"):
        p = run_dir / outcome / kind / f"{seed}_{outcome}.{suffix}"
        if p.exists():
            return p
    flat = run_dir / f"{seed}.{suffix}"
    return flat if flat.exists() else None


def iter_traces(run_dir):
    """Every episode trace in a run, whichever layout it was written under."""
    found = sorted(run_dir.glob("*/trace/*.jsonl"))
    return found or sorted(run_dir.glob("*.jsonl"))