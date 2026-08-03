"""Define repository, data, configuration, and result paths."""

from __future__ import annotations

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


def results_dir(task, exp):
    d = RESULTS / task / exp
    d.mkdir(parents=True, exist_ok=True)
    return d