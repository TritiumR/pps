"""Provide the task hooks used for offline MPC score labeling."""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent


def _load_registry():
    """Load the task registry in package or path-based execution."""
    try:
        from . import registry
    except ImportError:
        from mujoco_eval.tasks import registry
    return registry


registry = _load_registry()

_FK_FIT_CANDIDATES = (
    _HERE / "fk_fit_stack_d0.json",
    _HERE.parent / "bench/fk_fits/fk_fit_stack_d0.json",
)

# Which task's ladder the hooks serve. generate-cache loads this module by PATH, so it cannot
# pass --task through the import; the env var is the one channel both sides already share.
# Defaults to stack, which is what the hooks hard-coded before.
MG_TASK = os.environ.get("MG_OFFLINE_TASK", "stack")
_TASK_EXPLICIT = "MG_OFFLINE_TASK" in os.environ


def configure_task(name):
    """Select the task whose ladder these hooks serve, explicitly.

    C5. MG_TASK was resolved from the environment at IMPORT time and defaulted to "stack", and
    nothing in the tree ever set MG_OFFLINE_TASK -- so `generate-cache --task square` labelled
    square frames with the STACK ladder unless the variable happened to be exported by hand.
    Callers must now say which task they mean.
    """
    global MG_TASK, _TASK_EXPLICIT
    if name not in registry.MG_TASKS:
        raise SystemExit(f"[offline-labels] unknown task {name!r}; "
                         f"known: {sorted(registry.MG_TASKS)}")
    MG_TASK, _TASK_EXPLICIT = name, True
    print(f"[offline-labels] task={name}", flush=True)


def require_explicit_task():
    """Fail loudly when a non-stack cache is about to be labelled with the stack default."""
    if not _TASK_EXPLICIT:
        raise SystemExit(
            "[offline-labels] task was never selected, so the hooks would silently use the "
            "'stack' ladder. Call offline_labels.configure_task(<task>) (or export "
            "MG_OFFLINE_TASK) before generating labels.")


def _task():
    """Return the registry entry for the task being labelled."""
    try:
        return registry.MG_TASKS[MG_TASK]
    except KeyError:
        raise SystemExit(
            f"[offline-labels] MG_OFFLINE_TASK={MG_TASK!r} is not in the registry "
            f"(have {sorted(registry.MG_TASKS)})") from None

_UNREPRESENTABLE_TERMS = {
    "rekep_subgoal": "no VLM constraint offline",
    "rekep_path": "no VLM path functions offline",
    "consistency": "no previous-chunk plan_ref offline",
    "release_retreat": "no sensed 'released' offline",
    "place_approach_rate": "needs the stall-release bridge's 'overshoot_on'",
}
_UNREPRESENTABLE_GEOM = {
    "release_on_stall": "reads bridge 'seat_contact'",
    "release_on_subgoal": "reads the VLM constraint",
}


def check_objects_available(demo):
    """Fail if the ladder reads an object the offline context cannot supply.

    check_representable covers cost TERMS; it did not cover the object poses those terms read.
    Square's peg is static so the converter never wrote it, and the ladder died mid-run on a
    KeyError instead of being refused up front.
    """
    have = set(_canonical_objects(
        {n: None for n in demo.get("states/rigid_object", {})})) | set(_static_objects())
    try:
        from ..grounding.gt import TASKS
    except ImportError:
        from mujoco_eval.grounding.gt import TASKS

    spec = TASKS.get(MG_TASK, {})
    need = set(spec.get("grasp_objs", ())) | ({spec["place_obj"]} if spec.get("place_obj") else set())
    if missing := sorted(need - have):
        raise ValueError(
            f"[offline-labels] {MG_TASK}: the ladder reads {missing} but the offline context has "
            f"{sorted(have)}. Add them to _static_objects (if fixed) or reconvert (if they move).")


def check_representable(cost_cfg):
    bad = [
        f"{n} ({why})"
        for n, why in _UNREPRESENTABLE_TERMS.items()
        if float(cost_cfg["cost"]["terms"].get(n, 0.0)) != 0.0
    ]
    geom = cost_cfg["cost"].get("geometry", {})
    bad += [
        f"geometry.{k} ({why})"
        for k, why in _UNREPRESENTABLE_GEOM.items()
        if float(geom.get(k, 0.0) or 0.0) != 0.0
    ]
    if bad:
        raise ValueError(
            "MG offline context cannot represent: "
            + "; ".join(bad)
            + ". Labels would come from a different active cost than evaluation."
        )


def _fk_fit_path():
    """Return the fit the EVAL harness uses for this task.

    Not the fit named after the task: eval resolves square through the stack fit, and labels
    scored under a different FK than evaluation would describe a different robot.
    """
    try:                                                 # single source of truth; this module is
        from ..eval import _FK_FIT_NAME                  # loaded by path, so both forms are needed
    except ImportError:
        from mujoco_eval.eval import _FK_FIT_NAME

    name = _FK_FIT_NAME.get(MG_TASK, "fk_fit_stack_d0.json")
    for base in (_HERE, _HERE.parent / "bench/fk_fits"):
        if (path := base / name).exists():
            return path
    return next(p for p in _FK_FIT_CANDIDATES if p.exists())


def prepare_planner(planner):
    """Configure the planner with the fitted PandaGripper TCP frame."""
    from sim_free_mpc.fk import PandaFK

    fit_path = _fk_fit_path()
    with open(fit_path) as fh:
        fit = json.load(fh)
    q_off = fit["orientation"][fit["stored_quat_convention"]]["R_off_quat_wxyz"]
    planner.fk = PandaFK(
        ee_offset=tuple(fit["tcp_offset_link8"]),
        ee_offset_quat_wxyz=tuple(q_off),
    )


def attach_priority_cost(planner, cost_cfg):
    """Attach the evaluation priority cost to the planner."""
    from vlm_dp.cost import guard_cost
    from vlm_dp.cost.base_cost import CompositeCost

    if planner.config.cost_style != "priority":
        raise ValueError(
            f"MG labelling needs cost_style='priority', "
            f"got {planner.config.cost_style!r}"
        )
    check_representable(cost_cfg)
    planner.cost = guard_cost(
        CompositeCost(
            cost_cfg["cost"]["terms"],
            cost_cfg["cost"]["geometry"],
        )
    )


def _canonical_objects(objects):
    """Rename converted body names to the names the ladder reads.

    The converter records MuJoCo body names (SquareNut, coffee_machine_body); gt.py's ladder uses
    the eval aliases (nut, coffee_machine). _BODY_ALIASES already maps one to the other, so invert
    it rather than keep a second table that can drift.
    """
    try:
        from ..env.mujoco_env import _BODY_ALIASES
    except ImportError:
        from mujoco_eval.env.mujoco_env import _BODY_ALIASES

    inverse = {}
    for canonical, bodies in _BODY_ALIASES.items():
        for body in ((bodies,) if isinstance(bodies, str) else bodies):
            # Converted names drop the _main/_root suffix the env appends.
            inverse.setdefault(body, canonical)
            inverse.setdefault(body.rsplit("_", 1)[0], canonical)
    return {inverse.get(name, name): value for name, value in objects.items()}


def _static_objects():
    """Static scene objects the converter never wrote, from gt.py's own constants.

    The peg, the can seat and the kitchen fixtures never move, so gt.py hands EVALUATION a
    constant rather than a sim body -- and labels must read that same constant. Converting a body
    pose for them would create a second source of truth that could silently disagree.
    """
    try:
        from ..grounding import gt
    except ImportError:
        from mujoco_eval.grounding import gt

    statics = {
        "square": {"peg1": gt.PEG_POS},
        "can": {"bin2_q3": gt.CAN_SEAT},
    }.get(MG_TASK, {})
    return {name: {"pos": np.asarray(pos, dtype=np.float32)} for name, pos in statics.items()}


def episode_signals(demo):
    """Extract the episode signals for the task being labelled."""
    return _task().episode_signals(demo)


def frame_context(context, demo, step, sig, *, heuristic=True):
    """Build the cost context for one training frame of the task being labelled."""
    del heuristic
    base = dict(context)
    objects = _canonical_objects(context.get("objects", {}))
    objects.update(_static_objects())
    base["objects"] = {
        n: np.asarray(o["pos"], dtype=np.float32)
        for n, o in objects.items()
    }
    eef = np.asarray(demo["obs/eef_pos"])
    base["eef_step_motion"] = (
        float(np.linalg.norm(eef[step] - eef[step - 1]))
        if step > 0
        else 0.0
    )
    return _task().frame_context(base, sig, step)