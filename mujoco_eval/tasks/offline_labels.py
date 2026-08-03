"""Provide the task hooks used for offline MPC score labeling."""

from __future__ import annotations

import json
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


def prepare_planner(planner):
    """Configure the planner with the fitted PandaGripper TCP frame."""
    from sim_free_mpc.fk import PandaFK

    fit_path = next(p for p in _FK_FIT_CANDIDATES if p.exists())
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


def episode_signals(demo):
    """Extract the stack episode signals."""
    return registry.stack_episode_signals(demo)


def frame_context(context, demo, step, sig, *, heuristic=True):
    """Build the stack cost context for one training frame."""
    del heuristic
    base = dict(context)
    base["objects"] = {
        n: np.asarray(o["pos"], dtype=np.float32)
        for n, o in context.get("objects", {}).items()
    }
    eef = np.asarray(demo["obs/eef_pos"])
    base["eef_step_motion"] = (
        float(np.linalg.norm(eef[step] - eef[step - 1]))
        if step > 0
        else 0.0
    )
    return registry._stack_frame_context(base, sig, step)