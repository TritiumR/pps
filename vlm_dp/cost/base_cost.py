"""Composite cost used by the sim_free_mpc planner."""

from __future__ import annotations

import types

import torch

from vlm_dp.cost.terms import CostInputs, TERMS

# Roles let selection gate on feasibility, rank on task progress, and use
# execution priors only for shaping.
TERM_ROLES = {
    # Task progress
    "reach": "task",
    "terminal_reach": "task",
    "rekep_subgoal": "task",
    "rekep_keypose": "task",
    "rekep_path": "task",
    "straddle": "task",
    "tip_z": "task",
    "yaw": "task",
    "grasp_axis": "task",
    "center_region": "task",
    "aperture_region": "task",
    "grasp_region": "task",
    "close_gripper": "task",
    "grasp_commit": "task",
    "release_gripper": "task",
    "carry_hold": "task",
    "regrasp_penalty": "task",
    "hook_pull": "task",
    "press_axis": "task",
    "lift_xy": "task",
    "lift_z": "task",
    "lift_terminal": "task",
    "lift_reach": "task",
    "place_reach": "task",
    "place_xy": "task",
    "place_z": "task",
    "place_carry_height": "task",
    "place_descent": "task",
    "place_terminal": "task",
    "place_approach_above": "task",
    "place_setdown": "task",
    "insert_funnel": "task",
    "not_hold": "task",
    # Physical feasibility
    "collision": "feasibility",
    "clear": "feasibility",
    "carry_clear": "feasibility",
    "floor": "feasibility",
    "grasp_approach_corridor": "feasibility",
    "release_retreat": "feasibility",
    "place_retreat": "feasibility",
    "release_rise_first": "feasibility",
    # Execution and search priors
    "smooth": "prior",
    "gripper_smooth": "prior",
    "joint_delta": "prior",
    "orientation": "prior",
    "consistency": "prior",
    "grasp_descend_rate": "prior",
    "grasp_standoff": "prior",
    "place_approach_rate": "prior",
    "carry_liftoff": "prior",
    "carry_altitude": "prior",
    "descend_gate": "prior",
}

ROLES = ("feasibility", "task", "prior")


def check_roles(term_names):
    """Require every configured term to have an explicit role."""
    missing = sorted(name for name in term_names if name not in TERM_ROLES)
    if missing:
        raise ValueError(
            f"cost terms have no TERM_ROLES entry: {missing}. Classify each as "
            f"one of {ROLES} -- 'feasibility' only if a violation is unsafe or "
            "'prior' if it only shapes sampler behavior."
        )


# Fallbacks for callers that construct the cost without a configuration.
DEFAULT_TERMS = {
    "reach": 25.0,
    "terminal_reach": 40.0,
    "rekep_subgoal": 25.0,
    "rekep_path": 200.0,
    "smooth": 0.08,
    "joint_delta": 0.005,
    "orientation": 3.0,
    "consistency": 30.0,
    "straddle": 30.0,
    "collision": 20.0,
    "floor": 40.0,
    "tip_z": 80.0,
    "yaw": 5.0,
    "center_region": 120.0,
    "aperture_region": 80.0,
    "close_gripper": 2.0,
    "carry_hold": 40.0,
    "place_descent": 50.0,
}

DEFAULT_GEOM = {
    "ee_r": 0.035,
    "coll_clear": 0.02,
    "finger_r": 0.012,
    "open_half": 0.04,
    "tool_back": 0.06,
    "tcp_to_tip": 0.0,
    "center_scale": 0.12,
    "aperture_margin": 0.006,
    "close_z_scale": 0.025,
    "release_xy": 0.06,
    "release_z": 0.035,
    "place_descend_radius": 0.18,
    "place_seat_radius": 0.10,
    "place_carry_clear": 0.10,
}


class CompositeCost:
    """Compute the weighted sum of configured cost terms."""

    def __init__(self, terms=None, geom=None, extents=None):
        weights = DEFAULT_TERMS if terms is None else terms
        check_roles(weights)

        self.term_fns = [
            (TERMS[name], float(weight))
            for name, weight in weights.items()
        ]
        self.geom = types.SimpleNamespace(
            **(DEFAULT_GEOM if geom is None else geom)
        )
        self.extents = extents or {}

    def target(self, context, device, dtype):
        """Return the current target as a tensor."""
        return torch.as_tensor(
            context["target"],
            device=device,
            dtype=dtype,
        )

    def __call__(self, *, real_actions, ee_pos, ee_quat=None, context):
        objects = context.get("objects", {})
        extents = {
            name: obj["extents"]
            for name, obj in objects.items()
            if "extents" in obj
        } or self.extents

        inputs = CostInputs(
            real_actions,
            ee_pos,
            ee_quat,
            context,
            extents,
            self.geom,
        )
        cost = ee_pos.new_zeros(ee_pos.shape[0])
        by_role = {
            role: ee_pos.new_zeros(ee_pos.shape[0])
            for role in ROLES
        }
        terms = {}

        for fn, weight in self.term_fns:
            value = weight * fn(inputs)
            terms[fn.__name__] = value.detach()
            cost = cost + value
            by_role[TERM_ROLES.get(fn.__name__, "prior")] += value

        self.last_terms = terms
        self.last_cost_feasibility = by_role["feasibility"].detach()
        self.last_cost_task = by_role["task"].detach()
        self.last_cost_prior = by_role["prior"].detach()

        # Stage names use underscores for per-stage steering overrides.
        self.last_stage = (
            f"place_{context.get('place_target')}"
            if context.get("place_target")
            else (
                f"lift_{context.get('payload')}"
                if context.get("payload")
                else f"grasp_{context.get('grasp_obj')}"
            )
        )

        return cost