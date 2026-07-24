"""CompositeCost sums weight * TERMS[name](inputs) over a config's cost.terms.

cost.geometry supplies gripper and collision geometry. Exposes the call signature the sim_free_mpc
planner expects.
"""
from __future__ import annotations

import types

import torch

from vlm_dp.cost.terms import TERMS, CostInputs

# Fallbacks for callers that build the cost without a config, such as diagnostics. The runner uses YAML.
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
    # Grasp geometry, self-gating to grasp stages.
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
    """Weighted sum of config-selected cost terms, the interface sim_free_mpc's planner calls."""

    def __init__(self, terms=None, geom=None, extents=None):
        weights = DEFAULT_TERMS if terms is None else terms
        self.term_fns = [(TERMS[name], float(w)) for name, w in weights.items()]
        self.geom = types.SimpleNamespace(**(DEFAULT_GEOM if geom is None else geom))
        self.extents = extents or {}

    def target(self, context, device, dtype):
        return torch.as_tensor(context["target"], device=device, dtype=dtype)

    def __call__(self, *, real_actions, ee_pos, ee_quat=None, context):
        objects = context.get("objects", {})
        extents = {n: o["extents"] for n, o in objects.items() if "extents" in o} or self.extents
        inputs = CostInputs(real_actions, ee_pos, ee_quat, context, extents, self.geom)
        cost = ee_pos.new_zeros(ee_pos.shape[0])
        terms = {}
        for fn, w in self.term_fns:
            v = w * fn(inputs)
            terms[fn.__name__] = v.detach()
            cost = cost + v
        # Planner diagnostics contract, read as planner._last_cost_term_diagnostics.
        self.last_terms = terms
        self.last_stage = ("place" if context.get("place_target") else
                           "carry" if context.get("payload") else
                           f"grasp {context.get('grasp_obj')}")
        return cost
