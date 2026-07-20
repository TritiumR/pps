"""Config-driven composite cost for the VLM-DP base.

CompositeCost(terms, geom, extents) sums weight * cost_terms.TERMS[name](inputs) for each
{name: weight} in terms (the config's cost.terms); geom is the gripper/collision geometry
(cost.geometry). It exposes the __call__(*, real_actions, ee_pos, ee_quat, context) signature the
sim_free_mpc planner expects. Change the cost mix purely in config; implement new terms in cost_terms.
"""
from __future__ import annotations

import types

import torch

from vlm_base.cost_terms import TERMS, CostInputs

# Fallbacks for callers that build the cost without a config (e.g. diagnostics); the runner uses the YAML.
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
    # Grasp geometry (from grasp_flow); these self-gate to grasp stages.
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
    """Weighted sum of config-selected cost terms; the interface sim_free_mpc's planner calls."""

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
        for fn, w in self.term_fns:
            cost = cost + w * fn(inputs)
        return cost
