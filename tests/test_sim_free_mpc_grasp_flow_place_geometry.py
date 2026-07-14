from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sim_free_mpc.costs_explore import (  # noqa: E402
    _WEIGHT_OBJECT_HALF_HEIGHT,
    _WEIGHT_OBJECT_HORIZONTAL_RADIUS,
)
from sim_free_mpc.costs_grasp_flow import (  # noqa: E402
    _GRASP_FLOW_PLACE_SLOT_GAP,
    _GRASP_FLOW_PLACE_Y_OFFSET,
    GraspFlowCostWeights,
    GraspFlowStateCost,
    _default_place_x_offset,
)


def _context(object_name: str) -> dict:
    return {
        "objects": {
            object_name: {"pos": torch.tensor([0.0, 0.0, 1.0])},
            "scale": {"pos": torch.tensor([0.5, 0.0, 0.8])},
        },
        "eef_pos": torch.tensor([0.0, 0.0, 1.0]),
    }


def test_default_place_slots_clear_both_object_radii():
    separation = _default_place_x_offset("apple") - _default_place_x_offset("pear")
    required = (
        _WEIGHT_OBJECT_HORIZONTAL_RADIUS["pear"]
        + _WEIGHT_OBJECT_HORIZONTAL_RADIUS["apple"]
        + _GRASP_FLOW_PLACE_SLOT_GAP
    )

    assert separation == pytest.approx(required)


def test_place_target_uses_distinct_object_slots_and_heights():
    cost_fn = GraspFlowStateCost("weight")
    pear_target = cost_fn._place_target("pear", _context("pear"), torch.device("cpu"), torch.float32)
    apple_target = cost_fn._place_target("apple", _context("apple"), torch.device("cpu"), torch.float32)

    assert pear_target is not None
    assert apple_target is not None
    assert pear_target[0] < apple_target[0]
    assert pear_target[1] == pytest.approx(_GRASP_FLOW_PLACE_Y_OFFSET)
    assert apple_target[1] == pytest.approx(_GRASP_FLOW_PLACE_Y_OFFSET)
    assert pear_target[2] - apple_target[2] == pytest.approx(
        _WEIGHT_OBJECT_HALF_HEIGHT["pear"] - _WEIGHT_OBJECT_HALF_HEIGHT["apple"]
    )


def test_place_release_tolerance_scales_with_object_geometry():
    weights = GraspFlowCostWeights(
        reach=0.0,
        terminal=0.0,
        orient=0.0,
        floor=0.0,
        smooth=0.0,
        local=0.0,
        yaw=0.0,
        straddle=0.0,
        tip_z=0.0,
        center_region=0.0,
        aperture_region=0.0,
        close_gripper=0.0,
        gripper_smooth=0.0,
        soft_grasp=0.0,
        lift_reach=0.0,
        lift_terminal=0.0,
        lift_xy=0.0,
        lift_z=0.0,
        lift_gripper=0.0,
        place_reach=0.0,
        place_terminal=0.0,
        place_xy=0.0,
        place_z=0.0,
        place_carry_height=0.0,
        place_gripper=1.0,
        clear=0.0,
        transit=0.0,
        path=0.0,
    )
    cost_fn = GraspFlowStateCost("weight", weights)

    def opening_cost(object_name: str, xy_error: float, z_error: float) -> torch.Tensor:
        context = _context(object_name)
        target = cost_fn._place_target(object_name, context, torch.device("cpu"), torch.float32)
        context["eef_pos"] = context["objects"][object_name]["pos"].clone()
        tcp_pos = (target + torch.tensor([xy_error, 0.0, z_error])).view(1, 1, 3)
        actions = torch.zeros((1, 1, 8), dtype=torch.float32)
        terms, _ = cost_fn._place_terms(
            object_name=object_name,
            real_actions=actions,
            tcp_pos=tcp_pos,
            context=context,
        )
        return terms["place_gripper"]

    assert opening_cost("pear", 0.065, 0.045) < torch.tensor(1e-6)
    assert opening_cost("apple", 0.059, 0.034) < torch.tensor(1e-6)
