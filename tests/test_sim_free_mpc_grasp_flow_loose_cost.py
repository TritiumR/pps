from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_grasp_flow_loose import (  # noqa: E402
    GraspFlowCostWeights,
    GraspFlowStateCost as LooseGraspFlowStateCost,
)
from sim_free_mpc.costs_explore import (  # noqa: E402
    _WEIGHT_OBJECT_HALF_HEIGHT,
)
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig  # noqa: E402


def test_planner_selects_loose_grasp_flow_cost() -> None:
    planner = SimFreeMPC(
        policy=None,
        config=SimFreeMPCConfig(
            task_name="Isaac-Weight-Droid-Visuomotor-v0",
            cost_style="grasp_flow_loose",
        ),
    )

    assert isinstance(planner.cost, LooseGraspFlowStateCost)
    assert planner.cost.weights.tip_z == 0.0
    assert planner.cost.weights.center_region == 120.0
    assert planner.cost.weights.aperture_region == 0.0
    assert planner.cost.weights.yaw == 0.0
    assert planner.cost.weights.close_gripper == 20.0
    assert not hasattr(planner.cost.weights, "recovery_open_gripper")


def test_held_pear_stays_in_place_stage_when_xy_on_scale() -> None:
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    context = {
        "subtasks": {
            "grasp_pear": torch.tensor(True),
            "pear_on_scale": torch.tensor(True),
            "grasp_apple": torch.tensor(False),
        }
    }

    assert cost._place_object_name(context) == "pear"
    assert cost._grasp_object_name(context) is None


def test_lift_origin_is_recorded_before_grasp_and_frozen() -> None:
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    device = torch.device("cpu")
    dtype = torch.float32

    before_grasp = {
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 0.23])}},
        "subtasks": {"grasp_pear": torch.tensor(False)},
    }
    cost._update_lift_start("pear", before_grasp, device, dtype)
    assert cost._lift_start_z["pear"] == pytest.approx(0.23)

    grasped = {
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 0.31])}},
        "subtasks": {"grasp_pear": torch.tensor(True)},
    }
    cost._update_lift_start("pear", grasped, device, dtype)
    assert "pear" in cost._lift_start_frozen
    assert cost._lift_start_z["pear"] == pytest.approx(0.23)

    lifted = {
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 0.65])}},
        "subtasks": {"grasp_pear": torch.tensor(True)},
    }
    cost._reset_lift_progress(None)
    cost._update_lift_start("pear", lifted, device, dtype)
    assert cost._lift_start_z["pear"] == pytest.approx(0.23)
    assert cost._lift_target_z_value("pear", lifted, lifted["objects"]["pear"]["pos"]) == pytest.approx(0.38)


def test_planner_episode_reset_clears_lift_origin() -> None:
    planner = SimFreeMPC(
        policy=None,
        config=SimFreeMPCConfig(
            task_name="Isaac-Weight-Droid-Visuomotor-v0",
            cost_style="grasp_flow_loose",
        ),
    )
    planner.cost._lift_start_z["pear"] = 0.23
    planner.cost._lift_start_frozen.add("pear")

    planner.reset_episode()

    assert planner.cost._lift_start_z == {}
    assert planner.cost._lift_start_frozen == set()


def _release_context(*, pear_on_scale: bool, pear_z_offset: float = 0.0):
    context = {
        "eef_pos": torch.tensor([0.0, 0.0, 0.0]),
        "objects": {
            "scale": {"pos": torch.tensor([0.0, 0.0, 0.0])},
            "pear": {"pos": torch.zeros(3)},
        },
        "subtasks": {
            "grasp_pear": torch.tensor(True),
            "pear_on_scale": torch.tensor(pear_on_scale),
            "grasp_apple": torch.tensor(False),
        },
    }
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    target = cost._place_target("pear", context, torch.device("cpu"), torch.float32)
    assert target is not None
    context["objects"]["pear"]["pos"] = target + torch.tensor([0.0, 0.0, pear_z_offset])
    return context


def _release_gripper_costs(context, *, predicted_xy_offset: float):
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[predicted_xy_offset, 0.0, 0.0]],
            [[predicted_xy_offset, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    terms, _ = cost._place_terms(
        object_name="pear",
        real_actions=real_actions,
        tcp_pos=tcp_pos,
        context=context,
    )
    return terms["place_gripper"], cost.last_debug


@pytest.mark.parametrize("predicted_xy_offset", [0.0, 0.25])
def test_release_uses_observed_readiness_not_candidate_prediction(
    predicted_xy_offset: float,
) -> None:
    gripper_cost, debug = _release_gripper_costs(
        _release_context(pear_on_scale=True),
        predicted_xy_offset=predicted_xy_offset,
    )

    assert debug["release_ready"][0] == pytest.approx(1.0)
    assert debug["desired_gripper_first"][0] == pytest.approx(0.0)
    assert gripper_cost[0] < gripper_cost[1]


@pytest.mark.parametrize(
    ("pear_on_scale", "pear_z_offset"),
    [(False, 0.0), (True, 0.05)],
)
def test_observed_release_requires_on_scale_and_release_height(
    pear_on_scale: bool,
    pear_z_offset: float,
) -> None:
    gripper_cost, debug = _release_gripper_costs(
        _release_context(pear_on_scale=pear_on_scale, pear_z_offset=pear_z_offset),
        predicted_xy_offset=0.0,
    )

    assert debug["release_ready"][0] == pytest.approx(0.0)
    assert gripper_cost[1] < gripper_cost[0]


def test_apple_release_uses_observed_geometry_without_on_scale_flag() -> None:
    context = {
        "eef_pos": torch.tensor([0.0, 0.0, 0.0]),
        "objects": {
            "scale": {"pos": torch.tensor([0.0, 0.0, 0.0])},
            "apple": {"pos": torch.zeros(3)},
        },
        "subtasks": {
            "grasp_pear": torch.tensor(False),
            "pear_on_scale": torch.tensor(True),
            "grasp_apple": torch.tensor(True),
        },
    }
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    target = cost._place_target("apple", context, torch.device("cpu"), torch.float32)
    assert target is not None
    context["objects"]["apple"]["pos"] = target

    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [[[0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    terms, _ = cost._place_terms(
        object_name="apple",
        real_actions=real_actions,
        tcp_pos=tcp_pos,
        context=context,
    )

    assert cost.last_debug["release_ready"][0] == pytest.approx(1.0)
    assert terms["place_gripper"][0] < terms["place_gripper"][1]


def _raw_grasp_subtasks(*, grasp_pear: bool) -> dict[str, bool]:
    return {
        "grasp_pear": grasp_pear,
        "pear_on_scale": False,
        "grasp_apple": False,
    }


def test_grasp_confirmation_rejects_single_step_contact_pulses_and_is_reversible() -> None:
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    context = {"subtasks": _raw_grasp_subtasks(grasp_pear=False)}

    cost.observe_subtasks(_raw_grasp_subtasks(grasp_pear=False))
    cost.observe_subtasks(_raw_grasp_subtasks(grasp_pear=False))
    assert cost._grasp_object_name(context) == "pear"
    assert cost._place_object_name(context) is None

    cost.observe_subtasks(_raw_grasp_subtasks(grasp_pear=True))
    assert cost._grasp_object_name(context) == "pear"
    assert cost._place_object_name(context) is None

    cost.observe_subtasks(_raw_grasp_subtasks(grasp_pear=True))
    assert cost._grasp_object_name(context) is None
    assert cost._place_object_name(context) == "pear"

    cost.observe_subtasks(_raw_grasp_subtasks(grasp_pear=False))
    assert cost._place_object_name(context) == "pear"

    cost.observe_subtasks(_raw_grasp_subtasks(grasp_pear=False))
    assert cost._grasp_object_name(context) == "pear"
    assert cost._place_object_name(context) is None


def test_grasp_has_no_recovery_gripper_gate() -> None:
    cost = LooseGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")
    context = {
        "eef_pos": torch.tensor([0.10, 0.0, 0.0]),
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 0.0])}},
        "subtasks": _raw_grasp_subtasks(grasp_pear=False),
    }
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.zeros((2, 1, 3), dtype=torch.float32)

    terms, _ = cost._grasp_terms(
        object_name="pear",
        real_actions=real_actions,
        tcp_pos=tcp_pos,
        tcp_quat=None,
        context=context,
    )

    assert "recovery_open_gripper" not in terms
    assert not any(key.startswith("recovery_") for key in cost.last_debug)


def test_straddle_uses_effective_radius_at_each_finger_height() -> None:
    cost = LooseGraspFlowStateCost(
        "Isaac-Weight-Droid-Visuomotor-v0",
        weights=GraspFlowCostWeights(straddle=30.0),
    )
    half_height = _WEIGHT_OBJECT_HALF_HEIGHT["pear"]
    context = {
        "eef_pos": torch.zeros(3),
        "objects": {"pear": {"pos": torch.zeros(3)}},
        "subtasks": _raw_grasp_subtasks(grasp_pear=False),
    }
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[0.0, 0.0, half_height]]],
        dtype=torch.float32,
    )
    tcp_quat = torch.zeros((2, 1, 4), dtype=torch.float32)
    tcp_quat[..., 0] = 1.0

    terms, _ = cost._grasp_terms(
        object_name="pear",
        real_actions=real_actions,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        context=context,
    )

    assert terms["straddle"][0] > 0.0
    assert terms["straddle"][1] == pytest.approx(0.0, abs=1e-7)
