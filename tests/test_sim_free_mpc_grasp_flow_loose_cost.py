from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_grasp_flow_loose import (  # noqa: E402
    GraspFlowStateCost as LooseGraspFlowStateCost,
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
    assert planner.cost.weights.center_region == 0.0
    assert planner.cost.weights.aperture_region == 0.0
    assert planner.cost.weights.yaw == 0.0
    assert planner.cost.weights.close_gripper == 0.0


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
