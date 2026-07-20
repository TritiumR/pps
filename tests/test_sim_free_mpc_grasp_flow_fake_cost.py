from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_grasp_flow_fake import GraspFlowStateCost as FakeGraspFlowStateCost  # noqa: E402
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig  # noqa: E402


def test_fake_grasp_flow_swaps_pick_targets() -> None:
    cost = FakeGraspFlowStateCost("Isaac-Weight-Droid-Visuomotor-v0")

    assert cost._grasp_object_name({"pear_on_scale": False, "grasp_pear": False}) == "apple"
    assert cost._grasp_object_name({"pear_on_scale": True, "grasp_apple": False}) == "pear"


def test_planner_selects_fake_grasp_flow_cost() -> None:
    planner = SimFreeMPC(
        policy=None,
        config=SimFreeMPCConfig(
            task_name="Isaac-Weight-Droid-Visuomotor-v0",
            cost_style="grasp_flow_fake",
        ),
    )

    assert isinstance(planner.cost, FakeGraspFlowStateCost)
