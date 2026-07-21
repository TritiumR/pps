from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

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
