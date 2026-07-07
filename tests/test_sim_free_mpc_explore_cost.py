from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_explore import ExploreCostWeights, ExploreStateCost  # noqa: E402


def _base_context():
    return {
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)}},
        "subtasks": {"grasp_pear": torch.tensor(False)},
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "a_local": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        "tcp_to_tip_z": 0.03,
        "gripper_center_region_radius": 0.01,
    }


def test_explore_cost_penalizes_object_outside_gripper_center_region():
    weights = ExploreCostWeights(
        reach=0.0,
        terminal=0.0,
        orient=0.0,
        floor=0.0,
        smooth=0.0,
        local=0.0,
        yaw=0.0,
        straddle=0.0,
        tip_z=0.0,
        center_region=10.0,
        aperture_region=0.0,
        close_gripper=0.0,
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = ExploreStateCost("weight", weights)
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.03]], [[-0.05, 0.0, 1.03]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_base_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_explore_cost_penalizes_hovering_above_object_center_region():
    weights = ExploreCostWeights(
        reach=0.0,
        terminal=0.0,
        orient=0.0,
        floor=0.0,
        smooth=0.0,
        local=0.0,
        yaw=0.0,
        straddle=0.0,
        tip_z=0.0,
        center_region=10.0,
        aperture_region=0.0,
        close_gripper=0.0,
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = ExploreStateCost("weight", weights)
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.03]], [[0.0, 0.0, 1.08]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_base_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_explore_pick_reach_tracks_tip_not_tcp_above_target():
    weights = ExploreCostWeights(
        reach=1.0,
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
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = ExploreStateCost("weight", weights)
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.03]], [[0.0, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_base_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_explore_cost_encourages_closing_when_object_is_centered():
    weights = ExploreCostWeights(
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
        close_gripper=2.0,
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = ExploreStateCost("weight", weights)
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.03]], [[0.0, 0.0, 1.03]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_base_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_explore_cost_can_disable_gripper_close_term_for_arm_only_planners():
    weights = ExploreCostWeights(
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
        close_gripper=2.0,
        clear=0.0,
        transit=0.0,
        gripper=0.0,
        path=0.0,
    )
    cost_fn = ExploreStateCost("weight", weights)
    real_actions = torch.zeros((1, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.03]]], dtype=torch.float32)
    tcp_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    context = {**_base_context(), "optimize_gripper": False}

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert torch.allclose(cost[0], torch.tensor(0.0))
