from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_grasp_flow import GraspFlowCostWeights, GraspFlowStateCost  # noqa: E402


def _pick_context():
    return {
        "objects": {"pear": {"pos": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)}},
        "subtasks": {"grasp_pear": torch.tensor(False)},
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "a_local": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
        "tcp_to_tip_z": 0.0,
        "gripper_center_region_radius": 0.01,
    }


def _weights(**overrides):
    values = {
        "reach": 0.0,
        "terminal": 0.0,
        "orient": 0.0,
        "floor": 0.0,
        "smooth": 0.0,
        "local": 0.0,
        "yaw": 0.0,
        "straddle": 0.0,
        "tip_z": 0.0,
        "center_region": 0.0,
        "aperture_region": 0.0,
        "close_gripper": 0.0,
        "close_gripper_first": 0.0,
        "gripper_smooth": 0.0,
        "soft_grasp": 0.0,
        "lift_reach": 0.0,
        "lift_terminal": 0.0,
        "lift_xy": 0.0,
        "lift_z": 0.0,
        "lift_gripper": 0.0,
        "place_reach": 0.0,
        "place_terminal": 0.0,
        "place_xy": 0.0,
        "place_z": 0.0,
        "place_carry_height": 0.0,
        "place_gripper": 0.0,
        "place_obstacle": 0.0,
        "clear": 0.0,
        "transit": 0.0,
        "path": 0.0,
    }
    values.update(overrides)
    return GraspFlowCostWeights(**values)


def test_grasp_flow_pick_reach_tracks_tcp_by_default():
    cost_fn = GraspFlowStateCost("weight", _weights(reach=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.00]], [[0.0, 0.0, 1.03]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_grasp_flow_penalizes_object_outside_gripper_center_region():
    cost_fn = GraspFlowStateCost("weight", _weights(center_region=10.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.00]], [[-0.05, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_grasp_flow_encourages_closing_only_when_centered():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper=2.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.00]], [[0.0, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_grasp_flow_encourages_closing_inside_center_region():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper=2.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor([[[0.008, 0.0, 1.00]], [[0.008, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_grasp_flow_encourages_immediate_closing_inside_center_region():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper_first=2.0))
    real_actions = torch.ones((2, 2, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor(
        [
            [[0.0, 0.0, 1.00], [0.0, 0.0, 1.00]],
            [[0.0, 0.0, 1.00], [0.0, 0.0, 1.00]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]
    assert cost_fn.last_debug["grasp_close_gate_first"][0] > torch.tensor(0.99)


def test_grasp_flow_first_close_term_penalizes_early_closing_before_centered():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper_first=2.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor([[[-0.08, 0.0, 1.00]], [[-0.08, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert cost[0] > cost[1]
    assert cost_fn.last_debug["grasp_close_gate_first"][0] < torch.tensor(0.01)
    assert cost_fn.last_debug["grasp_first_close_command"][0] < torch.tensor(0.01)


def test_grasp_flow_first_close_requires_stable_predicted_geometry():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper_first=2.0))
    real_actions = torch.zeros((2, 3, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[0.0, 0.0, 1.00], [-0.08, 0.0, 1.00], [0.0, 0.0, 1.00]],
            [[0.0, 0.0, 1.00], [-0.08, 0.0, 1.00], [0.0, 0.0, 1.00]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]] * 3,
            [[1.0, 0.0, 0.0, 0.0]] * 3,
        ],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert cost_fn.last_debug["grasp_close_gate_first"][0] > torch.tensor(0.99)
    assert cost_fn.last_debug["grasp_stable_close_gate"][0] < torch.tensor(0.01)
    assert cost_fn.last_debug["grasp_first_close_command"][0] < torch.tensor(0.01)
    assert cost[0] > cost[1]


def test_grasp_flow_apple_close_gate_uses_center_radius_floor():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper=2.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor([[[0.019, 0.0, 1.00]], [[0.019, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = {
        "objects": {"apple": {"pos": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)}},
        "subtasks": {
            "grasp_pear": torch.tensor(False),
            "pear_on_scale": torch.tensor(True),
            "grasp_apple": torch.tensor(False),
        },
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "a_local": torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
    }

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost[0] < cost[1]


def test_grasp_flow_encourages_opening_when_not_centered():
    cost_fn = GraspFlowStateCost("weight", _weights(close_gripper=2.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor([[[-0.08, 0.0, 1.00]], [[-0.08, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert cost[0] < cost[1]


def test_grasp_flow_grasp_phase_has_no_place_terms():
    cost_fn = GraspFlowStateCost("weight", _weights(reach=1.0, place_reach=1.0, place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.00]], [[0.0, 0.0, 1.03]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert not any(name.startswith("place_") for name in cost_fn.last_terms)
    assert not any(name.startswith("release_") for name in cost_fn.last_debug)
    assert "reach" in cost_fn.last_terms


def _place_context(object_name="pear", *, lifted=True, object_z=None, scale_quat=None):
    if object_z is None:
        object_z = 1.21 if lifted else 1.0
    scale = {"pos": torch.tensor([0.5, 0.0, 1.0], dtype=torch.float32)}
    if scale_quat is not None:
        scale["quat"] = torch.tensor(scale_quat, dtype=torch.float32)
    return {
        "objects": {
            object_name: {"pos": torch.tensor([0.0, 0.0, object_z], dtype=torch.float32)},
            "scale": scale,
        },
        "subtasks": {
            "grasp_pear": torch.tensor(object_name == "pear"),
            "pear_on_scale": torch.tensor(object_name == "apple"),
            "grasp_apple": torch.tensor(object_name == "apple"),
        },
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "eef_pos": torch.tensor([0.0, 0.0, object_z], dtype=torch.float32),
        f"{object_name}_lift_start_z": 1.0,
    }


def _place_target_z(object_name="pear"):
    half_height = 0.0620635 if object_name == "pear" else 0.037665
    return 1.0 + 0.0272255 + 0.0523800 + half_height + 0.01


def _place_target_x():
    return 0.5 - 0.0470425


def _place_target_y():
    return 0.0


def test_grasp_flow_tail_stage_prioritizes_lift_before_place_gate_opens():
    cost_fn = GraspFlowStateCost("weight", _weights(lift_z=1.0, lift_gripper=1.0, place_reach=10.0, place_xy=10.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.20]], [[0.0, 0.0, 1.00]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(
        real_actions=real_actions,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        context=_place_context(lifted=False),
    )

    assert cost[0] < cost[1]
    assert "lift_z" in cost_fn.last_terms
    assert "place_reach" in cost_fn.last_terms
    assert "place_xy" in cost_fn.last_terms
    assert cost_fn.last_terms["place_xy"].max() < torch.tensor(1e-3)
    assert cost_fn.last_stage == "place_pear"


def test_grasp_flow_tail_stage_smoothly_blends_place_xy_during_lift():
    cost_fn = GraspFlowStateCost("weight", _weights(place_xy=1.0))
    real_actions = torch.zeros((1, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    tcp_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context(object_z=1.0))
    low = cost_fn.last_terms["place_xy"].clone()

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context(object_z=1.10))
    mid = cost_fn.last_terms["place_xy"].clone()

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context(object_z=1.20))
    high = cost_fn.last_terms["place_xy"].clone()

    assert low < torch.tensor(1e-6)
    assert mid > low + torch.tensor(1e-3)
    assert high > mid + torch.tensor(1e-3)


def test_grasp_flow_default_place_target_is_centered_on_scale_y():
    cost_fn = GraspFlowStateCost("weight", _weights(place_xy=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z()]],
            [[_place_target_x(), _place_target_y() - 0.05, _place_target_z()]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < torch.tensor(1e-4)
    assert cost[1] > cost[0]


def test_grasp_flow_place_target_uses_scale_orientation():
    cost_fn = GraspFlowStateCost("weight", _weights())
    device = torch.device("cpu")
    dtype = torch.float32
    target = cost_fn._place_target(
        "pear",
        _place_context(scale_quat=[0.70710677, 0.0, 0.0, 0.70710677]),
        device,
        dtype,
    )

    assert target is not None
    assert torch.allclose(target[:2], torch.tensor([0.5, -0.0470425]), atol=1e-5)


def test_grasp_flow_tail_stage_keeps_gripper_closed_before_lift_ready():
    cost_fn = GraspFlowStateCost("weight", _weights(lift_gripper=1.0, place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor(
        [[[_place_target_x(), _place_target_y(), _place_target_z()]], [[_place_target_x(), _place_target_y(), _place_target_z()]]],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context(lifted=False))

    assert cost[0] < cost[1]
    assert "place_gripper" in cost_fn.last_terms
    assert "lift_gripper" not in cost_fn.last_terms


def test_grasp_flow_place_phase_keeps_gripper_closed_until_near_scale():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor(
        [[[0.0, 0.0, _place_target_z()]], [[0.0, 0.0, _place_target_z()]]],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < torch.tensor(1e-4)
    assert cost[1] > cost[0]


def test_grasp_flow_place_release_requires_near_z():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z() + 0.08]],
            [[_place_target_x(), _place_target_y(), _place_target_z() + 0.08]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < torch.tensor(1e-2)
    assert cost[1] > cost[0]


def test_grasp_flow_place_keeps_gripper_closed_above_scale_surface():
    cost_fn = GraspFlowStateCost("weight", _weights(lift_gripper=1.0, place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z() + 0.08]],
            [[_place_target_x(), _place_target_y(), _place_target_z() + 0.08]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < cost[1]


def test_grasp_flow_place_keeps_gripper_closed_on_partial_release_gate():
    cost_fn = GraspFlowStateCost("weight", _weights(lift_gripper=1.0, place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 1.0
    real_actions[1, 0, 7] = 0.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x() + 0.035, _place_target_y(), _place_target_z()]],
            [[_place_target_x() + 0.035, _place_target_y(), _place_target_z()]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < cost[1]


def test_grasp_flow_place_opens_when_near_scale_xy_and_z():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [[[_place_target_x(), _place_target_y(), _place_target_z()]], [[_place_target_x(), _place_target_y(), _place_target_z()]]],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < cost[1]
    assert cost[1] > cost[0]


def test_grasp_flow_place_opens_with_small_release_error():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x() + 0.01, _place_target_y(), _place_target_z() + 0.01]],
            [[_place_target_x() + 0.01, _place_target_y(), _place_target_z() + 0.01]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert cost[0] < cost[1]


def test_grasp_flow_apple_release_z_uses_object_radius_floor():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z("apple") + 0.03]],
            [[_place_target_x(), _place_target_y(), _place_target_z("apple") + 0.03]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context("apple"))

    assert cost[0] < cost[1]
    assert cost_fn.last_debug["release_z_radius"][0] > torch.tensor(0.08)
    assert cost_fn.last_debug["release_command_final"][0] > torch.tensor(0.0)


def test_grasp_flow_place_records_release_debug_terms():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z()]],
            [[_place_target_x() + 0.1, _place_target_y(), _place_target_z() + 0.08]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context())

    assert set(cost_fn.last_debug) >= {
        "release_xy_to_place_first",
        "release_xy_to_place_final",
        "release_place_z_err_abs_first",
        "release_place_z_err_abs_final",
        "release_xy_gate_first",
        "release_xy_gate_final",
        "release_z_gate_first",
        "release_z_gate_final",
        "release_gate_first",
        "release_gate_final",
        "release_command_first",
        "release_command_final",
        "desired_gripper_first",
        "desired_gripper_final",
    }
    assert cost_fn.last_debug["release_command_final"][0] > cost_fn.last_debug["release_command_final"][1]


def test_grasp_flow_releases_pear_observed_on_scale_even_when_tcp_moved_away():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[0.0, -0.35, 1.15]],
            [[0.0, -0.35, 1.15]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = _place_context()
    context["objects"]["pear"]["pos"] = torch.tensor(
        [_place_target_x(), _place_target_y(), _place_target_z()],
        dtype=torch.float32,
    )
    context["subtasks"]["pear_on_scale"] = torch.tensor(True)

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost_fn.last_stage == "place_pear"
    assert cost_fn.last_debug["release_observed_placed_gate"][0] > torch.tensor(0.99)
    assert cost_fn.last_debug["release_command_first"][0] > torch.tensor(0.99)
    assert cost[0] < cost[1]


def test_grasp_flow_stops_chasing_pear_once_observed_on_scale():
    cost_fn = GraspFlowStateCost(
        "weight",
        _weights(lift_reach=1.0, lift_xy=1.0, lift_z=1.0, place_reach=1.0, place_xy=1.0, place_z=1.0),
    )
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [
            [[0.0, -0.35, 1.15]],
            [[0.0, -0.35, 1.15]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = _place_context()
    context["objects"]["pear"]["pos"] = torch.tensor(
        [_place_target_x(), _place_target_y(), _place_target_z()],
        dtype=torch.float32,
    )
    context["subtasks"]["pear_on_scale"] = torch.tensor(True)

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-6)


def test_grasp_flow_apple_release_xy_uses_object_radius_floor():
    cost_fn = GraspFlowStateCost("weight", _weights(place_gripper=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    real_actions[0, 0, 7] = 0.0
    real_actions[1, 0, 7] = 1.0
    tcp_pos = torch.tensor(
        [
            [[_place_target_x() + 0.019, _place_target_y(), _place_target_z("apple")]],
            [[_place_target_x() + 0.019, _place_target_y(), _place_target_z("apple")]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context("apple"))

    assert cost[0] < cost[1]


def test_grasp_flow_place_phase_supports_apple():
    cost_fn = GraspFlowStateCost("weight", _weights(place_reach=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z("apple")]],
            [[_place_target_x() + 0.1, _place_target_y(), _place_target_z("apple")]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context("apple"))

    assert cost[0] < torch.tensor(1e-4)
    assert cost[1] > cost[0]


def test_grasp_flow_place_stage_runs_after_lift_done():
    cost_fn = GraspFlowStateCost("weight", _weights(lift_z=1.0, place_z=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z() + 0.08]],
            [[_place_target_x(), _place_target_y(), _place_target_z()]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_place_context(lifted=True))

    assert cost_fn.last_stage == "place_pear"
    assert "place_z" in cost_fn.last_terms
    assert "lift_z" in cost_fn.last_terms
    assert cost_fn.last_terms["place_z"][1] < cost_fn.last_terms["place_z"][0]


def test_grasp_flow_place_does_not_penalize_object_already_on_scale():
    cost_fn = GraspFlowStateCost("weight", _weights(place_obstacle=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    target_z = _place_target_z("apple")
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), target_z]],
            [[_place_target_x() + 0.16, _place_target_y(), target_z]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = _place_context("apple")
    context["objects"]["pear"] = {
        "pos": torch.tensor([_place_target_x(), _place_target_y(), _place_target_z()], dtype=torch.float32)
    }

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert "place_obstacle" not in cost_fn.last_terms
    assert torch.allclose(cost, torch.zeros_like(cost))


def test_grasp_flow_grasp_phase_switches_to_apple_after_pear_on_scale():
    cost_fn = GraspFlowStateCost("weight", _weights(reach=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.3, 0.0, 1.0]], [[0.0, 0.0, 1.0]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = {
        "objects": {
            "pear": {"pos": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)},
            "apple": {"pos": torch.tensor([0.3, 0.0, 1.0], dtype=torch.float32)},
        },
        "subtasks": {
            "grasp_pear": torch.tensor(False),
            "pear_on_scale": torch.tensor(True),
            "grasp_apple": torch.tensor(False),
        },
        "joint_pos": torch.zeros(7, dtype=torch.float32),
        "tcp_to_tip_z": 0.0,
    }

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert torch.allclose(cost[0], torch.tensor(0.0))
    assert cost[1] > cost[0]


def test_grasp_flow_keeps_placing_apple_after_grasp_flicker():
    cost_fn = GraspFlowStateCost("weight", _weights(place_reach=1.0, reach=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z("apple")]],
            [[0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = _place_context("apple")
    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)
    context["subtasks"]["grasp_apple"] = torch.tensor(False)

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost_fn.last_stage == "place_apple"
    assert "place_reach" in cost_fn.last_terms
    assert "reach" not in cost_fn.last_terms
    assert cost[0] < cost[1]


def test_grasp_flow_keeps_placing_pear_until_grasp_released():
    cost_fn = GraspFlowStateCost("weight", _weights(place_reach=1.0, reach=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor(
        [
            [[_place_target_x(), _place_target_y(), _place_target_z()]],
            [[0.3, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    context = _place_context()
    context["subtasks"]["pear_on_scale"] = torch.tensor(True)

    cost = cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost_fn.last_stage == "place_pear"
    assert "place_reach" in cost_fn.last_terms
    assert "reach" not in cost_fn.last_terms
    assert cost[0] < cost[1]


def test_grasp_flow_stores_weighted_term_vectors_for_debug():
    cost_fn = GraspFlowStateCost("weight", _weights(reach=1.0, close_gripper=2.0, soft_grasp=1.0))
    real_actions = torch.zeros((2, 1, 8), dtype=torch.float32)
    tcp_pos = torch.tensor([[[0.0, 0.0, 1.00]], [[0.0, 0.0, 1.03]]], dtype=torch.float32)
    tcp_quat = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    cost_fn(real_actions=real_actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=_pick_context())

    assert set(cost_fn.last_terms) >= {"reach", "close_gripper"}
    assert cost_fn.last_terms["reach"].shape == (2,)
