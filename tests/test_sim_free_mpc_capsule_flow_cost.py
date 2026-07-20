from __future__ import annotations

import math
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.costs_capsule_flow import (  # noqa: E402
    CapsuleFlowCostWeights,
    CapsuleFlowStateCost,
)
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig  # noqa: E402


def _zero_weights(**overrides) -> CapsuleFlowCostWeights:
    values = {field.name: 0.0 for field in fields(CapsuleFlowCostWeights)}
    values.update(overrides)
    return CapsuleFlowCostWeights(**values)


def _identity_quat() -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def _base_context(*, lid_open: bool = False, pod_grasped: bool = False) -> dict:
    return {
        "objects": {
            "capsule": {
                "pos": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                "quat": _identity_quat(),
            },
            "capsule_lid": {
                "pos": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
                "quat": _identity_quat(),
            },
            "can": {
                "pos": torch.tensor([0.4, 0.0, 0.2], dtype=torch.float32),
                "quat": _identity_quat(),
            },
        },
        "subtasks": {
            "open_coffee_lid": torch.tensor(lid_open),
            "grasp_pod": torch.tensor(pod_grasped),
        },
        "capsule_lid_joint_pos": torch.tensor(-0.1, dtype=torch.float32),
        "eef_pos": torch.tensor([0.0, -0.25, 1.0], dtype=torch.float32),
        "gripper_pos": torch.zeros(2, dtype=torch.float32),
        "joint_pos": torch.zeros(7, dtype=torch.float32),
    }


def _trajectory(batch: int = 1, horizon: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = torch.zeros((batch, horizon, 8), dtype=torch.float32)
    tcp_pos = torch.zeros((batch, horizon, 3), dtype=torch.float32)
    tcp_quat = torch.zeros((batch, horizon, 4), dtype=torch.float32)
    tcp_quat[..., 0] = 1.0
    return actions, tcp_pos, tcp_quat


@pytest.mark.parametrize(
    ("lid_open", "pod_grasped", "expected_stage"),
    [
        (False, False, "open_lid"),
        (True, False, "grasp_pod"),
        (True, True, "place_pod"),
    ],
)
def test_capsule_flow_switches_between_three_stages(lid_open, pod_grasped, expected_stage):
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(reach=1.0, open_reach=1.0, place_reach=1.0))
    actions, tcp_pos, tcp_quat = _trajectory()
    context = _base_context(lid_open=lid_open, pod_grasped=pod_grasped)

    cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost.shape == (1,)
    assert cost_fn.last_stage == expected_stage


def test_open_stage_requires_live_lid_pose_and_joint_angle():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(open_reach=1.0))
    actions, tcp_pos, tcp_quat = _trajectory()
    context = _base_context()
    del context["objects"]["capsule_lid"]
    del context["capsule_lid_joint_pos"]

    with pytest.raises(ValueError, match="capsule_lid_joint_pos"):
        cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)


def test_open_stage_reconstructs_lid_pose_from_capsule_root_and_joint():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(open_reach=1.0))
    actions, tcp_pos, tcp_quat = _trajectory()
    context = _base_context()
    del context["objects"]["capsule_lid"]
    context["capsule_lid_joint_pos"] = torch.tensor(0.0)
    context["eef_pos"] = torch.tensor([-0.0073589, -0.2551588, 0.3963653])
    tcp_pos[0, 0] = context["eef_pos"]

    cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost_fn.last_stage == "open_lid"
    assert torch.allclose(cost, torch.zeros_like(cost), atol=1e-6)


def test_open_pull_target_advances_upward_along_hinge_arc():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(open_reach=1.0))
    actions, tcp_pos, tcp_quat = _trajectory()
    context = _base_context()
    context["gripper_pos"] = torch.full((2,), 0.5, dtype=torch.float32)
    tcp_pos[0, 0] = context["eef_pos"]

    _, target, _ = cost_fn._open_terms(
        real_actions=actions,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        context=context,
    )

    assert target[2] > context["objects"]["capsule_lid"]["pos"][2]
    assert cost_fn.last_debug["lid_pull_mode"].item() == pytest.approx(1.0)


def test_open_stage_penalizes_tcp_above_lid_contact_trajectory():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(open_above_lid=1.0))
    actions, tcp_pos, tcp_quat = _trajectory(batch=2)
    context = _base_context()
    lid_contact_z = context["objects"]["capsule_lid"]["pos"][2]
    tcp_pos[0, 0] = torch.tensor([0.0, -0.25, lid_contact_z - 0.05])
    tcp_pos[1, 0] = torch.tensor([0.0, -0.25, lid_contact_z + 0.05])

    cost = cost_fn(
        real_actions=actions,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        context=context,
    )

    assert cost[0] == pytest.approx(0.0)
    assert cost[1] == pytest.approx(0.05**2, abs=1e-6)


def test_open_stage_keeps_gripper_closed_during_approach_and_pull():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(open_gripper=1.0))
    actions, tcp_pos, tcp_quat = _trajectory(batch=2, horizon=2)
    context = _base_context()
    actions[0, :, 7] = 1.0
    actions[1, :, 7] = 0.0

    approach_cost = cost_fn(
        real_actions=actions,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        context=context,
    )

    assert approach_cost[0] < approach_cost[1]
    assert cost_fn.last_debug["lid_approach_mode"][0].item() == pytest.approx(1.0)

    context["gripper_pos"] = torch.full((2,), 0.5, dtype=torch.float32)
    pull_cost = cost_fn(
        real_actions=actions,
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        context=context,
    )

    assert pull_cost[0] < pull_cost[1]
    assert cost_fn.last_debug["lid_pull_mode"][0].item() == pytest.approx(1.0)


def test_open_stage_releases_gripper_after_lid_reaches_threshold():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(open_gripper=1.0))
    actions, tcp_pos, tcp_quat = _trajectory(batch=2)
    context = _base_context()
    context["capsule_lid_joint_pos"] = torch.tensor(-0.5)
    actions[0, :, 7] = 0.0
    actions[1, :, 7] = 1.0
    tcp_pos[:] = context["eef_pos"]

    cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost[0] < cost[1]
    assert cost_fn.last_debug["lid_release_mode"][0].item() == pytest.approx(1.0)


def test_grasp_center_uses_scaled_pod_mesh_offset_and_orientation():
    cost_fn = CapsuleFlowStateCost("capsule")
    context = _base_context(lid_open=True)
    half = math.sqrt(0.5)
    context["objects"]["can"]["quat"] = torch.tensor([half, 0.0, 0.0, half])

    center = cost_fn._pod_grasp_center(context, torch.device("cpu"), torch.float32)

    assert center is not None
    expected = context["objects"]["can"]["pos"] + torch.tensor([0.0008, -0.0433, 0.0221])
    assert torch.allclose(center, expected, atol=1e-4)


def test_grasp_stage_closes_only_when_centered():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(grasp_gripper=1.0))
    actions, tcp_pos, tcp_quat = _trajectory(batch=2)
    context = _base_context(lid_open=True)
    center = cost_fn._pod_grasp_center(context, torch.device("cpu"), torch.float32)
    assert center is not None
    tcp_pos[:] = center
    actions[0, :, 7] = 1.0
    actions[1, :, 7] = 0.0

    cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost[0] < cost[1]
    assert "place_gripper" not in cost_fn.last_terms


def test_place_target_is_transformed_from_capsule_local_frame():
    cost_fn = CapsuleFlowStateCost("capsule")
    context = _base_context(lid_open=True, pod_grasped=True)
    half = math.sqrt(0.5)
    context["objects"]["capsule"]["pos"] = torch.tensor([0.5, -0.2, 0.1])
    context["objects"]["capsule"]["quat"] = torch.tensor([half, 0.0, 0.0, half])
    context["capsule_pod_place_local"] = (0.1, 0.0, 0.27)

    target = cost_fn._place_target(context, torch.device("cpu"), torch.float32)

    assert torch.allclose(target, torch.tensor([0.5, -0.1, 0.37]), atol=1e-5)


def test_place_release_requires_both_xy_and_z_alignment():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(place_gripper=1.0))
    context = _base_context(lid_open=True, pod_grasped=True)
    target = cost_fn._place_target(context, torch.device("cpu"), torch.float32)
    context["objects"]["can"]["pos"] = target.clone()
    context["eef_pos"] = torch.tensor([0.0, 0.0, 0.6])

    actions, tcp_pos, tcp_quat = _trajectory(batch=2)
    tcp_pos[:] = context["eef_pos"]
    actions[0, :, 7] = 0.0
    actions[1, :, 7] = 1.0
    near_cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    tcp_pos[:, :, 0] += 0.08
    far_cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert near_cost[0] < near_cost[1]
    assert far_cost[1] < far_cost[0]


def test_place_stage_penalizes_low_carry_height_until_xy_aligned():
    cost_fn = CapsuleFlowStateCost("capsule", _zero_weights(place_carry_height=1.0))
    context = _base_context(lid_open=True, pod_grasped=True)
    target = cost_fn._place_target(context, torch.device("cpu"), torch.float32)
    context["objects"]["can"]["pos"] = target.clone()
    context["eef_pos"] = torch.tensor([0.0, 0.0, 0.6])
    actions, tcp_pos, tcp_quat = _trajectory(batch=2)
    tcp_pos[0, 0] = context["eef_pos"] + torch.tensor([0.1, 0.0, 0.0])
    tcp_pos[1, 0] = context["eef_pos"] + torch.tensor([0.1, 0.0, 0.1])

    cost = cost_fn(real_actions=actions, tcp_pos=tcp_pos, tcp_quat=tcp_quat, context=context)

    assert cost[1] < cost[0]


def test_capsule_flow_is_registered_for_action_prox_mpc():
    planner = SimFreeMPC(
        object(),
        SimFreeMPCConfig(task_name="capsule", cost_style="capsule_flow", action_dims=8),
    )
    assert isinstance(planner.cost, CapsuleFlowStateCost)

    def fake_optimize(x_t, _policy_inputs, _context, *, alpha_bar):
        planner.cost.last_stage = "place_pod"
        planner.cost.last_terms = {"place_gripper": torch.tensor([0.1, 0.2])}
        result = SimpleNamespace(
            costs=torch.tensor([1.0, 2.0]),
            weights=torch.tensor([0.75, 0.25]),
            noise_scale=torch.tensor([0.1]),
        )
        return x_t.detach().clone(), result, 8, 0.1

    planner._optimize_action_prox_chunk = fake_optimize
    x_t = torch.zeros((1, 2, 8), dtype=torch.float32)

    next_x, diagnostics = planner.step_mbd_score_action_prox(
        x_t,
        {},
        {},
        iteration=1,
        num_iterations=3,
    )

    assert next_x.shape == x_t.shape
    assert diagnostics["update_mode"] == "mbd_score_action_prox"
    assert diagnostics["cost_style"] == "capsule_flow"
    assert diagnostics["cost_stage"] == "place_pod"
    assert diagnostics["term_place_gripper_weighted"] == pytest.approx(0.125)
