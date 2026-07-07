from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.accel_planner import AccelActionMPC, AccelMPCConfig  # noqa: E402


def test_integrate_zero_accel_and_zero_velocity_keeps_q0():
    planner = AccelActionMPC(AccelMPCConfig(control_frequency=10.0, num_samples=2, iterations=1))
    q0 = torch.tensor([0.1, -0.2, 0.3, -1.0, 0.2, 1.0, -0.1])
    qd0 = torch.zeros(7)
    u = torch.zeros(3, 4, 7)

    q_traj = planner.integrate(u, q0, qd0)

    assert q_traj.shape == (3, 4, 7)
    assert torch.allclose(q_traj, q0.view(1, 1, 7).expand_as(q_traj))


def test_integrate_constant_accel_uses_semi_implicit_euler():
    planner = AccelActionMPC(
        AccelMPCConfig(
            control_frequency=2.0,
            num_samples=2,
            iterations=1,
            clamp_joint_limits=False,
        )
    )
    q0 = torch.zeros(7)
    qd0 = torch.zeros(7)
    u = torch.zeros(1, 3, 7)
    u[..., 0] = 2.0

    q_traj = planner.integrate(u, q0, qd0)

    expected_first_joint = torch.tensor([0.5, 1.5, 3.0])
    assert torch.allclose(q_traj[0, :, 0], expected_first_joint)
    assert torch.allclose(q_traj[0, :, 1:], torch.zeros(3, 6))


def test_plan_returns_real_action_chunk_without_policy_decode():
    planner = AccelActionMPC(
        AccelMPCConfig(
            task_name="custom",
            num_samples=4,
            iterations=1,
            noise=0.0,
            temperature=1.0,
            control_frequency=15.0,
            cost_style="ref_style",
        )
    )
    context = {
        "joint_pos": torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        "joint_vel": torch.zeros(7),
        "eef_pos": torch.zeros(3),
    }
    gripper = torch.full((5, 1), 0.5)

    actions, diagnostics = planner.plan(context=context, gripper_traj=gripper)

    assert actions.shape == (5, 8)
    assert torch.allclose(actions[:, :7], context["joint_pos"].view(1, 7).expand(5, 7))
    assert torch.allclose(actions[:, 7], torch.full((5,), 0.5))
    assert diagnostics["update_mode"] == "accel_mppi"
    assert diagnostics["optimize_space"] == "accel_action"


def test_hybrid_samples_use_arm_accel_and_direct_gripper_position():
    planner = AccelActionMPC(AccelMPCConfig(control_frequency=10.0, num_samples=2, iterations=1))
    q0 = torch.tensor([0.1, -0.2, 0.3, -1.0, 0.2, 1.0, -0.1])
    qd0 = torch.zeros(7)
    samples = torch.zeros(2, 3, 8)
    samples[0, :, 7] = -0.5
    samples[1, :, 7] = 1.5

    q_traj, gripper = planner._hybrid_samples_to_actions(samples, q0, qd0)

    assert torch.allclose(q_traj, q0.view(1, 1, 7).expand_as(q_traj))
    assert torch.allclose(gripper[0], torch.zeros(3, 1))
    assert torch.allclose(gripper[1], torch.ones(3, 1))


def test_plan_mbd_score_returns_real_action_chunk_without_policy_decode():
    planner = AccelActionMPC(
        AccelMPCConfig(
            task_name="custom",
            num_samples=4,
            iterations=1,
            noise=0.0,
            temperature=1.0,
            control_frequency=15.0,
            cost_style="ref_style",
        )
    )
    context = {
        "joint_pos": torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        "joint_vel": torch.zeros(7),
        "eef_pos": torch.zeros(3),
    }
    gripper = torch.full((5, 1), 0.5)

    actions, diagnostics = planner.plan_mbd_score(
        context=context,
        gripper_traj=gripper,
        num_iterations=2,
        score_scale=1.0,
    )

    assert actions.shape == (5, 8)
    assert torch.allclose(actions[:, :7], context["joint_pos"].view(1, 7).expand(5, 7))
    assert torch.allclose(actions[:, 7], torch.full((5,), 0.5))
    assert diagnostics["update_mode"] == "accel_mbd_score"
    assert diagnostics["optimize_space"] == "accel_action"
    assert diagnostics["score_norm"] == 0.0
