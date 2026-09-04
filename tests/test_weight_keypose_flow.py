from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc import SimFreeMPC, SimFreeMPCConfig  # noqa: E402
from sim_free_mpc.costs_weight_keypose import WeightKeyposeStateCost  # noqa: E402
from sim_free_mpc.planner import _apply_action_keypose_l1_descent  # noqa: E402
from sim_free_mpc.weight_release import WeightReleaseDetector  # noqa: E402


def _weight_context(*, grasp_pear: bool = False) -> dict:
    return {
        "subtasks": {
            "grasp_pear": grasp_pear,
            "grasp_apple": False,
            "pear_on_scale": False,
        },
        "eef_pos": torch.tensor([0.45, 0.0, 0.30]),
        "objects": {
            "pear": {"pos": torch.tensor([0.45, 0.0, 0.24])},
            "apple": {"pos": torch.tensor([0.55, 0.1, 0.22])},
            "mango": {"pos": torch.tensor([0.70, 0.2, 0.20])},
            "cabbage": {"pos": torch.tensor([0.70, -0.2, 0.20])},
            "scale": {"pos": torch.tensor([0.30, -0.35, 0.15])},
        },
    }


def test_weight_keypose_cost_ignores_action_prefix() -> None:
    cost_fn = WeightKeyposeStateCost("weight")
    actions = torch.zeros((2, 16, 8), dtype=torch.float32)
    tcp = torch.zeros((2, 16, 3), dtype=torch.float32)
    actions[:, -1] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    tcp[:, -1] = torch.tensor([0.45, 0.0, 0.24])
    changed = actions.clone()
    changed[:, :-1] = 100.0
    changed_tcp = tcp.clone()
    changed_tcp[:, :-1] = -100.0

    first = cost_fn(
        real_actions=actions,
        tcp_pos=tcp,
        tcp_quat=None,
        context=_weight_context(),
    )
    second = cost_fn(
        real_actions=changed,
        tcp_pos=changed_tcp,
        tcp_quat=None,
        context=_weight_context(),
    )

    assert torch.equal(first, second)
    assert not {
        "smooth",
        "local",
        "gripper_smooth",
        "lift_reach",
        "lift_z",
        "place_carry_height",
        "transit",
        "path",
    }.intersection(cost_fn.last_terms)


def test_weight_keypose_place_has_no_lift_or_carry_height_terms() -> None:
    cost_fn = WeightKeyposeStateCost("weight")
    actions = torch.zeros((3, 16, 8), dtype=torch.float32)
    tcp = torch.full((3, 16, 3), 0.45, dtype=torch.float32)

    costs = cost_fn(
        real_actions=actions,
        tcp_pos=tcp,
        tcp_quat=None,
        context=_weight_context(grasp_pear=True),
    )

    assert costs.shape == (3,)
    assert cost_fn.last_stage == "place_pear_keypose"
    assert "place_terminal" in cost_fn.last_terms
    assert not any(
        name.startswith("lift") or name == "place_carry_height"
        for name in cost_fn.last_terms
    )


def test_weight_keypose_open_phase_unconditionally_prefers_open_gripper() -> None:
    cost_fn = WeightKeyposeStateCost("weight")
    context = _weight_context(grasp_pear=True)
    context["subtasks"]["open_gripper_pear"] = True
    tcp = torch.full((2, 16, 3), 0.45, dtype=torch.float32)
    actions = torch.zeros((2, 16, 8), dtype=torch.float32)
    actions[1, -1, 7] = 1.0

    costs = cost_fn(
        real_actions=actions,
        tcp_pos=tcp,
        tcp_quat=None,
        context=context,
    )

    assert cost_fn.last_stage == "open_gripper_pear_keypose"
    assert "open_gripper" in cost_fn.last_terms
    assert not any(name.startswith("place_") for name in cost_fn.last_terms)
    assert costs[0] < costs[1]
    assert cost_fn.last_terms["open_gripper"].tolist() == pytest.approx([0.0, 40.0])


def test_weight_release_detector_triggers_over_scale_and_latches_until_release() -> None:
    detector = WeightReleaseDetector(
        ee_speed_threshold=0.05,
        scale_xy_radius=0.12,
        min_height=0.0,
        control_frequency=15.0,
    )
    placing = {
        "grasp_pear": True,
        "grasp_apple": False,
        "pear_on_scale": False,
    }

    first, first_debug = detector.update(
        eef_pos=[0.50, 1.30, 0.40],
        scale_top_pos=[0.50, 1.30, 0.31],
        subtasks=placing,
    )
    assert "open_gripper_pear" not in first
    assert not first_debug["active"]

    triggered, trigger_debug = detector.update(
        eef_pos=[0.501, 1.30, 0.40],
        scale_top_pos=[0.50, 1.30, 0.31],
        subtasks=placing,
    )
    assert triggered["open_gripper_pear"]
    assert trigger_debug["triggered"]
    assert trigger_debug["eef_speed_mps"] == pytest.approx(0.015)

    latched, _ = detector.update(
        eef_pos=[0.55, 1.30, 0.40],
        scale_top_pos=[0.50, 1.30, 0.31],
        subtasks=placing,
    )
    assert latched["open_gripper_pear"]

    released, release_debug = detector.update(
        eef_pos=[0.55, 1.30, 0.40],
        scale_top_pos=[0.50, 1.30, 0.31],
        subtasks={**placing, "grasp_pear": False, "pear_on_scale": True},
    )
    assert "open_gripper_pear" not in released
    assert not release_debug["active"]



def test_action_keypose_l1_descent_is_capped_and_time_ramped() -> None:
    clean = torch.zeros((1, 4, 8), dtype=torch.float32)
    clean[:, 3, :] = 1.0

    result, maximum, mean = _apply_action_keypose_l1_descent(
        clean,
        keypose_index=3,
        active_dims=8,
        step_size=0.6,
        time_ramp=True,
        include_gripper=True,
    )

    assert torch.allclose(
        result[0, :3, 0],
        torch.tensor([0.2, 0.4, 0.6]),
    )
    assert torch.equal(result[0, 3], torch.ones(8))
    assert maximum.item() == pytest.approx(0.6)
    assert mean.item() == pytest.approx(0.4)


def test_action_keypose_l1_can_exclude_gripper() -> None:
    clean = torch.zeros((1, 3, 8), dtype=torch.float32)
    clean[:, 2, :] = 1.0

    result, _, _ = _apply_action_keypose_l1_descent(
        clean,
        keypose_index=2,
        active_dims=8,
        step_size=0.4,
        time_ramp=False,
        include_gripper=False,
    )

    assert torch.allclose(
        result[0, :2, :7],
        torch.full((2, 7), 0.4),
    )
    assert torch.equal(result[0, :2, 7], torch.zeros(2))


def test_flow_average_guides_keypose_then_extends_l1_to_prefix() -> None:
    planner = SimFreeMPC(
        object(),
        SimFreeMPCConfig(
            task_name="weight",
            cost_style="weight_keypose",
            num_samples=256,
            iterations=1,
            noise=0.5,
            temperature=0.05,
            keypose_index=3,
            keypose_steering_coeff=1.0,
            keypose_action_steering_coeff=1.0,
            keypose_action_l1_step=0.3,
        ),
    )

    def fake_cost(
        self,
        samples,
        _x_template,
        _active_dims,
        _policy_inputs,
        _context,
    ):
        return torch.sum((samples[:, -1, :] - 0.8) ** 2, dim=-1)

    planner._cost_active_samples = MethodType(fake_cost, planner)
    x_t = torch.zeros((1, 4, 8), dtype=torch.float32)
    policy_velocity = torch.zeros_like(x_t)

    guided_velocity, diagnostics = planner.flow_average_weight_keypose(
        x_t,
        policy_velocity,
        {},
        {},
        time_value=0.5,
    )

    guided_clean = x_t - 0.5 * guided_velocity
    assert guided_clean[:, 3, :].mean().item() > 0.2
    assert torch.any(guided_clean[:, :3, :].abs() > 0.0)
    assert torch.all(guided_clean[:, :3, :].abs() <= 0.3 + 1e-6)
    assert diagnostics["update_mode"] == "weight_keypose_flow_averaging"
    assert diagnostics["keypose_action_l1_max"] > 0.0


def test_flow_average_keeps_pure_noise_step_unsteered() -> None:
    planner = SimFreeMPC(
        object(),
        SimFreeMPCConfig(
            task_name="weight",
            cost_style="weight_keypose",
            num_samples=8,
            iterations=1,
            keypose_index=3,
        ),
    )

    def zero_cost(
        self,
        samples,
        _x_template,
        _active_dims,
        _policy_inputs,
        _context,
    ):
        return torch.zeros(samples.shape[0], dtype=samples.dtype)

    planner._cost_active_samples = MethodType(zero_cost, planner)
    x_t = torch.zeros((1, 4, 8), dtype=torch.float32)
    policy_velocity = torch.full_like(x_t, 0.25)

    guided_velocity, diagnostics = planner.flow_average_weight_keypose(
        x_t,
        policy_velocity,
        {},
        {},
        time_value=1.0,
    )

    assert torch.equal(guided_velocity, policy_velocity)
    assert diagnostics["keypose_action_l1_max"] == 0.0
