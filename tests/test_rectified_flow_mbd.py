from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PPS_ROOT = Path(__file__).resolve().parents[1]
if str(PPS_ROOT) not in sys.path:
    sys.path.insert(0, str(PPS_ROOT))


from sim_free_mpc.rectified_flow_mbd import (
    bounded_gaussian_log_prob,
    FlowBlendCoefficients,
    RectifiedFlowMBD,
    RectifiedFlowMBDConfig,
    TokenBlockLayout,
    blend_policy_mbd_flows,
    flow_matching_clean_proposal_scale,
    memoryless_sde_kl_blocks,
    recenter_particle_noise,
    rectified_flow_from_score,
    rectified_flow_score,
)


def test_blockwise_flow_blend_and_kl_are_disjoint() -> None:
    layout = TokenBlockLayout(trajectory_start=1, keypose_index=3)
    policy = torch.zeros((2, 4, 2))
    mbd = torch.full_like(policy, 2.0)

    guided = blend_policy_mbd_flows(
        policy,
        mbd,
        layout=layout,
        coefficients=FlowBlendCoefficients(
            action=0.25,
            trajectory=0.5,
            keypose=0.75,
        ),
    )

    torch.testing.assert_close(guided[:, :1], torch.full((2, 1, 2), 0.5))
    torch.testing.assert_close(guided[:, 1:3], torch.full((2, 2, 2), 1.0))
    torch.testing.assert_close(guided[:, 3:], torch.full((2, 1, 2), 1.5))
    kl = memoryless_sde_kl_blocks(
        guided,
        dt=-0.1,
        diffusion=1.0,
        layout=layout,
    )
    torch.testing.assert_close(
        kl["total"], kl["action"] + kl["trajectory"] + kl["keypose"]
    )


def test_rectified_flow_score_conversion_round_trips() -> None:
    generator = torch.Generator().manual_seed(7)
    x_t = torch.randn((3, 4, 2), generator=generator)
    velocity = torch.randn((3, 4, 2), generator=generator)

    score = rectified_flow_score(x_t, velocity, time_value=0.4)
    recovered = rectified_flow_from_score(x_t, score, time_value=0.4)

    torch.testing.assert_close(recovered, velocity)
    assert flow_matching_clean_proposal_scale(
        time_value=0.4, noise_multiplier=0.5
    ) == pytest.approx(1.0 / 3.0)


def test_external_cost_callback_drives_clean_proposal_mean() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=2048,
            temperature=0.05,
            proposal_std=0.8,
        )
    )
    center = torch.zeros((2, 3, 1))
    lower = torch.full((1,), -2.0)
    upper = torch.full((1,), 2.0)
    calls = []

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        calls.append(candidates.shape)
        return torch.square(candidates - 1.0).mean(dim=(2, 3))

    result = engine.optimize_clean_trajectories(
        center,
        lower=lower,
        upper=upper,
        cost_fn=cost_fn,
        generator=torch.Generator().manual_seed(11),
        proposal_scale=0.8,
    )

    assert calls == [(2, 2048, 3, 1)]
    assert result.costs.shape == (2, 2048)
    assert result.weights.shape == (2, 2048)
    torch.testing.assert_close(result.candidates[:, 0], center)
    assert torch.all(result.mean > 0.5)
    np.testing.assert_allclose(result.weights.sum(dim=1).numpy(), 1.0, atol=1e-6)


def test_shifted_proposal_uses_policy_target_importance_ratio() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=64,
            temperature=0.1,
            proposal_std=1.0,
            proposal_sampler="clipped_gaussian",
        )
    )
    proposal_center = torch.full((1, 1, 1), 0.75)
    target_center = torch.zeros_like(proposal_center)
    lower = torch.full((1,), -8.0)
    upper = torch.full((1,), 8.0)

    result = engine.optimize_clean_trajectories(
        proposal_center,
        target_center=target_center,
        lower=lower,
        upper=upper,
        cost_fn=lambda candidates: torch.zeros(candidates.shape[:2]),
        generator=torch.Generator().manual_seed(19),
        proposal_scale=1.0,
    )

    values = result.candidates[..., 0, 0]
    expected_logits = -0.5 * values.square() + 0.5 * (
        values - proposal_center[0, 0, 0]
    ).square()
    expected_weights = torch.softmax(expected_logits, dim=1)
    torch.testing.assert_close(result.weights, expected_weights)
    assert not torch.equal(result.candidates[:, 0], proposal_center)
    assert result.diagnostics is not None
    assert result.diagnostics["importance_density_correction"] is True


def test_policy_blended_mixture_uses_balance_heuristic_weights() -> None:
    proposal_count = 64
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=proposal_count,
            temperature=0.1,
            proposal_std=1.0,
            proposal_sampler="clipped_gaussian",
        )
    )
    blended_center = torch.full((1, 1, 1), 0.75)
    policy_center = torch.zeros_like(blended_center)
    lower = torch.full((1,), -8.0)
    upper = torch.full((1,), 8.0)

    result = engine.optimize_clean_trajectories(
        blended_center,
        target_center=policy_center,
        mixture_with_target=True,
        lower=lower,
        upper=upper,
        cost_fn=lambda candidates: torch.zeros(candidates.shape[:2]),
        generator=torch.Generator().manual_seed(23),
        proposal_scale=1.0,
    )

    assert result.candidates.shape == (1, 2 * proposal_count, 1, 1)
    policy_log_prob = bounded_gaussian_log_prob(
        result.candidates,
        center=policy_center,
        scale=1.0,
        lower=lower,
        upper=upper,
        sampler="clipped_gaussian",
    )
    blended_log_prob = bounded_gaussian_log_prob(
        result.candidates,
        center=blended_center,
        scale=1.0,
        lower=lower,
        upper=upper,
        sampler="clipped_gaussian",
    )
    mixture_log_prob = torch.logaddexp(
        policy_log_prob, blended_log_prob
    ) - np.log(2.0)
    expected_weights = torch.softmax(
        policy_log_prob - mixture_log_prob, dim=1
    ).to(dtype=result.weights.dtype)
    torch.testing.assert_close(result.weights, expected_weights)
    assert result.diagnostics is not None
    assert (
        result.diagnostics["importance_proposal"]
        == "half_policy_half_previous_blended"
    )
    assert (
        result.diagnostics["importance_policy_component_samples"]
        == proposal_count
    )
    assert (
        result.diagnostics["importance_shifted_component_samples"]
        == proposal_count
    )
    assert result.diagnostics["importance_total_samples"] == 2 * proposal_count


def test_weighted_three_center_mixture_uses_actual_sample_fractions() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=8,
            temperature=0.1,
            proposal_std=1.0,
            proposal_sampler="truncated_gaussian",
        )
    )
    target = torch.zeros((1, 1, 1))
    previous_blended = torch.full_like(target, 1.25)
    previous_replan = torch.full_like(target, -1.5)
    centers = (target, previous_blended, previous_replan)
    counts = (5, 3, 2)
    labels = ("policy", "previous_blended", "previous_replan_keypose")
    lower = torch.full((1,), -8.0)
    upper = torch.full((1,), 8.0)

    result = engine.optimize_clean_trajectories(
        previous_blended,
        target_center=target,
        proposal_mixture_centers=centers,
        proposal_mixture_counts=counts,
        proposal_mixture_labels=labels,
        lower=lower,
        upper=upper,
        cost_fn=lambda candidates: torch.zeros(candidates.shape[:2]),
        generator=torch.Generator().manual_seed(31),
        proposal_scale=1.0,
    )

    assert result.candidates.shape == (1, sum(counts), 1, 1)
    component_logits = torch.stack(
        [
            bounded_gaussian_log_prob(
                result.candidates, center=center, scale=1.0,
                lower=lower, upper=upper, sampler="truncated_gaussian",
            )
            + np.log(count / sum(counts))
            for center, count in zip(centers, counts, strict=True)
        ],
        dim=0,
    )
    target_log_prob = bounded_gaussian_log_prob(
        result.candidates, center=target, scale=1.0,
        lower=lower, upper=upper, sampler="truncated_gaussian",
    )
    expected = torch.softmax(
        target_log_prob - torch.logsumexp(component_logits, dim=0), dim=1
    ).to(result.weights.dtype)
    torch.testing.assert_close(result.weights, expected)
    assert result.diagnostics["importance_proposal"] == (
        "weighted_multi_center_gaussian_mixture"
    )
    assert result.diagnostics["importance_mixture_component_samples"] == {
        label: count for label, count in zip(labels, counts, strict=True)
    }
    assert result.diagnostics["importance_total_samples"] == sum(counts)


def test_log_acceptance_tilt_is_exactly_importance_corrected() -> None:
    count = 20000
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=count,
            temperature=1.0,
            proposal_std=1.0,
            proposal_sampler="truncated_gaussian",
        )
    )
    center = torch.zeros((1, 1, 1))
    lower = torch.full((1,), -8.0)
    upper = torch.full((1,), 8.0)
    tilted_scale = 0.8

    def log_acceptance(samples: torch.Tensor) -> torch.Tensor:
        value = samples[..., 0, 0]
        return -0.5 * value.square() * (1.0 / tilted_scale**2 - 1.0)

    result = engine.optimize_clean_trajectories(
        center,
        lower=lower,
        upper=upper,
        cost_fn=lambda candidates: torch.zeros(candidates.shape[:2]),
        generator=torch.Generator().manual_seed(202),
        proposal_scale=1.0,
        proposal_log_acceptance_fn=log_acceptance,
    )

    squared = result.candidates[..., 0, 0].square()
    tilted_second_moment = float(squared.mean())
    corrected_second_moment = float((result.weights * squared).sum())
    assert 0.58 < tilted_second_moment < 0.70
    assert 0.90 < corrected_second_moment < 1.10
    assert result.diagnostics["importance_density_correction"] is True
    assert result.diagnostics["proposal_log_acceptance_tilt"] is True
    assert result.diagnostics["importance_proposal"] == (
        "bounded_gaussian_mixture_times_log_acceptance_tilt"
    )


def test_clipped_gaussian_log_prob_uses_boundary_probability_mass() -> None:
    samples = torch.tensor([[[[-1.0]], [[0.25]], [[1.0]]]])
    center = torch.tensor([[[0.2]]])
    lower = torch.tensor([-1.0])
    upper = torch.tensor([1.0])
    scale = 0.7

    actual = bounded_gaussian_log_prob(
        samples,
        center=center,
        scale=scale,
        lower=lower,
        upper=upper,
        sampler="clipped_gaussian",
    )
    normal = torch.distributions.Normal(0.0, 1.0)
    expected = torch.stack(
        (
            torch.log(normal.cdf(torch.tensor((-1.0 - 0.2) / scale))),
            normal.log_prob(torch.tensor((0.25 - 0.2) / scale))
            - np.log(scale),
            torch.log(normal.cdf(torch.tensor(-(1.0 - 0.2) / scale))),
        )
    ).reshape(1, 3)
    torch.testing.assert_close(actual, expected.to(torch.float64))


def test_score_guidance_uses_cost_callback_but_not_task_or_robot_types() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=1024,
            temperature=0.1,
            proposal_std=0.5,
        )
    )
    x_t = torch.zeros((2, 3, 1))
    policy_velocity = torch.zeros_like(x_t)
    lower = torch.full((1,), -2.0)
    upper = torch.full((1,), 2.0)

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        return torch.square(candidates - 1.0).mean(dim=(2, 3))

    initial = engine.guide_score(
        x_t,
        policy_velocity,
        time_value=1.0,
        lower=lower,
        upper=upper,
        cost_fn=cost_fn,
        generator=torch.Generator().manual_seed(3),
        trajectory_coefficient=1.0,
    )
    torch.testing.assert_close(initial.guided_flow, policy_velocity)
    assert initial.proposals.proposal_scale == 0.0

    guided = engine.guide_score(
        x_t,
        policy_velocity,
        time_value=0.5,
        lower=lower,
        upper=upper,
        cost_fn=cost_fn,
        generator=torch.Generator().manual_seed(3),
        trajectory_coefficient=1.0,
    )
    assert torch.all(guided.proposals.mean > 0.2)
    assert not torch.allclose(guided.guided_flow, policy_velocity)


def test_recenter_particle_noise_sets_exact_mean_and_preserves_residuals() -> None:
    noise = torch.tensor(
        [[[-2.0, 1.0]], [[1.0, 3.0]], [[4.0, -1.0]]]
    )
    requested_mean = torch.tensor([[[0.75, -0.25]]])

    recentered = recenter_particle_noise(noise, mean=requested_mean)

    torch.testing.assert_close(
        recentered.mean(dim=0, keepdim=True), requested_mean
    )
    torch.testing.assert_close(
        recentered - recentered.mean(dim=0, keepdim=True),
        noise - noise.mean(dim=0, keepdim=True),
    )


def test_sequential_guidance_freezes_optimized_keypose_for_waypoints() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=2048,
            temperature=0.05,
            proposal_std=0.8,
        )
    )
    x_t = torch.zeros((2, 3, 1))
    policy_velocity = torch.zeros_like(x_t)
    lower = torch.full((1,), -2.0)
    upper = torch.full((1,), 2.0)
    waypoint_inputs: list[torch.Tensor] = []

    def keypose_cost(candidates: torch.Tensor) -> torch.Tensor:
        return torch.square(candidates[..., -1, 0] - 1.0)

    def waypoint_cost(candidates: torch.Tensor) -> torch.Tensor:
        waypoint_inputs.append(candidates.detach().clone())
        return torch.square(candidates[..., :-1, 0] + 1.0).mean(dim=-1)

    result = engine.guide_keypose_then_waypoints(
        x_t,
        policy_velocity,
        time_value=0.5,
        lower=lower,
        upper=upper,
        keypose_cost_fn=keypose_cost,
        waypoint_cost_fn=waypoint_cost,
        generator=torch.Generator().manual_seed(13),
        waypoint_coefficient=1.0,
        keypose_coefficient=0.5,
    )

    assert result.waypoints is not None
    torch.testing.assert_close(
        result.fixed_keypose, result.keypose.proposals.mean
    )
    assert len(waypoint_inputs) == 1
    conditioned = waypoint_inputs[0][..., -1:, :]
    expected = result.fixed_keypose[:, None, :, :].expand_as(conditioned)
    torch.testing.assert_close(conditioned, expected)
    assert torch.all(result.fixed_keypose > 0.5)
    assert torch.all(result.waypoints.proposals.mean < -0.5)



def test_sequential_guidance_supports_distinct_proposal_populations() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=5,
            temperature=0.1,
            proposal_std=0.5,
        )
    )
    x_t = torch.zeros((1, 3, 1))
    policy_velocity = torch.zeros_like(x_t)
    lower = torch.full((1,), -2.0)
    upper = torch.full((1,), 2.0)

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        return torch.square(candidates).mean(dim=(2, 3))

    result = engine.guide_keypose_then_waypoints(
        x_t,
        policy_velocity,
        time_value=0.5,
        lower=lower,
        upper=upper,
        keypose_cost_fn=cost_fn,
        waypoint_cost_fn=cost_fn,
        generator=torch.Generator().manual_seed(17),
        waypoint_coefficient=1.0,
        keypose_coefficient=1.0,
        keypose_proposals_per_particle=11,
        waypoint_proposals_per_particle=7,
    )

    assert result.keypose.proposals.candidates.shape == (1, 11, 1, 1)
    assert result.waypoints is not None
    assert result.waypoints.proposals.candidates.shape == (1, 7, 2, 1)



def test_sequential_guidance_supports_waypoint_only_temperature() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=16,
            temperature=0.01,
            proposal_std=0.5,
        )
    )
    x_t = torch.zeros((1, 3, 1))
    policy_velocity = torch.zeros_like(x_t)
    bounds = torch.full((1,), 2.0)

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        return torch.square(candidates).mean(dim=(2, 3))

    result = engine.guide_keypose_then_waypoints(
        x_t,
        policy_velocity,
        time_value=0.5,
        lower=-bounds,
        upper=bounds,
        keypose_cost_fn=cost_fn,
        waypoint_cost_fn=cost_fn,
        generator=torch.Generator().manual_seed(23),
        waypoint_coefficient=1.0,
        keypose_coefficient=1.0,
        waypoint_temperature=0.5,
    )

    assert result.waypoints is not None
    assert result.keypose.proposals.diagnostics["beta_schedule"] == [0.0, 100.0]
    assert result.waypoints.proposals.diagnostics["beta_schedule"] == [0.0, 2.0]

def test_sequential_guidance_can_condition_on_guided_keypose() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=1024,
            temperature=0.05,
            proposal_std=0.8,
        )
    )
    x_t = torch.zeros((1, 3, 1))
    policy_velocity = torch.zeros_like(x_t)
    lower = torch.full((1,), -2.0)
    upper = torch.full((1,), 2.0)
    waypoint_inputs: list[torch.Tensor] = []

    def keypose_cost(candidates: torch.Tensor) -> torch.Tensor:
        return torch.square(candidates[..., -1, 0] - 1.0)

    def waypoint_cost(candidates: torch.Tensor) -> torch.Tensor:
        waypoint_inputs.append(candidates.detach().clone())
        return torch.square(candidates[..., :-1, 0]).mean(dim=-1)

    result = engine.guide_keypose_then_waypoints(
        x_t,
        policy_velocity,
        time_value=0.5,
        lower=lower,
        upper=upper,
        keypose_cost_fn=keypose_cost,
        waypoint_cost_fn=waypoint_cost,
        generator=torch.Generator().manual_seed(1),
        keypose_generator=torch.Generator().manual_seed(2),
        waypoint_generator=torch.Generator().manual_seed(3),
        waypoint_coefficient=0.4,
        keypose_coefficient=0.9,
        condition_waypoints_on_guided_keypose=True,
    )

    expected = torch.clamp(
        x_t[:, -1:, :] - 0.5 * result.keypose.guided_flow,
        min=lower,
        max=upper,
    )
    torch.testing.assert_close(result.waypoint_conditioning_keypose, expected)
    assert not torch.allclose(
        result.waypoint_conditioning_keypose, result.fixed_keypose
    )
    conditioned = waypoint_inputs[0][..., -1:, :]
    torch.testing.assert_close(
        conditioned,
        result.waypoint_conditioning_keypose[:, None].expand_as(conditioned),
    )


def test_sequential_keypose_rng_is_independent_of_waypoint_population() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=5,
            temperature=0.1,
            proposal_std=0.5,
        )
    )
    x_t = torch.zeros((1, 3, 1))
    policy_velocity = torch.zeros_like(x_t)
    lower = torch.full((1,), -2.0)
    upper = torch.full((1,), 2.0)

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        return torch.square(candidates).mean(dim=(2, 3))

    def run(waypoint_samples: int):
        return engine.guide_keypose_then_waypoints(
            x_t,
            policy_velocity,
            time_value=0.5,
            lower=lower,
            upper=upper,
            keypose_cost_fn=cost_fn,
            waypoint_cost_fn=cost_fn,
            generator=torch.Generator().manual_seed(11),
            keypose_generator=torch.Generator().manual_seed(13),
            waypoint_generator=torch.Generator().manual_seed(17),
            waypoint_coefficient=0.6,
            keypose_coefficient=0.9,
            keypose_proposals_per_particle=19,
            waypoint_proposals_per_particle=waypoint_samples,
            condition_waypoints_on_guided_keypose=True,
        )

    small = run(3)
    large = run(37)
    torch.testing.assert_close(
        small.keypose.proposals.candidates,
        large.keypose.proposals.candidates,
    )
    torch.testing.assert_close(small.fixed_keypose, large.fixed_keypose)
    torch.testing.assert_close(
        small.waypoint_conditioning_keypose,
        large.waypoint_conditioning_keypose,
    )


def test_pps_mbd_module_has_no_downstream_simulator_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "sim_free_mpc"
        / "rectified_flow_mbd.py"
    ).read_text()
    for forbidden in ("grill_sim_infra", "import mujoco", "import warp"):
        assert forbidden not in source


def test_adaptive_smc_is_optional_and_uses_gradient_callback_for_mala() -> None:
    engine = RectifiedFlowMBD(
        RectifiedFlowMBDConfig(
            proposals_per_particle=512,
            temperature=0.1,
            proposal_std=0.8,
            proposal_sampler="truncated_gaussian",
            inference_sampler="adaptive_smc",
            smc_target_ess_fraction=0.5,
            smc_resample_ess_fraction=0.5,
            smc_mala_steps=1,
            smc_mala_step_size=0.15,
        )
    )
    center = torch.zeros((1, 1, 1))
    lower = torch.full((1,), -3.0)
    upper = torch.full((1,), 3.0)

    def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.square(candidates[..., 0] - 1.0).mean(dim=-1)

    with torch.inference_mode():
        result = engine.optimize_clean_trajectories(
            center,
            lower=lower,
            upper=upper,
            cost_fn=cost_fn,
            gradient_cost_fn=cost_fn,
            generator=torch.Generator().manual_seed(31),
            proposal_scale=0.8,
        )

    assert result.diagnostics is not None
    assert result.diagnostics["sampler"] == "adaptive_tempered_smc"
    assert result.diagnostics["beta_schedule"][0] == 0.0
    assert result.diagnostics["beta_schedule"][-1] == 10.0
    assert result.diagnostics["gradient_cost_evaluation_calls"] > 0
    assert result.diagnostics["mala_acceptance_rate"] is not None
    assert float(result.mean.item()) > 0.5
