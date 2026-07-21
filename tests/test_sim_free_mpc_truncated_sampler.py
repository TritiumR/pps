from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc.action_space import decode_model_action_chunks  # noqa: E402
from sim_free_mpc.dial_sampler import DIALSampler, DIALSamplerConfig  # noqa: E402
from sim_free_mpc.fk import PANDA_JOINT_LIMITS  # noqa: E402
from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig  # noqa: E402
from sim_free_mpc.truncated_sampler import (  # noqa: E402
    sample_truncated_model_action_chunks,
)


def _policy_and_inputs(*, use_quantile_norm: bool):
    if use_quantile_norm:
        action_stats = SimpleNamespace(
            q01=-torch.ones(8),
            q99=torch.ones(8),
        )
        state_stats = SimpleNamespace(
            q01=-torch.ones(8),
            q99=torch.ones(8),
        )
    else:
        action_stats = SimpleNamespace(
            mean=torch.zeros(8),
            std=torch.tensor([0.5] * 7 + [1.0]),
        )
        state_stats = SimpleNamespace(
            mean=torch.zeros(8),
            std=torch.ones(8),
        )
    policy = SimpleNamespace(
        _metadata={
            "output_norm_stats": {
                "actions": action_stats,
                "state": state_stats,
            },
            "use_quantile_norm": use_quantile_norm,
        }
    )
    return policy, {"state": torch.zeros(1, 8)}


def _assert_valid(decoded: torch.Tensor, current: torch.Tensor, delta: float):
    joints = decoded[..., :7]
    limits = torch.as_tensor(PANDA_JOINT_LIMITS, dtype=decoded.dtype)
    assert torch.all(joints >= limits[:, 0] - 1e-6)
    assert torch.all(joints <= limits[:, 1] + 1e-6)

    previous = current.view(1, 7).expand(joints.shape[0], -1)
    for step in range(joints.shape[1]):
        assert torch.all((joints[:, step] - previous).abs() <= delta + 1e-6)
        previous = joints[:, step]

    assert torch.all(decoded[..., 7] >= 0.0)
    assert torch.all(decoded[..., 7] <= 1.0)


@pytest.mark.parametrize("use_quantile_norm", [False, True])
def test_truncated_sampler_generates_only_valid_autoregressive_chunks(
    use_quantile_norm,
):
    policy, policy_inputs = _policy_and_inputs(
        use_quantile_norm=use_quantile_norm
    )
    mean = torch.zeros(6, 8)
    mean[:, 7] = 0.5
    current = torch.zeros(7)
    delta = 0.15

    samples = sample_truncated_model_action_chunks(
        policy,
        policy_inputs,
        mean,
        noise_scale=1.0,
        num_samples=256,
        current_joint_pos=current,
        max_joint_delta=delta,
        generator=torch.Generator().manual_seed(7),
    )
    decoded = decode_model_action_chunks(
        policy,
        policy_inputs,
        samples,
        apply_clamp=False,
    ).real_actions

    assert samples.shape == (256, 6, 8)
    _assert_valid(decoded, current, delta)


def test_weighted_mean_of_truncated_candidates_remains_valid():
    policy, policy_inputs = _policy_and_inputs(use_quantile_norm=False)
    mean = torch.zeros(5, 8)
    mean[:, 7] = 0.5
    current = torch.zeros(7)
    delta = 0.15
    samples = sample_truncated_model_action_chunks(
        policy,
        policy_inputs,
        mean,
        noise_scale=0.8,
        num_samples=128,
        current_joint_pos=current,
        max_joint_delta=delta,
        generator=torch.Generator().manual_seed(11),
    )
    weights = torch.softmax(torch.randn(128), dim=0)
    weighted_mean = torch.sum(weights[:, None, None] * samples, dim=0, keepdim=True)
    decoded_mean = decode_model_action_chunks(
        policy,
        policy_inputs,
        weighted_mean,
        apply_clamp=False,
    ).real_actions

    _assert_valid(decoded_mean, current, delta)


def test_legacy_optimizer_uses_custom_proposal():
    sampler = DIALSampler(
        DIALSamplerConfig(num_samples=4, iterations=1, noise=0.3)
    )
    captured = {}

    def proposal(mean, scale, num_samples, _generator):
        captured["scale"] = scale.clone()
        return mean.unsqueeze(0).expand(num_samples, -1, -1).clone()

    sampler.proposal_fn = proposal
    result = sampler.optimize(
        torch.zeros(3, 2),
        lambda samples: samples.square().sum(dim=(1, 2)),
    )

    assert result.samples.shape == (4, 3, 2)
    assert captured["scale"].shape == (3,)


def test_truncated_sampler_uses_low_frequency_knots_when_interpolation_is_enabled():
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(
        sampler="truncated",
        interpolate=True,
        control_frequency=15.0,
        interpolate_frequency=5.0,
    )

    assert planner._interpolation_knot_count(12) == 4


@pytest.mark.parametrize("interpolation_method", ["linear", "bspline"])
def test_truncated_interpolation_preserves_high_frequency_joint_delta(
    interpolation_method,
):
    policy, policy_inputs = _policy_and_inputs(use_quantile_norm=False)
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(
        sampler="truncated",
        interpolate=True,
        control_frequency=40.0,
        interpolate_frequency=5.0,
        interpolation_method=interpolation_method,
        joint_delta_clip=0.15,
    )
    output_horizon = 40
    proposal_horizon = planner._interpolation_knot_count(output_horizon)
    mean = torch.zeros(proposal_horizon, 8)
    mean[:, 7] = 0.5
    max_joint_delta = planner._truncated_max_joint_deltas(
        proposal_horizon,
        output_horizon,
        device=mean.device,
        dtype=mean.dtype,
    )

    samples = sample_truncated_model_action_chunks(
        policy,
        policy_inputs,
        mean,
        noise_scale=1.0,
        num_samples=256,
        current_joint_pos=torch.zeros(7),
        max_joint_delta=max_joint_delta,
        generator=torch.Generator().manual_seed(17),
    )
    interpolated = planner._interpolate_control_points(samples, output_horizon)
    decoded = decode_model_action_chunks(
        policy,
        policy_inputs,
        interpolated,
        apply_clamp=False,
    ).real_actions

    assert proposal_horizon == 5
    assert max_joint_delta[0] == pytest.approx(0.15)
    assert torch.all(max_joint_delta[1:] > max_joint_delta[0])
    _assert_valid(decoded, torch.zeros(7), 0.15)
