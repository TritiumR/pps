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
    FlowBlendCoefficients,
    RectifiedFlowMBD,
    RectifiedFlowMBDConfig,
    TokenBlockLayout,
    blend_policy_mbd_flows,
    flow_matching_clean_proposal_scale,
    memoryless_sde_kl_blocks,
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


def test_pps_mbd_module_has_no_downstream_simulator_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "sim_free_mpc"
        / "rectified_flow_mbd.py"
    ).read_text()
    for forbidden in ("grill_sim_infra", "import mujoco", "import warp"):
        assert forbidden not in source
