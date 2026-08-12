from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PPS_ROOT = Path(__file__).resolve().parents[1]
if str(PPS_ROOT) not in sys.path:
    sys.path.insert(0, str(PPS_ROOT))

from sim_free_mpc.tempered_smc import (
    TemperedSMCConfig,
    adaptive_tempered_smc,
    choose_next_beta,
    mala_rejuvenate,
    resample_indices,
)


def quadratic_cost(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(
        torch.square(values - 1.0), dim=tuple(range(2, values.ndim))
    )


def full_quadratic_cost(values: torch.Tensor):
    return quadratic_cost(values), None


def test_adaptive_beta_hits_target_conditional_ess() -> None:
    costs = torch.linspace(0.0, 4.0, 256)[None]
    beta, ess = choose_next_beta(
        0.0,
        final_beta=10.0,
        log_weights=torch.full_like(costs, -torch.log(torch.tensor(256.0))),
        costs=costs,
        eligible=None,
        target_ess=128.0,
        tolerance=1e-5,
        bisection_steps=48,
    )
    assert 0.0 < beta < 10.0
    assert float(ess.item()) == pytest.approx(128.0, rel=2e-4)


def test_systematic_and_stratified_resampling_are_vectorized() -> None:
    weights = torch.tensor([[0.0, 0.0, 0.25, 0.75], [1.0, 0.0, 0.0, 0.0]])
    for method in ("systematic", "stratified"):
        indices = resample_indices(
            weights,
            method=method,
            generator=torch.Generator().manual_seed(7),
        )
        assert indices.shape == weights.shape
        assert torch.all(indices[0] >= 2)
        assert torch.all(indices[1] == 0)


def test_mala_mh_correction_preserves_base_gaussian() -> None:
    generator = torch.Generator().manual_seed(19)
    center = torch.zeros((1, 1, 1))
    particles = torch.randn((1, 4096, 1, 1), generator=generator)
    costs = torch.zeros((1, 4096))

    def zero_full(values: torch.Tensor):
        return torch.zeros(values.shape[:2]), None

    def zero_gradient(values: torch.Tensor):
        # Preserve an autograd path while contributing no cost gradient.
        return torch.sum(values * 0.0, dim=(2, 3))

    moved, _, _, diagnostics = mala_rejuvenate(
        particles,
        costs,
        None,
        center=center,
        scale=1.0,
        beta=10.0,
        lower=torch.tensor([-8.0]),
        upper=torch.tensor([8.0]),
        full_cost_fn=zero_full,
        gradient_cost_fn=zero_gradient,
        steps=8,
        step_size=0.4,
        generator=generator,
    )
    assert diagnostics["proposed"] == 8 * 4096
    assert 0.7 < diagnostics["accepted"] / diagnostics["proposed"] <= 1.0
    assert abs(float(torch.mean(moved))) < 0.05
    assert float(torch.var(moved)) == pytest.approx(1.0, abs=0.08)


def test_tempered_smc_reaches_sharp_quadratic_target_and_logs_work() -> None:
    generator = torch.Generator().manual_seed(23)
    center = torch.zeros((1, 1, 1))
    initial = torch.randn((1, 4096, 1, 1), generator=generator)
    result = adaptive_tempered_smc(
        initial,
        center=center,
        scale=1.0,
        lower=torch.tensor([-8.0]),
        upper=torch.tensor([8.0]),
        full_cost_fn=full_quadratic_cost,
        gradient_cost_fn=quadratic_cost,
        config=TemperedSMCConfig(
            final_beta=10.0,
            target_ess_fraction=0.5,
            resample_ess_fraction=0.5,
            resampling_method="systematic",
            mala_steps=2,
            mala_step_size=0.2,
        ),
        generator=generator,
    )
    diagnostics = result.diagnostics
    assert diagnostics["beta_schedule"][0] == 0.0
    assert diagnostics["beta_schedule"][-1] == 10.0
    assert all(
        left < right
        for left, right in zip(
            diagnostics["beta_schedule"], diagnostics["beta_schedule"][1:]
        )
    )
    assert diagnostics["full_cost_evaluation_calls"] >= 2
    assert diagnostics["gradient_cost_evaluation_calls"] >= 2
    assert diagnostics["mala_acceptance_rate"] is not None
    # Adaptive stages hit 0.5N and resample there; they must not create a
    # numerical micro-stage whose accumulated ESS falls below the target.
    assert all(
        float(stage_ess[0]) >= 0.5 * initial.shape[1] - 0.1
        for stage_ess in diagnostics["ess_per_stage"]
    )
    weighted_mean = torch.sum(
        result.weights[..., None, None] * result.particles, dim=1
    )
    # N(0,1) exp[-10 * .5(y-1)^2] has mean 10/11.
    assert float(weighted_mean.item()) == pytest.approx(10.0 / 11.0, abs=0.06)



def test_final_beta_mala_runs_once_and_logs_lineage_and_geometry() -> None:
    generator = torch.Generator().manual_seed(31)
    center = torch.zeros((1, 1, 1))
    initial = torch.randn((1, 512, 1, 1), generator=generator)
    result = adaptive_tempered_smc(
        initial,
        center=center,
        scale=1.0,
        lower=torch.tensor([-8.0]),
        upper=torch.tensor([8.0]),
        full_cost_fn=full_quadratic_cost,
        gradient_cost_fn=quadratic_cost,
        config=TemperedSMCConfig(
            final_beta=10.0,
            target_ess_fraction=0.5,
            resample_ess_fraction=0.5,
            mala_steps=1,
            mala_step_size=0.05,
            mala_schedule="final_beta",
        ),
        generator=generator,
    )
    diagnostics = result.diagnostics
    assert len(diagnostics["resampling_events"]) >= 1
    assert len(diagnostics["mala_events"]) == 1
    assert diagnostics["mala_events"][0]["beta"] == 10.0
    assert diagnostics["full_cost_evaluation_calls"] == 2
    assert diagnostics["gradient_cost_evaluation_calls"] == 2
    ancestry = diagnostics["final_pre_mala_ancestry"]
    assert ancestry["unique_ancestors"][0] <= initial.shape[1]
    before = diagnostics["final_pre_mala_geometric_diversity"]
    after = diagnostics["final_post_mala_geometric_diversity"]
    for values in (before, after):
        assert values["mean_pairwise_distance"][0] >= 0.0
        assert values["covariance_trace"][0] >= 0.0
        assert 0.0 <= values["near_duplicate_fraction"][0] <= 1.0
