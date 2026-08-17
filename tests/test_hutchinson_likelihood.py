import math

import torch

from sim_free_mpc.hutchinson_likelihood import (
    estimate_probability_flow_log_likelihood,
    hutchinson_divergence,
)


def test_hutchinson_is_exact_for_diagonal_linear_field() -> None:
    state = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    diagonal = torch.tensor([0.5, -0.25, 2.0])
    velocity = diagonal * state
    probes = torch.tensor([[[1.0, -1.0, 1.0]], [[-1.0, 1.0, -1.0]]])
    actual = hutchinson_divergence(velocity, state, probes)
    torch.testing.assert_close(actual, diagonal.sum().reshape(1))


def test_linear_probability_flow_matches_change_of_variables() -> None:
    coefficient = 0.2
    initial = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    estimate = estimate_probability_flow_log_likelihood(
        lambda state, _time: coefficient * state,
        initial,
        num_steps=2000,
        num_probes=1,
        time_start=0.0,
        time_end=1.0,
    )
    terminal = initial * math.exp(coefficient)
    expected_prior = -0.5 * (
        terminal.square().sum(dim=1) + initial.shape[1] * math.log(2.0 * math.pi)
    )
    expected = expected_prior + coefficient * initial.shape[1]
    torch.testing.assert_close(estimate.log_prob, expected, atol=2.0e-4, rtol=2.0e-4)


def test_batched_per_item_seeds_match_independent_estimates() -> None:
    initial = torch.tensor(
        [[[0.25, -0.5], [0.75, 0.1]], [[-0.2, 0.4], [0.3, -0.7]]],
        dtype=torch.float32,
    )
    seeds = [17, 29]
    velocity_fn = lambda state, _time: 0.15 * state.roll(1, dims=-1)

    batched = estimate_probability_flow_log_likelihood(
        velocity_fn, initial, num_steps=4, num_probes=2, seed=seeds
    )
    independent = [
        estimate_probability_flow_log_likelihood(
            velocity_fn, initial[index : index + 1], num_steps=4, num_probes=2, seed=seed
        )
        for index, seed in enumerate(seeds)
    ]

    torch.testing.assert_close(
        batched.log_prob, torch.cat([estimate.log_prob for estimate in independent])
    )
    torch.testing.assert_close(
        batched.divergence_integral,
        torch.cat([estimate.divergence_integral for estimate in independent]),
    )
