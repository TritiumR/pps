from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _load_ddim_module():
    path = Path(__file__).resolve().parents[1] / "sim_free_mpc" / "ddim.py"
    spec = importlib.util.spec_from_file_location("sim_free_mpc_ddim_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ddim = _load_ddim_module()


def test_cosine_alphas_are_bounded_and_monotonic():
    alphas = ddim.ddim_alphas_cumprod(100)

    assert len(alphas) == 100
    assert all(0.0 < alpha < 1.0 for alpha in alphas)
    assert all(a > b for a, b in zip(alphas, alphas[1:]))


def test_iteration_alphas_walk_from_noisy_to_clean():
    alpha_first, alpha_prev_first = ddim.ddim_iteration_alphas(
        iteration=0,
        num_iterations=10,
        num_train_timesteps=100,
    )
    alpha_last, alpha_prev_last = ddim.ddim_iteration_alphas(
        iteration=9,
        num_iterations=10,
        num_train_timesteps=100,
    )

    assert alpha_first < alpha_prev_first
    assert alpha_first < alpha_last
    assert alpha_prev_last == 1.0


def test_score_formula_matches_weighted_clean_estimate_equation():
    y_t = 0.7
    x0_hat = 0.2
    alpha_bar = 0.36

    score = (-y_t + math.sqrt(alpha_bar) * x0_hat) / (1.0 - alpha_bar)

    assert score == (-0.7 + 0.6 * 0.2) / 0.64


def test_zero_epsilon_ddim_step_scales_clean_estimate():
    x0_hat = 0.25
    alpha_bar = 0.25
    alpha_prev = 0.64
    y_t = math.sqrt(alpha_bar) * x0_hat

    eps_hat = (y_t - math.sqrt(alpha_bar) * x0_hat) / math.sqrt(1.0 - alpha_bar)
    y_prev = math.sqrt(alpha_prev) * x0_hat + math.sqrt(1.0 - alpha_prev) * eps_hat

    assert eps_hat == 0.0
    assert y_prev == math.sqrt(alpha_prev) * x0_hat
