from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc import SimFreeMPC, SimFreeMPCConfig  # noqa: E402


def test_bspline_basis_is_partition_of_unity():
    basis = SimFreeMPC._bspline_basis(
        4,
        11,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert basis.shape == (11, 4)
    assert torch.allclose(basis.sum(dim=1), torch.ones(11), atol=1e-6)
    assert torch.all(basis >= 0.0)


def test_bspline_resample_preserves_clamped_endpoints():
    control_points = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, -1.0],
            [4.0, 0.5],
        ],
        dtype=torch.float32,
    )

    resampled = SimFreeMPC._bspline_resample(control_points, 13)

    assert resampled.shape == (13, 2)
    assert torch.allclose(resampled[0], control_points[0], atol=1e-6)
    assert torch.allclose(resampled[-1], control_points[-1], atol=1e-6)


def test_interpolate_control_points_uses_bspline_when_enabled():
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(interpolate=True)
    control_points = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]], dtype=torch.float32)

    resampled = planner._interpolate_control_points(control_points, 9)

    assert resampled.shape == (1, 9, 1)
    assert torch.allclose(resampled[:, 0], control_points[:, 0], atol=1e-6)
    assert torch.allclose(resampled[:, -1], control_points[:, -1], atol=1e-6)


def test_accel_parameterization_reconstructs_trajectory():
    sequence = torch.tensor(
        [
            [0.2, -0.1, 0.0],
            [0.4, -0.3, 0.1],
            [0.45, -0.2, 0.15],
            [0.3, 0.0, 0.2],
        ],
        dtype=torch.float32,
    )

    accel_code = SimFreeMPC._trajectory_to_accel_code(sequence)
    reconstructed = SimFreeMPC._accel_code_to_trajectory(accel_code.unsqueeze(0))[0]

    assert torch.allclose(reconstructed, sequence, atol=1e-6)


def test_step_ddim_updates_only_active_dims(monkeypatch):
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(action_dims=2, noise=0.0, flow_eps=1e-6)

    x_t = torch.tensor([[[0.2, -0.4, 1.5], [0.3, -0.5, 1.7]]], dtype=torch.float32)

    def fake_optimize(x_t_arg, _policy_inputs, _context, *, alpha_bar):
        x0_hat = x_t_arg.detach().clone()
        x0_hat[:, :, :2] = x_t_arg[:, :, :2] / (alpha_bar**0.5)
        result = SimpleNamespace(
            costs=torch.tensor([1.0, 2.0], dtype=x_t_arg.dtype),
            weights=torch.tensor([0.75, 0.25], dtype=x_t_arg.dtype),
            noise_scale=torch.zeros(x_t_arg.shape[1], dtype=x_t_arg.dtype),
        )
        return x0_hat, result, 2, 0.0

    monkeypatch.setattr(planner, "_optimize_ddim_clean_chunk", fake_optimize)

    next_x, diagnostics = planner.step_ddim(
        x_t,
        {},
        {},
        iteration=1,
        num_iterations=3,
        step_scale=1.0,
    )

    assert next_x.shape == x_t.shape
    assert torch.allclose(next_x[:, :, 2:], x_t[:, :, 2:])
    assert torch.isfinite(next_x).all()
    assert diagnostics["update_mode"] == "ddim"


def test_step_mbd_score_updates_only_active_dims(monkeypatch):
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(action_dims=2, noise=0.0, flow_eps=1e-6)

    x_t = torch.tensor([[[0.2, -0.4, 1.5], [0.3, -0.5, 1.7]]], dtype=torch.float32)

    def fake_optimize(x_t_arg, _policy_inputs, _context, *, alpha_bar):
        x0_hat = x_t_arg.detach().clone()
        x0_hat[:, :, :2] = x_t_arg[:, :, :2] / (alpha_bar**0.5)
        result = SimpleNamespace(
            costs=torch.tensor([1.0, 2.0], dtype=x_t_arg.dtype),
            weights=torch.tensor([0.75, 0.25], dtype=x_t_arg.dtype),
            noise_scale=torch.zeros(x_t_arg.shape[1], dtype=x_t_arg.dtype),
        )
        return x0_hat, result, 2, 0.0

    monkeypatch.setattr(planner, "_optimize_ddim_clean_chunk", fake_optimize)

    next_x, diagnostics = planner.step_mbd_score(
        x_t,
        {},
        {},
        iteration=1,
        num_iterations=3,
        score_scale=1.0,
    )

    assert next_x.shape == x_t.shape
    assert torch.allclose(next_x[:, :, 2:], x_t[:, :, 2:])
    assert torch.isfinite(next_x).all()
    assert diagnostics["update_mode"] == "mbd_score"


def test_action_prox_mbd_score_uses_noisy_center_and_scaled_forward_noise_std():
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(action_dims=2, noise=0.25, flow_eps=1e-6)
    captured = {}

    class FakeSampler:
        def optimize_with_noise_scale(self, initial_mean, _cost_fn, *, noise_scale):
            captured["initial_mean"] = initial_mean.detach().clone()
            captured["noise_scale"] = float(noise_scale)
            return SimpleNamespace(
                mean=initial_mean + 0.05,
                costs=torch.tensor([1.0, 2.0], dtype=initial_mean.dtype),
                weights=torch.tensor([0.75, 0.25], dtype=initial_mean.dtype),
                samples=initial_mean.unsqueeze(0),
                noise_scale=torch.full((initial_mean.shape[0],), float(noise_scale), dtype=initial_mean.dtype),
            )

    planner.sampler = FakeSampler()

    x_t = torch.tensor([[[0.2, -0.4, 1.5], [0.3, -0.5, 1.7]]], dtype=torch.float32)
    next_x, diagnostics = planner.step_mbd_score_action_prox(
        x_t,
        {},
        {},
        iteration=1,
        num_iterations=3,
        score_scale=1.0,
    )

    assert torch.allclose(captured["initial_mean"], x_t[0, :, :2])
    assert captured["noise_scale"] == pytest.approx(
        planner.config.noise * (1.0 - diagnostics["alpha_bar"]) ** 0.5
    )
    assert next_x.shape == x_t.shape
    assert torch.allclose(next_x[:, :, 2:], x_t[:, :, 2:])
    assert diagnostics["update_mode"] == "mbd_score_action_prox"
    assert diagnostics["proposal_center"] == "current_noisy_action"

    _, warm_diagnostics = planner.step_mbd_score_action_warm(
        x_t,
        {},
        {},
        iteration=1,
        num_iterations=3,
        score_scale=1.0,
    )
    assert warm_diagnostics["update_mode"] == "mbd_score_action_warm"
    assert warm_diagnostics["proposal_noise_scale"] == pytest.approx(
        planner.config.noise * (1.0 - warm_diagnostics["alpha_bar"]) ** 0.5
    )


def test_action_warm_start_shifts_previous_normalized_action():
    planner = object.__new__(SimFreeMPC)
    planner.reset_action_warm()

    fallback = torch.zeros((1, 4, 2), dtype=torch.float32)
    initial, started = planner.warm_start_noise(fallback, shift_steps=2)
    assert not started
    assert initial is fallback

    previous = torch.tensor(
        [[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]],
        dtype=torch.float32,
    )
    planner.set_warm_action(previous)
    initial, started = planner.warm_start_noise(fallback, shift_steps=2)

    assert started
    assert torch.equal(
        initial,
        torch.tensor(
            [[[4.0, 5.0], [6.0, 7.0], [6.0, 7.0], [6.0, 7.0]]],
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize("use_quantile_norm", [False, True])
def test_action_warm_start_rebases_deltas_to_current_state(use_quantile_norm):
    planner = object.__new__(SimFreeMPC)
    if use_quantile_norm:
        stats = SimpleNamespace(q01=-torch.ones(8), q99=torch.ones(8))
    else:
        stats = SimpleNamespace(mean=torch.zeros(8), std=torch.ones(8))
    planner.policy = SimpleNamespace(
        _metadata={
            "output_norm_stats": {"actions": stats, "state": stats},
            "use_quantile_norm": use_quantile_norm,
        }
    )
    planner.reset_action_warm()

    previous = torch.zeros((1, 3, 8), dtype=torch.float32)
    previous[..., :7] = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1)
    previous[..., 7] = 0.75
    previous_state = torch.zeros((1, 8), dtype=torch.float32)
    current_state = torch.zeros((1, 8), dtype=torch.float32)
    current_state[..., :7] = 0.05
    planner.set_warm_action(previous, state=previous_state)

    initial, started = planner.warm_start_noise(
        torch.zeros_like(previous),
        shift_steps=1,
        current_state=current_state,
    )

    assert started
    expected_arm = torch.tensor([0.15, 0.25, 0.25]).view(1, 3, 1).expand(-1, -1, 7)
    assert torch.allclose(initial[..., :7], expected_arm, atol=1e-6)
    assert torch.allclose(initial[..., 7], torch.full((1, 3), 0.75), atol=1e-6)


def test_step_from_score_mbd_matches_base_score_numerator_update():
    planner = object.__new__(SimFreeMPC)
    planner.config = SimFreeMPCConfig(action_dims=2, flow_eps=1e-6)
    x_t = torch.tensor([[[0.2, -0.4, 1.5], [0.3, -0.5, 1.7]]], dtype=torch.float32)
    score = torch.zeros_like(x_t)
    score[:, :, :2] = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]], dtype=torch.float32)

    next_x = planner.step_from_score(
        x_t,
        score,
        iteration=1,
        num_iterations=3,
        update_mode="mbd_score",
        active_dims=2,
    )

    assert next_x.shape == x_t.shape
    assert torch.allclose(next_x[:, :, 2:], x_t[:, :, 2:])
    assert torch.isfinite(next_x).all()
