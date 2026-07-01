from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_free_mpc import SimFreeMPC, SimFreeMPCConfig  # noqa: E402


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
