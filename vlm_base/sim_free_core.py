"""Shared driver core for the sim_free_mpc-based tasks.

Provides what a driver needs to run SimFreeMPC without the pi0.5 checkpoint: a mock policy that
supplies only the decode norm-stats, the MPC construction, the B-spline horizon smoother, a NaN guard,
and the per-chunk plan/decode helpers. The engine (SimFreeMPC, the decode, ddim, FK, DIAL sampler)
is imported unchanged from sim_free_mpc.

Heavy imports stay at module top: this module is only imported from inside a task's run(), i.e.
after the Isaac app has booted (see vlm_base/main.py).
"""
from __future__ import annotations

import numpy as np
import torch

from sim_free_mpc.planner import SimFreeMPC, SimFreeMPCConfig
from sim_free_mpc.action_space import decode_model_action_chunks  # re-exported for task drivers
from sim_free_mpc.ddim import ddim_iteration_alphas               # re-exported for warm-start


class _Stats:
    """Minimal stand-in for pi0's per-dim output-norm stats (the only thing the decode fast-path reads)."""

    def __init__(self, mean=None, std=None, q01=None, q99=None):
        self.mean, self.std, self.q01, self.q99 = mean, std, q01, q99


# pi05_droid_jointpos quantile norm stats (7 joints + gripper); the decode fast-path reads only these,
# so no checkpoint is needed.
_A_Q01 = np.array([-0.286921, -0.459455, -0.285718, -0.474518, -0.466063, -0.441598, -0.532181, 0.0], np.float32)
_A_Q99 = np.array([0.285562, 0.527008, 0.283199, 0.470136, 0.453369, 0.500121, 0.532004, 0.9998], np.float32)
_S_Q01 = np.array([-0.827973, -0.839831, -0.842548, -2.77302, -1.84262, 1.17166, -2.04726, 0.0], np.float32)
_S_Q99 = np.array([0.899652, 1.38547, 0.692028, -0.454204, 1.7321, 3.4673, 2.1985, 0.991], np.float32)


def build_policy(real_stats: bool, action_std: float):
    """Mock policy carrying only the decode norm-stats (no checkpoint). Returns (policy, state_stats).

    real_stats: use the pi05_droid_jointpos quantile norm (exact decode). Otherwise an
    identity/action_std stand-in (arm delta = model * action_std + current joints).
    """
    if real_stats:
        actions_stats = _Stats(q01=_A_Q01, q99=_A_Q99)
        state_stats = _Stats(q01=_S_Q01, q99=_S_Q99)
        use_qn = True
    else:
        amean = np.zeros(8, np.float32); amean[7] = 0.5
        astd = np.full(8, float(action_std), np.float32); astd[7] = 0.5
        actions_stats = _Stats(mean=amean, std=astd)
        state_stats = _Stats(mean=np.zeros(8, np.float32), std=np.ones(8, np.float32))
        use_qn = False

    class MockPolicy:
        _metadata = {"use_quantile_norm": use_qn,
                     "output_norm_stats": {"actions": actions_stats, "state": state_stats}}

    return MockPolicy(), state_stats


def build_mpc(policy, *, num_samples, iterations, noise, temperature, joint_delta_clip, interpolate,
              task_name="auto", cost_style="priority", action_dims=8):
    """Construct the SimFreeMPC and its config. Returns (mpc, cfg).

    task_name + cost_style select the engine's builtin cost (e.g. cost_style="grasp_flow" ->
    the full grasp+lift+place GraspFlowStateCost). Drivers that replace mpc.cost (the base's
    CompositeCost) leave both at their defaults; the sim_free_mbd diagnostic passes them to run a
    builtin cost directly.
    """
    cfg = SimFreeMPCConfig(task_name=task_name, cost_style=cost_style, num_samples=num_samples,
                           iterations=iterations, noise=noise, temperature=temperature,
                           action_dims=action_dims, joint_delta_clip=joint_delta_clip, interpolate=interpolate)
    return SimFreeMPC(policy, cfg), cfg


def apply_horizon_basis(mpc, basis: str, knots: int):
    """Replace the linear knot-resample with a smoother basis; the cubic B-spline works best here.

    basis: linear | cubic (Catmull-Rom, overshoots) | bspline (approximating C2, no overshoot) | rbf.
    """
    if basis == "linear" and knots <= 0:
        return
    _cache = {}

    def _resample_matrix(n_in, n_out, device, dtype):
        B = _cache.get((n_in, n_out))
        if B is None:
            B = torch.zeros(n_out, n_in, dtype=torch.float64)
            if n_in == 1:
                B[:, 0] = 1.0
            elif basis == "rbf":  # KMPPI global RBF kernel: B = K(Hs,Tk) @ inv(K(Tk,Tk))
                Tk = torch.linspace(0, n_out - 1, n_in, dtype=torch.float64)
                Hs = torch.linspace(0, n_out - 1, n_out, dtype=torch.float64)
                sigma = (n_out - 1) / max(n_in - 1, 1)
                kf = lambda a, bb: torch.exp(-((a[:, None] - bb[None, :]) ** 2) / (2 * sigma ** 2))
                B = kf(Hs, Tk) @ torch.linalg.inv(kf(Tk, Tk) + 1e-6 * torch.eye(n_in, dtype=torch.float64))
            else:
                for h in range(n_out):
                    u = h * (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
                    i = min(max(int(u), 0), n_in - 1)
                    f = u - i
                    if basis == "cubic":  # Catmull-Rom (C1, interpolating -> overshoots)
                        taps = [(-1, -0.5 * f**3 + f**2 - 0.5 * f), (0, 1.5 * f**3 - 2.5 * f**2 + 1.0),
                                (1, -1.5 * f**3 + 2.0 * f**2 + 0.5 * f), (2, 0.5 * f**3 - 0.5 * f**2)]
                    elif basis == "bspline":  # uniform cubic B-spline (C2, approximating, convex-hull)
                        taps = [(-1, (1 - 3 * f + 3 * f**2 - f**3) / 6.0), (0, (4 - 6 * f**2 + 3 * f**3) / 6.0),
                                (1, (1 + 3 * f + 3 * f**2 - 3 * f**3) / 6.0), (2, f**3 / 6.0)]
                    else:  # linear (C0)
                        taps = [(0, 1.0 - f), (1, f)]
                    for off, w in taps:
                        B[h, min(max(i + off, 0), n_in - 1)] += w
            _cache[(n_in, n_out)] = B
        return B.to(device=device, dtype=dtype)

    def _basis_resample(sequence, output_horizon):
        n_in = sequence.shape[-2]
        if n_in == output_horizon:
            return sequence
        if n_in == 1:
            return sequence.expand(*sequence.shape[:-2], output_horizon, sequence.shape[-1])
        mat = _resample_matrix(n_in, output_horizon, sequence.device, sequence.dtype)
        return torch.einsum("oi,...id->...od", mat, sequence)

    if basis in ("cubic", "bspline", "rbf"):
        mpc._linear_resample = _basis_resample  # replace the linear resample with the smoother basis
    if knots > 0:
        mpc._interpolation_knot_count = lambda horizon: max(2, min(knots, horizon))


def guard_cost(cost):
    """Wrap a cost so NaN/inf map to a large finite value before the sampler's softmax."""

    class _Guarded:
        def __init__(self, c):
            self._c = c

        def __call__(self, **kw):
            return torch.nan_to_num(self._c(**kw), nan=1e12, posinf=1e12, neginf=1e12)

        def target(self, *a, **k):
            return self._c.target(*a, **k)

    return _Guarded(cost)


def apply_arm_only_smoothing(mpc):
    """Smooth the arm control points but keep the gripper channel (dim 7) sharp.

    The engine reduces the chunk to control points and B-spline-expands them; that smoothing keeps the
    arm stable but flattens the gripper's close into a curve the sampler cannot push past ~0.35. Expanding
    the gripper channel linearly instead lets it step closed at the grasp instant while the arm keeps the
    smooth basis. Assumes the engine's interpolation is on (so there is a control-point reduction to expand).
    """
    cls = type(mpc)

    def interp(sequence, output_horizon):
        smoothed = cls._bspline_resample(sequence, output_horizon)   # arm: smooth (stable)
        if sequence.shape[-1] > 7:
            sharp = cls._linear_resample(sequence, output_horizon)   # gripper: linear (can snap closed)
            smoothed = smoothed.clone()
            smoothed[..., 7] = sharp[..., 7]
        return smoothed

    mpc._interpolate_control_points = interp


def policy_inputs(E, state_stats, real_stats: bool):
    """Build the decode's policy_inputs (a normalized current-state so the unnormalize round-trips)."""
    st = torch.zeros(8, device=E.device, dtype=torch.float32)
    if real_stats:
        q01 = torch.as_tensor(state_stats.q01, device=E.device, dtype=torch.float32)
        q99 = torch.as_tensor(state_stats.q99, device=E.device, dtype=torch.float32)
        st[:7] = 2.0 * (E.q0() - q01[:7]) / (q99[:7] - q01[:7] + 1e-6) - 1.0
    else:
        st[:7] = E.q0()  # identity state stats: decode adds this back directly
    return {"state": st}


def plan_chunk(mpc, x_init, pin, ctx, *, mode, update, denoise_iters, score_scale, dt, it_start=0):
    """Produce the model-space chunk x_0 to decode and execute (reverse update, or DIAL mean)."""
    if mode == "mean":
        target, _, _ = mpc._optimize_chunk(x_init, pin, ctx)  # single DIAL optimize, no reverse update
        return target
    x = x_init
    N = denoise_iters
    for it in range(it_start, N):
        if update == "mbd_score":
            x, _ = mpc.step_mbd_score(x, pin, ctx, iteration=it, num_iterations=N, score_scale=score_scale)
        elif update == "score_space":
            x, _ = mpc.step_score_space(x, pin, ctx, step_scale=score_scale)
        elif update == "ddim":
            x, _ = mpc.step_ddim(x, pin, ctx, iteration=it, num_iterations=N, step_scale=score_scale)
        else:  # flow: step() returns flow = -(target - x)/dt -> x_next = target = x - flow*dt
            flow, _ = mpc.step(x, pin, ctx, dt=dt)
            x = x - flow * dt
    return x


def sdedit_warm_start(x_carry, it_start, denoise_iters, num_train_timesteps, H, device, noise=None):
    """SDEdit warm start: forward-diffuse x_carry to step it_start (sqrt(ab)*carry + sqrt(1-ab)*noise).

    The reverse loop then runs only the last denoise_iters - it_start steps, for temporal coherence.
    noise defaults to fresh Gaussian noise of shape [1, H, 8].
    """
    ab, _ = ddim_iteration_alphas(iteration=it_start, num_iterations=denoise_iters,
                                  num_train_timesteps=num_train_timesteps)
    ab_t = torch.tensor(float(ab), device=device, dtype=torch.float32)
    if noise is None:
        noise = torch.randn(1, H, 8, device=device)
    return torch.sqrt(ab_t) * x_carry + torch.sqrt(1.0 - ab_t) * noise


def read_subtask_flags(env):
    """Current env subtask_terms flags (task progress, e.g. grasp_pear / pear_on_scale).

    Returns {flag: bool} from the env's observation manager; empty when the env exposes no such group.
    """
    try:
        group = env.env.observation_manager.compute_group("subtask_terms")
        return {key: bool(val.detach().flatten()[0].item()) for key, val in group.items()}
    except Exception:
        return {}
