"""Weight-space PPS steering for the MBD base (prototype).

Instead of adding a proxy residual to the score AFTER the cost-weighted mean (additive PPS), fold the steer
INTO the softmax weights so it is part of the optimize (and therefore survives the inner-iteration
re-optimization that erases an additive correction):

    J_steer_i = ( ||a_i - x0_task||^2 - ||a_i - x0_ref||^2 ) / steer_bandwidth        # steer_bandwidth ~ 2 sigma_k^2
    w_i       = softmax( -J(a_i; o)/lam  -  gamma * J_steer_i )

x0_ref is the base's own unsteered clean estimate at this inner iteration (the plain-softmax mean of the SAME
candidates), so gamma=0 is exactly the base. Everything lives in the sampler's native control-point space
[K, D] (K knots), because that is what the DIAL optimize samples over.

The tilt is algebraically an exponential tilt of the MC weights along (x0_task - x0_ref): the ||a_i||^2 term
cancels, leaving  -gamma*J_steer_i = (gamma/bw) * a_i . (x0_task - x0_ref) + const .
"""
from __future__ import annotations

import glob
import json

import numpy as np
import torch


def make_weight_steer(x0_task_cp: torch.Tensor, gamma: float, bandwidth: float | None = None, stats=None):
    """Build the sampler hook (logits[N], samples[N,K,D], scale_view) -> logits[N].

    x0_task_cp: [K, D] control-point target (device/dtype are matched to `samples` inside the hook).
    bandwidth:  overrides 2*sigma_k^2 when given (the `steer_bandwidth` guardrail knob).
    stats:      when a list is passed, append per-call {ess, pull_norm, pull_cosine, N} for the offline logs.
    """
    g = float(gamma)

    @torch.no_grad()
    def hook(logits: torch.Tensor, samples: torch.Tensor, scale_view: torch.Tensor) -> torch.Tensor:
        tgt = x0_task_cp.to(device=samples.device, dtype=samples.dtype)
        w0 = torch.softmax(logits, dim=0)                       # base (unsteered) weights
        x0_ref = (w0[:, None, None] * samples).sum(dim=0)       # [K, D] base clean estimate this iteration
        d_task = ((samples - tgt[None]) ** 2).sum(dim=(1, 2))   # [N]
        d_ref = ((samples - x0_ref[None]) ** 2).sum(dim=(1, 2))  # [N]
        bw = float(bandwidth) if bandwidth is not None else 2.0 * float((scale_view ** 2).mean())
        out = logits - g * (d_task - d_ref) / max(bw, 1e-8)
        if stats is not None:
            w = torch.softmax(out, dim=0)
            x0_steer = (w[:, None, None] * samples).sum(dim=0)
            pull = x0_steer - x0_ref
            direction = tgt - x0_ref
            denom = float(pull.norm() * direction.norm()) + 1e-8
            stats.append({
                "ess": 1.0 / float((w ** 2).sum()),             # 1/sum(w_i^2), normalized weights
                "pull_norm": float(pull.norm()),                # ||x0_hat(gamma) - x0_hat(0)|| (this iteration)
                "pull_cosine": float((pull * direction).sum()) / denom,
                "N": int(samples.shape[0]),
            })
        return out

    return hook


def to_control_points(mpc, target_hd: torch.Tensor, active_dims: int) -> torch.Tensor:
    """Resample a full-horizon target [H, D] to the sampler's control-point space [K, active_dims]."""
    horizon = target_hd.shape[0]
    opt_horizon = mpc._interpolation_knot_count(horizon)
    return mpc._control_point_resample(target_hd[:, :active_dims], opt_horizon)


def load_oracle_targets(task: str, demos_path: str, labels_glob: str, horizon: int) -> dict[int, torch.Tensor]:
    """Per-subtask (phase = stage%3) oracle target: the mean demo action chunk for that phase, in model space.

    A constant-per-subtask target (the accepted simplification for this test). Returns {phase: [H, D]}.
    """
    import h5py

    from vlm_base.demo_ctx import demo_task_items
    from vlm_base.sim_free_core import _A_Q01, _A_Q99, _load_droid_norm_stats

    ref = np.load(sorted(glob.glob(labels_glob))[0])
    obj_names = [str(n) for n in ref["obj_names"]]
    extents = {n: np.asarray(ref["obs_obj_ext"][0][i], np.float32) for i, n in enumerate(obj_names)}
    loaded = _load_droid_norm_stats()
    a_q01, a_q99 = (loaded[0], loaded[1]) if loaded is not None else (_A_Q01, _A_Q99)
    with open("task_prompts.json", encoding="utf-8") as f:
        meta = json.load(f)[task]

    by_phase: dict[int, list] = {}
    with h5py.File(demos_path, "r") as f:
        root = f["data"] if "data" in f else f
        for key in root.keys():
            for feat, clean in demo_task_items(root[key], obj_names, extents, a_q01, a_q99, horizon,
                                               meta["grasp_objs"], meta["place_obj"]):
                by_phase.setdefault(int(feat["stage"]) % 3, []).append(np.asarray(clean, np.float32))
    out = {ph: torch.tensor(np.mean(np.stack(v), axis=0)) for ph, v in by_phase.items()}
    print(f"[weight_steer] oracle targets per phase: "
          f"{ {ph: tuple(t.shape) for ph, t in out.items()} }", flush=True)
    return out
