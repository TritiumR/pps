"""Per-replan steering authority: how much of the keypose guidance to apply, and when.

The endpoint_particles arm applies its guidance at every replan unconditionally. On coffee that
converts four episodes the expert loses and loses eight the expert wins, so the arm is not weak --
it is INDISCRIMINATE. This module supplies the missing decision: a scalar lambda in [0, 1] per
replan, read off a state signal that is computable at rollout time from the frozen proxy alone.

lambda semantics, fixed here and consumed by runner.keypose_fk_chunk:

* lambda = 0 -- the replan short-circuits to the proxy's OWN default chain (the `--steer expert`
  plan) and none of the particle machinery runs. This is an exact identity, not an approximation:
  the chain is fetched with the same server-side seed the expert arm uses, and skipping the
  particle draw leaves the global RNG stream untouched.
* 0 < lambda < 1 -- the KL-capped goal-row displacement is scaled by lambda, and the Feynman-Kac
  log-potential that resamples particles is scaled by the same lambda (an exponent tempering:
  lambda -> 0 flattens the potential toward uniform).
* lambda = 1 -- every expression reduces to `x * 1.0`, so the arm is bitwise endpoint_particles.

The gap between lambda = 0 and lambda = 0+ is deliberate and reported, not hidden: one chain of
the proxy is a different estimator from six resampled particles, and no continuous knob bridges
them. The binary rule R1 only ever asks for the two endpoints; the ramp R2 is the honest
continuous relaxation of the tilt, with the same discontinuity at the origin.

Two candidate signals, both frozen-proxy-only (no privileged state, no outcome labels):

* `kp_dispersion` (U) -- draw K chunks from the proxy varying ONLY the initial noise, take each
  draw's keypose row, FK it to a world EE position, and report the mean pairwise distance in mm.
  It measures the proxy's own disagreement about where the phase ends. High U = the proxy has no
  committed opinion, so a cost that does is worth listening to.
* `plan_cost` (C) -- the planner's task-bucket cost of the proxy's DEFAULT plan at this state.
  High C = the plan the proxy would execute is one the cost dislikes. Related to, but not the
  same as, VLS's relative stage-reward improvement: this is the level of the current plan's cost,
  not a delta against a reference plan. It is a competitor baseline here, not a VLS reproduction.

Both are z-scored per bridge stage against constants frozen on the calibration seeds, so a stage
whose cost is large for structural reasons does not monopolise the gate.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from .best_of import BatchExpert, observation, pairwise, tcp_paths
from .chunk_cost import chunk_costs
from .keypose_fk import cost_weights

# Dispersion draws live in their own seed block so they can never collide with best_of's extra
# draws or with the server's own per-replan chain seed.
_SEED_BLOCK = 7_700_017
_SEED_STRIDE = 4096

SIGNALS = ("kp_dispersion", "plan_cost")


def dispersion_seeds(base_seed, replan_idx, k):
    """K distinct seeds for the dispersion draw at one replan."""
    head = int(base_seed) * 100003 + _SEED_BLOCK + int(replan_idx) * _SEED_STRIDE
    return [head + i for i in range(int(k))]


def default_chain(client, *, seed, num_iterations, env):
    """The proxy's own default plan at this state, as real absolute joints [H_proxy, D].

    Same call, same seed convention and same observation as ProxySteering.begin_replan, so a
    lambda = 0 replan executes byte-identically to the `--steer expert` arm.
    """
    chain, server_s = client.chain(
        seed=int(seed), num_iterations=int(num_iterations),
        joint_pos=env.q0().numpy().astype(np.float32),
        gripper_pos=float(np.clip(env.gripper_q() / 0.080, 0.0, 1.0)),
        table=env.rgb("agentview", hw=224), wrist=env.rgb("robot0_eye_in_hand", hw=224))
    return np.asarray(chain[-1], dtype=np.float32), float(server_s)


def keypose_dispersion(batch, planner, env, ctx, *, seeds, num_iterations, keypose_row=None):
    """Mean pairwise distance (mm) between the keypose EE positions of K noise-only redraws.

    One batched prefix forward covers all K draws, so the cost is one extra chain per replan
    rather than K of them.
    """
    t0 = time.perf_counter()
    chunks = batch.draw(observation(env), seeds, int(num_iterations))
    row = int(chunks.shape[1] - 1 if keypose_row is None else keypose_row)
    tcp = tcp_paths(planner, ctx, chunks)[:, row]
    d = pairwise(tcp)
    return {"u_mm": float(d.mean() * 1000.0) if d.size else 0.0,
            "u_max_mm": float(d.max() * 1000.0) if d.size else 0.0,
            "u_k": int(len(seeds)), "u_row": row,
            "u_wall_s": round(time.perf_counter() - t0, 3)}


def plan_cost(planner, real_chunk, ctx):
    """Task-bucket cost of one real joint chunk, plus the two neighbouring buckets.

    `task` is the signal the calibration freezes on. `task_no_ch` is what the kfk arm ranks
    proposals with, and `total` adds feasibility; both are carried so the choice is auditable
    rather than asserted.
    """
    out = {}
    chunk = np.asarray(real_chunk, dtype=np.float32)[None]
    for bucket in ("task", "task_no_ch", "total"):
        out[f"c_{bucket}"] = float(np.asarray(chunk_costs(planner, chunk, ctx,
                                                          bucket=bucket)).reshape(-1)[0])
    return out


# D's proposal cloud, in units of the proxy's own per-row action std. The endpoint arm's
# flow_matched schedule sweeps sigma from 40 down to 0.01 across its 11 levels; the first three
# levels are clipped by the +/-4 support box and the last three are narrower than the controller's
# own tracking error. These three sample the usable middle. `mid` is the pre-designated primary --
# the geometric centre of the unclipped range -- and the other two are the sensitivity check.
D_SIGMAS = {"tight": 0.17, "mid": 0.50, "wide": 1.29}


def cost_preferred_goal(planner, ctx, chunk, *, action_rows, keypose_row, sigma, std_rows,
                        n_proposals, temperature, generator, bucket="task_no_ch"):
    """Cost-weighted mean of a proposal cloud on the goal rows, in REAL joint space.

    The same operator the endpoint arm applies per denoise level, evaluated once on the proxy's
    FINAL clean chunk. Drawing in real space rather than model space is exact, not an
    approximation: decode is affine, so a Gaussian of width `sigma` in normalised space is a
    Gaussian of width `sigma * std_rows` in joints. The scale comes from the PROXY's own
    action_std_rows because that is the only per-row normalisation defined over the whole 21-row
    chunk -- the planner's stops at its execution horizon, which is exactly where the goal block
    begins.
    """
    goal = slice(int(action_rows), int(keypose_row) + 1)
    base = torch.as_tensor(np.asarray(chunk), dtype=torch.float32)
    scale = float(sigma) * torch.as_tensor(np.asarray(std_rows)[goal], dtype=torch.float32)
    noise = torch.randn(int(n_proposals), *base[goal].shape, generator=generator)
    cloud = base.unsqueeze(0).repeat(int(n_proposals), 1, 1)
    cloud[:, goal] = base[goal].unsqueeze(0) + scale.unsqueeze(0) * noise
    costs = chunk_costs(planner, cloud.numpy(), ctx, bucket=bucket,
                        keypose_row=int(keypose_row), action_rows=int(action_rows))
    weights, ess, _ = cost_weights(costs, temperature)
    return (weights.view(-1, 1, 1) * cloud).sum(dim=0), ess, costs


def semantic_disagreement(planner, env, ctx, chunk, *, action_rows, keypose_row, std_rows,
                          n_proposals, temperature, seed, sigmas=None):
    """How far the COST wants the keypose moved from where the PROXY put it, in world mm.

    U asks whether the proxy is uncertain. D asks whether it is confidently pointed somewhere the
    cost disagrees with -- a different failure, and the one a steering gate would actually want to
    fire on. `cos` is the angle between the two keypose bearings as seen from the arm's current
    TCP, so a large D that is merely "further along the same line" is separable from a large D
    that is "a different place entirely". `ess` says whether the cost had an opinion at all.
    """
    t0 = time.perf_counter()
    sigmas = D_SIGMAS if sigmas is None else sigmas
    chunk = np.asarray(chunk, dtype=np.float32)
    row = int(keypose_row)
    tcp_now = np.asarray(env.tcp(), dtype=np.float64).reshape(3)
    proxy_ee = np.asarray(tcp_paths(planner, ctx, chunk[None]), dtype=np.float64)[0, row]
    out = {}
    for name, sigma in sigmas.items():
        gen = torch.Generator().manual_seed(int(seed) * 7717 + int(round(sigma * 1000)))
        pref, ess, _ = cost_preferred_goal(
            planner, ctx, chunk, action_rows=action_rows, keypose_row=row, sigma=sigma,
            std_rows=std_rows, n_proposals=n_proposals, temperature=temperature, generator=gen)
        pref_ee = np.asarray(tcp_paths(planner, ctx, pref.numpy()[None]), dtype=np.float64)[0, row]
        u, v = proxy_ee - tcp_now, pref_ee - tcp_now
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        out[f"d_{name}_mm"] = float(np.linalg.norm(pref_ee - proxy_ee) * 1000.0)
        out[f"d_{name}_cos"] = float(u @ v / denom) if denom > 1e-12 else float("nan")
        out[f"d_{name}_ess"] = round(float(ess), 2)
    out["d_reach_mm"] = float(np.linalg.norm(proxy_ee - tcp_now) * 1000.0)
    out["d_wall_s"] = round(time.perf_counter() - t0, 3)
    return out


class AuthorityRule:
    """A frozen signal -> lambda map: per-stage z-score constants, then a gate or a ramp."""

    def __init__(self, spec):
        self.signal = str(spec["signal"])
        self.norm = {int(k): (float(v[0]), float(v[1])) for k, v in spec["stage_norm"].items()}
        self.fallback = (float(spec["fallback_norm"][0]), float(spec["fallback_norm"][1]))
        self.tau = float(spec["tau"])
        self.s_lo = float(spec["s_lo"])
        self.s_hi = float(spec["s_hi"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(open(path).read()))

    def z(self, value, stage_idx):
        """Stage-normalised signal. Unseen stages fall back to the pooled constants."""
        mu, sd = self.norm.get(int(stage_idx), self.fallback)
        return (float(value) - mu) / max(sd, 1e-9)

    def lam(self, value, stage_idx, mode):
        """Authority for one replan, from the normalised signal."""
        z = self.z(value, stage_idx)
        if mode == "gate":
            return (1.0 if z > self.tau else 0.0), z
        if mode == "ramp":
            span = max(self.s_hi - self.s_lo, 1e-9)
            return float(np.clip((z - self.s_lo) / span, 0.0, 1.0)), z
        raise ValueError(f"AuthorityRule has no mode {mode!r}")


def stage_norm(values, stages, min_count=8):
    """Per-stage (mean, std) of a calibration signal, plus the pooled fallback.

    A stage with fewer than `min_count` observations cannot support its own constants, so it is
    left out and resolves through the fallback at rollout time.
    """
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(stages, dtype=int)
    norm = {}
    for k in np.unique(s):
        sel = v[s == k]
        if len(sel) >= int(min_count):
            norm[int(k)] = (float(sel.mean()), float(sel.std(ddof=1) + 1e-9))
    return norm, (float(v.mean()), float(v.std(ddof=1) + 1e-9))


def zscore(values, stages, norm, fallback):
    """Apply frozen per-stage constants to a signal vector."""
    out = np.empty(len(values), dtype=np.float64)
    for i, (v, k) in enumerate(zip(values, stages)):
        mu, sd = norm.get(int(k), fallback)
        out[i] = (float(v) - mu) / max(sd, 1e-9)
    return out
