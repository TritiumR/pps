"""Draw N proxy chunks per replan and execute the one a ranking picks (opt-in).

The expert path spends one denoise chain per replan and executes it unconditionally. Its ONLY
source of randomness is the initial noise x_T, so redrawing x_T is the whole candidate space --
there is no sampler, no cost and no clamp to add variation. This module makes that space
measurable and, behind `--expert_best_of`, selectable.

Two design points that keep the arm honest:

* Index 0 is the DEFAULT chain, fetched by the untouched `ProxySteering.begin_replan`. When the
  ranking abstains the executed chunk is byte-identical to `--steer expert`, so N=1 is an exact
  control rather than an approximate one.
* Indices >= 1 come from a batched host-side DDIM that replicates the server's `chain` update
  exactly (serve_mg_proxy_score.py, x0 branch), driven through `embed` + `score_batch` so N draws
  cost one prefix forward instead of N. `score_batch` autocasts to fp16 on CUDA where `chain` does
  not, so the extra draws carry ~1e-3 relative error against the fp32 path; that is far below the
  centimetre-scale spreads this is used to measure, but it is why index 0 is never redrawn here.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .. import paths
paths.ensure_repo_on_path()

from sim_free_mpc.ddim import ddim_iteration_alphas

_OPEN_APERTURE = 0.080
# Extra draws live in their own seed block, strided by replan. A naive base + replan + i collides
# with the NEXT replan's own draws, which would silently reuse one chunk across two replans.
_SEED_BLOCK = 1_000_003
_SEED_STRIDE = 1024
# Below this the candidates disagree by less than the controller's own tracking error, so the
# ranking has nothing to say; keep the default draw rather than churn on noise.
_ABSTAIN_M = 0.005
_DUP_RAD = 1e-4


def sample_seeds(base_seed, replan_idx, n):
    """Seeds for n draws; index 0 reproduces the default chain's own noise."""
    head = int(base_seed) * 100003 + int(replan_idx)
    return [head] + [int(base_seed) * 100003 + _SEED_BLOCK + int(replan_idx) * _SEED_STRIDE + i
                     for i in range(1, int(n))]


def observation(env):
    """Capture the proxy server's observation arguments for one replan."""
    return {"joint_pos": env.q0().numpy().astype(np.float32),
            "gripper_pos": float(np.clip(env.gripper_q() / _OPEN_APERTURE, 0.0, 1.0)),
            "table": env.rgb("agentview", hw=224),
            "wrist": env.rgb("robot0_eye_in_hand", hw=224)}


class BatchExpert:
    """Run the proxy's own reverse chain for a batch of initial noises."""

    def __init__(self, client):
        info = client.ready_info
        if info.get("action_norm") != "demo_delta":
            raise ValueError(f"best-of-N needs a demo_delta proxy; got {info.get('action_norm')!r}")
        self.client = client
        self.mean = np.asarray(info["action_mean_rows"], dtype=np.float32)
        self.std = np.asarray(info["action_std_rows"], dtype=np.float32)
        self.horizon = int(info["action_horizon"])
        self.dim = int(info["action_dim"])
        self.train_steps = int(info.get("ddim_num_train_timesteps", 100))
        self.last_wall_s = float("nan")

    def generators(self, seeds):
        """One CPU generator per draw, so a (seed, level) pair always yields the same noise."""
        return [torch.Generator().manual_seed(int(s)) for s in seeds]

    def noise(self, seeds, scale=1.0, gens=None):
        """Initial iterates for the given seeds, drawn exactly as the server draws them.

        `scale` widens x_T to N(0, scale^2). The model was trained on unit-variance x_T, so
        anything but 1.0 reads the score field off-distribution at level 0 -- which is the
        question the probe asks, not a bug.
        """
        gens = self.generators(seeds) if gens is None else gens
        out = np.empty((len(seeds), self.horizon, self.dim), dtype=np.float32)
        for i, gen in enumerate(gens):
            out[i] = torch.randn((1, self.horizon, self.dim), generator=gen).numpy()[0]
        return out * float(scale)

    def _level_noise(self, gens):
        """Fresh z ~ N(0, I) per draw for the stochastic-DDIM term."""
        out = np.empty((len(gens), self.horizon, self.dim), dtype=np.float32)
        for i, gen in enumerate(gens):
            out[i] = torch.randn((1, self.horizon, self.dim), generator=gen).numpy()[0]
        return out

    def draw(self, obs_kw, seeds, num_iterations, chunk=0, noise_scale=1.0, eta=0.0):
        """Return len(seeds) clean chunks as REAL absolute joints, [N, proxy_horizon, D].

        `noise_scale` and `eta` are the two widening knobs. eta is the standard DDIM
        stochasticity coefficient: sigma_t = eta * sqrt((1-a_prev)/(1-a)) * sqrt(1 - a/a_prev),
        with the deterministic direction shortened to sqrt(1 - a_prev - sigma_t^2) so the marginal
        is preserved. eta=0 recovers the server's own update exactly and never touches the
        generators, so the defaults stay bit-for-bit what they were.
        """
        t0 = time.perf_counter()
        obs, _ = self.client.embed(**obs_kw)
        gens = self.generators(seeds)
        y = self.noise(seeds, scale=noise_scale, gens=gens)
        eta = float(eta)
        x0 = None
        for it in range(int(num_iterations)):
            score = self.score(obs, y, it, num_iterations, chunk)
            alpha, alpha_prev = ddim_iteration_alphas(
                iteration=it, num_iterations=int(num_iterations),
                num_train_timesteps=self.train_steps)
            a, a_prev = float(alpha), float(alpha_prev)
            beta = max(1.0 - a, 1e-6)
            x0 = (y + beta * score) / math.sqrt(max(a, 1e-6))
            eps = -math.sqrt(beta) * score
            sigma = 0.0
            if eta > 0.0:
                sigma = eta * math.sqrt(max(1.0 - a_prev, 0.0) / beta) * math.sqrt(
                    max(1.0 - a / max(a_prev, 1e-12), 0.0))
            dir_c = math.sqrt(max(1.0 - a_prev - sigma ** 2, 0.0))
            y = math.sqrt(max(a_prev, 0.0)) * x0 + dir_c * eps
            if sigma > 0.0:
                y = y + sigma * self._level_noise(gens)
        real = x0 * (self.std[None] + 1e-6) + self.mean[None]
        real[..., :7] += np.asarray(obs_kw["joint_pos"], dtype=np.float32)[None, None, :7]
        self.last_wall_s = time.perf_counter() - t0
        return real.astype(np.float32)

    def score(self, obs, y, iteration, num_iterations, chunk=0):
        """One level's score field, split into request-sized blocks when N is large."""
        step = int(chunk) if chunk else y.shape[0]
        parts = [self.client.score_batch(obs=obs, iteration=int(iteration),
                                         num_iterations=int(num_iterations),
                                         x=y[lo:lo + step])[0]
                 for lo in range(0, y.shape[0], step)]
        return np.concatenate(parts, axis=0).astype(np.float32)


def tcp_paths(planner, ctx, joints):
    """World TCP positions for joint rows of any leading shape [..., >=7]."""
    q = torch.as_tensor(np.asarray(joints)[..., :7], dtype=torch.float32)
    ee = planner.fk.forward(q).ee_pos
    root_pos, root_quat = ctx.get("robot_root_pos"), ctx.get("robot_root_quat")
    if root_pos is not None and root_quat is not None:
        from sim_free_mpc.fk import transform_points_wxyz
        ee = transform_points_wxyz(
            torch.as_tensor(np.asarray(root_pos), dtype=ee.dtype),
            torch.as_tensor(np.asarray(root_quat), dtype=ee.dtype), ee)
    return ee.detach().cpu().numpy()


def phase(env, held, ctx):
    """Privileged progress reference: (carried object position or None, goal point).

    Phase-aware on purpose. Before the grasp the only progress that exists is toward the CAN, and
    ranking a pre-grasp chunk by its distance to the bin rewards exactly the mode error these
    episodes already fail in. Once the object is held the goal becomes the place point, and the
    quantity that has to arrive is the OBJECT, not the gripper.
    """
    if held is None:
        obj = ctx.get("grasp_obj") or ctx.get("payload")
        goal = env.object_pose(obj)[0] if obj else ctx["target"]
        return None, np.asarray(goal, dtype=np.float64)
    goal = ctx.get("place_point")
    goal = ctx["target"] if goal is None else goal
    return (np.asarray(env.object_pose(held)[0], dtype=np.float64),
            np.asarray(goal, dtype=np.float64))


def progress_scores(tcp_term, held_pos, tcp_now, goal):
    """Terminal distance to the phase goal; a held object rides the gripper rigidly."""
    pred = np.asarray(tcp_term, dtype=np.float64)
    if held_pos is not None:
        pred = pred + (np.asarray(held_pos, dtype=np.float64)
                       - np.asarray(tcp_now, dtype=np.float64))[None]
    return np.linalg.norm(pred - np.asarray(goal, dtype=np.float64)[None], axis=-1)


def select(scores, margin=_ABSTAIN_M):
    """argmin with an abstain margin, measured against the default draw."""
    best = int(np.argmin(scores))
    if float(scores[0] - scores[best]) < float(margin):
        return 0, True
    return best, False


def chunk_linf(chunks):
    """Upper-triangle pairwise L-infinity distances over the action joints of [N, H, D] draws."""
    a = np.asarray(chunks, dtype=np.float64)[..., :7]
    a = a.reshape(a.shape[0], -1)
    if a.shape[0] < 2:
        return np.zeros(0)
    d = np.abs(a[:, None, :] - a[None, :, :]).max(axis=-1)
    return d[np.triu_indices(a.shape[0], 1)]


def duplicate_count(chunks, radius=_DUP_RAD):
    """Pairs of draws that agree to `radius` in every joint of every action row."""
    d = chunk_linf(chunks)
    return int((d < float(radius)).sum())


def pairwise(points):
    """Upper-triangle pairwise Euclidean distances of [N, 3] points."""
    p = np.asarray(points, dtype=np.float64)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    iu = np.triu_indices(p.shape[0], 1)
    return d[iu]


def best_of_chunk(planner, env, ctx, args, bridge, steer):
    """Draw --expert_best_of chunks and execute the best-ranked one.

    Returns the runner's (plan, stats, rec) triple. `rec` is merged into the replan record the
    same way select_chunk's is, so a run can be audited for where selection changed the plan.
    """
    t0 = time.perf_counter()
    n = max(int(getattr(args, "expert_best_of", 0)), 1)
    sigma = float(getattr(args, "expert_noise_scale", 1.0))
    eta = float(getattr(args, "expert_ddim_eta", 0.0))
    wide = sigma != 1.0 or eta > 0.0
    steer.begin_replan(env, args.num_steps + 1)
    # Draw 0 is the server's fp32 chain when sampling is unmodified, so abstaining reproduces
    # --steer expert exactly. Under widening it has to be a WIDENED draw instead: otherwise a
    # no-harm arm at N=1 would still execute the unwidened default and measure nothing.
    chunks = [] if wide else [steer.expert_chunk()]
    draw_s = float("nan")
    if wide or n > 1:
        batch = getattr(steer, "_bon_batch", None)
        if batch is None:
            batch = steer._bon_batch = BatchExpert(steer.client)
        seeds = sample_seeds(steer.base_seed, steer.replan_idx - 1, n)
        seeds = seeds if wide else seeds[1:]
        extra = batch.draw(observation(env), seeds, args.num_steps + 1,
                           chunk=int(getattr(args, "expert_best_of_chunk", 0)),
                           noise_scale=sigma, eta=eta)
        chunks.extend(np.asarray(extra)[:, : args.horizon])
        draw_s = batch.last_wall_s
    chunks = np.stack(chunks).astype(np.float32)

    tcp = tcp_paths(planner, ctx, chunks)
    held = bridge.world.held() if bridge is not None else None
    held_pos, goal = phase(env, held, ctx)
    scores = progress_scores(tcp[:, -1], held_pos, env.tcp(), goal)
    idx, abstain = select(scores)
    spread = pairwise(tcp[:, -1]) if n > 1 else np.zeros(0)
    rec = {"bon_n": int(n), "bon_index": int(idx), "bon_abstain": bool(abstain),
           "bon_sigma": sigma, "bon_eta": eta,
           "bon_phase": ("held" if held_pos is not None else "free"),
           "bon_scores": [round(float(v), 4) for v in scores],
           "bon_gap": round(float(scores[0] - scores[idx]), 4),
           "bon_spread_med": round(float(np.median(spread)) if spread.size else 0.0, 4),
           "bon_spread_max": round(float(spread.max()) if spread.size else 0.0, 4),
           "bon_dupes": duplicate_count(chunks),
           "bon_draw_s": round(float(draw_s), 3),
           "bon_wall_s": round(time.perf_counter() - t0, 3)}
    return chunks[idx], {}, rec
