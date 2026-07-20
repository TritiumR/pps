"""Geometry-conditioned score proxy for score-space PPS (Config B).

The vlm_base base is geometry-driven: its MBD score depends on the ReKep ctx (object positions, target,
stage), not on camera pixels, because the policy is a norm-stats-only MockPolicy with no image encoder. So
the score-space PPS proxies condition on that same geometry. This is the faithful analog of the paper's
image-conditioned proxies for an image-conditioned base: a proxy conditions on whatever the base holds
fixed while it denoises.

The architecture mirrors ProxyScorePytorch's prefix/suffix/score-head split, with the image DINO prefix
replaced by a small geometry encoder. The DDIM schedule and score convention are shared with the engine
(sim_free_mpc.ddim), so s_ref and s_task live in the same score space as s_base and combine as
s = s_base + gamma * (s_task - s_ref) at every denoise step.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
from torch import nn

from sim_free_mpc.ddim import ddim_alphas_cumprod


@dataclasses.dataclass
class GeomProxyConfig:
    action_dim: int = 8
    action_horizon: int = 15
    obj_in_dim: int = 7          # per object: pos(3) + extents(3) + is-grasp-target(1)
    state_dim: int = 8           # pin: 7 joints padded to 8 (gripper slot)
    num_stages: int = 3          # phase embedding table size: 0=grasp, 1=lift, 2=place
    width: int = 256
    depth: int = 4
    heads: int = 8
    ddim_num_train_timesteps: int = 100
    score_scale: float = 1.0     # score mode only: the model output is scaled by this
    predict: str = "score"       # "score" (regress s_base) or "x0" (predict the clean action, derive score)


def time_cond_for(iteration, num_iterations, num_train_timesteps=100):
    """Normalized DDIM time in [0,1] for a reverse iteration, matching the engine's timestep mapping."""
    step_ratio = int(num_train_timesteps) // int(num_iterations)
    timestep = (int(num_iterations) - 1 - int(iteration)) * step_ratio
    return timestep / max(float(num_train_timesteps - 1), 1.0)


def features_from_geom(obj_pos, obj_ext, grasp_idx, target, eef, stage, joints):
    """Assemble the model inputs from raw ctx geometry, as numpy arrays.

    Called by both the trainer (per recorded chunk) and the eval-time combine (per live ctx), so train and
    eval see identical features. Object positions and the target are centered on the current eef, which makes
    the conditioning translation invariant; the absolute eef is kept as its own token for height/workspace
    context. grasp_idx flags which object is the current grasp target.

    ``stage`` is the linear grounding stage index, reduced here to a phase (grasp/lift/place = stage % 3,
    since GTGrounding lays out three stages per object). Phase + grasp_idx are order independent, so a demo
    that grasps objects in a different order than the base still shares this conditioning with the reference.
    """
    obj_pos = np.asarray(obj_pos, np.float32)
    eef = np.asarray(eef, np.float32).reshape(3)
    is_grasp = np.zeros((obj_pos.shape[0], 1), np.float32)
    if 0 <= int(grasp_idx) < obj_pos.shape[0]:
        is_grasp[int(grasp_idx)] = 1.0
    obj_feats = np.concatenate([obj_pos - eef, np.asarray(obj_ext, np.float32), is_grasp], axis=1)
    state = np.zeros(8, np.float32)
    state[:7] = np.asarray(joints, np.float32).reshape(-1)[:7]
    return {
        "obj_feats": obj_feats.astype(np.float32),
        "target": (np.asarray(target, np.float32).reshape(3) - eef).astype(np.float32),
        "eef": eef.astype(np.float32),
        "stage": np.int64(int(stage) % 3),   # phase: grasp/lift/place, order independent (see docstring)
        "state": state,
    }


def _sinusoidal_time_embed(time, dim, min_period=4e-3, max_period=4.0):
    """[B] time in [0,1] -> [B, dim] sinusoidal embedding (same periods as ProxyScorePytorch)."""
    if dim % 2 != 0:
        raise ValueError(f"time embed dim ({dim}) must be even")
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    ang = time[:, None].to(torch.float32) / period[None, :] * 2 * math.pi
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class GeomScoreProxy(nn.Module):
    """Predicts the denoise score at (x_t, t) given the ReKep geometry the base conditions on.

    forward inputs (batch-first):
      obj_feats [B,K,obj_in_dim], target [B,3], eef [B,3], stage [B] long, state [B,state_dim],
      x_t [B,H,action_dim], time [B] in [0,1].
    Prefix tokens = one per object + a target token + an eef token + a stage token; suffix tokens = a state
    token + H action-time tokens. A bidirectional transformer mixes them; the H suffix outputs are the score.
    """

    def __init__(self, config: GeomProxyConfig):
        super().__init__()
        self.config = config
        w = config.width

        self.obj_proj = nn.Sequential(nn.Linear(config.obj_in_dim, w), nn.SiLU(), nn.Linear(w, w))
        self.target_proj = nn.Linear(3, w)
        self.eef_proj = nn.Linear(3, w)
        self.stage_emb = nn.Embedding(config.num_stages, w)
        self.state_proj = nn.Linear(config.state_dim, w)

        self.action_in = nn.Linear(config.action_dim, w)
        self.time_mlp = nn.Sequential(nn.Linear(2 * w, w), nn.SiLU(), nn.Linear(w, w))

        layer = nn.TransformerEncoderLayer(
            d_model=w, nhead=config.heads, dim_feedforward=4 * w, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.score_out = nn.Linear(w, config.action_dim)

    def _prefix(self, obj_feats, target, eef, stage):
        obj_tok = self.obj_proj(obj_feats)                                    # [B,K,w]
        extra = torch.stack([self.target_proj(target), self.eef_proj(eef),
                             self.stage_emb(stage)], dim=1)                   # [B,3,w]
        return torch.cat([obj_tok, extra], dim=1)                            # [B,K+3,w]

    def _suffix(self, state, x_t, time):
        w = self.config.width
        state_tok = self.state_proj(state)[:, None, :]                        # [B,1,w]
        act = self.action_in(x_t)                                            # [B,H,w]
        temb = _sinusoidal_time_embed(time, w)[:, None, :].expand_as(act)     # [B,H,w]
        act_time = self.time_mlp(torch.cat([act, temb], dim=2))              # [B,H,w]
        return torch.cat([state_tok, act_time], dim=1)                       # [B,1+H,w]

    def _head(self, obj_feats, target, eef, stage, state, x_t, time):
        prefix = self._prefix(obj_feats, target, eef, stage)
        suffix = self._suffix(state, x_t, time)
        h = self.encoder(torch.cat([prefix, suffix], dim=1))
        return self.score_out(h[:, -self.config.action_horizon:])

    def _alpha_bar(self, time):
        """alpha_bar for a batch of DDIM times in [0,1] (x0 mode, to derive the score)."""
        alphas = torch.as_tensor(ddim_alphas_cumprod(self.config.ddim_num_train_timesteps),
                                 device=time.device, dtype=torch.float32)
        idx = torch.round(time.clamp(0.0, 1.0) * (alphas.shape[0] - 1)).long()
        return alphas[idx]

    def predict_score(self, obj_feats, target, eef, stage, state, x_t, time):
        """Score [B,H,action_dim] in the engine's convention (used by sampling and the eval combine).

        x0 mode predicts the clean action (a bounded, well-conditioned target) and derives the score from it,
        so the score is accurate even at low noise where a direct score target would blow up (paper Sec. 3).
        """
        out = self._head(obj_feats, target, eef, stage, state, x_t, time)
        if self.config.predict == "x0":
            a = self._alpha_bar(time)[:, None, None]
            return (torch.sqrt(a) * out - x_t) / torch.clamp(1.0 - a, min=1e-3)  # flow_eps, matches engine
        return out * self.config.score_scale

    def loss(self, batch, *, mode):
        """Per-element MSE. reference: regress recorded s_base. task: x0- or score-match demo actions."""
        cfg = self.config
        obj, tgt, eef, stage, state = (batch["obj_feats"], batch["target"], batch["eef"],
                                       batch["stage"], batch["state"])
        if mode == "reference":
            x_t, time, s_base = batch["x"], batch["time"], batch["score"]
            if cfg.predict == "x0":
                a = self._alpha_bar(time)[:, None, None]
                beta = torch.clamp(1.0 - a, min=1e-3)
                x0_target = (x_t + beta * s_base) / torch.sqrt(a)   # base clean action from recorded (x, s_base, t)
                pred = self._head(obj, tgt, eef, stage, state, x_t, time)
                return torch.nn.functional.mse_loss(pred, x0_target, reduction="none")
            pred = self.predict_score(obj, tgt, eef, stage, state, x_t, time)
            return torch.nn.functional.mse_loss(pred, s_base, reduction="none")
        if mode != "task":
            raise ValueError(f"unknown mode {mode!r}")
        actions = batch["actions"][:, :, :cfg.action_dim]
        time, alpha = _sample_alpha(actions.shape[0], cfg, actions.device)
        noise = torch.randn_like(actions)
        beta = torch.clamp(1.0 - alpha, min=1e-6)
        sqrt_alpha, sqrt_beta = alpha.clamp(min=1e-6).sqrt(), beta.sqrt()
        x_t = sqrt_alpha[:, None, None] * actions + sqrt_beta[:, None, None] * noise
        if cfg.predict == "x0":
            pred = self._head(obj, tgt, eef, stage, state, x_t, time)
            return torch.nn.functional.mse_loss(pred, actions, reduction="none")   # bounded target
        target_score = -noise / sqrt_beta[:, None, None]
        pred = self.predict_score(obj, tgt, eef, stage, state, x_t, time)
        return torch.nn.functional.mse_loss(pred, target_score, reduction="none")


def _sample_alpha(bsize, cfg: GeomProxyConfig, device):
    """Random DDIM (time, alpha_bar) pair for the task score-matching path."""
    alphas = torch.tensor(ddim_alphas_cumprod(cfg.ddim_num_train_timesteps), device=device, dtype=torch.float32)
    idx = torch.randint(0, alphas.shape[0], (bsize,), device=device)
    time = idx.to(torch.float32) / max(float(alphas.shape[0] - 1), 1.0)
    return time, alphas[idx]


def load_geom_proxy(path, device="cuda"):
    """Load a trained GeomScoreProxy from a train_geom_proxy checkpoint (state_dict + config)."""
    ckpt = torch.load(path, map_location=device)
    model = GeomScoreProxy(GeomProxyConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


class GeomSteer:
    """Score-space PPS steering term s_task - s_ref, evaluated per denoise iteration (paper Eq. 4).

    Built once per rollout from the two trained proxies. set_chunk() binds the current chunk's geometry
    (a features_from_geom dict); correction() then returns gamma * (s_task - s_ref) for a denoise state
    (x_t, time), which the engine adds to its own s_base before the reverse step. At gamma=0 the term is
    exactly zero, so the base runs unchanged.
    """

    def __init__(self, ref_model, task_model, gamma, steer_step=0.0, only_task=False, device="cuda",
                 mode="additive", oracle_targets=None, bandwidth=None, stats=None):
        # ref_model None means base-as-reference: s_ref := the live s_base (their Alt #1), so s = (1-g)
        # s_base + g s_task. only_task rolls the task proxy out as the policy (no base).
        # mode="weight": weight-space PPS (fold the steer into the optimize's softmax); uses oracle_targets
        # {phase: [H,D]} and needs no ref/task proxies (x0_ref is the base's own per-iteration mean).
        self.ref = ref_model.eval() if ref_model is not None else None
        self.task = task_model.eval() if task_model is not None else None
        if (self.ref is not None and self.task is not None
                and self.ref.config.predict != self.task.config.predict):
            raise ValueError(
                f"reference/task predict modes differ ({self.ref.config.predict} vs "
                f"{self.task.config.predict}); the score-space combine requires them to match")
        self.gamma = float(gamma)
        self.steer_step = float(steer_step)   # only steer while denoise time >= this (protect convergence)
        self.only_task = bool(only_task)
        self.device = device
        self.mode = mode                      # "additive" (score-space) | "weight" (importance-reweighting)
        self.oracle_targets = oracle_targets  # weight mode: {phase: [H,D]} demo targets
        self.bandwidth = bandwidth            # weight mode: overrides 2*sigma^2 (steer_bandwidth)
        self.stats = stats                    # weight mode: optional list, per-call ESS/pull logging
        self._feat = None
        self._phase = 0

    @torch.no_grad()
    def task_score(self, x_t, time_scalar) -> torch.Tensor:
        f = self._feat
        t = torch.as_tensor([float(time_scalar)], device=self.device, dtype=torch.float32)
        args = (f["obj_feats"], f["target"], f["eef"], f["stage"], f["state"], x_t, t)
        return self.task.predict_score(*args)

    def set_chunk(self, feat: dict):
        self._feat = {k: torch.as_tensor(v, device=self.device)[None] for k, v in feat.items()}
        self._phase = int(torch.as_tensor(feat["stage"]).reshape(-1)[0].item()) % 3   # subtask phase (weight mode)

    @torch.no_grad()
    def weight_hook(self, mpc, active_dims, x_t=None, time=None):
        """Weight-space PPS hook: x0_task from the learned task proxy (self.task, per-state) if set, else the
        per-phase oracle. None if no target is available."""
        from vlm_base.weight_steer import make_weight_steer, to_control_points
        if self.task is not None and x_t is not None:
            f = self._feat
            t = torch.as_tensor([float(time)], device=self.device, dtype=torch.float32)
            tgt = self.task._head(f["obj_feats"], f["target"], f["eef"], f["stage"], f["state"], x_t, t)[0]
        else:
            tgt = self.oracle_targets.get(self._phase) if self.oracle_targets else None
        if tgt is None:
            return None
        x0_task_cp = to_control_points(mpc, tgt.to(self.device), active_dims)
        return make_weight_steer(x0_task_cp, self.gamma, self.bandwidth, self.stats)

    @torch.no_grad()
    def correction(self, x_t, time_scalar, s_base=None) -> torch.Tensor:
        f = self._feat
        t = torch.as_tensor([float(time_scalar)], device=self.device, dtype=torch.float32)
        args = (f["obj_feats"], f["target"], f["eef"], f["stage"], f["state"], x_t, t)
        s_ref = s_base if self.ref is None else self.ref.predict_score(*args)
        return self.gamma * (self.task.predict_score(*args) - s_ref)
