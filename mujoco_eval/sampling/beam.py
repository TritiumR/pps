"""Maintain and resample warm-start plan hypotheses across replans."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .. import paths

paths.ensure_repo_on_path()

from sim_free_mpc.fk import transform_points_wxyz  # noqa: E402


@dataclass(frozen=True)
class BeamConfig:
    """Configure beam hypotheses, warm starts, scoring, and resampling."""

    k: int = 1
    warm: float = 0.0
    resample_every: int = 0
    ema: float = 0.5
    w_rate: float = 1.0
    w_subgoal: float = 0.0
    stage_reset: bool = True


def _ee_goal(context):
    """Return the current end-effector position and payload-adjusted goal."""
    eef = np.asarray(context["eef_pos"], dtype=np.float64).reshape(3)
    target = np.asarray(context["target"], dtype=np.float64).reshape(3)
    payload = context.get("payload")
    objects = context.get("objects", {})

    if payload is not None and payload in objects:
        off = (
            np.asarray(objects[payload]["pos"], dtype=np.float64).reshape(3)
            - eef
        )
        return eef, target - off

    return eef, target


def _belief_ee(planner, plan, context):
    """Return the world-frame end-effector path for a decoded plan."""
    joints = torch.as_tensor(
        np.asarray(plan)[:, :7],
        dtype=torch.float32,
    ).unsqueeze(0)
    ee_pos = planner.fk.forward(joints).ee_pos
    root_pos = context.get("robot_root_pos")
    root_quat = context.get("robot_root_quat")

    if root_pos is not None and root_quat is not None:
        ee_pos = transform_points_wxyz(
            torch.as_tensor(root_pos, dtype=ee_pos.dtype),
            torch.as_tensor(root_quat, dtype=ee_pos.dtype),
            ee_pos,
        )

    return ee_pos[0].detach().cpu().numpy().astype(np.float64)


def _subgoal_residual(context, ee_end):
    """Return the subgoal residual at a candidate plan endpoint."""
    fn = context.get("constraint")
    keypoints = context.get("keypoints")

    if fn is None or keypoints is None:
        return None

    ee = torch.as_tensor(
        ee_end,
        dtype=torch.float32,
    ).reshape(1, 1, 3)
    kp = torch.as_tensor(
        np.asarray(keypoints),
        dtype=torch.float32,
    )[:, None, None, :]

    return float(torch.as_tensor(fn(ee, kp)).reshape(-1)[0])


class Hypothesis:
    """Store one warm plan, its source state, and accumulated credit."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.x_model = None
        self.state = None
        self.credit = 0.0
        self.seen = 0
        self.warm_used = False

    def observe(self, value, ema):
        self.credit = (
            value
            if self.seen == 0
            else (1.0 - ema) * self.credit + ema * value
        )
        self.seen += 1

    def adopt(self, other):
        self.x_model = (
            None
            if other.x_model is None
            else other.x_model.clone()
        )
        self.state = (
            None
            if other.state is None
            else other.state.clone()
        )
        self.credit = other.credit
        self.seen = other.seen


class Beam:
    """Manage warm-start hypotheses, scoring, and scheduled resampling."""

    def __init__(
        self,
        cfg: BeamConfig,
        planner,
        *,
        horizon: int,
        spi: int,
    ):
        if cfg.k < 1:
            raise ValueError(f"beam k must be >= 1, got {cfg.k}")
        if not 0.0 <= cfg.warm <= 1.0:
            raise ValueError(
                f"beam warm must be in [0,1], got {cfg.warm}"
            )

        self.cfg = cfg
        self.planner = planner
        self.horizon = int(horizon)
        self.spi = int(spi)
        self.hyps = [Hypothesis() for _ in range(cfg.k)]
        self._stage_key = None
        self._replans = 0

    def reset_episode(self):
        for hypothesis in self.hyps:
            hypothesis.clear()

        self._stage_key = None
        self._replans = 0

    def replan(self, infer, env, context, stage_key):
        """Extend all hypotheses and return the highest-valued plan."""
        state_now = torch.zeros(8, dtype=torch.float32)
        state_now[:7] = env.q0()

        if self.cfg.stage_reset and stage_key != self._stage_key:
            for hypothesis in self.hyps:
                hypothesis.clear()

        self._stage_key = stage_key

        eef, goal = _ee_goal(context)
        d_now = float(np.linalg.norm(eef - goal))
        plans = []
        stats = []
        values = []
        d_ends = []

        for hypothesis in self.hyps:
            out = {}
            plan, plan_stats, _ = infer(
                x0_init=self._init_tensor(
                    hypothesis,
                    state_now,
                ),
                out=out,
            )
            hypothesis.x_model = out["x_final"]
            hypothesis.state = state_now.clone()

            ee = _belief_ee(
                self.planner,
                plan,
                context,
            )
            d_end = float(np.linalg.norm(ee[-1] - goal))
            value = (
                -d_end
                + self.cfg.w_rate * (d_now - d_end)
            )

            if self.cfg.w_subgoal:
                residual = _subgoal_residual(
                    context,
                    ee[-1],
                )
                if residual is not None:
                    value -= self.cfg.w_subgoal * residual

            hypothesis.observe(value, self.cfg.ema)
            plans.append(plan)
            stats.append(plan_stats)
            values.append(value)
            d_ends.append(d_end)

        incumbent = int(np.argmax(values))
        self._replans += 1

        record = {
            "k": self.cfg.k,
            "incumbent": incumbent,
            "d_now": round(d_now, 4),
            "d_end": [
                round(value, 4)
                for value in d_ends
            ],
            "value": [
                round(value, 4)
                for value in values
            ],
            "credit": [
                round(float(hypothesis.credit), 4)
                for hypothesis in self.hyps
            ],
            "warm": sum(
                int(hypothesis.warm_used)
                for hypothesis in self.hyps
            ),
        }

        if (
            self.cfg.resample_every > 0
            and self.cfg.k > 1
            and self._replans % self.cfg.resample_every == 0
        ):
            record["resample"] = self._resample()

        return (
            plans[incumbent],
            stats[incumbent],
            record,
        )

    def _init_tensor(self, hypothesis, state_now):
        """Build the level-zero tensor for one hypothesis."""
        noise = torch.randn(
            1,
            self.horizon,
            8,
        )
        hypothesis.warm_used = False

        if (
            self.cfg.warm <= 0.0
            or hypothesis.x_model is None
        ):
            return noise

        self.planner.set_warm_action(
            hypothesis.x_model,
            state=hypothesis.state,
        )
        shifted, ok = self.planner.warm_start_noise(
            noise,
            shift_steps=self.spi,
            current_state=state_now,
        )
        self.planner.reset_action_warm()

        if not ok:
            return noise

        hypothesis.warm_used = True
        weight = float(self.cfg.warm)

        return (
            weight * shifted
            + math.sqrt(
                max(1.0 - weight * weight, 0.0)
            )
            * noise
        )

    def _resample(self):
        """Replace lower-credit hypotheses with copies of higher-credit plans."""
        order = sorted(
            range(self.cfg.k),
            key=lambda index: self.hyps[index].credit,
            reverse=True,
        )
        keep = order[: max(1, self.cfg.k // 2)]
        drop = order[len(keep):]

        for j, index in enumerate(drop):
            self.hyps[index].adopt(
                self.hyps[keep[j % len(keep)]]
            )

        return {
            "kept": keep,
            "dropped": drop,
        }