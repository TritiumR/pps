from __future__ import annotations

from typing import Literal

import torch


ScoreSteeringMode = Literal["full", "task"]


def steer_scale_for_stage(
    cost_stage: str | None,
    *,
    default: float,
    grasp: float | None = None,
    lift: float | None = None,
    place: float | None = None,
) -> float:
    """Return the optional stage-specific score-steering scale."""
    stage_type = (cost_stage or "").split("_", 1)[0]
    override = {"grasp": grasp, "lift": lift, "place": place}.get(stage_type)
    return float(default if override is None else override)


def combine_scores(
    base_score: torch.Tensor,
    task_score: torch.Tensor,
    *,
    mode: ScoreSteeringMode,
    steer_scale: float | torch.Tensor,
    ref_score: torch.Tensor | None = None,
    base_scale: float = 1.0,
) -> torch.Tensor:
    """Combine score fields on one shared diffusion state.

    steer_scale may be a scalar or a per-dim tensor broadcastable against the score's
    last axis (per-channel gamma, e.g. a distinct gripper-channel gain)."""
    if base_score.shape != task_score.shape:
        raise ValueError(
            "base/task score shapes must match: "
            f"base={tuple(base_score.shape)}, task={tuple(task_score.shape)}."
        )

    scaled_base = float(base_scale) * base_score
    if mode == "full":
        if ref_score is None:
            raise ValueError("Full score steering requires a reference score.")
        if ref_score.shape != task_score.shape:
            raise ValueError(
                "task/ref score shapes must match: "
                f"task={tuple(task_score.shape)}, ref={tuple(ref_score.shape)}."
            )
        residual = task_score - ref_score
    elif mode == "task":
        if ref_score is not None:
            raise ValueError("Task score steering must not receive a reference score.")
        residual = task_score - scaled_base
    else:
        raise ValueError(f"Unknown score steering mode: {mode!r}.")

    if isinstance(steer_scale, torch.Tensor):
        scale = steer_scale.to(device=residual.device, dtype=residual.dtype)
    else:
        scale = float(steer_scale)
    return scaled_base + scale * residual
