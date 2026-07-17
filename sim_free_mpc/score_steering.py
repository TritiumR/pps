from __future__ import annotations

from typing import Literal

import torch


ScoreSteeringMode = Literal["full", "task"]


def combine_scores(
    base_score: torch.Tensor,
    task_score: torch.Tensor,
    *,
    mode: ScoreSteeringMode,
    steer_scale: float,
    ref_score: torch.Tensor | None = None,
    base_scale: float = 1.0,
) -> torch.Tensor:
    """Combine score fields on one shared diffusion state."""
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

    return scaled_base + float(steer_scale) * residual
