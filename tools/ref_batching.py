"""Shared observation-grouped batching for ref-distillation A/B runs."""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class RefBatchLayout:
    global_target_batch: int
    global_observation_batch: int
    local_observation_batch: int
    targets_per_observation: int
    trajectories_per_observation: int
    world_size: int


def resolve_ref_batch_layout(
    *,
    global_target_batch: int,
    trajectories_per_observation: int,
    world_size: int,
    targets_per_observation: int | None,
) -> RefBatchLayout:
    """Resolve one identical grouped batch layout for score-cache A and action-chunk B."""
    k = int(trajectories_per_observation)
    targets = k if targets_per_observation is None else int(targets_per_observation)
    batch = int(global_target_batch)
    ranks = int(world_size)
    if k <= 0:
        raise ValueError("trajectories_per_observation must be positive.")
    if targets <= 0 or targets > k:
        raise ValueError(
            "targets_per_observation must be in [1, trajectories_per_observation]; "
            f"got targets={targets}, K={k}."
        )
    if batch <= 0 or batch % targets != 0:
        raise ValueError(
            "Global target batch must be divisible by targets_per_observation; "
            f"got batch={batch}, targets={targets}."
        )
    if ranks <= 0:
        raise ValueError("world_size must be positive.")
    global_observations = batch // targets
    if global_observations % ranks != 0:
        raise ValueError(
            "Global observation batch must be divisible by world_size so every DDP rank "
            "has the same loss weight; "
            f"got observations={global_observations}, world_size={ranks}."
        )
    return RefBatchLayout(
        global_target_batch=batch,
        global_observation_batch=global_observations,
        local_observation_batch=global_observations // ranks,
        targets_per_observation=targets,
        trajectories_per_observation=k,
        world_size=ranks,
    )


def sample_trajectory_indices(
    batch_size: int,
    trajectories_per_observation: int,
    targets_per_observation: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Choose distinct teacher trajectories for every observation in a batch."""
    return torch.stack(
        [
            torch.randperm(trajectories_per_observation, device=device)[
                :targets_per_observation
            ]
            for _ in range(batch_size)
        ]
    )


def sample_cached_label_indices(
    batch_size: int,
    trajectories_per_observation: int,
    labels_per_trajectory: int,
    targets_per_observation: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Choose one random cached scheduler level from each selected trajectory."""
    trajectory_indices = sample_trajectory_indices(
        batch_size,
        trajectories_per_observation,
        targets_per_observation,
        device=device,
    )
    level_indices = torch.randint(
        labels_per_trajectory,
        (batch_size, targets_per_observation),
        device=device,
    )
    return trajectory_indices * labels_per_trajectory + level_indices
