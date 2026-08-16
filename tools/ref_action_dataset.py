"""Teacher final-action view over a grouped MPC epsilon cache."""

from __future__ import annotations

import torch


class MPCActionChunkDataset(torch.utils.data.Dataset):
    """K clean teacher action chunks grouped under each cached observation.

    Every returned action is the state reached after the cache's ten MBD-score
    updates. Keeping K chunks grouped lets action-chunk B use the exact same number
    of observations and targets per optimizer step as score-cache A. Standard
    epsilon training re-noises every clean chunk independently on every visit.
    """

    def __init__(self, grouped_cache_dataset):
        self.grouped = grouped_cache_dataset
        self.data_config = grouped_cache_dataset.data_config
        self.metadata = grouped_cache_dataset.metadata
        self.demo_names = grouped_cache_dataset.unique_demo_names
        self.unique_demo_names = grouped_cache_dataset.unique_demo_names
        self.unique_step_indices = grouped_cache_dataset.unique_step_indices
        self.num_observations = grouped_cache_dataset.num_observations
        self.trajectories_per_observation = (
            grouped_cache_dataset.trajectories_per_observation
        )
        self.trajectory_observation_indices = (
            grouped_cache_dataset.trajectory_observation_indices
        )

    def __len__(self) -> int:
        return self.num_observations

    def __getitem__(self, observation_idx: int):
        inputs, states, _, _ = self.grouped[int(observation_idx)]
        clean_actions = states.reshape(
            self.trajectories_per_observation,
            self.grouped.labels_per_trajectory,
            *states.shape[1:],
        )
        return inputs, clean_actions[:, -1].to(torch.float32)
