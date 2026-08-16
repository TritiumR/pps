import pytest
import torch

from tools.ref_batching import (
    resolve_ref_batch_layout,
    sample_cached_label_indices,
    sample_trajectory_indices,
)


def test_k8_batch32_is_four_observations_with_eight_targets_each():
    layout = resolve_ref_batch_layout(
        global_target_batch=32,
        trajectories_per_observation=8,
        world_size=1,
        targets_per_observation=None,
    )
    assert layout.global_observation_batch == 4
    assert layout.local_observation_batch == 4
    assert layout.targets_per_observation == 8


def test_grouped_batch_requires_exact_divisibility():
    with pytest.raises(ValueError, match="target batch"):
        resolve_ref_batch_layout(
            global_target_batch=33,
            trajectories_per_observation=8,
            world_size=1,
            targets_per_observation=8,
        )
    with pytest.raises(ValueError, match="world_size"):
        resolve_ref_batch_layout(
            global_target_batch=32,
            trajectories_per_observation=8,
            world_size=3,
            targets_per_observation=8,
        )


def test_cached_labels_take_one_level_from_each_distinct_trajectory():
    torch.manual_seed(7)
    indices = sample_cached_label_indices(
        batch_size=5,
        trajectories_per_observation=8,
        labels_per_trajectory=11,
        targets_per_observation=8,
    )
    assert indices.shape == (5, 8)
    assert torch.all((indices % 11) < 11)
    for row in indices:
        assert torch.equal(torch.sort(row // 11).values, torch.arange(8))


def test_action_chunks_choose_distinct_trajectories():
    indices = sample_trajectory_indices(3, 8, 4)
    assert indices.shape == (3, 4)
    assert all(len(torch.unique(row)) == 4 for row in indices)

def test_action_chunk_dataset_groups_k_final_states_by_observation():
    from tools.ref_action_dataset import MPCActionChunkDataset

    class FakeGrouped:
        data_config = object()
        metadata = {}
        unique_demo_names = ["a", "b"]
        unique_step_indices = [0, 4]
        num_observations = 2
        num_trajectories = 6
        trajectories_per_observation = 3
        labels_per_trajectory = 4
        trajectory_observation_indices = torch.tensor([0, 0, 0, 1, 1, 1])

        def __getitem__(self, observation_idx):
            states = (
                torch.arange(3 * 4 * 2, dtype=torch.float32)
                .reshape(3 * 4, 2, 1)
                + observation_idx * 100
            )
            return {"observation_idx": observation_idx}, states, None, None

    dataset = MPCActionChunkDataset(FakeGrouped())
    inputs, chunks = dataset[1]
    assert len(dataset) == 2
    assert inputs["observation_idx"] == 1
    assert chunks.shape == (3, 2, 1)
    expected = (
        torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2, 1)[:, -1]
        + 100
    )
    torch.testing.assert_close(chunks, expected)
