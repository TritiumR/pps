from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


OPENPI_SRC = Path(__file__).resolve().parents[1] / "openpi" / "src"
if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))

from openpi.training.subphase_keypose_dataset import (  # noqa: E402
    SubphaseKeyposeDataset,
    executable_action_horizon,
)


class _RawDataset:
    def __init__(self, episode_indices: list[int], frame_indices: list[int]):
        self.columns = {
            "episode_index": episode_indices,
            "frame_index": frame_indices,
        }

    def __getitem__(self, key: str):
        return self.columns[key]


class _ActionPrefixDataset:
    def __init__(
        self,
        episode_indices: list[int],
        frame_indices: list[int],
        *,
        horizon: int,
        action_dim: int,
    ):
        self.hf_dataset = _RawDataset(episode_indices, frame_indices)
        self.horizon = horizon
        self.action_dim = action_dim

    def __len__(self) -> int:
        return len(self.hf_dataset["episode_index"])

    def __getitem__(self, index: int) -> dict:
        start = 10 * index
        actions = np.arange(
            start,
            start + self.horizon * self.action_dim,
            dtype=np.float32,
        ).reshape(self.horizon, self.action_dim)
        return {"actions": actions, "source_index": np.asarray(index)}


class _TransformedDataset:
    """Match the private source layout used by OpenPI's TransformedDataset."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        return {**self._dataset[index], "prompt": "open the lid"}


def _write_sidecar(
    path: Path,
    *,
    episode_ends: list[int],
    candidates: list[list[int]],
    policy_qpos: np.ndarray,
    episode_names: list[str] | None = None,
) -> None:
    width = max(len(row) for row in candidates)
    indices = np.full((len(candidates), width), -1, dtype=np.int64)
    mask = np.zeros_like(indices, dtype=bool)
    for row_index, row in enumerate(candidates):
        indices[row_index, : len(row)] = row
        mask[row_index, : len(row)] = True
    np.savez_compressed(
        path,
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
        episode_names=np.asarray(
            episode_names
            if episode_names is not None
            else [f"demo_{index}" for index in range(len(episode_ends))]
        ),
        future_qpos_indices=indices,
        future_qpos_mask=mask,
        policy_qpos=np.asarray(policy_qpos, dtype=np.float32),
    )


def test_appends_strictly_future_keypose_without_requiring_it_after_actions(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "keyposes.npz"
    policy_qpos = np.stack(
        [np.asarray([frame, 100 + frame], dtype=np.float32) for frame in range(6)]
    )
    # For row 0, the sole keypose is frame 1. It follows the current qpos but
    # precedes the end of the three-action prefix.
    candidates = [[1], [1, 2, 3], [2, 3], [3, 4, 5], [4, 5], [5]]
    _write_sidecar(
        sidecar,
        episode_ends=[6],
        candidates=candidates,
        policy_qpos=policy_qpos,
    )
    base = _ActionPrefixDataset(
        [0] * 6,
        list(range(6)),
        horizon=3,
        action_dim=2,
    )

    dataset = SubphaseKeyposeDataset(base, sidecar, seed=0)

    assert len(dataset) == 5
    sample = dataset[0]
    assert sample["actions"].shape == (4, 2)
    np.testing.assert_array_equal(sample["actions"][:3], base[0]["actions"])
    np.testing.assert_array_equal(sample["actions"][-1], policy_qpos[1])


def test_filters_past_candidates_and_never_crosses_episode_boundary(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "keyposes.npz"
    policy_qpos = np.stack(
        [np.asarray([frame, -frame], dtype=np.float32) for frame in range(8)]
    )
    candidates = [
        [2, 3],
        [2, 3],
        [2, 3],
        [3],
        [6, 7],
        [6, 7],
        [6, 7],
        [7],
    ]
    _write_sidecar(
        sidecar,
        episode_ends=[4, 8],
        candidates=candidates,
        policy_qpos=policy_qpos,
    )
    base = _ActionPrefixDataset(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 2, 3, 0, 1, 2, 3],
        horizon=2,
        action_dim=2,
    )

    dataset = SubphaseKeyposeDataset(base, sidecar, seed=4)

    # Terminal phase rows 3 and 7 have no candidate strictly after them.
    assert dataset.valid_dataset_indices.tolist() == [0, 1, 2, 4, 5, 6]
    for wrapper_index, source_index in enumerate(
        dataset.valid_dataset_indices.tolist()
    ):
        sample = dataset[wrapper_index]
        keypose_frame = int(sample["actions"][-1, 0])
        current_global = (
            source_index
            if source_index < 4
            else 4 + int(base.hf_dataset["frame_index"][source_index])
        )
        assert keypose_frame > current_global
        assert (keypose_frame < 4) == (source_index < 4)


def test_executable_action_horizon_reserves_final_keypose_token() -> None:
    assert executable_action_horizon(16) == 15
    with pytest.raises(ValueError, match="at least 2"):
        executable_action_horizon(1)


def test_finds_lerobot_metadata_below_transformed_dataset(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "keyposes.npz"
    policy_qpos = np.stack(
        [np.asarray([frame, 100 + frame], dtype=np.float32) for frame in range(3)]
    )
    _write_sidecar(
        sidecar,
        episode_ends=[3],
        candidates=[[1, 2], [2], [2]],
        policy_qpos=policy_qpos,
    )
    base = _ActionPrefixDataset(
        [0, 0, 0],
        [0, 1, 2],
        horizon=2,
        action_dim=2,
    )

    dataset = SubphaseKeyposeDataset(
        _TransformedDataset(_TransformedDataset(base)),
        sidecar,
        seed=0,
    )

    assert dataset.valid_dataset_indices.tolist() == [0, 1]
    sample = dataset[0]
    assert sample["prompt"] == "open the lid"
    assert sample["actions"].shape == (3, 2)
    assert sample["actions"][-1, 0] > 0


def test_maps_lexical_lerobot_episode_order_to_natural_sidecar_order(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "keyposes.npz"
    episode_names = [f"demo_{index}" for index in range(11)]
    policy_qpos = np.stack(
        [np.asarray([frame, -frame], dtype=np.float32) for frame in range(22)]
    )
    candidates = [
        [frame + 1] if frame % 2 == 0 else [frame]
        for frame in range(22)
    ]
    _write_sidecar(
        sidecar,
        episode_ends=list(range(2, 23, 2)),
        candidates=candidates,
        policy_qpos=policy_qpos,
        episode_names=episode_names,
    )
    # LeRobot episode 2 is demo_10 under lexicographic sorting. Its frame zero
    # is global sidecar frame 20 under the sidecar's natural numeric ordering.
    base = _ActionPrefixDataset(
        [2],
        [0],
        horizon=2,
        action_dim=2,
    )

    dataset = SubphaseKeyposeDataset(base, sidecar, seed=0)

    assert dataset._global_indices.tolist() == [20]
    sample = dataset[0]
    np.testing.assert_array_equal(sample["actions"][-1], policy_qpos[21])
