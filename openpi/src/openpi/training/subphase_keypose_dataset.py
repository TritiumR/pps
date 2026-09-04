"""Dataset wrapper for a terminal, randomly sampled subphase keypose.

The wrapped dataset returns an executable action prefix. This wrapper appends
one absolute joint-position keypose as the final temporal action token:

    [action_0, ..., action_(H-1), keypose]

Candidate keyposes come from the phase-future sidecar produced by
``tools/annotate_capsule_lid_subphases_wandb.py``. Candidates at or before the
current observation are rejected, but the sampled keypose is intentionally not
required to follow the complete executable action prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, SupportsIndex

import numpy as np


class _Dataset(Protocol):
    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]: ...

    def __len__(self) -> int: ...


def executable_action_horizon(model_action_horizon: int) -> int:
    """Return the action-prefix length for ``[actions..., keypose]`` outputs."""
    if model_action_horizon < 2:
        raise ValueError("action/keypose model horizon must be at least 2")
    return model_action_horizon - 1


def _integer_column(values: Sequence[Any], *, name: str) -> np.ndarray:
    result = np.asarray([int(np.asarray(value).item()) for value in values], dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {result.shape}")
    return result


def _find_hf_dataset(dataset: _Dataset) -> Any:
    """Find the LeRobot ``hf_dataset`` below OpenPI transform wrappers."""
    current: Any = dataset
    wrapper_chain: list[str] = []
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        wrapper_chain.append(type(current).__name__)
        if hasattr(current, "hf_dataset"):
            return current.hf_dataset
        # ``openpi.training.data_loader.TransformedDataset`` intentionally
        # exposes only the Dataset protocol and stores its source privately.
        # The keypose wrapper is inserted before the main normalization/model
        # transforms, but after the optional PromptFromLeRobotTask wrapper.
        if not hasattr(current, "_dataset"):
            break
        current = current._dataset
    raise TypeError(
        "could not find a LeRobot hf_dataset beneath wrapper chain "
        + " -> ".join(wrapper_chain)
    )


class SubphaseKeyposeDataset:
    """Append one randomly sampled, strictly future subphase keypose."""

    def __init__(
        self,
        dataset: _Dataset,
        sidecar_path: str | Path,
        *,
        action_keys: Sequence[str] = ("actions",),
        seed: int | None = None,
    ) -> None:
        if not action_keys:
            raise ValueError("at least one action key is required")
        self.dataset = dataset
        self.sidecar_path = Path(sidecar_path).expanduser().resolve()
        self.action_keys = tuple(action_keys)
        self.seed = seed
        self._rng: np.random.Generator | None = None

        with np.load(self.sidecar_path, allow_pickle=False) as archive:
            self.episode_ends = np.asarray(archive["episode_ends"], dtype=np.int64)
            self.episode_names = tuple(
                str(value) for value in np.asarray(archive["episode_names"])
            )
            self.future_indices = np.asarray(
                archive["future_qpos_indices"], dtype=np.int64
            )
            self.future_mask = np.asarray(
                archive["future_qpos_mask"], dtype=bool
            )
            self.policy_qpos = np.asarray(
                archive["policy_qpos"], dtype=np.float32
            )

        if self.future_indices.shape != self.future_mask.shape:
            raise ValueError(
                "future index/mask shape mismatch: "
                f"{self.future_indices.shape} vs {self.future_mask.shape}"
            )
        if len(self.policy_qpos) != len(self.future_indices):
            raise ValueError(
                "policy_qpos and future candidate rows must have equal length"
            )
        if not len(self.episode_ends) or int(self.episode_ends[-1]) != len(
            self.policy_qpos
        ):
            raise ValueError("sidecar episode ends do not cover policy_qpos")

        raw_dataset = _find_hf_dataset(dataset)
        self._episode_indices = _integer_column(
            raw_dataset["episode_index"], name="episode_index"
        )
        self._frame_indices = _integer_column(
            raw_dataset["frame_index"], name="frame_index"
        )
        if len(self._episode_indices) != len(dataset):
            raise ValueError(
                "LeRobot metadata length does not match the queried dataset"
            )
        if len(set(self.episode_names)) != len(self.episode_names):
            raise ValueError("sidecar episode names must be unique")

        # The IsaacLab-to-LeRobot converter assigns episode IDs by sorting the
        # HDF5 demo names lexicographically, while the annotation tool stores
        # sidecar rows in natural numeric order. Map through the shared demo
        # names instead of assuming that the numeric episode IDs match.
        converter_episode_names = tuple(sorted(self.episode_names))
        sidecar_episode_by_name = {
            name: index for index, name in enumerate(self.episode_names)
        }
        if np.any(self._episode_indices < 0) or np.any(
            self._episode_indices >= len(converter_episode_names)
        ):
            raise ValueError("LeRobot episode index is missing from the keypose sidecar")
        self._sidecar_episode_indices = np.asarray(
            [
                sidecar_episode_by_name[converter_episode_names[episode]]
                for episode in self._episode_indices
            ],
            dtype=np.int64,
        )

        self._global_indices = self._map_global_indices()
        self._eligible_candidates: list[np.ndarray] = []
        valid_dataset_indices: list[int] = []
        for dataset_index, global_index in enumerate(self._global_indices):
            candidates = self.future_indices[global_index][
                self.future_mask[global_index]
            ]
            # The keypose must be after the observation qpos, but need not be
            # after every target in the executable action prefix.
            candidates = candidates[candidates > global_index]
            episode = int(self._sidecar_episode_indices[dataset_index])
            episode_start = 0 if episode == 0 else int(self.episode_ends[episode - 1])
            episode_end = int(self.episode_ends[episode])
            candidates = candidates[
                (candidates >= episode_start) & (candidates < episode_end)
            ]
            if len(candidates):
                valid_dataset_indices.append(dataset_index)
                self._eligible_candidates.append(candidates.copy())

        self.valid_dataset_indices = np.asarray(
            valid_dataset_indices, dtype=np.int64
        )
        if not len(self.valid_dataset_indices):
            raise ValueError("sidecar contains no strictly future keyposes")

    def _map_global_indices(self) -> np.ndarray:
        global_indices = np.empty(len(self._episode_indices), dtype=np.int64)
        for index, (episode, frame) in enumerate(
            zip(self._sidecar_episode_indices, self._frame_indices, strict=True)
        ):
            episode_start = 0 if episode == 0 else int(self.episode_ends[episode - 1])
            episode_length = int(self.episode_ends[episode]) - episode_start
            if frame < 0 or frame >= episode_length:
                raise ValueError(
                    f"frame {frame} is outside sidecar episode {episode} "
                    f"with length {episode_length}"
                )
            global_indices[index] = episode_start + frame
        return global_indices

    def _generator(self) -> np.random.Generator:
        if self._rng is None:
            seed = self.seed
            if seed is None:
                try:
                    import torch

                    seed = int(torch.initial_seed())
                except (ImportError, RuntimeError):
                    seed = 0
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        wrapper_index = index.__index__()
        dataset_index = int(self.valid_dataset_indices[wrapper_index])
        sample = dict(self.dataset[dataset_index])

        candidates = self._eligible_candidates[wrapper_index]
        candidate = int(candidates[self._generator().integers(len(candidates))])
        keypose = self.policy_qpos[candidate]

        for key in self.action_keys:
            if key not in sample:
                raise KeyError(f"wrapped sample is missing action key {key!r}")
            actions = np.asarray(sample[key])
            if actions.ndim < 2:
                raise ValueError(
                    f"{key} must be an action sequence, got shape {actions.shape}"
                )
            if actions.shape[-1] != keypose.shape[-1]:
                raise ValueError(
                    f"{key} width {actions.shape[-1]} does not match "
                    f"keypose width {keypose.shape[-1]}"
                )
            sample[key] = np.concatenate(
                [actions, keypose.astype(actions.dtype, copy=False)[None]],
                axis=0,
            )
        return sample

    def __len__(self) -> int:
        return len(self.valid_dataset_indices)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "layout": "[executable actions..., subphase keypose]",
            "sidecar": str(self.sidecar_path),
            "source_rows": len(self.dataset),
            "eligible_rows": len(self),
            "strictly_future": True,
        }
