from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
_POINTCLOUD_NORM_STATS_CACHE: dict[tuple[str, tuple[int, ...] | None], _normalize.NormStats] = {}


def _patch_datasets_array_extension_to_pylist():
    try:
        import datasets.features.features as _hf_features
    except Exception:
        return

    # Newer Hugging Face Datasets serializes fixed-size sequences as ``List``.
    # datasets==2.21 (used by this LeRobot environment) calls the equivalent
    # feature ``Sequence`` and otherwise crashes while reading parquet metadata.
    if _hf_features._FEATURE_TYPES.get("List") is None:
        _hf_features._FEATURE_TYPES["List"] = _hf_features.Sequence

    to_pylist = getattr(_hf_features.ArrayExtensionArray, "to_pylist", None)
    if to_pylist is None or getattr(to_pylist, "_openpi_accepts_pyarrow_kwargs", False):
        return

    def _to_pylist_compat(self, *args, **kwargs):
        return to_pylist(self)

    _to_pylist_compat._openpi_accepts_pyarrow_kwargs = True
    _hf_features.ArrayExtensionArray.to_pylist = _to_pylist_compat


_patch_datasets_array_extension_to_pylist()


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError(
            "Subclasses of IterableDataset should implement __iter__."
        )

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError(
            "Subclasses of DataLoader should implement data_config."
        )

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [
                    jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)
                ]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(
                    data_rng, shape=shape, minval=-1.0, maxval=1.0
                )
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class EpisodeSubsetSafeLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """
    A wrapper around LeRobotDataset that correctly handles random episode subsets.
    
    The issue: When you pass a sparse list of episode IDs (e.g., [7, 29, 92]),
    LeRobotDataset builds episode_data_index with length = number of selected episodes (3),
    but __getitem__ uses the actual episode ID (e.g., 29) to index into it, causing IndexError.
    
    This wrapper maps episode IDs to their position in the subset.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build mapping from episode ID to position in the subset
        self._ep_id_to_pos = (
            {ep_id: pos for pos, ep_id in enumerate(self.episodes)}
            if self.episodes is not None
            else None
        )

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        ep_id = item["episode_index"].item()
        
        # Map episode ID to position for indexing episode_data_index
        ep_pos = self._ep_id_to_pos[ep_id] if self._ep_id_to_pos is not None else ep_id

        query_indices = None
        if self.delta_indices is not None:
            # Use ep_pos (not ep_id) for episode_data_index lookups
            query_indices, padding = self._get_query_indices(idx, ep_pos)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            # Use ep_id (not ep_pos) for video file paths - they use real episode IDs
            video_frames = self._query_videos(query_timestamps, ep_id)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        return item


def _resolve_episode_indices_for_pointcloud_norm(
    data_config: _config.DataConfig,
    *,
    total_episodes: int,
) -> list[int] | None:
    if data_config.pointcloud_norm_episode_indices is not None:
        episodes = list(data_config.pointcloud_norm_episode_indices)
    elif data_config.episode_indices is not None:
        episodes = list(data_config.episode_indices)
    elif data_config.num_episodes is not None and data_config.num_episodes <= total_episodes:
        episodes = list(range(data_config.num_episodes))
    else:
        return None

    invalid_episodes = [episode for episode in episodes if episode < 0 or episode >= total_episodes]
    if invalid_episodes:
        raise ValueError(
            f"Pointcloud normalization episode indices out of range for dataset with "
            f"{total_episodes} episodes: {invalid_episodes}"
        )
    return episodes


def _with_pointcloud_norm_stats(
    data_config: _config.DataConfig,
    *,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
    skip_norm_stats: bool,
) -> _config.DataConfig:
    if skip_norm_stats or data_config.repo_id in (None, "fake"):
        return data_config
    if model_config.model_type not in (
        _model.ModelType.PROXY_POINTCLOUD,
        _model.ModelType.PROXY_DP3,
    ) and not getattr(model_config, "use_pointcloud_prefix", False):
        return data_config
    if data_config.norm_stats is None or "pointcloud" in data_config.norm_stats:
        return data_config

    episodes = _resolve_episode_indices_for_pointcloud_norm(
        data_config,
        total_episodes=dataset_meta.total_episodes,
    )
    cache_key = (data_config.repo_id, None if episodes is None else tuple(episodes))
    if cache_key in _POINTCLOUD_NORM_STATS_CACHE:
        pointcloud_stats = _POINTCLOUD_NORM_STATS_CACHE[cache_key]
    else:
        logging.info(
            "Computing global pointcloud normalization stats for %s episodes from %s.",
            "all" if episodes is None else len(episodes),
            data_config.repo_id,
        )
        stats_dataset = EpisodeSubsetSafeLeRobotDataset(
            data_config.repo_id,
            episodes=episodes,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        running_stats = _normalize.RunningStats()
        for idx in range(len(stats_dataset)):
            item = stats_dataset[idx]
            point_position = np.asarray(item["point_position"], dtype=np.float32)
            point_color = np.asarray(item["point_color"], dtype=np.float32)
            if point_position.ndim != 2 or point_color.ndim != 2:
                raise ValueError(
                    "Expected unbatched pointcloud coord/color with shape "
                    f"(num_points, channels), got {point_position.shape} and {point_color.shape}."
                )
            if point_position.shape != point_color.shape:
                raise ValueError(
                    f"Pointcloud coord/color shape mismatch: {point_position.shape} vs {point_color.shape}."
                )
            pointcloud = np.concatenate([point_position, point_color], axis=-1)
            xyz_valid = np.isfinite(pointcloud[..., :3]).all(axis=-1)
            if not np.any(xyz_valid):
                continue
            running_stats.update(np.nan_to_num(pointcloud[xyz_valid], nan=0.0, posinf=255.0, neginf=0.0))

        pointcloud_stats = running_stats.get_statistics()
        _POINTCLOUD_NORM_STATS_CACHE[cache_key] = pointcloud_stats

    return dataclasses.replace(
        data_config,
        norm_stats={**data_config.norm_stats, "pointcloud": pointcloud_stats},
    )


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    # Determine which episodes to load. Explicit episode indices take precedence
    # over the first-N episode shortcut so train/validation splits can be
    # episode-disjoint.
    episodes = None
    total_episodes = dataset_meta.total_episodes
    if data_config.episode_indices is not None:
        episodes = list(data_config.episode_indices)
        invalid_episodes = [episode for episode in episodes if episode < 0 or episode >= total_episodes]
        if invalid_episodes:
            raise ValueError(
                f"Episode indices out of range for dataset with {total_episodes} episodes: {invalid_episodes}"
            )
        logging.info(
            f"Loading {len(episodes)} explicitly selected episodes out of {total_episodes}: "
            f"{episodes[:10]}{'...' if len(episodes) > 10 else ''}"
        )
    elif data_config.num_episodes is not None:
        if data_config.num_episodes > total_episodes:
            logging.warning(
                f"Requested {data_config.num_episodes} episodes but dataset only has {total_episodes}. "
                "Loading all episodes."
            )
            episodes = None
        else:
            # Use consecutive episodes starting from 0.
            # Note: LeRobotDataset doesn't fully support non-consecutive episode indices,
            # so we load the first N episodes. Use shuffle=True in the data loader
            # to randomize the order of samples during training.
            episodes = list(range(data_config.num_episodes))
            logging.info(
                f"Loading first {data_config.num_episodes} out of {total_episodes} episodes"
            )

    dataset = EpisodeSubsetSafeLeRobotDataset(
        data_config.repo_id,
        episodes=episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
        shuffle_buffer_size=data_config.shuffle_buffer_size,
        num_shards=num_shards,
        shard_index=shard_index,
    )


def transform_dataset(
    dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    subset_indices: list[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        subset_indices: If provided, only use these indices from the dataset (for train/val split).
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        subset_indices=subset_indices,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    subset_indices: list[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        subset_indices: If provided, only use these indices from the dataset (for train/val split).
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    print("length of dataset", len(dataset))
    if data_config.repo_id not in (None, "fake"):
        data_config = _with_pointcloud_norm_stats(
            data_config,
            model_config=model_config,
            action_horizon=action_horizon,
            dataset_meta=lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id),
            skip_norm_stats=skip_norm_stats,
        )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Apply subset if indices provided (for train/val split)
    if subset_indices is not None:
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        logging.info(f"Using subset of dataset with {len(dataset)} samples")

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size (total across all GPUs).
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    # For PyTorch DDP, shard the RLDS data and reduce batch size per GPU
    num_shards = 1
    shard_index = 0
    local_batch_size = batch_size
    if framework == "pytorch" and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_batch_size = batch_size // world_size
        num_shards = world_size
        shard_index = rank
        logging.info(
            f"RLDS DDP: rank={rank}, world_size={world_size}, "
            f"local_batch_size={local_batch_size}"
        )

    dataset = create_rlds_dataset(
        data_config,
        action_horizon,
        local_batch_size,
        shuffle=shuffle,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    dataset = transform_iterable_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True
    )

    if framework == "pytorch":
        data_loader = RLDSDataLoaderPytorch(dataset, num_batches=num_batches)
    else:
        data_loader = RLDSDataLoader(dataset, sharding=sharding, num_batches=num_batches)

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError(
                "Data loading with multiple processes is not supported."
            )

        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)})."
            )

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        epoch = 0
        while True:
            # Update DistributedSampler epoch for proper shuffling across restarts
            sampler = self._data_loader.sampler
            if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                sampler.set_epoch(epoch)
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(
                        lambda x: jax.make_array_from_process_local_data(
                            self._sharding, x
                        ),
                        batch,
                    )
                else:
                    yield jax.tree.map(torch.as_tensor, batch)
            epoch += 1


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError(
                "Data loading with multiple processes is not supported."
            )

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    batch,
                )


class RLDSDataLoaderPytorch:
    """RLDS data loader for PyTorch training.

    Wraps the TF-based DroidRldsDataset (which handles RLDS loading, idle filtering,
    action chunking, shuffling, and batching internally) and converts the resulting
    numpy arrays to torch tensors. The TF pipeline is deliberately kept CPU-only so
    it does not conflict with PyTorch's GPU usage.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # Exhausted one pass; restart the infinite TF dataset.
                num_items += 1
                # Convert numpy arrays to torch tensors. Non-numeric leaves (e.g.
                # string prompt arrays that survive transforms) are passed through
                # unchanged so downstream code can handle them.
                yield jax.tree.map(
                    lambda x: torch.as_tensor(x) if x.dtype.kind in ("i", "u", "f", "b") else x,
                    batch,
                )


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: TorchDataLoader | RLDSDataLoader | RLDSDataLoaderPytorch,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            # Return noise if present in batch (for distillation training)
            noise = batch.get("noise", None)
            yield observation, actions, noise
