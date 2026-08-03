"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
import sys
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.proxy_config as proxy_config
import openpi.models.proxy_score_config as proxy_score_config
import openpi.models.proxy_sound_config as proxy_sound_config
import openpi.models.proxy_dp3_config as proxy_dp3_config
import openpi.models.residual_config as residual_config
import openpi.models.residual_v_config as residual_v_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.vlm_action_expert_config as vlm_action_expert_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Number of episodes to load from the dataset. If None, all episodes will be loaded.
    # Only used for LeRobot datasets. Loads the first N episodes (0 to num_episodes-1).
    num_episodes: int | None = None

    # Specific episode indices to load from the dataset. If provided, overrides num_episodes.
    # Only used for LeRobot datasets.
    episode_indices: list[int] | None = None

    # Episode indices used to fit pointcloud normalization stats. This is separate
    # from episode_indices so train/validation splits can share one fixed global
    # normalization fitted over the selected dataset subset.
    pointcloud_norm_episode_indices: list[int] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    # Shuffle buffer size for RLDS data loader. Reduce for smaller datasets to avoid
    # CPU RAM exhaustion -- the buffer holds *decoded* images so 250k frames ≈ 86 GB raw.
    # For a ~10 GB subset (~6k episodes, ~900k frames), 10_000 is more than sufficient.
    shuffle_buffer_size: int = 250_000


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                inputs = [
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                ]
                if getattr(model_config, "use_pointcloud_prefix", False):
                    inputs.append(_transforms.ResizePointCloud(model_config.pointcloud_num_points))
                inputs += [
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
                return _transforms.Group(
                    inputs=inputs,
                )
            case _model.ModelType.PROXY_POINTCLOUD | _model.ModelType.PROXY_DP3:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                inputs = [
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                ]
                if getattr(model_config, "use_pointcloud_prefix", False):
                    inputs.append(_transforms.ResizePointCloud(model_config.pointcloud_num_points))
                inputs += [
                    _transforms.TokenizePrompt(
                        _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
                return _transforms.Group(
                    inputs=inputs,
                )
            case _model.ModelType.PROXY | _model.ModelType.PROXY_SCORE | _model.ModelType.PROXY_SOUND:
                # return _transforms.Group(
                #     inputs=[
                #         _transforms.ResizeImages(224, 224),
                #         _transforms.PadStatesAndActions(model_config.action_dim),
                #     ],
                # )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.RESIDUAL:
                # Residual policy uses same transforms as PROXY
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {}
                    if model_config.fast_model_tokenizer_kwargs is None
                    else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs
                            ),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs
                            ),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    # norm stats file name
    norm_stats_dir: str | None = None

    # Number of episodes to load from the dataset. If None, all episodes will be loaded.
    # Loads the first N episodes (0 to num_episodes-1).
    num_episodes: int | None = None

    # Specific episode indices to load from the dataset. If provided, overrides num_episodes.
    episode_indices: list[int] | None = None

    # Specific episode indices to fit pointcloud normalization stats.
    pointcloud_norm_episode_indices: list[int] | None = None

    @abc.abstractmethod
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        """Create a data config."""

    def create_base_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=(
                self._load_norm_stats(
                    epath.Path(self.assets.assets_dir or assets_dirs),
                    asset_id,
                    (
                        epath.Path(self.norm_stats_dir)
                        if self.norm_stats_dir is not None
                        else None
                    ),
                )
            ),
            use_quantile_norm=model_config.model_type
            not in (
                _model.ModelType.PI0,
                _model.ModelType.PROXY,
                _model.ModelType.PROXY_SCORE,
                _model.ModelType.PROXY_SOUND,
                _model.ModelType.RESIDUAL,
            ),
            num_episodes=self.num_episodes,
            episode_indices=self.episode_indices,
            pointcloud_norm_episode_indices=self.pointcloud_norm_episode_indices,
        )

    def _load_norm_stats(
        self,
        assets_dir: epath.Path,
        asset_id: str | None,
        norm_stats_dir: epath.Path | None,
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = (
                str(assets_dir / asset_id)
                if norm_stats_dir is None
                else str(norm_stats_dir)
            )
            norm_stats = _normalize.load(
                norm_stats_dir
                if norm_stats_dir is not None
                else _download.maybe_download(data_assets_dir)
            )
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=GroupFactory
    )
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = (
        "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"
    )
    use_quantile_norm: bool = False
    # Shuffle buffer size. Decoded DROID images are large (~350 KB/frame for 2 cameras),
    # so 250k frames ≈ 86 GB RAM. Reduce for smaller datasets.
    shuffle_buffer_size: int = 250_000

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert (
            self.rlds_data_dir is not None
        ), "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
            use_quantile_norm=self.use_quantile_norm,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDJointPosDataConfig(DataConfigFactory):
    """
    Example data config for custom joint position dataset in LeRobot format.
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # We assume absolute joint *position* actions, so we should apply an additional delta transform.
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(
                    model_type=model_config.model_type,
                    use_pointcloud=getattr(model_config, "use_pointcloud_prefix", False),
                ),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDJointPosPointCloudDataConfig(DataConfigFactory):
    """DROID joint-position config that keeps RGB inputs and adds point-cloud inputs."""

    default_prompt: str | None = None
    use_quantile_norm: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/pointcloud_coord": "point_position",
                        "observation/pointcloud_color": "point_color",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(
                    model_type=model_config.model_type,
                    use_pointcloud=getattr(model_config, "use_pointcloud_prefix", False),
                ),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
        )


@dataclasses.dataclass(frozen=True)
class ProxySoundLeRobotDROIDJointPosDataConfig(DataConfigFactory):
    """DROID joint-position data config that also repacks two log-mel sound observations."""

    default_prompt: str | None = None
    use_quantile_norm: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "observation/mic1_log_mel": "mic1_log_mel",
                        "observation/mic2_log_mel": "mic2_log_mel",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(
                    model_type=model_config.model_type,
                    use_sound=getattr(model_config, "use_sound_prefix", False),
                ),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
        )


@dataclasses.dataclass(frozen=True)
class ProxyLeRobotDROIDJointPosDataConfig(DataConfigFactory):
    """
    Example data config for custom joint position dataset in LeRobot format.
    """

    default_prompt: str | None = None
    use_quantile_norm: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # We assume absolute joint *position* actions, so we should apply an additional delta transform.
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
        )


@dataclasses.dataclass(frozen=True)
class _OverlayThermalDroidImages:
    """Blend thermal RGB image streams into the corresponding normal RGB streams."""

    alpha: float = 0.5

    def __call__(self, data: dict) -> dict:
        pairs = (
            (
                "observation/exterior_image_1_left",
                "observation/thermal_exterior_image_1_left",
            ),
            (
                "observation/wrist_image_left",
                "observation/thermal_wrist_image_left",
            ),
        )
        for rgb_key, thermal_key in pairs:
            rgb = np.asarray(data[rgb_key])
            thermal = np.asarray(data[thermal_key])
            if rgb.shape != thermal.shape:
                raise ValueError(
                    f"RGB/thermal image shape mismatch for {rgb_key}: "
                    f"{rgb.shape} vs {thermal.shape}"
                )

            blended = (1.0 - self.alpha) * rgb.astype(np.float32) + self.alpha * thermal.astype(
                np.float32
            )
            if np.issubdtype(rgb.dtype, np.integer):
                blended = np.clip(blended, 0, np.iinfo(rgb.dtype).max).astype(rgb.dtype)
            else:
                blended = blended.astype(rgb.dtype)
            data[rgb_key] = blended
        return data


@dataclasses.dataclass(frozen=True)
class ProxyThermalLeRobotDROIDJointPosDataConfig(DataConfigFactory):
    """
    Joint-position LeRobot config that overlays thermal RGB streams onto the
    normal RGB DROID camera streams before feeding the proxy image model.
    """

    default_prompt: str | None = None
    use_quantile_norm: bool = False
    thermal_alpha: float = 0.5

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/thermal_exterior_image_1_left": "thermal_exterior_image_1_left",
                        "observation/thermal_wrist_image_left": "thermal_wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                _OverlayThermalDroidImages(alpha=self.thermal_alpha),
                droid_policy.DroidInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
        )


@dataclasses.dataclass(frozen=True)
class ProxyPointCloudLeRobotDROIDJointPosDataConfig(DataConfigFactory):
    """
    Joint-position LeRobot config for point-cloud proxy policies.

    This keeps the same action/state transforms as the image proxy config and only
    changes the repacked observation keys to match the point-cloud converter.
    """

    default_prompt: str | None = None
    use_quantile_norm: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/pointcloud_coord": "point_position",
                        "observation/pointcloud_color": "point_color",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
        )


@dataclasses.dataclass(frozen=True)
class ProxyLeRobotDROIDJointVelDataConfig(DataConfigFactory):
    """
    Example data config for custom joint velocity dataset in LeRobot format.
    """

    default_prompt: str | None = None
    use_quantile_norm: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # We assume absolute joint *velocity* actions, so we should apply no delta transform.
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type),
            ],
            outputs=[
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Use quantile normalization to match pi05_droid (joint velocity uses PI05 norm stats)
            use_quantile_norm=self.use_quantile_norm,
        )


@dataclasses.dataclass(frozen=True)
class DistillationDROIDDataConfig(DataConfigFactory):
    """
    Data config for distillation training where:
    - Actions are stored as full action chunks (action_horizon, action_dim) from teacher outputs
    - Teacher outputs are in absolute form (after AbsoluteActions transform)
    - Noise is stored alongside each observation-action pair
    - No delta_timestamps sequencing needed (actions already stored as chunks)
    - Still need DeltaActions transform to convert teacher's absolute outputs → deltas for student training
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",  # Full action chunk (10, 8) in absolute form
                        "noise": "noise",  # Initial noise used for generation (10, 8)
                    }
                )
            ]
        )

        # Teacher outputs absolute actions, but student learns deltas (like teacher did)
        # Apply DeltaActions to convert: absolute → deltas for training
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Empty tuple: don't use delta_timestamps for actions (already stored as full chunks)
            action_sequence_keys=(),
        )


@dataclasses.dataclass(frozen=True)
class DistillationPointCloudDROIDDataConfig(DataConfigFactory):
    """
    Distillation data config for point-cloud proxy policies.

    Actions/noise are already stored as full chunks. The point cloud is repacked from
    the LeRobot point_position/point_color fields into the existing DROID point-cloud
    input path, so the generic data loader can stay unchanged.
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/pointcloud_coord": "point_position",
                        "observation/pointcloud_color": "point_color",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "noise": "noise",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(),
        )


@dataclasses.dataclass(frozen=True)
class DistillationDROIDDataAblationConfig(DataConfigFactory):
    """
    Data config for distillation training where:
    - Actions are stored as full action chunks (action_horizon, action_dim) from teacher outputs
    - Teacher outputs are in absolute form (after AbsoluteActions transform)
    - Noise is stored alongside each observation-action pair
    - No delta_timestamps sequencing needed (actions already stored as chunks)
    - Still need DeltaActions transform to convert teacher's absolute outputs → deltas for student training
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",  # Full action chunk (10, 8) in absolute form
                    }
                )
            ]
        )

        # Teacher outputs absolute actions, but student learns deltas (like teacher did)
        # Apply DeltaActions to convert: absolute → deltas for training
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[
                droid_policy.DroidInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Empty tuple: don't use delta_timestamps for actions (already stored as full chunks)
            action_sequence_keys=(),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config
    )

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=_optimizer.AdamW
    )
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(
        default_factory=nnx.Nothing
    )

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # Number of selected episodes to reserve for validation. Zero disables validation.
    validation_num_episodes: int = 0
    # How often (in steps) to run validation when validation_num_episodes > 0.
    validation_interval: int = 1000
    # Number of validation batches per validation pass.
    validation_num_batches: int = 20

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Distillation-specific fields (for distill_pytorch.py)
    # Config name for the teacher model (e.g., "pi0_aloha_sim").
    teacher_config_name: str | None = None
    # Checkpoint directory for the teacher model (e.g., "./checkpoints/pi0_aloha_sim/exp/10000").
    teacher_checkpoint_dir: str | None = None
    # Number of distillation steps to use during training.
    num_distill_steps: int = 10

    teacher_flow_path_noise_std: float = 0.0

    use_noise_for_distill: bool = True
    # If true, action distillation reuses the teacher rollout's initial noise for
    # the student flow-matching path. Defaults to fresh student noise, matching
    # standard flow-matching training.
    use_teacher_noise_for_action_distill: bool = False

    # Residual policy training fields (for train_residual_pytorch.py)
    # Config name for the VLA model (e.g., "pi0_aloha_sim").
    vla_config_name: str | None = None
    # Checkpoint directory for the VLA model (e.g., "./checkpoints/pi0_aloha_sim/exp/10000").
    vla_checkpoint_dir: str | None = None
    # Number of inference steps for VLA model during training.
    num_vla_steps: int = 10

    # Residual-V (velocity-conditioned residual) training field
    # (for train_residual_v_pytorch.py). Number of (x_0, t) path points sampled
    # per observation per training step. Effective residual batch size is
    # batch_size * num_paths_per_obs.
    num_paths_per_obs: int = 10

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (
            pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name
        ).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    ##########################################################
    # PI0.5-DROID-Jointpos configs                           #
    ##########################################################
    TrainConfig(
        name="pi05_droid_jointpos",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                # inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                inputs=[
                    droid_policy.DroidInputs(model_type=ModelType.PI05),
                    _transforms.DeltaActions(_transforms.make_bool_mask(7, -1)),
                ],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    ##########################################################
    # Custom PI0.5-DROID-Jointpos finetune configs           #
    ##########################################################
    # Spoon
    TrainConfig(
        name="pi05_droid_jointpos_lora_spoon",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Spoon new
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_spoon_new",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_coffee_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/coffee_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_jeans_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/jeans_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/jeans_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_spoon_new_2x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_spoon_new_4x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_spoon_new_8x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Shoe
    TrainConfig(
        name="pi05_droid_jointpos_lora_shoe",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/shoe",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Flower
    TrainConfig(
        name="pi05_droid_jointpos_lora_flower",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Flower new
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_flower_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_flower_new_2x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_flower_new_4x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_flower_new_8x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Bread
    TrainConfig(
        name="pi05_droid_jointpos_lora_bread",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/bread",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bread"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Coffee
    TrainConfig(
        name="pi05_droid_jointpos_lora_coffee",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Tissue
    TrainConfig(
        name="pi05_droid_jointpos_lora_tissue",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/tissue",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Drawer new
    TrainConfig(
        name="pi05_droid_jointpos_lora_drawer_new",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_drawer_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_drawer_new_2x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_drawer_new_4x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_drawer_new_8x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # New real DROID tasks
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_shoe_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/shoe_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_mug_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/mug_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/mug_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_sweep_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/sweep_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/sweep_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tissue_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/tissue_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_lever_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/lever_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/lever_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    ##########################################################
    # PI0-DROID-Jointpos finetune configs                    #
    ##########################################################
    TrainConfig(
        name="pi0_droid_jointpos_lora_spoon_jar",
        save_interval=200,
        keep_period=200,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=100_000,
        batch_size=128,  # for 4 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_box_shoes",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/shoe",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5000,
        batch_size=32,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_flower_vase",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5000,
        batch_size=32,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_flower_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_coffee_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/coffee_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_jeans_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/jeans_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/jeans_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_spoon",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_spoon_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_tissue",
        save_interval=500,
        keep_period=500,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/tissue_lora",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue_lora"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=20_000,
        batch_size=64,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_coffee",
        save_interval=500,
        keep_period=500,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=20_000,
        batch_size=64,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_bread",
        save_interval=500,
        keep_period=500,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/bread",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bread"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=20_000,
        batch_size=64,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_drawer_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_drawer_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # New real DROID tasks
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_shoe_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/shoe_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_mug_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/mug_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/mug_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_sweep_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/sweep_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/sweep_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_tissue_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/tissue_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_only_lora_lever_new",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/lever_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/lever_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=64,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),

    ##########################################################
    # Aloha inference configs                                #
    ##########################################################
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    ##########################################################
    # PI0-DROID inference configs                            #
    ##########################################################
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_droid_jointpos",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                # inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                inputs=[
                    droid_policy.DroidInputs(model_type=ModelType.PI0),
                    _transforms.DeltaActions(_transforms.make_bool_mask(7, -1)),
                ],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    # Jacobian analysis config - full PI0 model with a real dataset for analysis scripts
    TrainConfig(
        name="pi0_droid_jointpos_jacobian",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        batch_size=2,
    ),
    # Jacobian analysis config - full PI05 model with a real dataset for analysis scripts
    TrainConfig(
        name="pi05_droid_jacobian",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid/assets/droid",
        ),
        batch_size=2,
    ),
    TrainConfig(
        name="proxy_droid_water_jointpos",  # in total 33M params
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_water",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_water"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=100_000,
        save_interval=20000,
        batch_size=64,
    ),
    TrainConfig(
        name="proxy_real_droid_water_jointpos",  # in total 33M params
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/water",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/water"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=100_000,
        save_interval=20000,
        batch_size=64,
    ),
    # Water random configs
    TrainConfig(
        name="proxy_droid_water_random_jointpos",  # in total 33M params
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_water_random",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_water_random"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=20_000,
        batch_size=64,
    ),
    # Inference configs
    TrainConfig(
        name="proxy_real_droid_inference_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="proxy_real_droid_inference_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="proxy_real_droid_inference_freeze_dino_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            freeze_dino_encoder=True,
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/inference_freeze_dino",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference_freeze_dino"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="proxy_real_droid_inference_jointvel",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="residual_real_droid_inference_jointpos",
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="residual_real_droid_inference_pi05_jointpos",
        model=residual_config.ResidualConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="proxy_real_droid_inference_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    TrainConfig(
        name="proxy_multitask_8task_inference_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/inference",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/inference"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=1,
        save_interval=1,
        batch_size=1,
    ),
    # Bread task configs
    TrainConfig(
        name="proxy_real_droid_bread_jointpos",  # Bread task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/bread",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bread"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_bread_pi05_jointpos",  # Bread task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/bread",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bread"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="residual_real_droid_bread_jointpos",  # Residual policy for bread task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/bread",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bread"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    ##########################################################
    # Residual-V (velocity-conditioned residual) DROID configs #
    ##########################################################
    # Residual-V: trained along the steer/expert probability path, conditioned
    # on the frozen VLA's v_base at each path point. Launched with
    # `scripts/train_residual_v_pytorch.py`. batch_size here is B_obs
    # (observations per step); effective residual batch is
    # batch_size * num_paths_per_obs.
    TrainConfig(
        name="residual_v_real_droid_bread_jointpos",
        model=residual_v_config.ResidualVConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/bread",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bread"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=8,
        num_paths_per_obs=10,
        vla_config_name="pi0_droid_jointpos",
        vla_checkpoint_dir="checkpoints/pytorch/pi0_droid_jointpos",
    ),
    TrainConfig(
        name="residual_v_real_droid_spoon_jointpos",
        model=residual_v_config.ResidualVConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=2_000,
        batch_size=8,
        num_paths_per_obs=10,
        vla_config_name="pi0_droid_jointpos",
        vla_checkpoint_dir="checkpoints/pytorch/pi0_droid_jointpos",
    ),
    # Box task configs
    TrainConfig(
        name="proxy_real_droid_box_jointpos",  # Box task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/box",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/box"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    # Spoon task configs
    TrainConfig(
        name="proxy_real_droid_spoon_jointpos",  # Spoon task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_new_jointpos",  # Spoon new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_pi05_jointpos",  # Spoon pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_new_pi05_jointpos",  # Spoon pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_new_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_new_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_new_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/spoon_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_spoon_jointvel",  # Spoon task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="residual_real_droid_spoon_jointpos",  # Residual policy for spoon task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/spoon",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    # Shoe task configs
    TrainConfig(
        name="proxy_real_droid_shoe_jointpos",  # Shoe task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/shoe",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_shoe_pi05_jointpos",  # Shoe pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/shoe",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_shoe_new_jointpos",  # Shoe new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/shoe_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_shoe_new_pi05_jointpos",  # Shoe new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/shoe_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_coffee_new_jointpos",  # Coffee new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/coffee_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_coffee_new_pi05_jointpos",  # Coffee new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/coffee_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_jeans_new_jointpos",  # Jeans new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/jeans_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/jeans_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_jeans_new_pi05_jointpos",  # Jeans new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/jeans_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/jeans_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="residual_real_droid_jeans_new_pi05_jointpos",
        model=residual_config.ResidualConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/jeans_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/jeans_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=20_001,
        save_interval=20_000,
        batch_size=64,
        vla_config_name="pi05_droid_jointpos",
        vla_checkpoint_dir="checkpoints/pytorch/pi05_droid_jointpos",
    ),
    TrainConfig(
        name="residual_real_droid_shoe_new_pi05_jointpos",
        model=residual_config.ResidualConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/shoe_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=20_001,
        save_interval=20_000,
        batch_size=64,
        vla_config_name="pi05_droid_jointpos",
        vla_checkpoint_dir="checkpoints/pytorch/pi05_droid_jointpos",
    ),
    TrainConfig(
        name="residual_real_droid_shoe_jointpos",  # Residual policy for shoe task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/shoe",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_shoe_jointvel",  # Shoe task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/shoe_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/shoe_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    # Duster task configs
    TrainConfig(
        name="proxy_real_droid_duster_jointpos",  # Duster task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/duster",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/duster"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    # Bottle task configs
    TrainConfig(
        name="proxy_real_droid_bottle_jointpos",  # Bottle task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/bottle",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/bottle"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    # Mug task configs
    TrainConfig(
        name="proxy_real_droid_mug_new_jointpos",  # Mug new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/mug_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/mug_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_mug_new_pi05_jointpos",  # Mug new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/mug_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/mug_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="residual_real_droid_mug_new_pi05_jointpos",
        model=residual_config.ResidualConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/mug_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/mug_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=20_001,
        save_interval=20_000,
        batch_size=64,
        vla_config_name="pi05_droid_jointpos",
        vla_checkpoint_dir="checkpoints/pytorch/pi05_droid_jointpos",
    ),
    # Coffee task configs
    TrainConfig(
        name="proxy_real_droid_coffee_jointpos",  # Coffee task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_coffee_pi05_jointpos",  # Coffee pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="residual_real_droid_coffee_jointpos",  # Residual policy for coffee task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/coffee",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_coffee_jointvel",  # Coffee task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/coffee_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/coffee_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    # Drawer task configs
    TrainConfig(
        name="proxy_real_droid_drawer_jointpos",  # Drawer task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_drawer_jointvel",  # Drawer task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    # Drawer new task configs
    TrainConfig(
        name="proxy_real_droid_drawer_new_jointpos",  # Drawer new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_droid_new_jointpos",  # Alias for drawer new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_drawer_new_pi05_jointpos",  # Drawer new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_drawer_new_pi05_jointpos_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_drawer_new_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_drawer_new_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_drawer_new_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="residual_real_droid_drawer_new_jointpos",  # Residual policy for drawer new task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/drawer_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/drawer_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    # Flower task configs
    # TrainConfig(
    #     name="proxy_real_droid_flower_mid_jointpos",  # Flower task real world
    #     model=proxy_config.ProxyConfig(
    #         action_horizon=10,
    #         action_dim=8,
    #         action_expert_variant="gemma_12m",
    #         dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    #     ),
    #     data=ProxyLeRobotDROIDJointPosDataConfig(
    #         # modify this line to point to our own dataset
    #         repo_id="cn356/flower_mid",
    #         base_config=DataConfig(prompt_from_task=True),
    #         assets=AssetsConfig(asset_id="cn356/flower_mid"),
    #         norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
    #     ),
    #     num_train_steps=40_000,
    #     save_interval=10_000,
    #     batch_size=8,
    # ),
    TrainConfig(
        name="proxy_real_droid_flower_jointpos",  # Flower task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_new_jointpos",  # Flower new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_pi05_jointpos",  # Flower pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_new_pi05_jointpos",  # Flower new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_new_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_new_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_new_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/flower_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_flower_jointvel",  # Flower task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="residual_real_droid_flower_jointpos",  # Residual policy for flower task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/flower",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/flower"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    # Tissue task configs
    TrainConfig(
        name="proxy_real_droid_tissue_jointpos",  # Tissue task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/tissue",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_tissue_pi05_jointpos",  # Tissue pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/tissue",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="proxy_real_droid_tissue_new_jointpos",  # Tissue new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/tissue_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_tissue_new_pi05_jointpos",  # Tissue new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/tissue_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_tissue_jointvel",  # Tissue task real world
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/tissue_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=8,
    ),
    TrainConfig(
        name="residual_real_droid_tissue_jointpos",  # Residual policy for tissue task
        model=residual_config.ResidualConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/tissue",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/tissue"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_000,
        save_interval=10_000,
        batch_size=32,
    ),
    # Sweep new task configs
    TrainConfig(
        name="proxy_real_droid_sweep_new_jointpos",  # Sweep new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/sweep_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/sweep_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_sweep_new_pi05_jointpos",  # Sweep new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/sweep_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/sweep_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    # Lever new task configs
    TrainConfig(
        name="proxy_real_droid_lever_new_jointpos",  # Lever new task real world
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/lever_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/lever_new"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_real_droid_lever_new_pi05_jointpos",  # Lever new pi05
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/lever_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/lever_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=40_000,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="residual_real_droid_lever_new_pi05_jointpos",
        model=residual_config.ResidualConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/lever_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/lever_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=20_001,
        save_interval=20_000,
        batch_size=64,
        vla_config_name="pi05_droid_jointpos",
        vla_checkpoint_dir="checkpoints/pytorch/pi05_droid_jointpos",
    ),
    TrainConfig(
        name="pi0_fast_droid_jointpos",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[
                    # droid_policy.DroidInputs(
                    #     action_dim=model.action_dim, model_type=ModelType.PI0_FAST
                    # )
                    droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)
                ],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    ##########################################################
    # DROID-Jointpos IsaacLab configs                  #
    ##########################################################
    # Cylinder task configs
    TrainConfig(
        name="proxy_isaaclab_droid_cylinder_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_cylinder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_cylinder"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_cylinder_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_cylinder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_cylinder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_cylinder",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_cylinder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_cylinder"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_cylinder",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_cylinder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_cylinder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    # Block task configs
    TrainConfig(
        name="proxy_isaaclab_droid_block_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_block",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_block"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_block_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_block",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_block"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_droid_jointpos_lora_block",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_block",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_block"),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi0_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        # Freeze non-LoRA parameters
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_dim=32,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_block",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_block",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_block"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),

    # Laptop task configs
    TrainConfig(
        name="proxy_isaaclab_droid_laptop_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_laptop",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_laptop"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_laptop_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_laptop",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_laptop"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_laptop_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_laptop",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_laptop"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_laptop_dp3_pi05_jointpos",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=1024,
            freeze_point_encoder=False,
            compile_sample_actions=False,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_laptop_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_laptop_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_laptop",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_laptop",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_laptop"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_laptop",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_laptop",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_laptop"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Oven task configs
    TrainConfig(
        name="proxy_isaaclab_droid_oven_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_oven_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_oven_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_oven_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_oven_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_oven",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_oven",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_oven_2x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_oven_4x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_oven_8x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_oven",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_oven"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Plate task configs
    TrainConfig(
        name="proxy_isaaclab_droid_plate_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_plate",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_plate"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_plate_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_plate",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_plate"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_plate_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_plate",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_plate"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_plate",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_plate",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_plate"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_plate",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_plate",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_plate"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Pot task configs
    TrainConfig(
        name="proxy_isaaclab_droid_pot_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_pot_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_pot_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_pot_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_pot_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_pot",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pot",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pot_2x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pot_4x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pot_8x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Pot pointcloud task configs (dataset: cn356/isaaclab_pot_pointcloud, 512 masked points)
    TrainConfig(
        name="proxy_isaaclab_droid_pot_dp3_pi05_jointpos",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=256,
            freeze_point_encoder=False,
            compile_sample_actions=False,
            pointcloud_position_noise_std=0.005,
            pointcloud_dropout_ratio=0.05,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pot_pointcloud_rgb",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pot_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pot_pointcloud",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_pointcloud_prefix=True,
            pointcloud_num_points=512,
            pointcloud_channels=6,
            pointcloud_prefix_tokens=16,
        ),
        data=LeRobotDROIDJointPosPointCloudDataConfig(
            repo_id="cn356/isaaclab_pot_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pot_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights and the new point-cloud prefix encoder.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*pointcloud_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    # Weight task configs
    TrainConfig(
        name="proxy_isaaclab_droid_weight_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="score_task_weight",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
            prediction_type="epsilon",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_weight_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_weight_score"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    TrainConfig(
        # Tea-task twin of score_task_weight, for loading tea task-proxy checkpoints.
        name="score_task_tea",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
            prediction_type="epsilon",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_tea_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_tea_score"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    TrainConfig(
        name="score_ref_weight",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_weight_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_weight_score"),
            # Task, ref, and base scores must use one normalized action space.
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    # Diagnostic twin of score_ref_weight read as an epsilon predictor: |ref| is flat across
    # denoise steps where a score-scaled field climbs ~1/sqrt(beta), so load a ref checkpoint here
    # to test whether it was trained on epsilon and only mislabelled.
    TrainConfig(
        name="score_ref_weight_eps",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
            prediction_type="epsilon",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_weight_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_weight_score"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_dp3_pi05_jointpos_masked",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=1024,
            freeze_point_encoder=False,
            compile_sample_actions=False,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight_masked_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight_masked_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_dp3_pi05_jointpos_masked_no_color",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=1024,
            freeze_point_encoder=False,
            compile_sample_actions=False,
            use_pc_color=False,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight_masked_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight_masked_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_weight_dp3_pi05_jointpos",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=1024,
            freeze_point_encoder=False,
            compile_sample_actions=False,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_weight",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_weight",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_weight_2x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_weight_4x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_weight_8x",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_weight",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_weight"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Can task configs
    TrainConfig(
        name="proxy_isaaclab_droid_can_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_can",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_can"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_can_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_can",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_can"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_can_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_can",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_can"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_can",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_can",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_can"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_can",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_can",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_can"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Pen task configs
    TrainConfig(
        name="proxy_isaaclab_droid_pen_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_pen",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_pen_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pen",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_pen_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pen",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_pen",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_pen",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pen",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pen",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Pen pointcloud task configs (dataset: cn356/isaaclab_pen_pointcloud, 512 masked points)
    TrainConfig(
        name="proxy_isaaclab_droid_pen_dp3_pi05_jointpos",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=512,
            freeze_point_encoder=False,
            compile_sample_actions=False,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pen_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pen_pointcloud_rgb",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_pen_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_pen_pointcloud",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_pointcloud_prefix=True,
            pointcloud_num_points=512,
            pointcloud_channels=6,
            pointcloud_prefix_tokens=16,
        ),
        data=LeRobotDROIDJointPosPointCloudDataConfig(
            repo_id="cn356/isaaclab_pen_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_pen_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights and the new point-cloud prefix encoder.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*pointcloud_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="proxy_sound_isaaclab_droid_phone_pi05_jointpos",
        model=proxy_sound_config.ProxySoundConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxySoundLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_phone_sound",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_phone_sound"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    # Drink task configs
    TrainConfig(
        name="proxy_isaaclab_droid_drink_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_drink",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_drink"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_drink_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_drink",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_drink"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_drink_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_drink",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_drink"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_lora_drink",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        # Add LoRA variants to the model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",  # LoRA for vision-language model
            action_expert_variant="gemma_300m_lora",  # LoRA for action decoder
        ),
        data=LeRobotDROIDJointPosDataConfig(
            # modify this line to point to our own dataset
            repo_id="cn356/isaaclab_drink",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_drink"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_drink",
        save_interval=10000,
        keep_period=10000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_drink",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_drink"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=10_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Tea task configs
    TrainConfig(
        name="proxy_isaaclab_droid_tea_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_score_isaaclab_droid_tea_pi05_jointpos",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_tea_new_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea_new",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea_new"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_hot_tea_pi05_jointpos_thermal",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyThermalLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_hot_tea_thermal",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_hot_tea_thermal"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_hot_tea_pi05_jointpos_thermal_40M",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyThermalLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_hot_tea_thermal",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_hot_tea_thermal"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_hot_tea",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_hot_tea_thermal",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_hot_tea_thermal"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_hot_tea_thermal",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=ProxyThermalLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_hot_tea_thermal",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_hot_tea_thermal"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
            thermal_alpha=0.5,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_tea_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_tea_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_tea_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_tea_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tea",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tea_2x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tea_4x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tea_8x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_tea_wood_dp3_pi05_jointpos_no_color",
        model=proxy_dp3_config.ProxyDP3Config(
            action_horizon=15,
            action_dim=8,
            num_points=256,
            pointnet_type="multi_stage_pointnet",
            encoder_output_dim=128,
            freeze_point_encoder=False,
            compile_sample_actions=False,
            use_pc_color=False,
            time_embedding_scale=100.0,
            down_dims=(128, 256, 384),
            pointcloud_position_noise_std=0.005,
            pointcloud_dropout_ratio=0.05,
            pointcloud_random_resample=True,
        ),
        data=ProxyPointCloudLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea_wood_pointcloud_huge",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea_wood_pointcloud_huge"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        validation_num_episodes=10,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tea_wood",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_tea_wood_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea_wood_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_tea_wood_pointcloud",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_pointcloud_prefix=True,
            pointcloud_num_points=1024,
            pointcloud_channels=6,
            pointcloud_prefix_tokens=16,
        ),
        data=LeRobotDROIDJointPosPointCloudDataConfig(
            repo_id="cn356/isaaclab_tea_wood_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea_wood_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights and the new point-cloud prefix encoder.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*pointcloud_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_phone",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_phone_sound",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_phone_sound"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_phone_sound",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_sound_prefix=True,
            sound_channels=2,
            sound_mel_bins=80,
            sound_time_bins=198,
            sound_prefix_tokens=18,
        ),
        data=ProxySoundLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_phone_sound",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_phone_sound"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        # Freeze every parameter except LoRA weights and the new sound prefix encoder.
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*sound_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="only_lora_phone_sound",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_sound_prefix=True,
            sound_channels=2,
            sound_mel_bins=80,
            sound_time_bins=198,
            sound_prefix_tokens=18,
        ),
        data=ProxySoundLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_phone_sound",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_phone_sound"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*sound_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="only_lora_tea_wood_pointcloud",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_pointcloud_prefix=True,
            pointcloud_num_points=1024,
            pointcloud_channels=6,
            pointcloud_prefix_tokens=16,
        ),
        data=LeRobotDROIDJointPosPointCloudDataConfig(
            repo_id="cn356/isaaclab_tea_wood_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea_wood_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*pointcloud_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="only_lora_tea_wood_poincloud",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
            use_pointcloud_prefix=True,
            pointcloud_num_points=1024,
            pointcloud_channels=6,
            pointcloud_prefix_tokens=16,
        ),
        data=LeRobotDROIDJointPosPointCloudDataConfig(
            repo_id="cn356/isaaclab_tea_wood_pointcloud",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_tea_wood_pointcloud"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*pointcloud_prefix_encoder.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="only_lora_hot_tea_thermal",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=ProxyThermalLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_hot_tea_thermal",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_hot_tea_thermal"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=False,
            thermal_alpha=0.5,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Holder task configs
    TrainConfig(
        name="proxy_isaaclab_droid_holder_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_holder_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_holder_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_holder_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_holder_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_holder",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_holder_2x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_holder_4x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_holder_8x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_holder",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_holder"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    # Capsule task configs
    TrainConfig(
        name="proxy_isaaclab_droid_capsule_pi05_jointpos",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        batch_size=32,
    ),
    TrainConfig(
        name="score_task_capsule",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
            prediction_type="epsilon",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_capsule_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_capsule_score"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    # score_task_capsule twin for the low-data demo-BC proxy, DINO frozen: train-bc sees ~2k
    # frames, so the 21.6M-param encoder is the overfitting surface. Same state dict.
    TrainConfig(
        name="score_task_stack_bc",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            freeze_dino_encoder=True,
            ddim_num_train_timesteps=100,
            prediction_type="epsilon",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_capsule_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_capsule_score"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    # score_task_stack_bc with the visual encoder TRAINED, matching the PPS paper's proxy
    # recipe (~33M trainable). The frozen twin underfits: train MAE == held-out MAE.
    TrainConfig(
        name="score_task_stack_bc_unfrozen",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            freeze_dino_encoder=False,
            ddim_num_train_timesteps=100,
            prediction_type="epsilon",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="local/isaaclab_capsule_score",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="local/isaaclab_capsule_score"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_score_isaaclab_droid_capsule_pi05_jointpos",
        model=proxy_score_config.ProxyScoreConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
            ddim_num_train_timesteps=100,
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pytorch/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_capsule_pi05_jointpos_vits16plus_40m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_40m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_capsule_pi05_jointpos_vits16plus_100m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_100m",
            dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_capsule_pi05_jointpos_vitb16_380m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_380m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=32,
    ),
    TrainConfig(
        name="proxy_isaaclab_droid_capsule_pi05_jointpos_vitb16_180m",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_180m",
            dino_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ),
        data=ProxyLeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
        ),
        num_train_steps=10_001,
        save_interval=10_000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_capsule",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r10",
            action_expert_variant="gemma_300m_lora_r24",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_capsule_2x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r20",
            action_expert_variant="gemma_300m_lora_r48",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_capsule_4x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r40",
            action_expert_variant="gemma_300m_lora_r96",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_droid_jointpos_only_lora_capsule_8x",
        save_interval=5000,
        keep_period=5000,  # Keep every checkpoint
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            paligemma_variant="gemma_2b_lora_r80",
            action_expert_variant="gemma_300m_lora_r192",
        ),
        data=LeRobotDROIDJointPosDataConfig(
            repo_id="cn356/isaaclab_capsule",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/isaaclab_capsule"),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/pi05_droid_jointpos/params"
        ),
        num_train_steps=5_001,
        batch_size=32,  # for 2 GPUs
        freeze_filter=nnx.All(
            nnx_utils.PathRegex(".*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        ema_decay=None,
    ),

    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    ##########################################################
    # General Mimic configs. Distillation
    ##########################################################
    TrainConfig(
        # This config is for distilling pi05 on the subset of DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        name="mimic_pi05_droid_10G",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid_10G",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/share/ma/scratch/chuanruo/openpi_data/droid_10G",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            # norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
            # 10 GB subset has ~6k episodes × ~150 frames = ~900k frames total.
            # 250k (default) would hold 86 GB of decoded images in RAM. Use 10k instead.
            shuffle_buffer_size=10_000,
        ),
        # lr_schedule=_optimizer.CosineDecaySchedule(
        #     warmup_steps=1_000,
        #     peak_lr=5e-5,
        #     decay_steps=1_000_000,
        #     decay_lr=5e-5,
        # ),
        num_train_steps=20_000,
        batch_size=8,
        # log_interval=100,
        save_interval=2_000,
        keep_period=2_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for distilling pi05 on the subset of DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        name="mimic_pi05_droid_50G",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid_50G",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="../data/droid_50G",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
            # 10 GB subset has ~6k episodes × ~150 frames = ~900k frames total.
            # 250k (default) would hold 86 GB of decoded images in RAM. Use 10k instead.
            shuffle_buffer_size=50_000,
        ),
        num_train_steps=20_000,
        batch_size=16,
        # log_interval=100,
        save_interval=2_000,
        keep_period=2_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for distilling pi05 on the subset of DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        name="mimic_droid_50G",
        model=proxy_config.ProxyConfig(
            action_horizon=10,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid_50G",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="../data/droid_50G",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_droid_jointpos/assets",
                asset_id="droid",
            ),
            norm_stats_dir="checkpoints/pi0_droid_jointpos/assets/droid",
            # use_quantile_norm=True,
            # 10 GB subset has ~6k episodes × ~150 frames = ~900k frames total.
            # 250k (default) would hold 86 GB of decoded images in RAM. Use 10k instead.
            shuffle_buffer_size=50_000,
        ),
        num_train_steps=20_000,
        batch_size=16,
        # log_interval=100,
        save_interval=2_000,
        keep_period=2_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for distilling pi05 on the subset of DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        name="mimic_pi05_droid_150G",
        model=proxy_config.ProxyConfig(
            action_horizon=15,
            action_dim=8,
            action_expert_variant="gemma_12m",
            dino_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid_150G",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/share/ma/scratch/chuanruo/openpi_data/droid_150G",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            # norm_stats_dir="checkpoints/pi05_droid_jointpos/assets/droid",
            use_quantile_norm=True,
            # 150G with 128 cores: AUTOTUNE parallelism + 50k buffer exceeded 512G RAM.
            # 20k saves ~42 GB across 4 DDP ranks while still covering ~3% of per-shard data.
            shuffle_buffer_size=20_000,
        ),
        num_train_steps=20_000,
        batch_size=16,
        # log_interval=100,
        save_interval=2_000,
        keep_period=2_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_fast_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params"
        ),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        ),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy"
        ),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy"
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/debug/debug/9/params"
        ),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"
        ),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # VLM Action Expert configs.
    # This model uses VLM encoder from pi05 and action expert from proxy (finetuned).
    #
    TrainConfig(
        name="vlm_proxy_droid_spoon_jointvel",
        model=vlm_action_expert_config.VLMActionExpertConfig(
            action_horizon=15,
            action_dim=8,
            paligemma_variant="gemma_2b",  # VLM encoder (frozen)
            action_expert_variant="gemma_12m",  # Action expert (finetuned)
            dtype="bfloat16",
        ),
        data=ProxyLeRobotDROIDJointVelDataConfig(
            # Modify this to point to your dataset
            repo_id="cn356/spoon_vel",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="cn356/spoon_vel"),
            norm_stats_dir="checkpoints/pi05_droid/assets/droid",
        ),
        # For PyTorch training, use pytorch_weight_path to load VLM encoder weights
        # The training script will only load VLM encoder weights for VLMActionExpert models
        pytorch_weight_path="checkpoints/pytorch/pi05_droid",
        num_train_steps=20000,
        save_interval=10000,
        batch_size=64,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    config = tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )
    return _apply_manual_model_overrides(config, sys.argv[1:])


def _apply_manual_model_overrides(config: TrainConfig, argv: list[str]) -> TrainConfig:
    if not dataclasses.is_dataclass(config.model):
        return config

    model_field_map = {field.name: field for field in dataclasses.fields(config.model)}
    pending_updates: dict[str, Any] = {}

    index = 0
    while index < len(argv):
        arg = argv[index]
        if not arg.startswith("--model."):
            index += 1
            continue

        key = arg[len("--model.") :].replace("-", "_")
        field = model_field_map.get(key)
        if field is None:
            index += 1
            continue

        next_arg = argv[index + 1] if index + 1 < len(argv) else None
        if field.type is bool and (next_arg is None or next_arg.startswith("--")):
            pending_updates[key] = True
            index += 1
            continue

        if next_arg is None:
            raise ValueError(f"Missing value for model override {arg}")

        pending_updates[key] = _coerce_model_override_value(next_arg, field.type)
        index += 2

    if not pending_updates:
        return config

    return dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, **pending_updates),
    )


def _coerce_model_override_value(value: str, field_type):
    if field_type is bool:
        lowered = value.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Invalid boolean override value: {value}")

    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    if field_type is str:
        return value

    field_type_str = str(field_type)
    if "int | None" in field_type_str or "Optional[int]" in field_type_str:
        return None if value.lower() == "none" else int(value)
    if "float | None" in field_type_str or "Optional[float]" in field_type_str:
        return None if value.lower() == "none" else float(value)
    if "bool | None" in field_type_str or "Optional[bool]" in field_type_str:
        return None if value.lower() == "none" else _coerce_model_override_value(value, bool)
    if "str | None" in field_type_str or "Optional[str]" in field_type_str:
        return None if value.lower() == "none" else value

    return value


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
