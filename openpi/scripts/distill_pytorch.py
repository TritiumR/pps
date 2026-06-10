"""
PyTorch distillation entrypoint with multi-GPU DDP support.

Each GPU loads batch_size // num_gpus samples in parallel. For RLDS datasets (e.g. DROID),
the tf.data pipeline is sharded across GPUs so each process reads a disjoint subset.

Usage
Single GPU:
  python scripts/distill_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_checkpoint_dir <teacher_ckpt>
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
      scripts/distill_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_checkpoint_dir <teacher_ckpt>
Multi-Node:
  torchrun --nnodes=<N> --nproc_per_node=<gpus> --node_rank=<rank> \
      --master_addr=<ip> --master_port=<port> \
      scripts/distill_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_checkpoint_dir <teacher_ckpt>

"""

import dataclasses
import gc
import logging
import os
import pathlib
import platform
import shutil
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config
import openpi.models.proxy_config
import openpi.models.proxy_sound_config
import openpi.models.proxy_dp3_config
import openpi.models.tokenizer as _tokenizer
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.proxy_dp3_pytorch
import openpi.models_pytorch.proxy_pytorch
import openpi.models_pytorch.proxy_sound_pytorch
import openpi.policies.droid_policy as droid_policy
import openpi.policies.policy_config as policy_config
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.shared.download as download
import openpi.transforms as _transforms


def _config_name_from_checkpoint_dir(checkpoint_dir):
    """Derive the training-config name from a checkpoint dir path.

    Checkpoints follow ".../checkpoints/<config_name>/<exp>/<step>" or
    ".../checkpoints/pytorch/<config_name>".
    """
    parts = [p for p in os.path.normpath(os.fspath(checkpoint_dir)).split(os.sep) if p]
    if "checkpoints" in parts:
        idx = parts.index("checkpoints") + 1
        # skip an optional framework wrapper segment (e.g. "pytorch")
        if idx < len(parts) and parts[idx] == "pytorch":
            idx += 1
        if idx < len(parts):
            return parts[idx]
    raise ValueError(
        f"Could not derive a teacher config name from checkpoint dir: {checkpoint_dir!r}. "
        "Expected '.../checkpoints/<config_name>/<exp>/<step>' or "
        "'.../checkpoints/pytorch/<config_name>'."
    )


def init_logging():
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


@dataclasses.dataclass(frozen=True)
class _DroidImagePointCloudInputs:
    """Build a single observation with both PI image inputs and proxy pointcloud inputs."""

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        base_image = droid_policy._parse_image(data["observation/exterior_image_1_left"])
        wrist_image = droid_policy._parse_image(data["observation/wrist_image_left"])
        image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
            "pointcloud": droid_policy._parse_pointcloud(data),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class _DistillHybridDataConfigFactory:
    """Wrap a student data factory so image teachers and pointcloud students share a batch."""

    base_factory: _config.DataConfigFactory
    teacher_model_config: _model.BaseModelConfig

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> _config.DataConfig:
        data_config = self.base_factory.create(assets_dirs, model_config)
        if not _needs_hybrid_image_pointcloud_loader(model_config, self.teacher_model_config):
            return data_config

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
                _DroidImagePointCloudInputs(),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        default_prompt = getattr(self.base_factory, "default_prompt", None)
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(
                        self.teacher_model_config.max_token_len
                    ),
                    discrete_state_input=getattr(
                        self.teacher_model_config, "discrete_state_input", False
                    ),
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        logging.info(
            "Using hybrid distillation data transforms: images for teacher and pointcloud for student."
        )
        return dataclasses.replace(
            data_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class _DroidThermalDistillInputs:
    """Build an image observation with normal RGB plus thermal RGB side channels."""

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        base_image = droid_policy._parse_image(data["observation/exterior_image_1_left"])
        wrist_image = droid_policy._parse_image(data["observation/wrist_image_left"])
        image_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        images = [base_image, wrist_image, np.zeros_like(base_image)]
        image_masks = [np.True_, np.True_, np.False_]

        thermal_keys = (
            "observation/thermal_exterior_image_1_left",
            "observation/thermal_wrist_image_left",
        )
        missing_thermal = [key for key in thermal_keys if key not in data]
        if missing_thermal:
            raise KeyError(
                "Thermal student config requires thermal image fields in the dataset. "
                f"Missing keys after repack: {missing_thermal}"
            )
        image_names.extend(("thermal_base_0_rgb", "thermal_left_wrist_0_rgb"))
        images.extend(
            (
                droid_policy._parse_image(
                    data["observation/thermal_exterior_image_1_left"]
                ),
                droid_policy._parse_image(data["observation/thermal_wrist_image_left"]),
            )
        )
        image_masks.extend((np.True_, np.True_))

        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class _DistillThermalDataConfigFactory:
    """Load normal RGB and thermal RGB so only the student sees overlaid images."""

    base_factory: _config.DataConfigFactory
    teacher_model_config: _model.BaseModelConfig

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> _config.DataConfig:
        data_config = self.base_factory.create(assets_dirs, model_config)
        if not _needs_thermal_rgb_distill_loader(
            self.base_factory, model_config, self.teacher_model_config
        ):
            return data_config

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
                _DroidThermalDistillInputs(),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                droid_policy.DroidOutputs(),
            ],
        )

        default_prompt = getattr(self.base_factory, "default_prompt", None)
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(
                        self.teacher_model_config.max_token_len
                    ),
                    discrete_state_input=getattr(
                        self.teacher_model_config, "discrete_state_input", False
                    ),
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        logging.info(
            "Using thermal distillation data transforms: normal RGB for teacher and RGB+thermal overlay for student."
        )
        return dataclasses.replace(
            data_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class _DistillImageDataConfigFactory:
    """Use normal RGB student data transforms but tokenize prompts for the PI teacher."""

    base_factory: _config.DataConfigFactory
    teacher_model_config: _model.BaseModelConfig

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> _config.DataConfig:
        data_config = self.base_factory.create(assets_dirs, model_config)
        if not _needs_image_teacher_tokenization_loader(
            model_config, self.teacher_model_config
        ):
            return data_config

        default_prompt = getattr(self.base_factory, "default_prompt", None)
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(
                        self.teacher_model_config.max_token_len
                    ),
                    discrete_state_input=getattr(
                        self.teacher_model_config, "discrete_state_input", False
                    ),
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        logging.info(
            "Using image distillation data transforms: RGB inputs with teacher-compatible tokenization."
        )
        return dataclasses.replace(data_config, model_transforms=model_transforms)


@dataclasses.dataclass(frozen=True)
class _DistillTeacherActionNormStatsDataConfigFactory:
    """Use the teacher's state/action normalization as the distillation action space."""

    base_factory: _config.DataConfigFactory
    teacher_norm_stats: dict[str, _transforms.NormStats] | None

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> _config.DataConfig:
        data_config = self.base_factory.create(assets_dirs, model_config)
        if self.teacher_norm_stats is None:
            return data_config

        norm_stats = dict(data_config.norm_stats or {})
        replaced_keys = []
        for key in ("state", "actions"):
            if key in self.teacher_norm_stats:
                norm_stats[key] = self.teacher_norm_stats[key]
                replaced_keys.append(key)

        if not replaced_keys:
            logging.warning(
                "Teacher norm stats do not contain state/actions; using student data normalization."
            )
            return data_config

        logging.info(
            "Using teacher normalization stats for distillation keys: %s",
            replaced_keys,
        )
        return dataclasses.replace(data_config, norm_stats=norm_stats)


def _needs_hybrid_image_pointcloud_loader(
    student_model_config: _model.BaseModelConfig,
    teacher_model_config: _model.BaseModelConfig,
) -> bool:
    return student_model_config.model_type in (
        _model.ModelType.PROXY_POINTCLOUD,
        _model.ModelType.PROXY_DP3,
    ) and teacher_model_config.model_type in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    )


def _needs_image_teacher_tokenization_loader(
    student_model_config: _model.BaseModelConfig,
    teacher_model_config: _model.BaseModelConfig,
) -> bool:
    return student_model_config.model_type in (
        _model.ModelType.PROXY,
        _model.ModelType.PROXY_SOUND,
    ) and teacher_model_config.model_type in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    )


def _needs_thermal_rgb_distill_loader(
    data_factory: _config.DataConfigFactory,
    student_model_config: _model.BaseModelConfig,
    teacher_model_config: _model.BaseModelConfig,
) -> bool:
    return (
        isinstance(data_factory, _config.ProxyThermalLeRobotDROIDJointPosDataConfig)
        and student_model_config.model_type == _model.ModelType.PROXY
        and teacher_model_config.model_type in (_model.ModelType.PI0, _model.ModelType.PI05)
    )


def _with_distill_data_config(
    config: _config.TrainConfig,
    teacher_train_config: _config.TrainConfig,
    teacher_norm_stats: dict[str, _transforms.NormStats] | None = None,
) -> _config.TrainConfig:
    if _needs_thermal_rgb_distill_loader(
        config.data, config.model, teacher_train_config.model
    ):
        distill_config = dataclasses.replace(
            config,
            data=_DistillThermalDataConfigFactory(
                config.data, teacher_train_config.model
            ),
        )
        return _with_teacher_action_norm_stats(distill_config, teacher_norm_stats)
    if not _needs_hybrid_image_pointcloud_loader(config.model, teacher_train_config.model):
        if _needs_image_teacher_tokenization_loader(
            config.model, teacher_train_config.model
        ):
            distill_config = dataclasses.replace(
                config,
                data=_DistillImageDataConfigFactory(
                    config.data, teacher_train_config.model
                ),
            )
            return _with_teacher_action_norm_stats(distill_config, teacher_norm_stats)
        return _with_teacher_action_norm_stats(config, teacher_norm_stats)
    distill_config = dataclasses.replace(
        config,
        data=_DistillHybridDataConfigFactory(config.data, teacher_train_config.model),
    )
    return _with_teacher_action_norm_stats(distill_config, teacher_norm_stats)


def _with_teacher_action_norm_stats(
    config: _config.TrainConfig,
    teacher_norm_stats: dict[str, _transforms.NormStats] | None,
) -> _config.TrainConfig:
    if teacher_norm_stats is None:
        return config
    return dataclasses.replace(
        config,
        data=_DistillTeacherActionNormStatsDataConfigFactory(
            config.data,
            teacher_norm_stats,
        ),
    )


def _load_teacher_norm_stats(
    teacher_train_config: _config.TrainConfig,
    teacher_checkpoint_dir: pathlib.Path,
) -> dict[str, _transforms.NormStats] | None:
    teacher_data_config = teacher_train_config.data.create(
        teacher_train_config.assets_dirs,
        teacher_train_config.model,
    )
    norm_stats = policy_config._load_checkpoint_norm_stats(
        teacher_checkpoint_dir,
        teacher_data_config.asset_id,
    )
    if norm_stats is not None:
        return norm_stats

    if teacher_data_config.norm_stats is not None:
        logging.info("Using teacher config normalization stats for distillation.")
        return teacher_data_config.norm_stats

    logging.warning(
        "Teacher normalization stats were not found in checkpoint or config. "
        "Distillation will use the student data normalization."
    )
    return None


def log_teacher_tokenization_config(
    config: _config.TrainConfig, teacher_train_config: _config.TrainConfig
):
    logging.info(
        "Distill loader teacher tokenization: student_model_type=%s teacher_model_type=%s max_token_len=%s tokenize_state_input=%s",
        config.model.model_type,
        teacher_train_config.model.model_type,
        getattr(teacher_train_config.model, "max_token_len", None),
        getattr(teacher_train_config.model, "discrete_state_input", False),
    )


def build_datasets(
    config: _config.TrainConfig,
    teacher_train_config: _config.TrainConfig,
    teacher_norm_stats: dict[str, _transforms.NormStats] | None = None,
):
    # Use the unified data loader with PyTorch framework
    loader_config = _with_distill_data_config(
        config,
        teacher_train_config,
        teacher_norm_stats,
    )
    data_loader = _data.create_data_loader(
        loader_config, framework="pytorch", shuffle=True
    )
    return data_loader, data_loader.data_config()


def _uses_thermal_overlay_student(config: _config.TrainConfig) -> bool:
    return isinstance(config.data, _config.ProxyThermalLeRobotDROIDJointPosDataConfig)


def _overlay_thermal_student_observation(observation, alpha: float):
    if observation.images is None:
        raise ValueError("Thermal overlay distillation requires image observations.")

    if "base_0_rgb" not in observation.images or "left_wrist_0_rgb" not in observation.images:
        raise ValueError("Thermal overlay distillation requires normal RGB image keys.")
    if (
        "thermal_base_0_rgb" not in observation.images
        or "thermal_left_wrist_0_rgb" not in observation.images
    ):
        raise ValueError(
            "Thermal student config requires thermal image keys in the observation."
        )

    images = dict(observation.images)
    alpha = float(alpha)
    images["base_0_rgb"] = (
        (1.0 - alpha) * images["base_0_rgb"] + alpha * images["thermal_base_0_rgb"]
    )
    images["left_wrist_0_rgb"] = (
        (1.0 - alpha) * images["left_wrist_0_rgb"]
        + alpha * images["thermal_left_wrist_0_rgb"]
    )
    return dataclasses.replace(observation, images=images)


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (
        global_step % config.save_interval == 0 and global_step > 0
    ) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            )
            safetensors.torch.load_model(
                model_to_load, safetensors_path, device=str(device)
            )
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(
                optimizer_path, map_location=device, weights_only=False
            )
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(
            ckpt_dir / "metadata.pt", map_location=device, weights_only=False
        )
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(
            f"Successfully loaded all checkpoint components from step {latest_step}"
        )
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
        device
    )
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(
                    f"No valid checkpoints found in {exp_checkpoint_dir} for resume"
                )
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume"
            )
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(
            f"Using existing experiment checkpoint directory: {config.checkpoint_dir}"
        )

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Load the teacher config before constructing the dataset: pointcloud students
    # need hybrid batches that also include the teacher's image inputs.
    teacher_checkpoint_dir = getattr(config, "teacher_checkpoint_dir", None)
    if teacher_checkpoint_dir is None:
        raise ValueError(
            "teacher_checkpoint_dir must be specified for distillation. "
            "Use --teacher_checkpoint_dir <checkpoint_dir>"
        )

    # The teacher config name is inferred from its checkpoint dir; pass
    # --teacher_config_name only to override the inferred value.
    teacher_config_name = getattr(config, "teacher_config_name", None)
    if teacher_config_name is None:
        teacher_config_name = _config_name_from_checkpoint_dir(teacher_checkpoint_dir)

    teacher_train_config = _config.get_config(teacher_config_name)
    log_teacher_tokenization_config(config, teacher_train_config)
    teacher_checkpoint_dir = pathlib.Path(teacher_checkpoint_dir)
    teacher_norm_stats = _load_teacher_norm_stats(
        teacher_train_config,
        teacher_checkpoint_dir,
    )

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(
        config,
        teacher_train_config,
        teacher_norm_stats,
    )

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(
            _with_distill_data_config(
                config,
                teacher_train_config,
                teacher_norm_stats,
            ),
            framework="pytorch",
            shuffle=False,
        )
        # print("length of sample_data_loader", len(sample_data_loader))
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions, noise = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        if "image" in sample_batch:
            images_to_log = []
            batch_size = next(iter(sample_batch["image"].values())).shape[0]
            for i in range(min(5, batch_size)):
                img_concatenated = torch.cat(
                    [img[i].permute(1, 2, 0) for img in sample_batch["image"].values()],
                    axis=1,
                )
                img_concatenated = img_concatenated.cpu().numpy()
                images_to_log.append(wandb.Image(img_concatenated))
            wandb.log({"camera_views": images_to_log}, step=0)
        pointcloud = sample_batch.get("pointcloud")
        if pointcloud is not None:
            num_points = pointcloud.shape[1]
            feature_dim = pointcloud.shape[2]
            wandb.log(
                {
                    "pointcloud_num_points": num_points,
                    "pointcloud_feature_dim": feature_dim,
                },
                step=0,
            )
            if isinstance(
                config.model,
                openpi.models.proxy_dp3_config.ProxyDP3Config,
            ):
                expected_num_points = config.model.num_points
                if num_points != expected_num_points:
                    logging.warning(
                        "Point cloud loader produced %s points per sample, but the model is configured for %s. "
                        "The model will resize the cloud before encoding.",
                        num_points,
                        expected_num_points,
                    )
                if feature_dim < 6 and config.model.use_pc_color:
                    logging.warning(
                        "Point cloud loader produced feature dim %s, but the model expects XYZ+RGB input.",
                        feature_dim,
                    )

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, noise
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    if is_main:
        logging.info(
            f"Loading teacher model: config={teacher_config_name}, checkpoint={teacher_checkpoint_dir}"
        )

    # print(f"teacher_train_config: {teacher_train_config.model.action_dim}")
    # print(f"teacher_train_config: {teacher_train_config.model.action_horizon}")
    # Load teacher model using the same mechanism as create_trained_policy
    teacher_weight_path = teacher_checkpoint_dir / "model.safetensors"
    if not teacher_weight_path.exists():
        raise FileNotFoundError(
            f"Teacher model checkpoint not found at {teacher_weight_path}"
        )

    teacher_model = teacher_train_config.model.load_pytorch(
        teacher_train_config, str(teacher_weight_path)
    )

    teacher_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # Move to device and set to eval mode
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False  # Freeze teacher model

    if is_main:
        logging.info(f"Loaded teacher model from {teacher_checkpoint_dir}")

    # Get number of distillation steps from config or use default
    num_distill_steps = getattr(config, "num_distill_steps", 10)
    teacher_flow_path_noise_std = getattr(config, "teacher_flow_path_noise_std", 0.0)
    use_noise_for_distill = getattr(config, "use_noise_for_distill", True)
    use_thermal_overlay_student = _uses_thermal_overlay_student(config)
    thermal_overlay_alpha = getattr(config.data, "thermal_alpha", 0.5)

    # Build student model
    if isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        model = openpi.models_pytorch.proxy_pytorch.ProxyPytorch(config.model).to(device)
    elif isinstance(config.model, openpi.models.proxy_dp3_config.ProxyDP3Config):
        model = openpi.models_pytorch.proxy_dp3_pytorch.ProxyDP3Pytorch(
            config.model
        ).to(device)
    elif isinstance(config.model, openpi.models.proxy_sound_config.ProxySoundConfig):
        model = openpi.models_pytorch.proxy_sound_pytorch.ProxySoundPytorch(
            config.model
        ).to(device)
    else:
        raise ValueError(
            "Student model must be ProxyConfig, ProxyDP3Config, or ProxySoundConfig for distillation."
        )
    model_cfg = config.model

    enable_gradient_checkpointing = False
    logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:128,expandable_segments:True"
        )
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            ),
            model_path,
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(
            "Distillation config: "
            f"teacher_config={teacher_config_name}, "
            f"teacher_checkpoint={teacher_checkpoint_dir}, "
            f"num_distill_steps={num_distill_steps}, "
            f"teacher_flow_path_noise_std={teacher_flow_path_noise_std}, "
            f"use_noise_for_distill={use_noise_for_distill}"
        )
        logging.info(
            f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}"
        )
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Training",
            disable=not is_main,
        )
        if is_main
        else None
    )

    # Synchronize all processes before training begins
    if use_ddp:
        dist.barrier()

    while global_step < config.num_train_steps:
        for observation, _, _ in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions, noise) tuple
            observation = jax.tree.map(
                lambda x: x.to(device), observation
            )  # noqa: PLW2901
            teacher_observation = observation
            student_observation = (
                _overlay_thermal_student_observation(
                    observation, alpha=thermal_overlay_alpha
                )
                if use_thermal_overlay_student
                else observation
            )

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Distillation forward pass:
            with torch.no_grad():
                noises, times, gradients, teacher_actions = (
                    teacher_model.forward_for_distill(
                        teacher_observation,
                        num_distill_steps,
                        teacher_flow_path_noise_std=teacher_flow_path_noise_std,
                    )
                )

            # Use the DDP forward path so reducer bookkeeping stays synchronized.
            losses = model(
                student_observation,
                teacher_actions,
                noises=noises,
                times=times,
                gradients=gradients,
                mode="distill",
                use_noise=use_noise_for_distill,
            )
            # losses shape: (batch_size, num_steps, action_horizon, action_dim)
            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.optimizer.clip_gradient_norm
            )

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": (
                            float(grad_norm)
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        ),
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"]
                        for info in infos
                        if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint on main process; barrier around actual saves to keep ranks in sync.
            is_save_step = (
                (global_step % config.save_interval == 0 and global_step > 0)
                or global_step == config.num_train_steps - 1
            )
            if use_ddp and is_save_step:
                dist.barrier()
            save_checkpoint(model, optim, global_step, config, is_main, data_config)
            if use_ddp and is_save_step:
                dist.barrier()

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step,
                    }
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
