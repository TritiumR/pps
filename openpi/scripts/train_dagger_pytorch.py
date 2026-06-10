"""
Single-GPU DAgger-style PyTorch training entrypoint.

This script combines:
1. Supervised student training on ground-truth actions, like `train_pytorch.py`
2. Ground-truth flow supervision on a teacher/student mixed rollout path.

Only the student model is trained. The teacher model is loaded once, frozen, and
used with the current student to generate a flow path:

    v_t = teacher_v_t + gamma * (student_v_t - teacher_v_t)

The student is then supervised at the visited path states with
`path_noise - ground_truth_action`.

Usage:
  python scripts/train_dagger_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_config_name <teacher_cfg> \
      --teacher_checkpoint_dir <teacher_ckpt> \
      --dagger_batch_size <dagger_batch_size> \
      --warm_up_student_steps <warmup_steps> \
      --dagger_min_time 0.01 \
      --dagger_target_max_norm 8
"""

import argparse
import dataclasses
import gc
import logging
import os
import pathlib
import platform
import shutil
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config
import openpi.models.proxy_config
import openpi.models.tokenizer as _tokenizer
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import openpi.models_pytorch.proxy_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.transforms as _transforms


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


def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return device


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclasses.dataclass(frozen=True)
class _DistillImageDataConfigFactory:
    """Use student data transforms but teacher-compatible tokenization."""

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
        return dataclasses.replace(data_config, model_transforms=model_transforms)


def _needs_image_teacher_tokenization_loader(
    student_model_config: _model.BaseModelConfig,
    teacher_model_config: _model.BaseModelConfig,
) -> bool:
    return student_model_config.model_type == _model.ModelType.PROXY and teacher_model_config.model_type in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    )


def _with_distill_data_config(
    config: _config.TrainConfig, teacher_train_config: _config.TrainConfig
) -> _config.TrainConfig:
    if _needs_image_teacher_tokenization_loader(
        config.model, teacher_train_config.model
    ):
        return dataclasses.replace(
            config,
            data=_DistillImageDataConfigFactory(config.data, teacher_train_config.model),
        )
    return config


def log_teacher_tokenization_config(
    config: _config.TrainConfig, teacher_train_config: _config.TrainConfig
):
    logging.info(
        "DAgger loader teacher tokenization: student_model_type=%s teacher_model_type=%s max_token_len=%s tokenize_state_input=%s",
        config.model.model_type,
        teacher_train_config.model.model_type,
        getattr(teacher_train_config.model, "max_token_len", None),
        getattr(teacher_train_config.model, "discrete_state_input", False),
    )


def build_datasets(
    config: _config.TrainConfig,
    teacher_train_config: _config.TrainConfig,
    *,
    batch_size: int | None = None,
):
    loader_config = _with_distill_data_config(config, teacher_train_config)
    if batch_size is not None:
        loader_config = dataclasses.replace(loader_config, batch_size=batch_size)
    data_loader = _data.create_data_loader(
        loader_config, framework="pytorch", shuffle=True
    )
    return data_loader, data_loader.data_config()


def get_latest_checkpoint_step(checkpoint_dir: pathlib.Path) -> int | None:
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def resolve_model_safetensors(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str).expanduser().resolve()

    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got: {path}")
        return path

    direct_model = path / "model.safetensors"
    if direct_model.exists():
        return direct_model

    latest_step = get_latest_checkpoint_step(path)
    if latest_step is None:
        raise FileNotFoundError(
            f"Could not find model.safetensors or numeric checkpoint subdirectories in {path}"
        )

    model_path = path / str(latest_step) / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Resolved checkpoint is missing model.safetensors: {model_path}")
    return model_path


def save_checkpoint(model, optimizer, global_step, config, data_config):
    if (
        global_step % config.save_interval == 0 and global_step > 0
    ) or global_step == config.num_train_steps - 1:
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        safetensors.torch.save_model(model, tmp_ckpt_dir / "model.safetensors")
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info("Saved checkpoint at step %s -> %s", global_step, final_ckpt_dir)

        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")
    safetensors.torch.load_model(model, safetensors_path, device=str(device))

    optimizer_path = ckpt_dir / "optimizer.pt"
    if not optimizer_path.exists():
        raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")
    optimizer_state_dict = torch.load(
        optimizer_path, map_location=device, weights_only=False
    )
    optimizer.load_state_dict(optimizer_state_dict)

    metadata = torch.load(
        ckpt_dir / "metadata.pt", map_location=device, weights_only=False
    )
    global_step = metadata.get("global_step", latest_step)
    logging.info("Resumed from checkpoint step %s", global_step)
    return global_step


def freeze_model(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_teacher_model(config: _config.TrainConfig, device: torch.device):
    teacher_config_name = getattr(config, "teacher_config_name", None)
    teacher_checkpoint_dir = getattr(config, "teacher_checkpoint_dir", None)

    if teacher_config_name is None or teacher_checkpoint_dir is None:
        raise ValueError(
            "teacher_config_name and teacher_checkpoint_dir must be specified. "
            "Use --teacher_config_name <config_name> --teacher_checkpoint_dir <checkpoint_dir>"
        )

    teacher_train_config = _config.get_config(teacher_config_name)
    teacher_weight_path = resolve_model_safetensors(teacher_checkpoint_dir)
    teacher_model = teacher_train_config.model.load_pytorch(
        teacher_train_config, str(teacher_weight_path)
    )

    if not hasattr(teacher_model, "denoise_step"):
        raise ValueError(
            f"Teacher model {type(teacher_model).__name__} does not support denoise_step()."
        )

    if hasattr(teacher_model, "paligemma_with_expert"):
        teacher_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    teacher_model = teacher_model.to(device)
    freeze_model(teacher_model)
    logging.info("Loaded teacher model from %s", teacher_weight_path)
    return teacher_model


def ensure_tensor_loss(losses, device):
    if isinstance(losses, list | tuple):
        return torch.stack(losses)
    if not isinstance(losses, torch.Tensor):
        return torch.tensor(losses, device=device, dtype=torch.float32)
    return losses


def slice_batch(batch, batch_size: int):
    return jax.tree.map(
        lambda x: x[:batch_size] if isinstance(x, torch.Tensor) else x,
        batch,
    )


def build_path_flow_targets(
    noises: torch.Tensor,
    times: torch.Tensor,
    actions: torch.Tensor,
    action_dim: int,
    min_time: float,
    target_max_norm: float,
) -> torch.Tensor:
    """Create path-conditioned targets (x_t - action) / t for every visited state."""
    path_noises = noises[:, :, :, :action_dim]
    actions = actions[:, :, :action_dim]
    if path_noises.shape[0] != actions.shape[0] or path_noises.shape[2:] != actions.shape[1:]:
        raise ValueError(
            "Path noises and ground-truth actions have incompatible shapes: "
            f"{path_noises.shape} vs {actions.shape}"
        )
    if times.shape != path_noises.shape[:2]:
        raise ValueError(
            "Times and path noises have incompatible shapes: "
            f"{times.shape} vs {path_noises.shape}"
        )

    clamped_times = torch.clamp(times, min=min_time)
    targets = (path_noises - actions[:, None, :, :]) / clamped_times[:, :, None, None]

    if target_max_norm > 0:
        target_norm = targets.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(target_max_norm / target_norm, max=1.0)
        targets = targets * scale

    return targets


def predict_proxy_flow(
    student_model: openpi.models_pytorch.proxy_pytorch.ProxyPytorch,
    prefix_embs: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    state: torch.Tensor,
    x_t: torch.Tensor,
    time_step: torch.Tensor,
) -> torch.Tensor:
    suffix_embs, suffix_pad_masks, _, adarms_cond = student_model.embed_suffix(
        state,
        x_t[:, :, : student_model.config.action_dim],
        time_step,
    )
    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    position_ids = position_ids.to(dtype=torch.long)

    hidden_states, _ = student_model.expert_model.forward(
        attention_mask=pad_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -student_model.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return student_model.action_out_proj(suffix_out)


@torch.no_grad()
def generate_dagger_flow_path(
    teacher_model: torch.nn.Module,
    student_model: openpi.models_pytorch.proxy_pytorch.ProxyPytorch,
    observation,
    num_steps: int,
    gamma: float | torch.Tensor,
    *,
    min_time: float,
    teacher_flow_path_noise_std: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out a path using teacher_v + gamma * (student_v - teacher_v)."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be greater than 0, got {num_steps}")
    if teacher_flow_path_noise_std < 0:
        raise ValueError(
            "teacher_flow_path_noise_std must be non-negative, got "
            f"{teacher_flow_path_noise_std}"
        )
    if min_time <= 0 or min_time >= 1:
        raise ValueError(f"min_time must be in (0, 1), got {min_time}")

    (
        teacher_images,
        teacher_img_masks,
        teacher_lang_tokens,
        teacher_lang_masks,
        teacher_state,
    ) = teacher_model._preprocess_observation(observation, train=False)
    teacher_state = torch.nn.functional.pad(
        teacher_state,
        (0, teacher_model.config.action_dim - teacher_state.shape[1]),
        mode="constant",
        value=0,
    )

    student_images, student_img_masks, student_state = student_model._preprocess_observation(
        observation,
        train=False,
    )

    bsize = teacher_state.shape[0]
    device = teacher_state.device
    student_action_dim = student_model.config.action_dim

    # if teacher_model.config.action_horizon != student_model.config.action_horizon:
    #     raise ValueError(
    #         "Teacher and student action horizons must match: "
    #         f"{teacher_model.config.action_horizon} vs {student_model.config.action_horizon}"
    #     )
    # if teacher_model.config.action_dim < student_action_dim:
    #     raise ValueError(
    #         "Teacher action_dim must be >= student action_dim for mixed rollout: "
    #         f"{teacher_model.config.action_dim} vs {student_action_dim}"
    #     )

    actions_shape = (
        bsize,
        teacher_model.config.action_horizon,
        teacher_model.config.action_dim,
    )
    x_t = teacher_model.sample_noise(actions_shape, device)
    time_schedule = teacher_model.sample_bin_times(bsize, num_steps, device)
    time_schedule = torch.clamp(time_schedule, min=min_time)
    end_time = torch.tensor(0.0, dtype=torch.float32, device=device).expand(bsize)
    time_schedule = torch.cat([time_schedule, end_time.unsqueeze(1)], dim=1)
    time_schedule = time_schedule.transpose(0, 1)

    teacher_prefix_embs, teacher_prefix_pad_masks, teacher_prefix_att_masks = (
        teacher_model.embed_prefix(
            teacher_images,
            teacher_img_masks,
            teacher_lang_tokens,
            teacher_lang_masks,
        )
    )
    teacher_prefix_att_2d_masks = make_att_2d_masks(
        teacher_prefix_pad_masks,
        teacher_prefix_att_masks,
    )
    teacher_prefix_position_ids = torch.cumsum(teacher_prefix_pad_masks, dim=1) - 1
    teacher_prefix_att_2d_masks_4d = teacher_model._prepare_attention_masks_4d(
        teacher_prefix_att_2d_masks
    )
    teacher_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    _, teacher_past_key_values = teacher_model.paligemma_with_expert.forward(
        attention_mask=teacher_prefix_att_2d_masks_4d,
        position_ids=teacher_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[teacher_prefix_embs, None],
        use_cache=True,
    )

    student_prefix_embs, student_prefix_pad_masks, _ = student_model.embed_prefix(
        student_images,
        student_img_masks,
    )

    current_time = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)
    noises = []
    times = []

    for target_time in time_schedule:
        if teacher_flow_path_noise_std > 0:
            x_t = x_t + torch.randn_like(x_t) * teacher_flow_path_noise_std

        noises.append(x_t.clone())
        times.append(current_time.clone())

        teacher_v_t = teacher_model.denoise_step(
            teacher_state,
            teacher_prefix_pad_masks,
            teacher_past_key_values,
            x_t,
            current_time,
        )
        student_v_t = predict_proxy_flow(
            student_model,
            student_prefix_embs,
            student_prefix_pad_masks,
            student_state,
            x_t,
            current_time,
        )

        v_t = teacher_v_t.clone()
        gamma_expanded = (
            gamma[:, None, None]
            if isinstance(gamma, torch.Tensor)
            else gamma
        )
        v_t[:, :, :student_action_dim] = teacher_v_t[:, :, :student_action_dim] + gamma_expanded * (
            student_v_t - teacher_v_t[:, :, :student_action_dim]
        )

        dt = (target_time - current_time)[:, None, None]
        x_t = x_t + dt * v_t
        current_time = target_time

    return (
        torch.stack(noises, dim=1),
        torch.stack(times, dim=1),
    )


def train_loop(
    config: _config.TrainConfig,
    *,
    action_loss_weight: float,
    action_batch_size: int | None,
    dagger_batch_size: int | None,
    dagger_gamma: float | None,
    dagger_gamma_min: float,
    dagger_gamma_max: float,
    dagger_min_time: float,
    dagger_target_max_norm: float,
    warm_up_student_steps: int,
):
    device = setup_device()
    set_seed(config.seed)

    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(config.checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    "Resuming from experiment checkpoint directory: %s at step %s",
                    config.checkpoint_dir,
                    latest_step,
                )
            else:
                raise FileNotFoundError(
                    f"No valid checkpoints found in {config.checkpoint_dir} for resume"
                )
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume"
            )
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Created experiment checkpoint directory: %s", config.checkpoint_dir)
    else:
        logging.info("Using existing experiment checkpoint directory: %s", config.checkpoint_dir)

    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    teacher_config_name = getattr(config, "teacher_config_name", None)
    if teacher_config_name is None:
        raise ValueError(
            "teacher_config_name must be specified. "
            "Use --teacher_config_name <config_name>"
        )
    teacher_train_config = _config.get_config(teacher_config_name)
    log_teacher_tokenization_config(config, teacher_train_config)

    effective_action_batch_size = (
        config.batch_size if action_batch_size is None else action_batch_size
    )
    effective_dagger_batch_size = (
        config.batch_size if dagger_batch_size is None else dagger_batch_size
    )
    warmup_loader_batch_size = effective_action_batch_size
    dagger_loader_batch_size = (
        max(effective_action_batch_size, effective_dagger_batch_size)
        if action_loss_weight > 0
        else effective_dagger_batch_size
    )

    if not isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        raise ValueError(
            "DAgger training currently requires the student model to be ProxyConfig."
        )

    student_model = openpi.models_pytorch.proxy_pytorch.ProxyPytorch(config.model).to(
        device
    )
    teacher_model = None

    if config.pytorch_weight_path is not None and not resuming:
        student_weight_path = resolve_model_safetensors(config.pytorch_weight_path)
        safetensors.torch.load_model(student_model, student_weight_path)
        logging.info("Loaded student weights from %s", student_weight_path)

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        student_model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(student_model, optim, config.checkpoint_dir, device)

    current_loader_batch_size = (
        dagger_loader_batch_size
        if global_step >= warm_up_student_steps
        else warmup_loader_batch_size
    )
    action_loader, action_data_config = build_datasets(
        config, teacher_train_config, batch_size=current_loader_batch_size
    )
    logging.info(
        "Built initial dataloader with batch_size=%s",
        current_loader_batch_size,
    )

    if global_step >= warm_up_student_steps and global_step < config.num_train_steps:
        teacher_model = load_teacher_model(config, device)

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    student_model.train()
    start_time = time.time()
    infos = []
    num_dagger_steps = getattr(config, "num_distill_steps", 10)
    teacher_flow_path_noise_std = getattr(config, "teacher_flow_path_noise_std", 0.0)

    logging.info("Running on: %s", platform.node())
    logging.info(
        "Training config: warmup_loader_batch_size=%s post_warmup_loader_batch_size=%s action_batch_size=%s dagger_batch_size=%s (default batch_size=%s) num_train_steps=%s",
        warmup_loader_batch_size,
        dagger_loader_batch_size,
        effective_action_batch_size,
        effective_dagger_batch_size,
        config.batch_size,
        config.num_train_steps,
    )
    logging.info(
        "DAgger config: action_loss_weight=%.3f warm_up_student_steps=%s num_dagger_steps=%s dagger_gamma=%s dagger_gamma_min=%.3f dagger_gamma_max=%.3f dagger_min_time=%.4f dagger_target_max_norm=%.3f teacher_flow_path_noise_std=%s",
        action_loss_weight,
        warm_up_student_steps,
        num_dagger_steps,
        "sampled" if dagger_gamma is None else f"{dagger_gamma:.3f}",
        dagger_gamma_min,
        dagger_gamma_max,
        dagger_min_time,
        dagger_target_max_norm,
        teacher_flow_path_noise_std,
    )
    logging.info(
        "Action dataset asset_id=%s",
        action_data_config.asset_id,
    )

    pbar = tqdm.tqdm(
        total=config.num_train_steps,
        initial=global_step,
        desc="Training",
    )

    action_iter = iter(action_loader)

    while global_step < config.num_train_steps:
        if (
            global_step >= warm_up_student_steps
            and current_loader_batch_size != dagger_loader_batch_size
        ):
            current_loader_batch_size = dagger_loader_batch_size
            action_loader, action_data_config = build_datasets(
                config, teacher_train_config, batch_size=current_loader_batch_size
            )
            action_iter = iter(action_loader)
            logging.info(
                "Warm-up finished at step %s; rebuilt dataloader with batch_size=%s",
                global_step,
                current_loader_batch_size,
            )

        if global_step >= warm_up_student_steps and teacher_model is None:
            teacher_model = load_teacher_model(config, device)

        batch_observation, batch_actions, batch_noise = next(action_iter)

        batch_observation = jax.tree.map(lambda x: x.to(device), batch_observation)
        batch_actions = batch_actions.to(torch.float32).to(device)
        if batch_noise is not None:
            batch_noise = batch_noise.to(torch.float32).to(device)

        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        action_loss = None
        dagger_loss = None
        action_grad_norm = None
        dagger_grad_norm = None
        gamma = None
        target_norm_mean = None
        target_norm_max = None
        target_clip_fraction = None
        in_student_warmup = global_step < warm_up_student_steps
        effective_action_loss_weight = 1.0 if in_student_warmup else action_loss_weight

        if effective_action_loss_weight > 0:
            action_observation = slice_batch(batch_observation, effective_action_batch_size)
            action_actions = batch_actions[:effective_action_batch_size]
            action_noise = (
                batch_noise[:effective_action_batch_size]
                if batch_noise is not None
                else None
            )
            action_losses = student_model(
                action_observation, action_actions, noise=action_noise
            )
            action_losses = ensure_tensor_loss(action_losses, device)
            action_loss = action_losses.mean()
            weighted_action_loss = action_loss * effective_action_loss_weight

            optim.zero_grad(set_to_none=True)
            weighted_action_loss.backward()
            action_grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            del action_losses, weighted_action_loss

        if not in_student_warmup:
            if teacher_model is None:
                raise RuntimeError("Teacher model was not loaded before DAgger training.")

            dagger_observation = slice_batch(
                batch_observation, effective_dagger_batch_size
            )
            dagger_actions = batch_actions[:effective_dagger_batch_size]

            if dagger_gamma is None:
                gamma = torch.empty(
                    effective_dagger_batch_size,
                    dtype=torch.float32,
                    device=device,
                ).uniform_(dagger_gamma_min, dagger_gamma_max)
            else:
                gamma = dagger_gamma

            student_model.eval()
            noises, times = generate_dagger_flow_path(
                teacher_model,
                student_model,
                dagger_observation,
                num_dagger_steps,
                gamma,
                min_time=dagger_min_time,
                teacher_flow_path_noise_std=teacher_flow_path_noise_std,
            )
            student_model.train()
            with torch.no_grad():
                unclipped_targets = build_path_flow_targets(
                    noises,
                    times,
                    dagger_actions,
                    student_model.config.action_dim,
                    dagger_min_time,
                    -1.0,
                )
                unclipped_target_norms = unclipped_targets.norm(dim=-1)
                target_norm_mean = float(unclipped_target_norms.mean().item())
                target_norm_max = float(unclipped_target_norms.max().item())
                if dagger_target_max_norm > 0:
                    target_clip_fraction = float(
                        (unclipped_target_norms > dagger_target_max_norm)
                        .to(torch.float32)
                        .mean()
                        .item()
                    )
                else:
                    target_clip_fraction = 0.0
                gt_gradients = build_path_flow_targets(
                    noises,
                    times,
                    dagger_actions,
                    student_model.config.action_dim,
                    dagger_min_time,
                    dagger_target_max_norm,
                )

            dagger_losses = student_model.forward_distill(
                dagger_observation,
                noises,
                times,
                gt_gradients,
                dagger_actions,
                use_noise=True,
            )
            dagger_loss = dagger_losses.mean()

            optim.zero_grad(set_to_none=True)
            dagger_loss.backward()
            dagger_grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            del dagger_losses

        total_loss_value = (
            (
                action_loss.item() * effective_action_loss_weight
                if action_loss is not None
                else 0.0
            )
            + (dagger_loss.item() if dagger_loss is not None else 0.0)
        )
        infos.append(
            {
                "loss": total_loss_value,
                "action_loss": action_loss.item() if action_loss is not None else None,
                "dagger_loss": dagger_loss.item() if dagger_loss is not None else None,
                "learning_rate": optim.param_groups[0]["lr"],
                "dagger_gamma": (
                    None
                    if gamma is None
                    else (
                        float(gamma.mean().item())
                        if isinstance(gamma, torch.Tensor)
                        else float(gamma)
                    )
                ),
                "student_warmup": float(in_student_warmup),
                "dagger_target_norm_mean": target_norm_mean,
                "dagger_target_norm_max": target_norm_max,
                "dagger_target_clip_fraction": target_clip_fraction,
                "action_grad_norm": (
                    float(action_grad_norm)
                    if isinstance(action_grad_norm, torch.Tensor)
                    else action_grad_norm
                ),
                "dagger_grad_norm": (
                    float(dagger_grad_norm)
                    if isinstance(dagger_grad_norm, torch.Tensor)
                    else dagger_grad_norm
                ),
            }
        )

        if global_step % config.log_interval == 0:
            elapsed = time.time() - start_time

            avg_loss = sum(info["loss"] for info in infos) / len(infos)
            avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
            dagger_gamma_vals = [
                info["dagger_gamma"]
                for info in infos
                if info["dagger_gamma"] is not None
            ]
            avg_dagger_gamma = (
                sum(dagger_gamma_vals) / len(dagger_gamma_vals)
                if len(dagger_gamma_vals) > 0
                else None
            )
            warmup_vals = [info["student_warmup"] for info in infos]
            avg_student_warmup = sum(warmup_vals) / len(warmup_vals)

            action_vals = [
                info["action_loss"]
                for info in infos
                if info["action_loss"] is not None
            ]
            dagger_vals = [
                info["dagger_loss"]
                for info in infos
                if info["dagger_loss"] is not None
            ]
            avg_action_loss = (
                sum(action_vals) / len(action_vals) if len(action_vals) > 0 else None
            )
            avg_dagger_loss = (
                sum(dagger_vals) / len(dagger_vals)
                if len(dagger_vals) > 0
                else None
            )
            action_grad_vals = [
                info["action_grad_norm"]
                for info in infos
                if info["action_grad_norm"] is not None
            ]
            dagger_grad_vals = [
                info["dagger_grad_norm"]
                for info in infos
                if info["dagger_grad_norm"] is not None
            ]
            target_norm_mean_vals = [
                info["dagger_target_norm_mean"]
                for info in infos
                if info["dagger_target_norm_mean"] is not None
            ]
            target_norm_max_vals = [
                info["dagger_target_norm_max"]
                for info in infos
                if info["dagger_target_norm_max"] is not None
            ]
            target_clip_fraction_vals = [
                info["dagger_target_clip_fraction"]
                for info in infos
                if info["dagger_target_clip_fraction"] is not None
            ]
            avg_action_grad_norm = (
                sum(action_grad_vals) / len(action_grad_vals)
                if len(action_grad_vals) > 0
                else None
            )
            avg_dagger_grad_norm = (
                sum(dagger_grad_vals) / len(dagger_grad_vals)
                if len(dagger_grad_vals) > 0
                else None
            )
            avg_target_norm_mean = (
                sum(target_norm_mean_vals) / len(target_norm_mean_vals)
                if len(target_norm_mean_vals) > 0
                else None
            )
            max_target_norm = (
                max(target_norm_max_vals)
                if len(target_norm_max_vals) > 0
                else None
            )
            avg_target_clip_fraction = (
                sum(target_clip_fraction_vals) / len(target_clip_fraction_vals)
                if len(target_clip_fraction_vals) > 0
                else None
            )

            if avg_dagger_loss is None:
                log_str = (
                    f"step={global_step} loss={avg_loss:.4f} action_loss={avg_action_loss:.4f} "
                    f"lr={avg_lr:.2e} student_warmup={avg_student_warmup:.2f} "
                    f"action_grad_norm={avg_action_grad_norm:.2f} time={elapsed:.1f}s"
                )
            elif avg_action_loss is None:
                log_str = (
                    f"step={global_step} loss={avg_loss:.4f} dagger_loss={avg_dagger_loss:.4f} "
                    f"lr={avg_lr:.2e} dagger_gamma={avg_dagger_gamma:.3f} "
                    f"student_warmup={avg_student_warmup:.2f} dagger_grad_norm={avg_dagger_grad_norm:.2f} "
                    f"target_norm_mean={avg_target_norm_mean:.2f} target_norm_max={max_target_norm:.2f} "
                    f"target_clip_frac={avg_target_clip_fraction:.3f} "
                    f"time={elapsed:.1f}s"
                )
            else:
                log_str = (
                    f"step={global_step} loss={avg_loss:.4f} "
                    f"action_loss={avg_action_loss:.4f} "
                    f"dagger_loss={avg_dagger_loss:.4f} "
                    f"lr={avg_lr:.2e} dagger_gamma={avg_dagger_gamma:.3f} "
                    f"student_warmup={avg_student_warmup:.2f} action_grad_norm={avg_action_grad_norm:.2f} "
                    f"dagger_grad_norm={avg_dagger_grad_norm:.2f} "
                    f"target_norm_mean={avg_target_norm_mean:.2f} target_norm_max={max_target_norm:.2f} "
                    f"target_clip_frac={avg_target_clip_fraction:.3f} time={elapsed:.1f}s"
                )

            logging.info(log_str)

            if config.wandb_enabled:
                log_payload = {
                    "loss": avg_loss,
                    "learning_rate": avg_lr,
                    "student_warmup": avg_student_warmup,
                    "step": global_step,
                    "time_per_step": elapsed / config.log_interval,
                }
                if avg_dagger_gamma is not None:
                    log_payload["dagger_gamma"] = avg_dagger_gamma
                if avg_action_loss is not None:
                    log_payload["action_loss"] = avg_action_loss
                if avg_dagger_loss is not None:
                    log_payload["dagger_loss"] = avg_dagger_loss
                if avg_action_grad_norm is not None:
                    log_payload["action_grad_norm"] = avg_action_grad_norm
                if avg_dagger_grad_norm is not None:
                    log_payload["dagger_grad_norm"] = avg_dagger_grad_norm
                if avg_target_norm_mean is not None:
                    log_payload["dagger_target_norm_mean"] = avg_target_norm_mean
                if max_target_norm is not None:
                    log_payload["dagger_target_norm_max"] = max_target_norm
                if avg_target_clip_fraction is not None:
                    log_payload["dagger_target_clip_fraction"] = avg_target_clip_fraction
                wandb.log(log_payload, step=global_step)

            start_time = time.time()
            infos = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        global_step += 1
        save_checkpoint(student_model, optim, global_step, config, action_data_config)

        pbar.update(1)
        pbar.set_postfix(
            {
                "loss": f"{total_loss_value:.4f}",
                "lr": f"{optim.param_groups[0]['lr']:.2e}",
                "step": global_step,
            }
        )

    pbar.close()

    if config.wandb_enabled:
        wandb.finish()


def parse_args() -> tuple[argparse.Namespace, _config.TrainConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--action_batch_size", type=int, default=None)
    parser.add_argument("--dagger_batch_size", type=int, default=None)
    parser.add_argument("--distill_batch_size", type=int, default=None)
    parser.add_argument("--dagger_gamma", type=float, default=None)
    parser.add_argument("--dagger_gamma_min", type=float, default=0.0)
    parser.add_argument("--dagger_gamma_max", type=float, default=1.0)
    parser.add_argument("--dagger_min_time", type=float, default=0.01)
    parser.add_argument("--dagger_target_max_norm", type=float, default=-1.0)
    parser.add_argument("--warm_up_student_steps", type=int, default=0)

    dagger_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        config = _config.cli()
    finally:
        sys.argv = original_argv

    if dagger_args.action_loss_weight < 0:
        raise ValueError("--action_loss_weight must be non-negative.")
    if dagger_args.action_batch_size is not None and dagger_args.action_batch_size <= 0:
        raise ValueError("--action_batch_size must be greater than 0.")
    if dagger_args.dagger_batch_size is None:
        dagger_args.dagger_batch_size = dagger_args.distill_batch_size
    if (
        dagger_args.dagger_batch_size is not None
        and dagger_args.dagger_batch_size <= 0
    ):
        raise ValueError("--dagger_batch_size must be greater than 0.")
    if dagger_args.dagger_min_time <= 0 or dagger_args.dagger_min_time >= 1:
        raise ValueError("--dagger_min_time must be in (0, 1).")
    if dagger_args.dagger_target_max_norm == 0 or dagger_args.dagger_target_max_norm < -1:
        raise ValueError("--dagger_target_max_norm must be positive, or -1 to disable clipping.")
    if dagger_args.dagger_gamma_min < 0 or dagger_args.dagger_gamma_max > 1:
        raise ValueError("--dagger_gamma_min/max must stay within [0, 1].")
    if dagger_args.dagger_gamma_min > dagger_args.dagger_gamma_max:
        raise ValueError("--dagger_gamma_min must be <= --dagger_gamma_max.")
    if dagger_args.dagger_gamma is not None and not (0 <= dagger_args.dagger_gamma <= 1):
        raise ValueError("--dagger_gamma must be in [0, 1].")
    if dagger_args.warm_up_student_steps < 0:
        raise ValueError("--warm_up_student_steps must be non-negative.")

    return dagger_args, config


def main():
    init_logging()
    dagger_args, config = parse_args()
    train_loop(
        config,
        action_loss_weight=dagger_args.action_loss_weight,
        action_batch_size=dagger_args.action_batch_size,
        dagger_batch_size=dagger_args.dagger_batch_size,
        dagger_gamma=dagger_args.dagger_gamma,
        dagger_gamma_min=dagger_args.dagger_gamma_min,
        dagger_gamma_max=dagger_args.dagger_gamma_max,
        dagger_min_time=dagger_args.dagger_min_time,
        dagger_target_max_norm=dagger_args.dagger_target_max_norm,
        warm_up_student_steps=dagger_args.warm_up_student_steps,
    )


if __name__ == "__main__":
    main()
