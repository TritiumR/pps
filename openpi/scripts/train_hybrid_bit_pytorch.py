"""Single-GPU bit-conditioned hybrid PyTorch training entrypoint.

This script trains one proxy policy on one dataset with two simultaneous
objectives:

1. Ground-truth flow matching with ``action_expert_bit = 1``.
2. Teacher-flow distillation with ``action_expert_bit = 0``.

Each data-loader batch is split into disjoint ground-truth and distillation
subsets. The subsets may have different sizes, but their losses are averaged
independently before weighting and summation. One backward pass and optimizer
step updates the student from both objectives.

The student config must set ``ProxyConfig.use_action_expert_bit=True``.

Usage:
  python scripts/train_hybrid_bit_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_config_name <teacher_cfg> \
      --teacher_checkpoint_dir <teacher_ckpt> \
      --gt_batch_size 16 --distill_batch_size 8
"""

import argparse
import dataclasses
import gc
import logging
import os
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

import openpi.models.proxy_config
import openpi.models_pytorch.proxy_pytorch
import openpi.training.config as _config

if __package__:
    from . import train_hybrid_pytorch as _hybrid
else:
    import train_hybrid_pytorch as _hybrid


def resolve_batch_sizes(
    config_batch_size: int,
    gt_batch_size: int | None,
    distill_batch_size: int | None,
) -> tuple[int, int]:
    """Resolve two positive subset sizes.

    With no overrides, ``config.batch_size`` is split as evenly as possible.
    With one override, the other subset receives the remainder. With both
    overrides, their sum becomes the data-loader batch size.
    """
    if config_batch_size <= 0:
        raise ValueError("config.batch_size must be greater than 0.")

    if gt_batch_size is None and distill_batch_size is None:
        if config_batch_size < 2:
            raise ValueError(
                "config.batch_size must be at least 2 when both hybrid batch "
                "sizes use their defaults."
            )
        gt_batch_size = config_batch_size // 2
        distill_batch_size = config_batch_size - gt_batch_size
    elif gt_batch_size is None:
        gt_batch_size = config_batch_size - distill_batch_size
    elif distill_batch_size is None:
        distill_batch_size = config_batch_size - gt_batch_size

    if gt_batch_size <= 0:
        raise ValueError(
            "gt_batch_size must be greater than 0; provide both batch sizes if "
            "their sum should exceed config.batch_size."
        )
    if distill_batch_size <= 0:
        raise ValueError(
            "distill_batch_size must be greater than 0; provide both batch sizes "
            "if their sum should exceed config.batch_size."
        )
    return gt_batch_size, distill_batch_size


def slice_batch(batch, start: int, end: int):
    """Slice every tensor leaf in an observation or other batch pytree."""
    return jax.tree.map(
        lambda x: x[start:end] if isinstance(x, torch.Tensor) else x,
        batch,
    )


def set_action_expert_bit(observation, value: int):
    """Return an observation carrying one binary token ID per batch item."""
    if value not in (0, 1):
        raise ValueError(f"action expert bit must be 0 or 1, got {value}")
    bit = torch.full(
        (observation.state.shape[0],),
        value,
        dtype=torch.long,
        device=observation.state.device,
    )
    return dataclasses.replace(observation, action_expert_bit=bit)


def split_and_condition_batch(
    observation,
    actions: torch.Tensor,
    noise: torch.Tensor | None,
    *,
    gt_batch_size: int,
    distill_batch_size: int,
):
    """Create disjoint bit-1 GT and bit-0 distillation subsets."""
    combined_batch_size = gt_batch_size + distill_batch_size
    observation_batch_size = observation.state.shape[0]
    if observation_batch_size != combined_batch_size:
        raise ValueError(
            "Observation batch size does not match the requested hybrid batch: "
            f"got {observation_batch_size}, expected {combined_batch_size}."
        )
    if actions.shape[0] != combined_batch_size:
        raise ValueError(
            "Action batch size does not match the requested hybrid batch: "
            f"got {actions.shape[0]}, expected {combined_batch_size}."
        )
    if noise is not None and noise.shape[0] != combined_batch_size:
        raise ValueError(
            "Noise batch size does not match the requested hybrid batch: "
            f"got {noise.shape[0]}, expected {combined_batch_size}."
        )

    gt_slice = slice(0, gt_batch_size)
    distill_slice = slice(gt_batch_size, combined_batch_size)

    gt_observation = set_action_expert_bit(
        slice_batch(observation, gt_slice.start, gt_slice.stop), 1
    )
    distill_observation = set_action_expert_bit(
        slice_batch(observation, distill_slice.start, distill_slice.stop), 0
    )

    gt_noise = noise[gt_slice] if noise is not None else None
    return (
        gt_observation,
        actions[gt_slice],
        gt_noise,
        distill_observation,
        actions[distill_slice],
    )


def combine_hybrid_losses(
    gt_loss: torch.Tensor,
    distill_loss: torch.Tensor,
    *,
    gt_loss_weight: float,
    distill_loss_weight: float,
) -> torch.Tensor:
    """Combine independently averaged GT and teacher-flow losses."""
    if gt_loss_weight <= 0 or distill_loss_weight <= 0:
        raise ValueError("Both hybrid loss weights must be greater than 0.")
    return gt_loss * gt_loss_weight + distill_loss * distill_loss_weight


def load_initial_student_weights(model, weight_path):
    """Load a student checkpoint, allowing only the newly added bit embedding."""
    missing, unexpected = safetensors.torch.load_model(
        model,
        weight_path,
        strict=False,
    )
    allowed_missing = {"action_expert_bit_embedding.weight"}
    disallowed_missing = set(missing) - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Student checkpoint does not match the bit-conditioned proxy model: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}"
        )
    if missing:
        logging.info(
            "Initialized %s randomly because it is absent from the source checkpoint",
            sorted(missing),
        )


def _validate_student_config(config: _config.TrainConfig):
    if not isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        raise ValueError(
            "Bit-conditioned hybrid training requires the student model to use "
            "ProxyConfig."
        )
    if not config.model.use_action_expert_bit:
        raise ValueError(
            "Bit-conditioned hybrid training requires "
            "ProxyConfig.use_action_expert_bit=True."
        )


def train_loop(
    config: _config.TrainConfig,
    *,
    gt_batch_size: int | None,
    distill_batch_size: int | None,
    gt_loss_weight: float,
    distill_loss_weight: float,
):
    _validate_student_config(config)
    effective_gt_batch_size, effective_distill_batch_size = resolve_batch_sizes(
        config.batch_size, gt_batch_size, distill_batch_size
    )
    loader_batch_size = effective_gt_batch_size + effective_distill_batch_size

    device = _hybrid.setup_device()
    _hybrid.set_seed(config.seed)

    resuming = False
    if config.resume:
        if not config.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does "
                "not exist for resume"
            )
        latest_step = _hybrid.get_latest_checkpoint_step(config.checkpoint_dir)
        if latest_step is None:
            raise FileNotFoundError(
                f"No valid checkpoints found in {config.checkpoint_dir} for resume"
            )
        resuming = True
        logging.info(
            "Resuming from experiment checkpoint directory: %s at step %s",
            config.checkpoint_dir,
            latest_step,
        )
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Created experiment checkpoint directory: %s", config.checkpoint_dir
        )
    else:
        logging.info(
            "Using existing experiment checkpoint directory: %s",
            config.checkpoint_dir,
        )

    _hybrid.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    if config.wandb_enabled:
        wandb.config.update(
            {
                "hybrid_gt_batch_size": effective_gt_batch_size,
                "hybrid_distill_batch_size": effective_distill_batch_size,
                "hybrid_gt_loss_weight": gt_loss_weight,
                "hybrid_distill_loss_weight": distill_loss_weight,
            },
            allow_val_change=True,
        )

    teacher_config_name = getattr(config, "teacher_config_name", None)
    if teacher_config_name is None:
        raise ValueError(
            "teacher_config_name must be specified. Use --teacher_config_name "
            "<config_name>"
        )
    teacher_train_config = _config.get_config(teacher_config_name)
    _hybrid.log_teacher_tokenization_config(config, teacher_train_config)

    data_loader, data_config = _hybrid.build_datasets(
        config,
        teacher_train_config,
        batch_size=loader_batch_size,
    )

    student_model = openpi.models_pytorch.proxy_pytorch.ProxyPytorch(
        config.model
    ).to(device)
    teacher_model = _hybrid.load_teacher_model(config, device)

    if config.pytorch_weight_path is not None and not resuming:
        student_weight_path = _hybrid.resolve_model_safetensors(
            config.pytorch_weight_path
        )
        load_initial_student_weights(student_model, student_weight_path)
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
        global_step = _hybrid.load_checkpoint(
            student_model, optim, config.checkpoint_dir, device
        )

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, decay_steps - warmup_steps),
        )
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cosine

    student_model.train()
    num_distill_steps = getattr(config, "num_distill_steps", 10)
    teacher_flow_path_noise_std = getattr(
        config, "teacher_flow_path_noise_std", 0.0
    )

    logging.info("Running on: %s", platform.node())
    logging.info(
        "Single-dataset hybrid batch: loader=%s gt(bit=1)=%s "
        "distill(bit=0)=%s config.batch_size=%s",
        loader_batch_size,
        effective_gt_batch_size,
        effective_distill_batch_size,
        config.batch_size,
    )
    logging.info(
        "Hybrid losses: gt_weight=%.3f distill_weight=%.3f "
        "num_distill_steps=%s use_noise_for_distill=%s "
        "teacher_flow_path_noise_std=%s",
        gt_loss_weight,
        distill_loss_weight,
        num_distill_steps,
        config.use_noise_for_distill,
        teacher_flow_path_noise_std,
    )
    logging.info("Dataset asset_id=%s", data_config.asset_id)

    progress = tqdm.tqdm(
        total=config.num_train_steps,
        initial=global_step,
        desc="Training",
    )
    data_iter = iter(data_loader)
    infos = []
    log_start_time = time.time()

    while global_step < config.num_train_steps:
        batch_observation, batch_actions, batch_noise = next(data_iter)
        batch_observation = jax.tree.map(lambda x: x.to(device), batch_observation)
        batch_actions = batch_actions.to(device=device, dtype=torch.float32)
        if batch_noise is not None:
            batch_noise = batch_noise.to(device=device, dtype=torch.float32)

        (
            gt_observation,
            gt_actions,
            gt_noise,
            distill_observation,
            _,
        ) = split_and_condition_batch(
            batch_observation,
            batch_actions,
            batch_noise,
            gt_batch_size=effective_gt_batch_size,
            distill_batch_size=effective_distill_batch_size,
        )

        for param_group in optim.param_groups:
            param_group["lr"] = lr_schedule(global_step)

        with torch.no_grad():
            noises, times, gradients, teacher_actions = (
                teacher_model.forward_for_distill(
                    distill_observation,
                    num_distill_steps,
                    teacher_flow_path_noise_std=teacher_flow_path_noise_std,
                )
            )

        gt_losses = student_model(
            gt_observation,
            gt_actions,
            noise=gt_noise,
        )
        gt_loss = _hybrid.ensure_tensor_loss(gt_losses, device).mean()

        distill_losses = student_model.forward_distill(
            distill_observation,
            noises,
            times,
            gradients,
            teacher_actions,
            use_noise=config.use_noise_for_distill,
        )
        distill_loss = _hybrid.ensure_tensor_loss(distill_losses, device).mean()

        total_loss = combine_hybrid_losses(
            gt_loss,
            distill_loss,
            gt_loss_weight=gt_loss_weight,
            distill_loss_weight=distill_loss_weight,
        )

        optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student_model.parameters(),
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

        infos.append(
            {
                "loss": total_loss.item(),
                "gt_loss": gt_loss.item(),
                "distill_loss": distill_loss.item(),
                "learning_rate": optim.param_groups[0]["lr"],
                "grad_norm": (
                    float(grad_norm)
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                ),
            }
        )

        if global_step % config.log_interval == 0:
            elapsed = time.time() - log_start_time
            num_logged_steps = len(infos)
            averages = {
                key: sum(info[key] for info in infos) / num_logged_steps
                for key in (
                    "loss",
                    "gt_loss",
                    "distill_loss",
                    "learning_rate",
                    "grad_norm",
                )
            }
            logging.info(
                "step=%s loss=%.4f gt_loss=%.4f distill_loss=%.4f "
                "lr=%.2e grad_norm=%.2f time=%.1fs",
                global_step,
                averages["loss"],
                averages["gt_loss"],
                averages["distill_loss"],
                averages["learning_rate"],
                averages["grad_norm"],
                elapsed,
            )

            if config.wandb_enabled:
                wandb.log(
                    {
                        **averages,
                        "step": global_step,
                        "time_per_step": elapsed / num_logged_steps,
                    },
                    step=global_step,
                )

            infos = []
            log_start_time = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        global_step += 1
        _hybrid.save_checkpoint(
            student_model,
            optim,
            global_step,
            config,
            data_config,
        )

        progress.update(1)
        progress.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "gt": f"{gt_loss.item():.4f}",
                "distill": f"{distill_loss.item():.4f}",
                "lr": f"{optim.param_groups[0]['lr']:.2e}",
                "step": global_step,
            }
        )

    progress.close()
    if config.wandb_enabled:
        wandb.finish()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args() -> tuple[argparse.Namespace, _config.TrainConfig]:
    """Parse the focused hybrid trainer CLI without expanding every repo config.

    The general training CLI builds a Tyro union containing all registered
    configs. This specialized entrypoint only needs a small set of TrainConfig
    overrides, so resolving the selected config directly is substantially
    faster and produces a compact, useful ``--help`` page.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train one bit-conditioned proxy with simultaneous GT (bit 1) and "
            "teacher-flow (bit 0) supervision."
        )
    )
    parser.add_argument("config_name")
    parser.add_argument("--exp_name", "--exp-name", required=True)
    parser.add_argument("--gt_batch_size", type=int, default=None)
    parser.add_argument("--distill_batch_size", type=int, default=None)
    parser.add_argument("--gt_loss_weight", type=float, default=1.0)
    parser.add_argument("--distill_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_train_steps", "--num-train-steps", type=int)
    parser.add_argument("--batch_size", "--batch-size", type=int)
    parser.add_argument("--num_workers", "--num-workers", type=int)
    parser.add_argument("--log_interval", "--log-interval", type=int)
    parser.add_argument("--save_interval", "--save-interval", type=int)
    parser.add_argument("--checkpoint_base_dir", "--checkpoint-base-dir")
    parser.add_argument("--teacher_config_name", "--teacher-config-name")
    parser.add_argument("--teacher_checkpoint_dir", "--teacher-checkpoint-dir")
    parser.add_argument("--pytorch_weight_path", "--pytorch-weight-path")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--wandb_enabled",
        "--wandb-enabled",
        nargs="?",
        const=True,
        type=_parse_bool,
    )
    parser.add_argument("--overwrite", nargs="?", const=True, type=_parse_bool)
    parser.add_argument("--resume", nargs="?", const=True, type=_parse_bool)
    hybrid_args = parser.parse_args()

    config = _config.get_config(hybrid_args.config_name)
    config_updates = {"exp_name": hybrid_args.exp_name}
    for field_name in (
        "num_train_steps",
        "batch_size",
        "num_workers",
        "log_interval",
        "save_interval",
        "checkpoint_base_dir",
        "teacher_config_name",
        "teacher_checkpoint_dir",
        "pytorch_weight_path",
        "seed",
        "wandb_enabled",
        "overwrite",
        "resume",
    ):
        value = getattr(hybrid_args, field_name)
        if value is not None:
            config_updates[field_name] = value
    config = dataclasses.replace(config, **config_updates)

    if hybrid_args.gt_batch_size is not None and hybrid_args.gt_batch_size <= 0:
        raise ValueError("--gt_batch_size must be greater than 0.")
    if (
        hybrid_args.distill_batch_size is not None
        and hybrid_args.distill_batch_size <= 0
    ):
        raise ValueError("--distill_batch_size must be greater than 0.")
    if hybrid_args.gt_loss_weight <= 0:
        raise ValueError("--gt_loss_weight must be greater than 0.")
    if hybrid_args.distill_loss_weight <= 0:
        raise ValueError("--distill_loss_weight must be greater than 0.")

    return hybrid_args, config


def main():
    _hybrid.init_logging()
    hybrid_args, config = parse_args()
    train_loop(
        config,
        gt_batch_size=hybrid_args.gt_batch_size,
        distill_batch_size=hybrid_args.distill_batch_size,
        gt_loss_weight=hybrid_args.gt_loss_weight,
        distill_loss_weight=hybrid_args.distill_loss_weight,
    )


if __name__ == "__main__":
    main()
