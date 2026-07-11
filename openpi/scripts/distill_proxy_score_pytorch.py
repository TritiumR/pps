"""
Legacy PyTorch score-proxy distillation entrypoint.

Do not use this for the current score-space PPS reference policy.  It distills
from a pi0/pi05 teacher action.  The MPC reference policy should be generated
and trained with scripts/train_mpc_proxy_score_pytorch.py so the supervised
target is the FK/cost MPC score from mbd_score_action_prox.

This mirrors scripts/distill_pytorch.py for teacher loading, teacher-compatible
data transforms, DDP, and checkpointing.  Unlike velocity distillation, the
teacher's final denoised action chunk is treated as a clean sample x0_ref, then
the student is trained with DDIM/VP denoising score matching:

  x_t = sqrt(alpha_t) * x0_ref + sqrt(1 - alpha_t) * eps
  score_target = -eps / sqrt(1 - alpha_t)

Usage:
  python scripts/distill_proxy_score_pytorch.py <proxy-score-config> \
      --exp_name reference \
      --teacher_checkpoint_dir checkpoints/pytorch/pi05_droid_jointpos
"""

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

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import distill_pytorch as _distill
import openpi.models.proxy_score_config
import openpi.models_pytorch.proxy_score_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _build_student_model(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(
            "distill_proxy_score_pytorch.py requires a ProxyScoreConfig. "
            f"Got {type(config.model).__name__} from config {config.name!r}."
        )
    return openpi.models_pytorch.proxy_score_pytorch.ProxyScorePytorch(config.model).to(device)


def _match_student_action_space(
    teacher_actions: torch.Tensor,
    model_cfg,
) -> torch.Tensor:
    action_horizon = model_cfg.action_horizon
    action_dim = model_cfg.action_dim
    if teacher_actions.shape[1] < action_horizon or teacher_actions.shape[2] < action_dim:
        raise ValueError(
            "Teacher action output is smaller than the score proxy action space: "
            f"teacher_actions={tuple(teacher_actions.shape)}, "
            f"student=(horizon={action_horizon}, dim={action_dim})"
        )
    return teacher_actions[:, :action_horizon, :action_dim].to(torch.float32).contiguous()


def _log_sample_batch(
    config: _config.TrainConfig,
    teacher_train_config: _config.TrainConfig,
    teacher_norm_stats,
):
    sample_data_loader = _data.create_data_loader(
        _distill._with_distill_data_config(
            config,
            teacher_train_config,
            teacher_norm_stats,
        ),
        framework="pytorch",
        shuffle=False,
    )
    observation, actions, _ = next(iter(sample_data_loader))
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
            images_to_log.append(wandb.Image(img_concatenated.cpu().numpy()))
        wandb.log({"camera_views": images_to_log}, step=0)

    del sample_data_loader, observation, actions, sample_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_teacher_model(
    teacher_train_config: _config.TrainConfig,
    teacher_checkpoint_dir: pathlib.Path,
    device: torch.device,
):
    teacher_weight_path = teacher_checkpoint_dir / "model.safetensors"
    if not teacher_weight_path.exists():
        raise FileNotFoundError(
            f"Teacher model checkpoint not found at {teacher_weight_path}"
        )

    teacher_model = teacher_train_config.model.load_pytorch(
        teacher_train_config,
        str(teacher_weight_path),
    )
    if hasattr(teacher_model, "paligemma_with_expert"):
        teacher_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if not hasattr(teacher_model, "forward_for_distill"):
        raise ValueError(
            f"Teacher model {type(teacher_model).__name__} does not support forward_for_distill()."
        )

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def train_loop(config: _config.TrainConfig):
    print("[score_ref] train_loop: setup_ddp", file=sys.stderr, flush=True)
    use_ddp, local_rank, device = _distill.setup_ddp()
    is_main = local_rank == 0
    print(
        f"[score_ref] train_loop: ddp ready use_ddp={use_ddp} local_rank={local_rank} device={device}",
        file=sys.stderr,
        flush=True,
    )
    _distill.set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        if not config.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume."
            )
        latest_step = _distill.get_latest_checkpoint_step(config.checkpoint_dir)
        if latest_step is None:
            raise FileNotFoundError(
                f"No valid checkpoints found in {config.checkpoint_dir} for resume."
            )
        resuming = True
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[score_ref] train_loop: checkpoint dir ready {config.checkpoint_dir}",
            file=sys.stderr,
            flush=True,
        )

    if is_main:
        print("[score_ref] train_loop: wandb init", file=sys.stderr, flush=True)
        _distill.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        print("[score_ref] train_loop: wandb ready", file=sys.stderr, flush=True)

    teacher_checkpoint_dir = getattr(config, "teacher_checkpoint_dir", None)
    if teacher_checkpoint_dir is None:
        raise ValueError(
            "teacher_checkpoint_dir must be specified for score distillation. "
            "Use --teacher_checkpoint_dir <checkpoint_dir>."
        )
    teacher_checkpoint_dir = pathlib.Path(teacher_checkpoint_dir)
    teacher_config_name = getattr(config, "teacher_config_name", None)
    if teacher_config_name is None:
        teacher_config_name = _distill._config_name_from_checkpoint_dir(
            teacher_checkpoint_dir
        )
    teacher_train_config = _config.get_config(teacher_config_name)
    print(
        f"[score_ref] train_loop: teacher config ready {teacher_config_name}",
        file=sys.stderr,
        flush=True,
    )
    _distill.log_teacher_tokenization_config(config, teacher_train_config)
    teacher_norm_stats = _distill._load_teacher_norm_stats(
        teacher_train_config,
        teacher_checkpoint_dir,
    )

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    loader, data_config = _distill.build_datasets(
        config,
        teacher_train_config,
        teacher_norm_stats,
    )
    print("[score_ref] train_loop: data loader ready", file=sys.stderr, flush=True)

    if is_main and config.wandb_enabled and not resuming:
        _log_sample_batch(config, teacher_train_config, teacher_norm_stats)

    if is_main:
        logging.info(
            "Loading teacher model: config=%s checkpoint=%s",
            teacher_config_name,
            teacher_checkpoint_dir,
        )
    teacher_model = _load_teacher_model(
        teacher_train_config,
        teacher_checkpoint_dir,
        device,
    )
    print("[score_ref] train_loop: teacher model ready", file=sys.stderr, flush=True)

    model = _build_student_model(config, device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )

    if config.pytorch_weight_path is not None:
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            ),
            model_path,
        )
        logging.info("Loaded student weights from %s", config.pytorch_weight_path)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = _distill.load_checkpoint(
            model,
            optim,
            config.checkpoint_dir,
            device,
        )

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    num_distill_steps = getattr(config, "num_distill_steps", 10)
    teacher_flow_path_noise_std = getattr(config, "teacher_flow_path_noise_std", 0.0)

    model.train()
    if is_main:
        logging.info(
            "Score distillation config: teacher_config=%s teacher_checkpoint=%s "
            "num_distill_steps=%s teacher_flow_path_noise_std=%s",
            teacher_config_name,
            teacher_checkpoint_dir,
            num_distill_steps,
            teacher_flow_path_noise_std,
        )
        logging.info(
            "Training config: batch_size=%s effective_batch_size=%s num_train_steps=%s",
            config.batch_size,
            effective_batch_size,
            config.num_train_steps,
        )
        logging.info("Running on: %s | world_size=%s", platform.node(), world_size)

    pbar = (
        tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Score distill",
            disable=not is_main,
        )
        if is_main
        else None
    )

    if use_ddp:
        dist.barrier()

    start_time = time.time()
    infos = []
    while global_step < config.num_train_steps:
        for observation, _, _ in loader:
            if global_step >= config.num_train_steps:
                break

            observation = _distill.move_to_device(observation, device)
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            with torch.no_grad():
                _, _, _, teacher_actions = teacher_model.forward_for_distill(
                    observation,
                    num_distill_steps,
                    teacher_flow_path_noise_std=teacher_flow_path_noise_std,
                )
                teacher_actions = _match_student_action_space(
                    teacher_actions,
                    config.model,
                )

            losses = model(observation, teacher_actions)
            loss = losses.mean()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            if is_main:
                infos.append(
                    {
                        "loss": float(loss.detach().cpu()),
                        "lr": float(optim.param_groups[0]["lr"]),
                        "grad_norm": float(grad_norm.detach().cpu())
                        if isinstance(grad_norm, torch.Tensor)
                        else float(grad_norm),
                    }
                )

            completed_step = global_step + 1
            if is_main and completed_step % config.log_interval == 0 and infos:
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["lr"] for info in infos) / len(infos)
                avg_grad_norm = sum(info["grad_norm"] for info in infos) / len(infos)
                logging.info(
                    "step=%s score_distill_loss=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                    completed_step,
                    avg_loss,
                    avg_lr,
                    avg_grad_norm,
                    elapsed,
                )
                if config.wandb_enabled:
                    wandb.log(
                        {
                            "score_distill_loss": avg_loss,
                            "learning_rate": avg_lr,
                            "grad_norm": avg_grad_norm,
                            "time_per_step": elapsed / config.log_interval,
                        },
                        step=completed_step,
                    )
                start_time = time.time()
                infos = []

            global_step = completed_step
            _distill.save_checkpoint(
                model,
                optim,
                global_step,
                config,
                is_main,
                data_config,
            )
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                    }
                )

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    _distill.cleanup_ddp()


def main():
    _distill.init_logging()
    print("[score_ref] main: parsing config", file=sys.stderr, flush=True)
    config = _config.cli()
    print(f"[score_ref] main: config ready {config.name}", file=sys.stderr, flush=True)
    train_loop(config)


if __name__ == "__main__":
    main()
