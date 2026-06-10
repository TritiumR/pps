"""
PyTorch action distillation entrypoint with multi-GPU DDP support.

This mirrors scripts/distill_pytorch.py for teacher loading and hybrid distill
data, but trains the student with the normal flow-matching loss from
scripts/train_pytorch.py. The teacher's final denoised action is used as the
ground-truth action. By default, the student's flow path samples fresh initial
noise, matching standard flow-matching training. Use
--use_teacher_noise_for_action_distill to start from the same initial noise
state used by the teacher rollout.

Usage
Single GPU:
  python scripts/distill_action_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_config_name <teacher_cfg> --teacher_checkpoint_dir <teacher_ckpt>
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
      scripts/distill_action_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_config_name <teacher_cfg> --teacher_checkpoint_dir <teacher_ckpt>
Multi-Node:
  torchrun --nnodes=<N> --nproc_per_node=<gpus> --node_rank=<rank> \
      --master_addr=<ip> --master_port=<port> \
      scripts/distill_action_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_config_name <teacher_cfg> --teacher_checkpoint_dir <teacher_ckpt>
"""

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

import distill_pytorch as _distill
import openpi.models.proxy_config
import openpi.models.proxy_dp3_config
import openpi.models_pytorch.proxy_dp3_pytorch
import openpi.models_pytorch.proxy_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _build_student_model(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
    if isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        return openpi.models_pytorch.proxy_pytorch.ProxyPytorch(config.model).to(device)
    if isinstance(config.model, openpi.models.proxy_dp3_config.ProxyDP3Config):
        return openpi.models_pytorch.proxy_dp3_pytorch.ProxyDP3Pytorch(config.model).to(device)
    raise ValueError(
        "Student model must be ProxyConfig or ProxyDP3Config for action distillation."
    )


def _match_student_action_space(
    teacher_actions: torch.Tensor,
    model_cfg,
) -> torch.Tensor:
    action_horizon = model_cfg.action_horizon
    action_dim = model_cfg.action_dim
    if teacher_actions.shape[1] < action_horizon or teacher_actions.shape[2] < action_dim:
        raise ValueError(
            "Teacher action output is smaller than the student action space: "
            f"teacher_actions={tuple(teacher_actions.shape)}, "
            f"student=(horizon={action_horizon}, dim={action_dim})"
        )

    return teacher_actions[:, :action_horizon, :action_dim].to(torch.float32).contiguous()


def _match_student_noise_space(
    teacher_initial_noise: torch.Tensor,
    model_cfg,
) -> torch.Tensor:
    action_horizon = model_cfg.action_horizon
    action_dim = model_cfg.action_dim
    if (
        teacher_initial_noise.shape[1] < action_horizon
        or teacher_initial_noise.shape[2] < action_dim
    ):
        raise ValueError(
            "Teacher initial noise is smaller than the student action space: "
            f"teacher_noise={tuple(teacher_initial_noise.shape)}, "
            f"student=(horizon={action_horizon}, dim={action_dim})"
        )

    return teacher_initial_noise[:, :action_horizon, :action_dim].to(torch.float32).contiguous()


def _log_sample_batch(
    config: _config.TrainConfig,
    teacher_train_config: _config.TrainConfig,
):
    sample_data_loader = _data.create_data_loader(
        _distill._with_distill_data_config(config, teacher_train_config),
        framework="pytorch",
        shuffle=False,
    )
    observation, actions, noise = next(iter(sample_data_loader))
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

    pointcloud = sample_batch.get("pointcloud")
    if pointcloud is not None:
        wandb.log(
            {
                "pointcloud_num_points": pointcloud.shape[1],
                "pointcloud_feature_dim": pointcloud.shape[2],
            },
            step=0,
        )

    del sample_batch, observation, actions, noise, sample_data_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Cleared sample batch and data loader from memory")


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = _distill.setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    _distill.set_seed(config.seed, local_rank)

    resuming = False
    if config.resume:
        if config.checkpoint_dir.exists():
            latest_step = _distill.get_latest_checkpoint_step(config.checkpoint_dir)
            if latest_step is None:
                raise FileNotFoundError(
                    f"No valid checkpoints found in {config.checkpoint_dir} for resume"
                )
            resuming = True
            logging.info(
                f"Resuming from experiment checkpoint directory: {config.checkpoint_dir} at step {latest_step}"
            )
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume"
            )
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {config.checkpoint_dir}")
    else:
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    if is_main:
        _distill.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    teacher_config_name = getattr(config, "teacher_config_name", None)
    teacher_checkpoint_dir = getattr(config, "teacher_checkpoint_dir", None)
    if teacher_config_name is None or teacher_checkpoint_dir is None:
        raise ValueError(
            "teacher_config_name and teacher_checkpoint_dir must be specified for distillation. "
            "Use --teacher_config_name <config_name> --teacher_checkpoint_dir <checkpoint_dir>"
        )

    teacher_train_config = _config.get_config(teacher_config_name)
    _distill.log_teacher_tokenization_config(config, teacher_train_config)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} "
        f"(total batch size across {world_size} GPUs: {config.batch_size})"
    )

    loader, data_config = _distill.build_datasets(config, teacher_train_config)

    if is_main and config.wandb_enabled and not resuming:
        _log_sample_batch(config, teacher_train_config)

    teacher_checkpoint_dir = pathlib.Path(teacher_checkpoint_dir)
    teacher_weight_path = teacher_checkpoint_dir / "model.safetensors"
    if not teacher_weight_path.exists():
        raise FileNotFoundError(
            f"Teacher model checkpoint not found at {teacher_weight_path}"
        )

    if is_main:
        logging.info(
            f"Loading teacher model: config={teacher_config_name}, checkpoint={teacher_checkpoint_dir}"
        )
    teacher_model = teacher_train_config.model.load_pytorch(
        teacher_train_config, str(teacher_weight_path)
    )
    if hasattr(teacher_model, "paligemma_with_expert"):
        teacher_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    if is_main:
        logging.info(f"Loaded teacher model from {teacher_checkpoint_dir}")

    num_distill_steps = getattr(config, "num_distill_steps", 10)
    teacher_flow_path_noise_std = getattr(config, "teacher_flow_path_noise_std", 0.0)
    use_teacher_noise_for_action_distill = getattr(
        config, "use_teacher_noise_for_action_distill", False
    )
    use_thermal_overlay_student = _distill._uses_thermal_overlay_student(config)
    thermal_overlay_alpha = getattr(config.data, "thermal_alpha", 0.5)

    model = _build_student_model(config, device)
    model_cfg = config.model
    enable_gradient_checkpointing = False
    logging.info("Gradient checkpointing is not supported for this model")

    if is_main and torch.cuda.is_available():
        _distill.log_memory_usage(device, 0, "after_model_creation")

    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:128,expandable_segments:True"
        )
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )

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

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = _distill.load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | "
            f"world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, "
            f"effective_batch_size={effective_batch_size}, "
            f"num_train_steps={config.num_train_steps}"
        )
        logging.info(
            "Action distillation config: "
            f"teacher_config={teacher_config_name}, "
            f"teacher_checkpoint={teacher_checkpoint_dir}, "
            f"num_distill_steps={num_distill_steps}, "
            f"teacher_flow_path_noise_std={teacher_flow_path_noise_std}, "
            f"use_teacher_noise_for_action_distill={use_teacher_noise_for_action_distill}"
        )
        logging.info(
            f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}"
        )
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, "
            f"decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, "
            f"weight_decay={config.optimizer.weight_decay}, "
            f"clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

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

    if use_ddp:
        dist.barrier()

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, _, _ in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)
            teacher_observation = observation
            student_observation = (
                _distill._overlay_thermal_student_observation(
                    observation, alpha=thermal_overlay_alpha
                )
                if use_thermal_overlay_student
                else observation
            )

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            with torch.no_grad():
                noises, _, _, teacher_actions = teacher_model.forward_for_distill(
                    teacher_observation,
                    num_distill_steps,
                    teacher_flow_path_noise_std=teacher_flow_path_noise_std,
                )
                teacher_initial_noise = noises[:, 0]
                teacher_actions = _match_student_action_space(
                    teacher_actions,
                    model_cfg,
                )
                student_noise = (
                    _match_student_noise_space(teacher_initial_noise, model_cfg)
                    if use_teacher_noise_for_action_distill
                    else None
                )

            losses = model(student_observation, teacher_actions, noise=student_noise)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            loss = losses.mean()

            loss.backward()

            if global_step < 5 and is_main and torch.cuda.is_available():
                _distill.log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.optimizer.clip_gradient_norm
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

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
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                vals = [
                    info["grad_norm"]
                    for info in infos
                    if info.get("grad_norm") is not None
                ]
                avg_grad_norm = sum(vals) / len(vals) if vals else None
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} "
                    f"grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

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
                infos = []

            global_step += 1
            is_save_step = (
                (global_step % config.save_interval == 0 and global_step > 0)
                or global_step == config.num_train_steps - 1
            )
            if use_ddp and is_save_step:
                dist.barrier()
            _distill.save_checkpoint(model, optim, global_step, config, is_main, data_config)
            if use_ddp and is_save_step:
                dist.barrier()

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                        "step": global_step,
                    }
                )

    if pbar is not None:
        pbar.close()

    if is_main and config.wandb_enabled:
        wandb.finish()

    _distill.cleanup_ddp()


def main():
    _distill.init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
