"""
Single-GPU hybrid-noise PyTorch training entrypoint.

This script combines:
1. Supervised student training on ground-truth actions, like `train_pytorch.py`
2. Ground-truth flow supervision on teacher-generated rollout states/times.

Only the student model is trained. The teacher model is loaded once, frozen, and
used to provide the input flow path. Unlike `train_hybrid_pytorch.py`, this
script does not distill teacher gradients; the target velocity is always
`teacher_initial_noise - ground_truth_action`, matching normal flow training.

Usage:
  python scripts/train_hybrid_noise_pytorch.py <config_name> --exp_name <run_name> \
      --teacher_config_name <teacher_cfg> \
      --teacher_checkpoint_dir <teacher_ckpt> \
      --distill_batch_size <distill_batch_size>
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
        "Hybrid-noise loader teacher tokenization: student_model_type=%s teacher_model_type=%s max_token_len=%s tokenize_state_input=%s",
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

    if not hasattr(teacher_model, "forward_for_distill"):
        raise ValueError(
            f"Teacher model {type(teacher_model).__name__} does not support forward_for_distill()."
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


def build_ground_truth_flow_targets(
    noises: torch.Tensor,
    actions: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Create normal flow targets for every teacher-path input state."""
    initial_noise = noises[:, 0, :, :action_dim]
    actions = actions[:, :, :action_dim]
    if initial_noise.shape != actions.shape:
        raise ValueError(
            "Teacher initial noise and ground-truth actions have incompatible shapes: "
            f"{initial_noise.shape} vs {actions.shape}"
        )

    flow_target = initial_noise - actions
    return flow_target[:, None, :, :].expand(
        -1, noises.shape[1], -1, -1
    )


def train_loop(
    config: _config.TrainConfig,
    *,
    action_loss_weight: float,
    action_batch_size: int | None,
    distill_batch_size: int | None,
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
    effective_distill_batch_size = (
        config.batch_size if distill_batch_size is None else distill_batch_size
    )
    loader_batch_size = max(effective_action_batch_size, effective_distill_batch_size)

    action_loader, action_data_config = build_datasets(
        config, teacher_train_config, batch_size=loader_batch_size
    )

    if not isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        raise ValueError(
            "Hybrid training currently requires the student model to be ProxyConfig."
        )

    student_model = openpi.models_pytorch.proxy_pytorch.ProxyPytorch(config.model).to(
        device
    )
    teacher_model = load_teacher_model(config, device)

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
    num_distill_steps = getattr(config, "num_distill_steps", 10)
    teacher_flow_path_noise_std = getattr(config, "teacher_flow_path_noise_std", 0.0)

    logging.info("Running on: %s", platform.node())
    logging.info(
        "Training config: loader_batch_size=%s action_batch_size=%s distill_batch_size=%s (default batch_size=%s) num_train_steps=%s",
        loader_batch_size,
        effective_action_batch_size,
        effective_distill_batch_size,
        config.batch_size,
        config.num_train_steps,
    )
    logging.info(
        "Hybrid-noise config: action_loss_weight=%.3f num_distill_steps=%s teacher_flow_path_noise_std=%s",
        action_loss_weight,
        num_distill_steps,
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
        batch_observation, batch_actions, batch_noise = next(action_iter)

        batch_observation = jax.tree.map(lambda x: x.to(device), batch_observation)
        batch_actions = batch_actions.to(torch.float32).to(device)
        if batch_noise is not None:
            batch_noise = batch_noise.to(torch.float32).to(device)

        action_observation = slice_batch(batch_observation, effective_action_batch_size)
        action_actions = batch_actions[:effective_action_batch_size]
        action_noise = (
            batch_noise[:effective_action_batch_size]
            if batch_noise is not None
            else None
        )

        teacher_path_observation = slice_batch(
            batch_observation, effective_distill_batch_size
        )
        teacher_path_actions = batch_actions[:effective_distill_batch_size]

        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        action_loss = None
        teacher_path_loss = None
        action_grad_norm = None
        teacher_path_grad_norm = None

        if action_loss_weight > 0:
            action_losses = student_model(
                action_observation, action_actions, noise=action_noise
            )
            action_losses = ensure_tensor_loss(action_losses, device)
            action_loss = action_losses.mean()
            weighted_action_loss = action_loss * action_loss_weight

            optim.zero_grad(set_to_none=True)
            weighted_action_loss.backward()
            action_grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            del action_losses, weighted_action_loss

        with torch.no_grad():
            noises, times, _, _ = teacher_model.forward_for_distill(
                teacher_path_observation,
                num_distill_steps,
                teacher_flow_path_noise_std=teacher_flow_path_noise_std,
            )
            gt_gradients = build_ground_truth_flow_targets(
                noises,
                teacher_path_actions,
                student_model.config.action_dim,
            )

        teacher_path_losses = student_model.forward_distill(
            teacher_path_observation,
            noises,
            times,
            gt_gradients,
            teacher_path_actions,
            use_noise=True,
        )
        teacher_path_loss = teacher_path_losses.mean()

        optim.zero_grad(set_to_none=True)
        teacher_path_loss.backward()
        teacher_path_grad_norm = torch.nn.utils.clip_grad_norm_(
            student_model.parameters(),
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

        del teacher_path_losses

        total_loss_value = (
            (action_loss.item() * action_loss_weight if action_loss is not None else 0.0)
            + teacher_path_loss.item()
        )
        infos.append(
            {
                "loss": total_loss_value,
                "action_loss": action_loss.item() if action_loss is not None else None,
                "teacher_path_loss": (
                    teacher_path_loss.item()
                    if teacher_path_loss is not None
                    else None
                ),
                "learning_rate": optim.param_groups[0]["lr"],
                "action_grad_norm": (
                    float(action_grad_norm)
                    if isinstance(action_grad_norm, torch.Tensor)
                    else action_grad_norm
                ),
                "teacher_path_grad_norm": (
                    float(teacher_path_grad_norm)
                    if isinstance(teacher_path_grad_norm, torch.Tensor)
                    else teacher_path_grad_norm
                ),
            }
        )

        if global_step % config.log_interval == 0:
            elapsed = time.time() - start_time

            avg_loss = sum(info["loss"] for info in infos) / len(infos)
            avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

            action_vals = [
                info["action_loss"]
                for info in infos
                if info["action_loss"] is not None
            ]
            teacher_path_vals = [
                info["teacher_path_loss"]
                for info in infos
                if info["teacher_path_loss"] is not None
            ]
            avg_action_loss = (
                sum(action_vals) / len(action_vals) if len(action_vals) > 0 else None
            )
            avg_teacher_path_loss = (
                sum(teacher_path_vals) / len(teacher_path_vals)
                if len(teacher_path_vals) > 0
                else None
            )
            action_grad_vals = [
                info["action_grad_norm"]
                for info in infos
                if info["action_grad_norm"] is not None
            ]
            teacher_path_grad_vals = [
                info["teacher_path_grad_norm"]
                for info in infos
                if info["teacher_path_grad_norm"] is not None
            ]
            avg_action_grad_norm = (
                sum(action_grad_vals) / len(action_grad_vals)
                if len(action_grad_vals) > 0
                else None
            )
            avg_teacher_path_grad_norm = (
                sum(teacher_path_grad_vals) / len(teacher_path_grad_vals)
                if len(teacher_path_grad_vals) > 0
                else None
            )

            log_str = (
                f"step={global_step} loss={avg_loss:.4f} "
                f"action_loss={avg_action_loss:.4f} "
                f"teacher_path_loss={avg_teacher_path_loss:.4f} "
                f"lr={avg_lr:.2e} action_grad_norm={avg_action_grad_norm:.2f} "
                f"teacher_path_grad_norm={avg_teacher_path_grad_norm:.2f} time={elapsed:.1f}s"
            )
            if avg_action_loss is None:
                log_str = (
                    f"step={global_step} loss={avg_loss:.4f} teacher_path_loss={avg_teacher_path_loss:.4f} "
                    f"lr={avg_lr:.2e} teacher_path_grad_norm={avg_teacher_path_grad_norm:.2f} "
                    f"time={elapsed:.1f}s"
                )
            elif avg_teacher_path_loss is None:
                log_str = (
                    f"step={global_step} loss={avg_loss:.4f} action_loss={avg_action_loss:.4f} "
                    f"lr={avg_lr:.2e} action_grad_norm={avg_action_grad_norm:.2f} "
                    f"time={elapsed:.1f}s"
                )

            logging.info(log_str)

            if config.wandb_enabled:
                log_payload = {
                    "loss": avg_loss,
                    "learning_rate": avg_lr,
                    "step": global_step,
                    "time_per_step": elapsed / config.log_interval,
                }
                if avg_action_loss is not None:
                    log_payload["action_loss"] = avg_action_loss
                if avg_teacher_path_loss is not None:
                    log_payload["teacher_path_loss"] = avg_teacher_path_loss
                if avg_action_grad_norm is not None:
                    log_payload["action_grad_norm"] = avg_action_grad_norm
                if avg_teacher_path_grad_norm is not None:
                    log_payload["teacher_path_grad_norm"] = avg_teacher_path_grad_norm
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
    parser.add_argument("--distill_batch_size", type=int, default=None)

    hybrid_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        config = _config.cli()
    finally:
        sys.argv = original_argv

    if hybrid_args.action_loss_weight < 0:
        raise ValueError("--action_loss_weight must be non-negative.")
    if hybrid_args.action_batch_size is not None and hybrid_args.action_batch_size <= 0:
        raise ValueError("--action_batch_size must be greater than 0.")
    if (
        hybrid_args.distill_batch_size is not None
        and hybrid_args.distill_batch_size <= 0
    ):
        raise ValueError("--distill_batch_size must be greater than 0.")

    return hybrid_args, config


def main():
    init_logging()
    hybrid_args, config = parse_args()
    train_loop(
        config,
        action_loss_weight=hybrid_args.action_loss_weight,
        action_batch_size=hybrid_args.action_batch_size,
        distill_batch_size=hybrid_args.distill_batch_size,
    )


if __name__ == "__main__":
    main()
