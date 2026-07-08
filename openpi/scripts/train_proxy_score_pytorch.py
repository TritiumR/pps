"""Train score-space proxy policies for PPS steering.

This entrypoint is intentionally separate from the velocity proxy trainer.  The
model learns denoising scores for the DDIM/MBD noising process used by the
sim-free MPC sampler, not rectified-flow velocities.

Usage:
  python scripts/train_proxy_score_pytorch.py proxy_score_local_mpc_weight_jointpos --exp_name task
"""

import dataclasses
import logging
import os
import shutil
import sys
import time

_OPENPI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPENPI_SRC_DIR = os.path.join(_OPENPI_DIR, "src")
if _OPENPI_SRC_DIR not in sys.path:
    sys.path.insert(0, _OPENPI_SRC_DIR)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import safetensors.torch
import torch
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.proxy_score_config
import openpi.models_pytorch.proxy_score_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
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


def move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: move_to_device(getattr(value, field.name), device)
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb_id_path = config.checkpoint_dir / "wandb_id.txt"
    if resuming and wandb_id_path.exists():
        wandb.init(id=wandb_id_path.read_text().strip(), resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        wandb_id_path.write_text(wandb.run.id)


def build_datasets(config: _config.TrainConfig):
    return _data.create_data_loader(config, framework="pytorch", shuffle=True)


def get_model_parameters(model):
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    if not is_main:
        return
    if not (
        (global_step % config.save_interval == 0 and global_step > 0)
        or global_step == config.num_train_steps
    ):
        return

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")
    torch.save({"global_step": global_step, "timestamp": time.time()}, tmp_ckpt_dir / "metadata.pt")

    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_ckpt_dir.exists():
        shutil.rmtree(final_ckpt_dir)
    tmp_ckpt_dir.rename(final_ckpt_dir)
    logging.info("Saved checkpoint at step %s -> %s", global_step, final_ckpt_dir)


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
    model_to_load = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    safetensors.torch.load_model(
        model_to_load,
        ckpt_dir / "model.safetensors",
        device=str(device),
    )
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
    logging.info("Resumed checkpoint %s", ckpt_dir)
    return latest_step


def ensure_tensor_loss(losses, device):
    if isinstance(losses, list | tuple):
        return torch.stack(losses)
    if not isinstance(losses, torch.Tensor):
        return torch.tensor(losses, device=device, dtype=torch.float32)
    return losses


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = local_rank == 0
    set_seed(config.seed, local_rank)

    if not isinstance(config.model, openpi.models.proxy_score_config.ProxyScoreConfig):
        raise ValueError(
            "train_proxy_score_pytorch.py requires a ProxyScoreConfig. "
            f"Got {type(config.model).__name__} from config {config.name!r}."
        )

    resuming = False
    if config.resume:
        if not config.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Experiment checkpoint directory {config.checkpoint_dir} does not exist for resume."
            )
        resuming = True
    elif config.overwrite and config.checkpoint_dir.exists():
        import shutil

        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    elif config.wandb_enabled:
        wandb.init(mode="disabled")

    train_loader = build_datasets(config)
    data_config = train_loader.data_config()

    model = openpi.models_pytorch.proxy_score_pytorch.ProxyScorePytorch(config.model).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optim = torch.optim.AdamW(
        get_model_parameters(model),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)

    def lr_schedule(step: int):
        warmup_steps = config.lr_schedule.warmup_steps
        peak_lr = config.lr_schedule.peak_lr
        decay_steps = config.lr_schedule.decay_steps
        end_lr = config.lr_schedule.decay_lr
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Score proxy")
        if is_main
        else None
    )
    infos = []
    start_time = time.time()
    data_iter = iter(train_loader)

    while global_step < config.num_train_steps:
        observation, actions, noise = next(data_iter)
        observation = move_to_device(observation, device)
        actions = actions.to(torch.float32).to(device)
        if noise is not None:
            noise = noise.to(torch.float32).to(device)

        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        losses = model(observation, actions, noise=noise)
        losses = ensure_tensor_loss(losses, device)
        loss = losses.mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            get_model_parameters(model),
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)

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
                "step=%s score_loss=%.4f lr=%.2e grad_norm=%.2f time=%.1fs",
                completed_step,
                avg_loss,
                avg_lr,
                avg_grad_norm,
                elapsed,
            )
            if config.wandb_enabled:
                wandb.log(
                    {
                        "score_loss": avg_loss,
                        "learning_rate": avg_lr,
                        "grad_norm": avg_grad_norm,
                        "time_per_step": elapsed / config.log_interval,
                    },
                    step=completed_step,
                )
            infos = []
            start_time = time.time()

        global_step = completed_step
        save_checkpoint(model, optim, global_step, config, is_main, data_config)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "score_loss": f"{loss.item():.4f}",
                    "lr": f"{optim.param_groups[0]['lr']:.2e}",
                }
            )

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
