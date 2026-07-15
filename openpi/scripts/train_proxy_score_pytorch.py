"""Train score-space proxy policies for PPS steering.

This entrypoint is intentionally separate from the velocity proxy trainer.  The
model learns denoising scores for the DDIM/MBD noising process used by the
sim-free MPC sampler, not rectified-flow velocities.

Usage:
  python scripts/train_proxy_score_pytorch.py score_task_weight --exp_name task
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
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

from openpi.models import model as _model
import openpi.models.proxy_score_config
import openpi.models_pytorch.proxy_score_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


TASK_CACHE_FORMAT_VERSION = 1
TASK_CACHE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
TASK_CACHE_ENV = "SCORE_TASK_CACHE_PATH"


def _norm_stats_fingerprint(norm_stats) -> str | None:
    if norm_stats is None:
        return None
    digest = hashlib.sha256()
    for key in sorted(norm_stats):
        digest.update(key.encode("utf-8"))
        stat = norm_stats[key]
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stat, field, None)
            if value is None:
                digest.update(f"{field}:None".encode("utf-8"))
                continue
            array = np.asarray(value)
            digest.update(field.encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _task_cache_metadata(config, data_config, *, num_samples: int) -> dict:
    return {
        "format_version": TASK_CACHE_FORMAT_VERSION,
        "config": config.name,
        "repo_id": data_config.repo_id,
        "num_samples": int(num_samples),
        "action_horizon": int(config.model.action_horizon),
        "action_dim": int(config.model.action_dim),
        "norm_stats_fingerprint": _norm_stats_fingerprint(data_config.norm_stats),
        "use_quantile_norm": bool(data_config.use_quantile_norm),
        "image_keys": list(TASK_CACHE_IMAGE_KEYS),
        "image_shape": [224, 224, 3],
    }


def _build_task_data_config(config):
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None and getattr(config.data, "norm_stats_dir", None):
        norm_stats_dir = pathlib.Path(getattr(config.data, "norm_stats_dir"))
        candidates = (
            norm_stats_dir,
            pathlib.Path(_OPENPI_DIR) / norm_stats_dir,
            pathlib.Path(_OPENPI_DIR).parent / norm_stats_dir,
        )
        for candidate in candidates:
            if (candidate / "norm_stats.json").exists():
                data_config = dataclasses.replace(
                    data_config,
                    norm_stats=_normalize.load(candidate),
                )
                break
    if data_config.norm_stats is None:
        raise FileNotFoundError(
            f"Normalization stats are required to build the {config.name!r} task cache."
        )
    return data_config


def _task_cache_matches(cache_path: pathlib.Path, metadata: dict) -> bool:
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        cached_metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if cached_metadata != metadata:
        return False
    required_files = (
        "images.npy",
        "image_masks.npy",
        "states.npy",
        "actions.npy",
        "tokenized_prompt.npy",
        "tokenized_prompt_mask.npy",
    )
    return all((cache_path / filename).exists() for filename in required_files)


def prepare_task_cache(config, cache_path: pathlib.Path, *, num_workers: int) -> None:
    data_config = _build_task_data_config(config)
    dataset = _data.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
    )
    dataset = _data.transform_dataset(dataset, data_config)
    metadata = _task_cache_metadata(config, data_config, num_samples=len(dataset))
    if _task_cache_matches(cache_path, metadata):
        logging.info("Reusing task score cache: %s", cache_path)
        return

    for stale_tmp_path in cache_path.parent.glob(f"{cache_path.name}.tmp-*"):
        shutil.rmtree(stale_tmp_path)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    tmp_path.mkdir(parents=True)

    num_samples = len(dataset)
    images = np.lib.format.open_memmap(
        tmp_path / "images.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(num_samples, len(TASK_CACHE_IMAGE_KEYS), 224, 224, 3),
    )
    image_masks = np.lib.format.open_memmap(
        tmp_path / "image_masks.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(num_samples, len(TASK_CACHE_IMAGE_KEYS)),
    )
    states = np.lib.format.open_memmap(
        tmp_path / "states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, config.model.action_dim),
    )
    actions = np.lib.format.open_memmap(
        tmp_path / "actions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, config.model.action_horizon, config.model.action_dim),
    )
    tokenized_prompt = np.lib.format.open_memmap(
        tmp_path / "tokenized_prompt.npy",
        mode="w+",
        dtype=np.int64,
        shape=(num_samples, config.model.max_token_len),
    )
    tokenized_prompt_mask = np.lib.format.open_memmap(
        tmp_path / "tokenized_prompt_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(num_samples, config.model.max_token_len),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        worker_init_fn=_data._worker_init_fn if num_workers > 0 else None,
        drop_last=False,
    )
    logging.info(
        "Building task score cache with %s samples, workers=%s at %s",
        num_samples,
        num_workers,
        cache_path,
    )
    cursor = 0
    for batch in tqdm.tqdm(loader, total=len(loader), desc="Task observation cache"):
        batch_size = int(batch["state"].shape[0])
        batch_slice = slice(cursor, cursor + batch_size)
        for image_idx, image_key in enumerate(TASK_CACHE_IMAGE_KEYS):
            image_batch = batch["image"][image_key].numpy()
            if image_batch.shape[1:] != (224, 224, 3) or image_batch.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 [B, 224, 224, 3] images for {image_key}, got "
                    f"shape={image_batch.shape} dtype={image_batch.dtype}."
                )
            images[batch_slice, image_idx] = image_batch
            image_masks[batch_slice, image_idx] = batch["image_mask"][image_key].numpy()
        states[batch_slice] = batch["state"].numpy().astype(np.float32, copy=False)
        actions[batch_slice] = batch["actions"].numpy().astype(np.float32, copy=False)
        tokenized_prompt[batch_slice] = batch["tokenized_prompt"].numpy()
        tokenized_prompt_mask[batch_slice] = batch["tokenized_prompt_mask"].numpy()
        cursor += batch_size

    if cursor != num_samples:
        raise RuntimeError(f"Cached {cursor} task samples, expected {num_samples}.")
    for array in (
        images,
        image_masks,
        states,
        actions,
        tokenized_prompt,
        tokenized_prompt_mask,
    ):
        array.flush()
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if cache_path.exists():
        shutil.rmtree(cache_path)
    tmp_path.rename(cache_path)
    logging.info("Finished task score cache: %s", cache_path)


class TaskScoreCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path: str, config, data_config):
        self.cache_path = pathlib.Path(cache_path)
        metadata = json.loads((self.cache_path / "metadata.json").read_text())
        expected_fields = {
            "format_version": TASK_CACHE_FORMAT_VERSION,
            "config": config.name,
            "repo_id": data_config.repo_id,
            "action_horizon": int(config.model.action_horizon),
            "action_dim": int(config.model.action_dim),
            "norm_stats_fingerprint": _norm_stats_fingerprint(data_config.norm_stats),
            "use_quantile_norm": bool(data_config.use_quantile_norm),
        }
        for key, expected_value in expected_fields.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Task score cache field {key!r} is stale: "
                    f"{metadata.get(key)!r} != {expected_value!r}."
                )
        self.num_samples = int(metadata["num_samples"])
        self.images = np.load(self.cache_path / "images.npy", mmap_mode="c")
        self.image_masks = np.load(self.cache_path / "image_masks.npy", mmap_mode="c")
        self.states = np.load(self.cache_path / "states.npy", mmap_mode="c")
        self.actions = np.load(self.cache_path / "actions.npy", mmap_mode="c")
        self.tokenized_prompt = np.load(
            self.cache_path / "tokenized_prompt.npy", mmap_mode="c"
        )
        self.tokenized_prompt_mask = np.load(
            self.cache_path / "tokenized_prompt_mask.npy", mmap_mode="c"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = {
            "image": {
                image_key: torch.from_numpy(self.images[idx, image_idx])
                for image_idx, image_key in enumerate(TASK_CACHE_IMAGE_KEYS)
            },
            "image_mask": {
                image_key: torch.as_tensor(
                    bool(self.image_masks[idx, image_idx]), dtype=torch.bool
                )
                for image_idx, image_key in enumerate(TASK_CACHE_IMAGE_KEYS)
            },
            "state": torch.from_numpy(self.states[idx]),
            "tokenized_prompt": torch.from_numpy(self.tokenized_prompt[idx]),
            "tokenized_prompt_mask": torch.from_numpy(self.tokenized_prompt_mask[idx]),
        }
        return inputs, torch.from_numpy(self.actions[idx])


class TaskScoreCacheLoader:
    def __init__(self, config, cache_path: str):
        self._data_config = _build_task_data_config(config)
        dataset = TaskScoreCacheDataset(cache_path, config, self._data_config)
        sampler = None
        world_size = 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=torch.distributed.get_rank(),
                shuffle=True,
                drop_last=True,
            )
        if config.batch_size % world_size != 0:
            raise ValueError(
                f"batch_size={config.batch_size} must be divisible by world_size={world_size}."
            )
        local_batch_size = config.batch_size // world_size
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def data_config(self):
        return self._data_config

    def __iter__(self):
        epoch = 0
        while True:
            sampler = self._loader.sampler
            if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                sampler.set_epoch(epoch)
            for inputs, actions in self._loader:
                yield inputs, actions, None
            epoch += 1


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        kwargs = {"backend": backend, "init_method": "env://"}
        if device.type == "cuda":
            kwargs["device_id"] = device
        torch.distributed.init_process_group(**kwargs)
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


def build_loader(config: _config.TrainConfig):
    task_cache_path = os.environ.get(TASK_CACHE_ENV)
    if task_cache_path:
        logging.info("Using cached task score dataset: %s", task_cache_path)
        return TaskScoreCacheLoader(config, task_cache_path)
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
    rank = torch.distributed.get_rank() if use_ddp else 0
    is_main = rank == 0
    set_seed(config.seed, rank)

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
    elif config.overwrite and config.checkpoint_dir.exists() and is_main:
        shutil.rmtree(config.checkpoint_dir)
        logging.info("Overwriting checkpoint directory: %s", config.checkpoint_dir)
    elif not config.overwrite and config.checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint directory {config.checkpoint_dir} already exists; use --resume or --overwrite."
        )

    if use_ddp:
        torch.distributed.barrier()

    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    elif config.wandb_enabled:
        wandb.init(mode="disabled")

    train_loader = build_loader(config)
    data_config = train_loader.data_config()

    model = openpi.models_pytorch.proxy_score_pytorch.ProxyScorePytorch(config.model).to(device)
    if not resuming and config.pytorch_weight_path is not None:
        init_path = os.fspath(config.pytorch_weight_path)
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, "model.safetensors")
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f"Initial model checkpoint not found: {init_path}")
        safetensors.torch.load_model(model, init_path, device=str(device))
        logging.info("Initialized score proxy from %s", init_path)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        get_model_parameters(model),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optimizer, config.checkpoint_dir, device)

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
    metrics = []
    start_time = time.time()
    data_iter = iter(train_loader)

    while global_step < config.num_train_steps:
        observation, actions, noise = next(data_iter)
        observation = move_to_device(observation, device)
        if isinstance(observation, dict):
            # Cached uint8 images stay compact through DataLoader and are converted,
            # permuted, and augmented as a single batch on the GPU.
            observation = _model.Observation.from_dict(observation)
        actions = actions.to(torch.float32).to(device)
        if noise is not None:
            noise = noise.to(torch.float32).to(device)

        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(global_step)

        losses = model(observation, actions, noise=noise)
        losses = ensure_tensor_loss(losses, device)
        loss = losses.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            get_model_parameters(model),
            max_norm=config.optimizer.clip_gradient_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        metrics.append(
            {
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm.detach().cpu())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm),
            }
        )

        completed_step = global_step + 1
        if is_main and completed_step % config.log_interval == 0 and metrics:
            elapsed = time.time() - start_time
            avg_loss = sum(item["loss"] for item in metrics) / len(metrics)
            avg_lr = sum(item["lr"] for item in metrics) / len(metrics)
            avg_grad_norm = sum(item["grad_norm"] for item in metrics) / len(metrics)
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
            metrics = []
            start_time = time.time()

        global_step = completed_step
        save_checkpoint(model, optimizer, global_step, config, is_main, data_config)
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "score_loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def main():
    init_logging()
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-task-cache":
        parser = argparse.ArgumentParser(description="Build the task score mmap cache.")
        parser.add_argument("--config", required=True)
        parser.add_argument("--cache-path", required=True)
        parser.add_argument("--num-workers", type=int, default=8)
        args = parser.parse_args(sys.argv[2:])
        config = _config.get_config(args.config)
        prepare_task_cache(
            config,
            pathlib.Path(args.cache_path),
            num_workers=args.num_workers,
        )
        return
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
