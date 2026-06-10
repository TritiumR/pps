"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import pathlib
import platform
import random
import shutil
import time

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
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.proxy_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.shared.download as download


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


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def create_train_val_split(dataset_length: int, val_ratio: float = 0.1, seed: int = 42):
    """
    Create deterministic train/val split indices.

    Args:
        dataset_length: Total number of samples in dataset
        val_ratio: Fraction of data to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices)
    """
    indices = list(range(dataset_length))
    random.seed(seed)
    random.shuffle(indices)

    val_size = int(dataset_length * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices


def get_dataset_length(config: _config.TrainConfig) -> int:
    """Get the total dataset length by creating a temporary loader."""
    temp_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    return len(temp_loader._data_loader._data_loader.dataset)


def _is_rlds(config: _config.TrainConfig) -> bool:
    """Check whether this config uses an RLDS (streaming) dataset."""
    data_config = config.data.create(config.assets_dirs, config.model)
    return data_config.rlds_data_dir is not None


def build_datasets_with_split(
    config: _config.TrainConfig, val_ratio: float = 0.1, seed: int = 42
):
    """
    Build train and validation data loaders with proper split.

    Uses the unified data loader with subset_indices parameter for cleaner implementation.
    For RLDS (streaming) datasets, index-based splitting is not supported; the full
    dataset is used for training and validation samples separate batches from the stream.

    Args:
        config: Training configuration
        val_ratio: Fraction of data for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_indices, data_config).
        val_indices is None for RLDS datasets.
    """
    if _is_rlds(config):
        train_loader = _data.create_data_loader(
            config, framework="pytorch", shuffle=True
        )
        logging.info(
            "RLDS streaming dataset: skipping index-based train/val split. "
            "Validation will sample fresh batches from the stream."
        )
        return train_loader, None, train_loader.data_config()

    # Torch map-style dataset: use index-based split
    dataset_length = get_dataset_length(config)
    train_indices, val_indices = create_train_val_split(dataset_length, val_ratio, seed)

    logging.info(
        f"Dataset split: {len(train_indices)} train samples, {len(val_indices)} val samples ({val_ratio*100:.0f}% val)"
    )

    train_loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        subset_indices=train_indices,
    )

    return train_loader, val_indices, train_loader.data_config()


def run_validation_distill(
    student_model,
    teacher_model,
    config: _config.TrainConfig,
    val_indices: list[int] | None,
    device,
    num_distill_steps: int,
    num_val_batches: int = 50,
) -> float:
    """
    Run validation for distillation on randomly sampled batches from the validation set.

    Uses the unified data loader with subset_indices for cleaner implementation.
    Each call randomly samples from the validation indices for variety.
    For RLDS datasets (val_indices is None), fresh batches are sampled from the stream.

    Args:
        student_model: The student model to evaluate
        teacher_model: The teacher model for generating targets
        config: Training configuration
        val_indices: Validation indices for the dataset, or None for RLDS datasets
        device: Device to run on
        num_distill_steps: Number of distillation steps
        num_val_batches: Number of batches to evaluate (default: 50)

    Returns:
        Average validation loss
    """
    student_model.eval()
    val_losses = []

    if val_indices is None:
        # RLDS streaming dataset: sample fresh batches from the stream
        val_loader = _data.create_data_loader(
            config,
            framework="pytorch",
            shuffle=True,
            num_batches=num_val_batches,
        )
    else:
        # Torch map-style dataset: use subset indices
        num_samples_needed = num_val_batches * config.batch_size
        if len(val_indices) > num_samples_needed:
            sampled_val_indices = random.sample(val_indices, num_samples_needed)
        else:
            sampled_val_indices = val_indices

        val_loader = _data.create_data_loader(
            config,
            framework="pytorch",
            shuffle=False,
            subset_indices=sampled_val_indices,
        )

    teacher_flow_path_noise_std = getattr(config, "teacher_flow_path_noise_std", 0.0)

    with torch.no_grad():
        for batch_idx, (observation, _, _) in enumerate(val_loader):
            if batch_idx >= num_val_batches:
                break

            # Move data to device
            observation = jax.tree.map(lambda x: x.to(device), observation)

            # Get teacher outputs
            noises, times, gradients, teacher_actions = (
                teacher_model.forward_for_distill(
                    observation,
                    num_distill_steps,
                    teacher_flow_path_noise_std=teacher_flow_path_noise_std,
                )
            )

            # Compute distillation loss
            losses = student_model.forward_distill(
                observation,
                noises,
                times,
                gradients,
                teacher_actions,
                use_noise=config.use_noise_for_distill,
            )

            val_losses.append(losses.mean().item())

    student_model.train()

    if len(val_losses) == 0:
        return float("nan")

    return sum(val_losses) / len(val_losses)


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

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Build train/val data loaders with proper split (5% validation)
    loader, val_indices, data_config = build_datasets_with_split(
        config, val_ratio=0.05, seed=config.seed
    )

    # Validation configuration
    val_interval = 5000  # Evaluate every 500 training steps
    num_val_batches = 50  # 50 batches * batch_size = validation frames

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(
            config, framework="pytorch", shuffle=False
        )
        # print("length of sample_data_loader", len(sample_data_loader))
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions, noise = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat(
                [img[i].permute(1, 2, 0) for img in sample_batch["image"].values()],
                axis=1,
            )
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, noise, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Load teacher model for distillation
    teacher_config_name = getattr(config, "teacher_config_name", None)
    teacher_checkpoint_dir = getattr(config, "teacher_checkpoint_dir", None)

    if teacher_config_name is None or teacher_checkpoint_dir is None:
        raise ValueError(
            "teacher_config_name and teacher_checkpoint_dir must be specified for distillation. "
            "Use --teacher_config_name <config_name> --teacher_checkpoint_dir <checkpoint_dir>"
        )

    if is_main:
        logging.info(
            f"Loading teacher model: config={teacher_config_name}, checkpoint={teacher_checkpoint_dir}"
        )

    # Get teacher training config
    teacher_train_config = _config.get_config(teacher_config_name)
    # print(f"teacher_train_config: {teacher_train_config.model.action_dim}")
    # print(f"teacher_train_config: {teacher_train_config.model.action_horizon}")
    teacher_checkpoint_dir = pathlib.Path(teacher_checkpoint_dir)

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

    # Build student model (ProxyPytorch)
    if not isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        raise ValueError(
            "Student model must be ProxyPytorch for distillation. Current model is not ProxyConfig."
        )

    model = openpi.models_pytorch.proxy_pytorch.ProxyPytorch(config.model).to(device)
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
            f"teacher_flow_path_noise_std={teacher_flow_path_noise_std}"
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

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, _, _ in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions, noise) tuple
            observation = jax.tree.map(
                lambda x: x.to(device), observation
            )  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Distillation forward pass:
            with torch.no_grad():
                noises, times, gradients, teacher_actions = (
                    teacher_model.forward_for_distill(
                        observation,
                        num_distill_steps,
                        teacher_flow_path_noise_std=teacher_flow_path_noise_std,
                    )
                )

            # Use student model to compute distillation loss
            losses = model.forward_distill(
                observation,
                noises,
                times,
                gradients,
                teacher_actions,
                use_noise=config.use_noise_for_distill,
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

            # Run validation every val_interval steps
            if is_main and global_step > 0 and (global_step % val_interval == 0):
                val_start_time = time.time()
                val_loss = run_validation_distill(
                    (
                        model.module
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                        else model
                    ),
                    teacher_model,
                    config,
                    val_indices,
                    device,
                    num_distill_steps=num_distill_steps,
                    num_val_batches=num_val_batches,
                )
                val_elapsed = time.time() - val_start_time
                logging.info(
                    f"step={global_step} val_loss={val_loss:.4f} val_time={val_elapsed:.1f}s ({num_val_batches} batches, random sample)"
                )

                # Log validation loss to wandb
                if config.wandb_enabled:
                    wandb.log({"val_loss": val_loss}, step=global_step)

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

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
