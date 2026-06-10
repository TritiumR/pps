"""PyTorch training entrypoint for the Residual-V (velocity-conditioned residual) policy.

Trains a 130M residual velocity model along the steer/expert probability path
``x_t = (1 - t) * x_0 + t * a_steer``. At each training step the frozen VLA base
model (pi0 / pi0.5) provides ``v_base_t = v_base(x_t, t, c)``, which the residual
uses as an extra conditioning signal in its suffix fusion. The training target
is ``v_res_target = (a_steer - x_0) - v_base_t``; at gamma=1 the combined field
``v_base + v_res`` exactly regresses onto the expert conditional-OT velocity.

Key batching strategy (to minimize VLA compute):
  - Data loader draws ``B_obs = config.batch_size`` observations per step.
  - For each observation we sample ``K = config.num_paths_per_obs`` path points
    ``(x_0, t)``. Effective residual training batch is ``B_res = B_obs * K``.
  - VLA's expensive PaliGemma prefix is computed ONCE at ``B_obs`` (via
    ``Pi0Pytorch.compute_prefix_cache``), its KV cache is expanded to ``B_res``
    by ``repeat_interleave``, and the cheap Gemma-expert ``denoise_step`` is
    called once at ``B_res`` to produce all ``v_base_t`` at once.
  - Residual model is trained at ``B_res``, receiving the same ``(x_t, t)``.

Usage
  Single GPU:
    python scripts/train_residual_v_pytorch.py <config_name> --exp_name <run_name>
  Multi-GPU (single node):
    torchrun --standalone --nnodes=1 --nproc_per_node=<N> \
      scripts/train_residual_v_pytorch.py <config_name> --exp_name <run_name>
  Multi-node:
    torchrun --nnodes=<N> --nproc_per_node=<G> --node_rank=<R> \
      --master_addr=<IP> --master_port=<P> \
      scripts/train_residual_v_pytorch.py <config_name> --exp_name <run_name>
"""

import dataclasses
import gc
import logging
import os
import pathlib
import platform
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

import openpi.models.pi0_config  # noqa: F401  (register config for tyro)
import openpi.models.residual_v_config  # noqa: F401
import openpi.models_pytorch.pi0_pytorch
import openpi.models_pytorch.residual_v_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


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


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

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
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def _is_ddp(model):
    return isinstance(model, torch.nn.parallel.DistributedDataParallel)


def _unwrap(model):
    return model.module if _is_ddp(model) else model


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    if not is_main:
        return

    if not (
        (global_step % config.save_interval == 0 and global_step > 0)
        or global_step == config.num_train_steps - 1
    ):
        return

    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(_unwrap(model), tmp_ckpt_dir / "model.safetensors")
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

    logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")
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
    safetensors.torch.load_model(_unwrap(model), safetensors_path, device=str(device))

    optimizer_path = ckpt_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer_state_dict = torch.load(
            optimizer_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict

    metadata = torch.load(
        ckpt_dir / "metadata.pt", map_location=device, weights_only=False
    )
    global_step = metadata.get("global_step", latest_step)
    logging.info(f"Resumed training from step {global_step}")
    return global_step


def get_latest_checkpoint_step(checkpoint_dir: pathlib.Path):
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def resolve_model_safetensors(path_str: str) -> pathlib.Path:
    """Resolve a VLA checkpoint path to a concrete model.safetensors file."""
    path = pathlib.Path(path_str).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got: {path}")
        return path
    direct_model = path / "model.safetensors"
    if direct_model.exists():
        return direct_model
    latest = get_latest_checkpoint_step(path)
    if latest is None:
        raise FileNotFoundError(
            f"Could not find model.safetensors or numeric checkpoint dirs in {path}"
        )
    resolved = path / str(latest) / "model.safetensors"
    if not resolved.exists():
        raise FileNotFoundError(f"Missing model.safetensors at {resolved}")
    return resolved


def log_memory_usage(device, step, phase="unknown"):
    if not torch.cuda.is_available():
        return
    ma = torch.cuda.memory_allocated(device) / 1e9
    mr = torch.cuda.memory_reserved(device) / 1e9
    stats = torch.cuda.memory_stats(device)
    pa = stats.get("allocated_bytes.all.peak", 0) / 1e9
    pr = stats.get("reserved_bytes.all.peak", 0) / 1e9
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
    logging.info(
        f"Step {step} ({phase}): GPU alloc={ma:.2f}GB reserved={mr:.2f}GB "
        f"peak_alloc={pa:.2f}GB peak_reserved={pr:.2f}GB{ddp_info}"
    )


# -----------------------------------------------------------------------------
# Path sampling + KV cache expansion utilities
# -----------------------------------------------------------------------------


def sample_bin_times_flat(batch_size: int, k: int, device) -> torch.Tensor:
    """Stratified time sampling: one t per uniform bin of [0, 1] per observation.

    Returns a tensor of shape ``(batch_size * k,)`` where the k consecutive
    entries for observation ``i`` are the K stratified draws for that obs.
    Using stratified (not i.i.d.) t reduces the within-obs gradient correlation
    and makes the K-paths trick close to fully independent in effective variance.
    """
    bin_width = 1.0 / k
    offsets = torch.arange(k, dtype=torch.float32, device=device) * bin_width
    # Shape (batch_size, k): one i.i.d. uniform within each bin, per obs
    u = torch.rand(batch_size, k, dtype=torch.float32, device=device) * bin_width
    t = offsets.unsqueeze(0) + u  # (B, K) in [0, 1]
    # Shift into (eps, 1] to avoid t=0 singularities (match proxy's sample_time)
    t = t * 0.999 + 0.001
    return t.reshape(batch_size * k)


def _expand_tensor_list(tensor_list, k: int):
    return [
        t.repeat_interleave(k, dim=0) if isinstance(t, torch.Tensor) else t
        for t in tensor_list
    ]


def expand_kv_cache(past_kv, k: int):
    """Replicate a transformers KV cache along the batch dimension by ``k``.

    Tries multiple known internal layouts (``key_cache``/``value_cache`` list,
    ``layers`` list of ``(k, v)`` tuples, or fallback on any tensor attribute).
    Mutates the cache in place where possible and returns it.
    """
    # Transformers >= 4.40: DynamicCache exposes key_cache / value_cache lists
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        past_kv.key_cache = _expand_tensor_list(past_kv.key_cache, k)
        past_kv.value_cache = _expand_tensor_list(past_kv.value_cache, k)
        return past_kv

    # Some cache impls expose a .layers list of objects with .keys/.values
    if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
        for layer in past_kv.layers:
            for attr in ("keys", "values", "k", "v", "key", "value"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    setattr(layer, attr, t.repeat_interleave(k, dim=0))
        return past_kv

    # Tuple of (k, v) tensors per layer
    if isinstance(past_kv, tuple | list):
        return type(past_kv)(
            tuple(x.repeat_interleave(k, dim=0) for x in layer)
            if isinstance(layer, tuple | list)
            else layer.repeat_interleave(k, dim=0)
            for layer in past_kv
        )

    raise RuntimeError(
        f"Unable to expand KV cache of type {type(past_kv)}. "
        "Update expand_kv_cache() to handle this layout."
    )


def replicate_observation(observation, k: int):
    """Return a new observation with every tensor leaf repeat_interleaved by k.

    The Observation class is a jax/flax struct pytree, so we use ``jax.tree.map``
    to walk its leaves and replicate each tensor along dim 0.
    """
    return jax.tree.map(
        lambda x: x.repeat_interleave(k, dim=0) if isinstance(x, torch.Tensor) else x,
        observation,
    )


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # ----- checkpoint / resume bookkeeping -----
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
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

    if not resuming:
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        logging.info(
            f"Using existing experiment checkpoint directory: {config.checkpoint_dir}"
        )

    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # ----- data loader -----
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using B_obs per GPU: {effective_batch_size} "
        f"(total B_obs across {world_size} GPUs: {config.batch_size})"
    )

    loader, data_config = build_datasets(config)

    # ----- load frozen VLA base model -----
    vla_config_name = getattr(config, "vla_config_name", None)
    vla_checkpoint_dir = getattr(config, "vla_checkpoint_dir", None)
    if vla_config_name is None or vla_checkpoint_dir is None:
        raise ValueError(
            "vla_config_name and vla_checkpoint_dir must be specified for residual-V training."
        )

    if is_main:
        logging.info(
            f"Loading VLA base model: config={vla_config_name}, checkpoint={vla_checkpoint_dir}"
        )

    vla_train_config = _config.get_config(vla_config_name)
    vla_weight_path = resolve_model_safetensors(vla_checkpoint_dir)

    # Ensure VLA's sample_actions isn't torch.compiled (we call denoise_step directly
    # so this is cosmetic, but avoids surprising import-time compilation).
    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "0")

    vla_model = vla_train_config.model.load_pytorch(vla_train_config, str(vla_weight_path))

    if hasattr(vla_model, "paligemma_with_expert"):
        vla_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    vla_model = vla_model.to(device)
    vla_model.eval()
    for param in vla_model.parameters():
        param.requires_grad = False

    if not hasattr(vla_model, "compute_prefix_cache") or not hasattr(
        vla_model, "denoise_step"
    ):
        raise ValueError(
            f"VLA model {type(vla_model).__name__} must expose compute_prefix_cache() "
            "and denoise_step() to support residual-V training."
        )

    vla_action_dim = vla_model.config.action_dim
    if is_main:
        logging.info(
            f"VLA loaded. action_dim={vla_action_dim}, action_horizon={vla_model.config.action_horizon}"
        )

    # ----- build residual-V model -----
    import openpi.models.residual_v_config as _residual_v_config

    if not isinstance(config.model, _residual_v_config.ResidualVConfig):
        raise ValueError(
            "Model must be ResidualVConfig for residual-V training. "
            f"Got {type(config.model).__name__}."
        )

    model = openpi.models_pytorch.residual_v_pytorch.ResidualVPytorch(config.model).to(
        device
    )
    model_cfg = config.model

    if vla_model.config.action_horizon != model_cfg.action_horizon:
        raise ValueError(
            "VLA and residual-V must have the same action_horizon. "
            f"Got VLA={vla_model.config.action_horizon}, residual={model_cfg.action_horizon}."
        )
    if model_cfg.action_dim > vla_action_dim:
        raise ValueError(
            "Residual action_dim cannot exceed VLA action_dim. "
            f"Got residual={model_cfg.action_dim}, VLA={vla_action_dim}."
        )

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

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

    # ----- optional: load residual weights from a prior checkpoint (fine-tune) -----
    if config.pytorch_weight_path is not None and not resuming:
        residual_weight_path = resolve_model_safetensors(config.pytorch_weight_path)
        logging.info(f"Loading residual weights from: {residual_weight_path}")
        safetensors.torch.load_model(_unwrap(model), str(residual_weight_path))

    # ----- optimizer + LR schedule -----
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
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # ----- training state -----
    K = int(getattr(config, "num_paths_per_obs", 10))
    action_dim = model_cfg.action_dim
    H = model_cfg.action_horizon

    model.train()
    start_time = time.time()
    infos = []

    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={world_size}"
        )
        logging.info(
            f"B_obs={effective_batch_size}, K={K}, B_res={effective_batch_size * K}, "
            f"global_B_obs={config.batch_size}, global_B_res={config.batch_size * K}, "
            f"num_train_steps={config.num_train_steps}"
        )
        logging.info(
            f"Residual-V config: action_dim={action_dim}, H={H}, "
            f"vla_action_dim={vla_action_dim}, dtype={model_cfg.dtype}"
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
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, gt_actions, _ in loader:
            if global_step >= config.num_train_steps:
                break

            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            gt_actions = gt_actions.to(device)

            B_obs = gt_actions.shape[0]
            B_res = B_obs * K

            # Truncate GT actions to residual's action_dim -> a_steer_8.
            a_steer = gt_actions[:, :, :action_dim].to(torch.float32)

            # ---- [STAGE 1] Sample K (noise, time) per observation ----
            # Noise is sampled in VLA's full action_dim space (pad with zeros for residual).
            noise_full = torch.randn(
                B_res, H, vla_action_dim, dtype=torch.float32, device=device
            )
            time_flat = sample_bin_times_flat(B_obs, K, device)  # (B_res,)

            # a_steer in VLA's full dim: first action_dim entries are a_steer, rest 0.
            a_steer_full = torch.zeros(
                B_res, H, vla_action_dim, dtype=torch.float32, device=device
            )
            # (B_obs, H, D) -> (B_obs, K, H, D) -> (B_res, H, D)
            a_steer_full[:, :, :action_dim] = (
                a_steer.unsqueeze(1)
                .expand(B_obs, K, H, action_dim)
                .reshape(B_res, H, action_dim)
            )

            # Construct path states using the same convention as pi0 / proxy training:
            #   x_t = t * noise + (1 - t) * a_steer
            #   u_t = noise - a_steer     (the path velocity)
            tt = time_flat[:, None, None]
            x_t_full = tt * noise_full + (1.0 - tt) * a_steer_full

            # Slices for residual model (8-dim)
            x_t = x_t_full[:, :, :action_dim]
            noise_slim = noise_full[:, :, :action_dim]
            a_steer_rep = a_steer_full[:, :, :action_dim]
            u_t = noise_slim - a_steer_rep

            # ---- [STAGE 2] Base VLA prefix: ONCE at B_obs (expensive, no_grad, bf16) ----
            with torch.no_grad():
                past_kv, prefix_pad_masks, vla_state = vla_model.compute_prefix_cache(
                    observation
                )

                # ---- [STAGE 3] Expand KV cache 8 -> 80 ----
                past_kv_exp = expand_kv_cache(past_kv, K)
                prefix_pad_masks_exp = prefix_pad_masks.repeat_interleave(K, dim=0)
                vla_state_exp = vla_state.repeat_interleave(K, dim=0)

                # ---- [STAGE 4] Base VLA denoise_step: ONCE at B_res ----
                v_base_full = vla_model.denoise_step(
                    vla_state_exp,
                    prefix_pad_masks_exp,
                    past_kv_exp,
                    x_t_full,
                    time_flat,
                )
                v_base = v_base_full[:, :, :action_dim].to(torch.float32).detach()

            # Release large tensors we no longer need before the grad-requiring forward.
            del (
                past_kv,
                past_kv_exp,
                prefix_pad_masks,
                prefix_pad_masks_exp,
                vla_state,
                vla_state_exp,
                v_base_full,
                noise_full,
                a_steer_full,
                x_t_full,
            )

            # ---- [STAGE 5] Residual-V forward + backward at B_res ----
            observation_K = replicate_observation(observation, K)

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            v_res = model(observation_K, x_t=x_t, time=time_flat, v_base=v_base)
            v_res_target = (u_t - v_base).detach()

            loss = torch.nn.functional.mse_loss(v_res, v_res_target)

            # Diagnostic: gamma=0 baseline loss (how much the base alone misses u_t).
            with torch.no_grad():
                loss_gamma0 = torch.nn.functional.mse_loss(v_base, u_t).detach()

            loss.backward()

            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

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
                        "loss_gamma0": float(loss_gamma0.item()),
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
                avg = lambda key: sum(i[key] for i in infos) / len(infos)  # noqa: E731

                avg_loss = avg("loss")
                avg_loss_g0 = avg("loss_gamma0")
                avg_lr = avg("learning_rate")
                grad_norms = [
                    i["grad_norm"]
                    for i in infos
                    if i.get("grad_norm") is not None
                ]
                avg_gn = sum(grad_norms) / len(grad_norms) if grad_norms else None

                msg = (
                    f"step={global_step} loss={avg_loss:.4f} "
                    f"loss_gamma0={avg_loss_g0:.4f} lr={avg_lr:.2e}"
                )
                if avg_gn is not None:
                    msg += f" grad_norm={avg_gn:.2f}"
                msg += f" time={elapsed:.1f}s"
                logging.info(msg)

                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "loss_gamma0": avg_loss_g0,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_gn is not None:
                        log_payload["grad_norm"] = avg_gn
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

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

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
