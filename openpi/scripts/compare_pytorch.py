"""
PyTorch comparison entrypoint for teacher/student flow vectors.

This script reuses the same config parsing, data loading, and model loading flow as
`scripts/distill_pytorch.py`, but runs comparison only. It samples denoising states
from the teacher model and compares the teacher flow vectors against the student flow
vectors on the same states. No training is performed.

Usage:
  python scripts/compare_pytorch.py <config_name> --exp_name <student_exp_name> \
      --teacher_config_name <teacher_cfg> --teacher_checkpoint_dir <teacher_ckpt>

Optional comparison-only flags:
  --compare_batches <N>          Number of batches to compare. Default: 100
  --log_every <N>                Log running metrics every N batches. Default: 10
  --student_checkpoint_dir <p>   Explicit student checkpoint directory or step dir.
  --use_train_preprocess         Use the student training preprocessing path.
"""

import argparse
import json
import logging
import os
import pathlib
import platform
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch
import tqdm

import openpi.models.proxy_config
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


def build_datasets(config: _config.TrainConfig, *, num_batches: int):
    data_loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=num_batches,
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


def freeze_model(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_teacher_model(config: _config.TrainConfig, device: torch.device) -> torch.nn.Module:
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


def load_student_model(
    config: _config.TrainConfig,
    device: torch.device,
    student_checkpoint_dir: str | None,
) -> tuple[torch.nn.Module, pathlib.Path]:
    if not isinstance(config.model, openpi.models.proxy_config.ProxyConfig):
        raise ValueError(
            "Student model must be ProxyPytorch for comparison. Current model is not ProxyConfig."
        )

    candidate_paths: list[str] = []
    if student_checkpoint_dir is not None:
        candidate_paths.append(student_checkpoint_dir)
    candidate_paths.append(str(config.checkpoint_dir))
    if config.pytorch_weight_path is not None:
        candidate_paths.append(config.pytorch_weight_path)

    student_weight_path = None
    last_error = None
    for candidate_path in candidate_paths:
        try:
            student_weight_path = resolve_model_safetensors(candidate_path)
            break
        except (FileNotFoundError, ValueError) as exc:
            last_error = exc

    if student_weight_path is None:
        raise FileNotFoundError(
            "Unable to resolve student checkpoint. Checked: "
            + ", ".join(candidate_paths)
        ) from last_error

    student_model = config.model.load_pytorch(config, str(student_weight_path))
    student_model = student_model.to(device)
    freeze_model(student_model)
    logging.info("Loaded student model from %s", student_weight_path)
    return student_model, student_weight_path


def predict_student_flows(
    student_model: torch.nn.Module,
    observation,
    noises: torch.Tensor,
    times: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    use_noise: bool,
    use_train_preprocess: bool,
) -> torch.Tensor:
    """Mirror ProxyPytorch.forward_distill(), but return predicted flow vectors."""
    images, img_masks, state = student_model._preprocess_observation(  # noqa: SLF001
        observation,
        train=use_train_preprocess,
    )

    prefix_embs, prefix_pad_masks, _ = student_model.embed_prefix(images, img_masks)

    action_dim = student_model.config.action_dim
    initial_noise = noises[:, 0, :, :action_dim]
    noises = noises[:, 1:, :, :action_dim]
    times = times[:, 1:]
    teacher_actions = teacher_actions[:, :, :action_dim]

    _, num_steps = times.shape
    predictions = []

    for step_idx in range(num_steps):
        noise_step = noises[:, step_idx, :, :]
        time_step = times[:, step_idx]

        time_expanded = time_step[:, None, None]
        if use_noise:
            x_t = noise_step
        else:
            x_t = time_expanded * initial_noise + (1 - time_expanded) * teacher_actions

        suffix_embs, suffix_pad_masks, _, adarms_cond = student_model.embed_suffix(
            state, x_t, time_step
        )

        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        attention_mask = pad_masks
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        position_ids = position_ids.to(dtype=torch.long)

        hidden_states, _ = student_model.expert_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = hidden_states[:, -student_model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        predictions.append(student_model.action_out_proj(suffix_out))

    return torch.stack(predictions, dim=1)


def compute_batch_metrics(
    teacher_flows: torch.Tensor,
    student_flows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    diff = student_flows - teacher_flows

    teacher_vec = teacher_flows.reshape(teacher_flows.shape[0], teacher_flows.shape[1], -1)
    student_vec = student_flows.reshape(student_flows.shape[0], student_flows.shape[1], -1)
    cosine = torch.nn.functional.cosine_similarity(student_vec, teacher_vec, dim=-1)

    return {
        "sq_error_sum": diff.pow(2).sum(dtype=torch.float64),
        "abs_error_sum": diff.abs().sum(dtype=torch.float64),
        "teacher_sq_sum": teacher_flows.pow(2).sum(dtype=torch.float64),
        "student_sq_sum": student_flows.pow(2).sum(dtype=torch.float64),
        "cosine_sum": cosine.sum(dtype=torch.float64),
        "element_count": torch.tensor(diff.numel(), dtype=torch.float64, device=diff.device),
        "vector_count": torch.tensor(cosine.numel(), dtype=torch.float64, device=diff.device),
        "batch_count": torch.tensor(teacher_flows.shape[0], dtype=torch.float64, device=diff.device),
    }


def format_running_metrics(stats: dict[str, torch.Tensor]) -> str:
    mse = (stats["sq_error_sum"] / stats["element_count"]).item()
    mae = (stats["abs_error_sum"] / stats["element_count"]).item()
    cosine = (stats["cosine_sum"] / stats["vector_count"]).item()
    teacher_rms = torch.sqrt(stats["teacher_sq_sum"] / stats["element_count"]).item()
    student_rms = torch.sqrt(stats["student_sq_sum"] / stats["element_count"]).item()
    teacher_sq_sum = torch.clamp(stats["teacher_sq_sum"], min=torch.finfo(torch.float64).eps)
    rel_l2 = torch.sqrt(stats["sq_error_sum"] / teacher_sq_sum).item()

    return (
        f"mse={mse:.6f} mae={mae:.6f} cosine={cosine:.6f} "
        f"teacher_rms={teacher_rms:.6f} student_rms={student_rms:.6f} rel_l2={rel_l2:.6f}"
    )


def stats_to_serializable(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.item()) for key, value in stats.items()}


def summarize_metrics(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    eps = torch.finfo(torch.float64).eps
    element_count = torch.clamp(stats["element_count"], min=eps)
    vector_count = torch.clamp(stats["vector_count"], min=eps)
    teacher_sq_sum = torch.clamp(stats["teacher_sq_sum"], min=eps)

    return {
        "mse": float((stats["sq_error_sum"] / element_count).item()),
        "mae": float((stats["abs_error_sum"] / element_count).item()),
        "cosine": float((stats["cosine_sum"] / vector_count).item()),
        "teacher_rms": float(torch.sqrt(stats["teacher_sq_sum"] / element_count).item()),
        "student_rms": float(torch.sqrt(stats["student_sq_sum"] / element_count).item()),
        "rel_l2": float(torch.sqrt(stats["sq_error_sum"] / teacher_sq_sum).item()),
    }


def compare_loop(
    config: _config.TrainConfig,
    *,
    compare_batches: int,
    log_every: int,
    student_checkpoint_dir: str | None,
    use_train_preprocess: bool,
    output_json: str | None,
):
    device = setup_device()
    set_seed(config.seed)

    loader, _ = build_datasets(config, num_batches=compare_batches)

    teacher_model = load_teacher_model(config, device)
    student_model, student_weight_path = load_student_model(
        config, device, student_checkpoint_dir
    )

    num_distill_steps = getattr(config, "num_distill_steps", 10)
    use_noise = getattr(config, "use_noise_for_distill", True)

    logging.info("Running on: %s", platform.node())
    logging.info(
        "Comparison config: compare_batches=%s log_every=%s num_distill_steps=%s use_noise=%s use_train_preprocess=%s",
        compare_batches,
        log_every,
        num_distill_steps,
        use_noise,
        use_train_preprocess,
    )
    logging.info("Student checkpoint: %s", student_weight_path)

    progress = tqdm.tqdm(total=compare_batches, desc="Comparing")

    stats = {}

    with torch.no_grad():
        for batch_idx, (observation, _, _) in enumerate(loader, start=1):
            observation = jax.tree.map(lambda x: x.to(device), observation)

            noises, times, teacher_gradients, teacher_actions = teacher_model.forward_for_distill(
                observation, num_distill_steps
            )

            action_dim = student_model.config.action_dim
            teacher_flows = teacher_gradients[:, 1:, :, :action_dim]
            student_flows = predict_student_flows(
                student_model,
                observation,
                noises,
                times,
                teacher_actions,
                use_noise=use_noise,
                use_train_preprocess=use_train_preprocess,
            )

            if teacher_flows.shape != student_flows.shape:
                raise ValueError(
                    f"Teacher/student flow shapes do not match: {teacher_flows.shape} vs {student_flows.shape}"
                )

            batch_stats = compute_batch_metrics(teacher_flows, student_flows)
            if not stats:
                stats = {k: v.clone() for k, v in batch_stats.items()}

                logging.info(
                    "Flow tensor shapes teacher=%s student=%s times=%s",
                    tuple(teacher_flows.shape),
                    tuple(student_flows.shape),
                    tuple(times[:, 1:].shape),
                )
                preview_dim = min(8, action_dim)
                logging.info(
                    "First flow slice teacher=%s student=%s",
                    teacher_flows[0, 0, 0, :preview_dim].detach().cpu().tolist(),
                    student_flows[0, 0, 0, :preview_dim].detach().cpu().tolist(),
                )
            else:
                for key, value in batch_stats.items():
                    stats[key] += value

            if batch_idx % log_every == 0 or batch_idx == compare_batches:
                logging.info(
                    "batch=%s/%s %s",
                    batch_idx,
                    compare_batches,
                    format_running_metrics(stats),
                )

            progress.update(1)

    progress.close()

    logging.info("Final comparison: %s", format_running_metrics(stats))

    output_path = (
        pathlib.Path(output_json).expanduser().resolve()
        if output_json is not None
        else student_weight_path.parent / "compare_pytorch.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "compare_batches": compare_batches,
        "log_every": log_every,
        "num_distill_steps": num_distill_steps,
        "use_noise_for_distill": bool(use_noise),
        "use_train_preprocess": bool(use_train_preprocess),
        "teacher_checkpoint": str(resolve_model_safetensors(config.teacher_checkpoint_dir)),
        "student_checkpoint": str(student_weight_path),
        "total_stats": stats_to_serializable(stats),
        "metrics": summarize_metrics(stats),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved comparison summary to %s", output_path)


def parse_args() -> tuple[argparse.Namespace, _config.TrainConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compare_batches", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--student_checkpoint_dir", type=str, default=None)
    parser.add_argument("--use_train_preprocess", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)

    compare_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        config = _config.cli()
    finally:
        sys.argv = original_argv

    if compare_args.compare_batches <= 0:
        raise ValueError("--compare_batches must be greater than 0.")
    if compare_args.log_every <= 0:
        raise ValueError("--log_every must be greater than 0.")

    return compare_args, config


def main():
    init_logging()
    compare_args, config = parse_args()
    compare_loop(
        config,
        compare_batches=compare_args.compare_batches,
        log_every=compare_args.log_every,
        student_checkpoint_dir=compare_args.student_checkpoint_dir,
        use_train_preprocess=compare_args.use_train_preprocess,
        output_json=compare_args.output_json,
    )


if __name__ == "__main__":
    main()
