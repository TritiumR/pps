"""
Compare teacher/student flow vectors on real-world recorded observations.

This script follows `scripts/compare_pytorch.py`, but instead of dataset batches
it reads real-world recording `.pkl` files (as used by `visualize_real.py`) and
uses recorded diffusion states from `step["vectors"]`.
"""

import argparse
import logging
import os
import pathlib
import pickle
import platform
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
import torch
import tqdm

import openpi.models.model as _model
import openpi.models.proxy_config
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


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


def load_input_norm_stats(
    config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    data_config: _config.DataConfig,
):
    # Match policy loading behavior for real-world inference as closely as possible:
    # 1) explicit norm_stats_dir from config
    # 2) checkpoint assets/<asset_id>
    # 3) fallback to data_config.norm_stats
    if getattr(config.data, "norm_stats_dir", None) is not None:
        project_root = pathlib.Path(__file__).resolve().parents[1]
        norm_stats_dir = project_root / config.data.norm_stats_dir
        try:
            return _checkpoints.load_norm_stats(norm_stats_dir=str(norm_stats_dir))
        except FileNotFoundError:
            logging.warning("norm_stats_dir not found at %s. Falling back.", norm_stats_dir)

    if data_config.asset_id is not None:
        try:
            return _checkpoints.load_norm_stats(
                assets_dir=checkpoint_dir / "assets", asset_id=data_config.asset_id
            )
        except FileNotFoundError:
            logging.warning(
                "Norm stats not found in checkpoint assets for asset_id=%s. Falling back to config norm stats.",
                data_config.asset_id,
            )

    return data_config.norm_stats


def build_input_transform(
    config: _config.TrainConfig,
    student_checkpoint_dir: pathlib.Path,
    *,
    default_prompt: str | None,
):
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = load_input_norm_stats(config, student_checkpoint_dir, data_config)

    input_transform = _transforms.compose(
        [
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ]
    )
    return input_transform


def raw_obs_to_model_observation(raw_obs: dict, input_transform, device):
    inputs = jax.tree.map(lambda x: x, raw_obs)
    inputs = input_transform(inputs)

    def _to_tensor(x):
        arr = np.asarray(x)
        if arr.dtype.kind in ("U", "S", "O"):
            raise TypeError(
                f"Non-numeric leaf remained after input transforms with dtype={arr.dtype}. "
                "Check prompt/tokenization inputs."
            )
        return torch.from_numpy(arr).to(device)[None, ...]

    inputs = jax.tree.map(_to_tensor, inputs)
    observation = _model.Observation.from_dict(inputs)
    return observation


def parse_recorded_vectors(step: dict, device: torch.device):
    if "vectors" not in step:
        raise KeyError("Missing 'vectors' in recording step.")

    vectors = step["vectors"]
    if len(vectors) == 0:
        raise ValueError("Empty 'vectors' in recording step.")

    vectors_np = np.asarray(vectors, dtype=np.float32)

    # Typical shapes:
    # - list[(1, H, D)] -> (L, 1, H, D) -> swap to (1, L, H, D)
    # - (L, H, D) -> add batch -> (1, L, H, D)
    # - (1, L, H, D) -> use directly
    if vectors_np.ndim == 4:
        if vectors_np.shape[0] == 1:
            vectors_np = vectors_np
        elif vectors_np.shape[1] == 1:
            vectors_np = np.swapaxes(vectors_np, 0, 1)
        else:
            raise ValueError(f"Unexpected vectors shape: {vectors_np.shape}")
    elif vectors_np.ndim == 3:
        vectors_np = vectors_np[None, ...]
    else:
        raise ValueError(f"Unexpected vectors shape: {vectors_np.shape}")

    x_states = torch.from_numpy(vectors_np).to(device=device, dtype=torch.float32)

    if x_states.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 recorded diffusion states, got shape={tuple(x_states.shape)}"
        )

    recorded_num_steps = int(step.get("num_steps", x_states.shape[1] - 1))
    if recorded_num_steps <= 0:
        raise ValueError(f"Invalid num_steps in recording: {recorded_num_steps}")

    if x_states.shape[1] != recorded_num_steps + 1:
        logging.warning(
            "Recorded vectors length (%s) != num_steps + 1 (%s). Deriving num_steps from vectors length.",
            x_states.shape[1],
            recorded_num_steps + 1,
        )
        recorded_num_steps = x_states.shape[1] - 1

    times_1d = 1.0 - torch.arange(
        x_states.shape[1], dtype=torch.float32, device=device
    ) / float(recorded_num_steps)
    times = times_1d.unsqueeze(0).expand(x_states.shape[0], -1)

    return x_states, times, recorded_num_steps


def predict_proxy_flows_from_states(
    model: torch.nn.Module,
    observation,
    x_states: torch.Tensor,
    times: torch.Tensor,
    *,
    use_train_preprocess: bool,
) -> torch.Tensor:
    images, img_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=use_train_preprocess
    )
    prefix_embs, prefix_pad_masks, _ = model.embed_prefix(images, img_masks)

    action_dim = model.config.action_dim
    if x_states.shape[-1] < action_dim:
        raise ValueError(
            f"Recorded vector dim {x_states.shape[-1]} is smaller than model action_dim {action_dim}"
        )

    x_steps = x_states[:, 1:, :, :action_dim]
    t_steps = times[:, 1:]

    flows = []
    for step_idx in range(t_steps.shape[1]):
        x_t = x_steps[:, step_idx, :, :]
        t_t = t_steps[:, step_idx]

        suffix_embs, suffix_pad_masks, _, adarms_cond = model.embed_suffix(
            state, x_t, t_t
        )
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        attention_mask = pad_masks
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        position_ids = position_ids.to(dtype=torch.long)

        hidden_states, _ = model.expert_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = hidden_states[:, -model.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        flows.append(model.action_out_proj(suffix_out))

    return torch.stack(flows, dim=1)


def predict_pi0_flows_from_states(
    model: torch.nn.Module,
    observation,
    x_states: torch.Tensor,
    times: torch.Tensor,
    *,
    use_train_preprocess: bool,
) -> torch.Tensor:
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=use_train_preprocess
    )

    state = torch.nn.functional.pad(
        state,
        (0, model.config.action_dim - state.shape[1]),
        mode="constant",
        value=0,
    )

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001

    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
        "eager"
    )
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    action_dim = model.config.action_dim
    if x_states.shape[-1] < action_dim:
        raise ValueError(
            f"Recorded vector dim {x_states.shape[-1]} is smaller than model action_dim {action_dim}"
        )

    x_steps = x_states[:, 1:, :, :action_dim]
    t_steps = times[:, 1:]

    flows = []
    for step_idx in range(t_steps.shape[1]):
        x_t = x_steps[:, step_idx, :, :]
        t_t = t_steps[:, step_idx]
        flows.append(
            model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_t,
            )
        )
    return torch.stack(flows, dim=1)


def predict_model_flows_from_states(
    model: torch.nn.Module,
    observation,
    x_states: torch.Tensor,
    times: torch.Tensor,
    *,
    use_train_preprocess: bool,
) -> torch.Tensor:
    if hasattr(model, "paligemma_with_expert") and hasattr(model, "denoise_step"):
        return predict_pi0_flows_from_states(
            model,
            observation,
            x_states,
            times,
            use_train_preprocess=use_train_preprocess,
        )
    if hasattr(model, "expert_model") and hasattr(model, "embed_suffix"):
        return predict_proxy_flows_from_states(
            model,
            observation,
            x_states,
            times,
            use_train_preprocess=use_train_preprocess,
        )
    raise ValueError(
        f"Unsupported model type for recorded-state flow prediction: {type(model).__name__}"
    )


def align_flow_shapes(
    teacher_flows: torch.Tensor,
    student_flows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    common_steps = min(teacher_flows.shape[1], student_flows.shape[1])
    common_horizon = min(teacher_flows.shape[2], student_flows.shape[2])
    common_dim = min(teacher_flows.shape[3], student_flows.shape[3])

    if common_steps <= 0 or common_horizon <= 0 or common_dim <= 0:
        raise ValueError(
            f"Invalid common flow shape between teacher={teacher_flows.shape} and student={student_flows.shape}"
        )

    return (
        teacher_flows[:, :common_steps, :common_horizon, :common_dim],
        student_flows[:, :common_steps, :common_horizon, :common_dim],
    )


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


def resolve_recording_dir(
    recording_dir: str | None,
    instruction: str | None,
    recording_exp_name: str | None,
) -> pathlib.Path:
    if recording_dir is not None:
        path = pathlib.Path(recording_dir).expanduser().resolve()
    else:
        if instruction is None or recording_exp_name is None:
            raise ValueError(
                "Either --recording_dir or both --instruction and --recording_exp_name must be provided."
            )
        path = (pathlib.Path("droid") / "results" / instruction / recording_exp_name).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Recording directory does not exist: {path}")
    return path


def iter_recorded_steps(recording_dir: pathlib.Path):
    pkl_files = sorted(recording_dir.glob("*.pkl"))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found in recording directory: {recording_dir}")

    for pkl_path in pkl_files:
        with pkl_path.open("rb") as f:
            steps = pickle.load(f)

        if not isinstance(steps, list):
            logging.warning("Skipping %s because it does not contain a list of steps.", pkl_path)
            continue

        for step_idx, step in enumerate(steps):
            yield pkl_path, step_idx, step


def compare_loop(
    config: _config.TrainConfig,
    *,
    recording_dir: str | None,
    instruction: str | None,
    recording_exp_name: str | None,
    max_real_steps: int | None,
    log_every: int,
    student_checkpoint_dir: str | None,
    default_prompt: str | None,
    use_train_preprocess: bool,
    use_teacher_flow_path: bool,
):
    device = setup_device()
    set_seed(config.seed)

    teacher_model = load_teacher_model(config, device)
    student_model, student_weight_path = load_student_model(
        config, device, student_checkpoint_dir
    )

    recording_path = resolve_recording_dir(
        recording_dir, instruction, recording_exp_name
    )
    input_transform = build_input_transform(
        config,
        student_weight_path.parent,
        default_prompt=default_prompt,
    )
    num_distill_steps = getattr(config, "num_distill_steps", 10)

    logging.info("Running on: %s", platform.node())
    logging.info(
        "Real comparison config: recording_dir=%s max_real_steps=%s log_every=%s use_train_preprocess=%s use_teacher_flow_path=%s num_distill_steps=%s",
        recording_path,
        max_real_steps,
        log_every,
        use_train_preprocess,
        use_teacher_flow_path,
        num_distill_steps,
    )
    logging.info("Student checkpoint: %s", student_weight_path)

    progress = tqdm.tqdm(
        total=max_real_steps if max_real_steps is not None else None,
        desc="Comparing real",
    )

    stats = {}
    processed = 0
    skipped = 0

    with torch.no_grad():
        for pkl_path, step_idx, step in iter_recorded_steps(recording_path):
            if max_real_steps is not None and processed >= max_real_steps:
                break

            if not isinstance(step, dict):
                skipped += 1
                continue
            if "observation" not in step:
                skipped += 1
                continue
            if (not use_teacher_flow_path) and "vectors" not in step:
                skipped += 1
                continue

            raw_obs = step["observation"]
            if not isinstance(raw_obs, dict):
                skipped += 1
                continue

            try:
                observation = raw_obs_to_model_observation(
                    raw_obs, input_transform, device
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logging.warning(
                    "Skipping %s step=%s due to input parse error: %s",
                    pkl_path.name,
                    step_idx,
                    exc,
                )
                continue

            try:
                if use_teacher_flow_path:
                    if not hasattr(teacher_model, "forward_for_distill"):
                        raise ValueError(
                            f"Teacher model {type(teacher_model).__name__} does not support forward_for_distill()."
                        )

                    noises, times, teacher_gradients, _ = teacher_model.forward_for_distill(
                        observation, num_distill_steps
                    )
                    x_states = noises
                    teacher_flows = teacher_gradients[:, 1:, :, :]
                    path_num_steps = num_distill_steps
                else:
                    x_states, times, path_num_steps = parse_recorded_vectors(step, device)
                    teacher_flows = predict_model_flows_from_states(
                        teacher_model,
                        observation,
                        x_states,
                        times,
                        use_train_preprocess=use_train_preprocess,
                    )

                student_flows = predict_model_flows_from_states(
                    student_model,
                    observation,
                    x_states,
                    times,
                    use_train_preprocess=use_train_preprocess,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logging.warning(
                    "Skipping %s step=%s due to model forward error: %s",
                    pkl_path.name,
                    step_idx,
                    exc,
                )
                continue

            teacher_cmp, student_cmp = align_flow_shapes(teacher_flows, student_flows)

            batch_stats = compute_batch_metrics(teacher_cmp, student_cmp)
            if not stats:
                stats = {k: v.clone() for k, v in batch_stats.items()}
                preview_dim = min(8, teacher_cmp.shape[-1], student_cmp.shape[-1])
                logging.info(
                    "First valid step: file=%s step=%s path_num_steps=%s teacher_shape=%s student_shape=%s",
                    pkl_path.name,
                    step_idx,
                    path_num_steps,
                    tuple(teacher_cmp.shape),
                    tuple(student_cmp.shape),
                )
                logging.info(
                    "First flow slice teacher=%s student=%s",
                    teacher_cmp[0, 0, 0, :preview_dim].detach().cpu().tolist(),
                    student_cmp[0, 0, 0, :preview_dim].detach().cpu().tolist(),
                )
            else:
                for key, value in batch_stats.items():
                    stats[key] += value

            processed += 1
            progress.update(1)

            if processed % log_every == 0:
                logging.info(
                    "real_step=%s %s",
                    processed,
                    format_running_metrics(stats),
                )

    progress.close()

    if processed == 0:
        raise RuntimeError(
            f"No valid recording steps processed from {recording_path}. "
            "Check that steps contain both 'observation' and 'vectors'."
        )

    logging.info(
        "Final comparison over %s real steps (skipped=%s): %s",
        processed,
        skipped,
        format_running_metrics(stats),
    )


def parse_args() -> tuple[argparse.Namespace, _config.TrainConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recording_dir", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--recording_exp_name", type=str, default=None)
    parser.add_argument("--max_real_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--student_checkpoint_dir", type=str, default=None)
    parser.add_argument("--default_prompt", type=str, default=None)
    parser.add_argument("--use_train_preprocess", action="store_true")
    parser.add_argument("--use_teacher_flow_path", action="store_true")

    compare_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        config = _config.cli()
    finally:
        sys.argv = original_argv

    if compare_args.log_every <= 0:
        raise ValueError("--log_every must be greater than 0.")
    if compare_args.max_real_steps is not None and compare_args.max_real_steps <= 0:
        raise ValueError("--max_real_steps must be greater than 0.")

    return compare_args, config


def main():
    init_logging()
    compare_args, config = parse_args()
    compare_loop(
        config,
        recording_dir=compare_args.recording_dir,
        instruction=compare_args.instruction,
        recording_exp_name=compare_args.recording_exp_name,
        max_real_steps=compare_args.max_real_steps,
        log_every=compare_args.log_every,
        student_checkpoint_dir=compare_args.student_checkpoint_dir,
        default_prompt=compare_args.default_prompt,
        use_train_preprocess=compare_args.use_train_preprocess,
        use_teacher_flow_path=compare_args.use_teacher_flow_path,
    )


if __name__ == "__main__":
    main()
