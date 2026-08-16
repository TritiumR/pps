#!/usr/bin/env python3
"""Offline teacher-forced and closed-loop diagnostics for cached ref trajectories."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np
import safetensors.torch
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI = ROOT / "openpi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPENPI / "src"))

from openpi.models import model as model_api  # noqa: E402
from openpi.models_pytorch.proxy_score_pytorch import ProxyScorePytorch  # noqa: E402
from openpi.training import config as training_config  # noqa: E402
from sim_free_mpc.ddim import ddim_iteration_alphas  # noqa: E402


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=pathlib.Path, required=True)
    parser.add_argument("--observation-cache-path", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config", default="score_ref_weight_eps")
    parser.add_argument("--attention", choices=("config", "causal", "bidirectional"), default="bidirectional")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-observations", type=int, default=0)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _load_cache(path: pathlib.Path) -> dict:
    cache = np.load(path, allow_pickle=False)
    metadata = json.loads(str(cache["metadata_json"].item()))
    labels_per_trajectory = int(metadata["labels_per_trajectory"])
    num_trajectories = int(metadata["num_trajectories"])
    if cache["x_t"].shape[0] != labels_per_trajectory * num_trajectories:
        raise ValueError("Cache does not contain a complete rectangular trajectory grid.")
    if "epsilon" in cache.files:
        epsilon = cache["epsilon"].astype(np.float32)
    elif "score" in cache.files:
        times = cache["time"].astype(np.float32)
        epsilon = np.empty_like(cache["score"], dtype=np.float32)
        for idx, time in enumerate(times):
            iteration = int(cache["iteration"][idx])
            alpha, _ = ddim_iteration_alphas(
                iteration=iteration,
                num_iterations=labels_per_trajectory,
                num_train_timesteps=int(metadata["ddim_num_train_timesteps"]),
            )
            del time
            epsilon[idx] = -math.sqrt(max(1.0 - alpha, 1e-6)) * cache["score"][idx]
    else:
        raise ValueError("Cache contains neither epsilon nor score targets.")
    return {
        "metadata": metadata,
        "x_t": cache["x_t"].astype(np.float32).reshape(num_trajectories, labels_per_trajectory, *cache["x_t"].shape[1:]),
        "epsilon": epsilon.reshape(num_trajectories, labels_per_trajectory, *epsilon.shape[1:]),
        "time": cache["time"].astype(np.float32).reshape(num_trajectories, labels_per_trajectory),
        "demo_name": cache["demo_name"].astype(str).reshape(num_trajectories, labels_per_trajectory)[:, 0],
        "step_index": cache["step_index"].astype(np.int64).reshape(num_trajectories, labels_per_trajectory)[:, 0],
    }


def _trajectory_observation_map(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return all trajectories and their first-appearance observation indices."""
    key_to_observation: dict[tuple[str, int], int] = {}
    observation_indices = []
    for demo, step in zip(cache["demo_name"], cache["step_index"], strict=True):
        key = (str(demo), int(step))
        if key not in key_to_observation:
            key_to_observation[key] = len(key_to_observation)
        observation_indices.append(key_to_observation[key])
    return (
        np.arange(len(observation_indices), dtype=np.int64),
        np.asarray(observation_indices, dtype=np.int64),
    )


def _select_observations(
    cache: dict, limit: int, seed: int, selected_observations: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    trajectory_indices, trajectory_observations = _trajectory_observation_map(cache)
    num_observations = int(trajectory_observations.max()) + 1
    if selected_observations is None:
        selected_observations = np.arange(num_observations, dtype=np.int64)
    if limit > 0 and len(selected_observations) > limit:
        rng = np.random.default_rng(seed)
        selected_observations = np.sort(
            rng.choice(selected_observations, size=limit, replace=False)
        )
    mask = np.isin(trajectory_observations, selected_observations)
    return trajectory_indices[mask], trajectory_observations[mask], len(selected_observations)


def _load_observation_arrays(path: pathlib.Path):
    return {
        "images": np.load(path / "images.npy", mmap_mode="r"),
        "image_masks": np.load(path / "image_masks.npy", mmap_mode="r"),
        "states": np.load(path / "states.npy", mmap_mode="r"),
        "tokenized_prompt": np.load(path / "tokenized_prompt.npy", mmap_mode="r"),
        "tokenized_prompt_mask": np.load(path / "tokenized_prompt_mask.npy", mmap_mode="r"),
    }


def _observation_batch(arrays, indices: np.ndarray, device: torch.device):
    batch = len(indices)
    prompt = torch.from_numpy(np.asarray(arrays["tokenized_prompt"]))
    prompt_mask = torch.from_numpy(np.asarray(arrays["tokenized_prompt_mask"]))
    payload = {
        "image": {
            key: torch.from_numpy(np.asarray(arrays["images"][indices, image_idx])).to(device)
            for image_idx, key in enumerate(IMAGE_KEYS)
        },
        "image_mask": {
            key: torch.from_numpy(np.asarray(arrays["image_masks"][indices, image_idx])).to(device)
            for image_idx, key in enumerate(IMAGE_KEYS)
        },
        "state": torch.from_numpy(np.asarray(arrays["states"][indices])).to(device),
        "tokenized_prompt": prompt.unsqueeze(0).expand(batch, -1).to(device),
        "tokenized_prompt_mask": prompt_mask.unsqueeze(0).expand(batch, -1).to(device),
    }
    return model_api.Observation.from_dict(payload)


def _cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    denom = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
    return (left * right).sum(dim=1) / torch.clamp(denom, min=1e-12)


def _mbd_score_step(
    x_t: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    iteration: int,
    num_levels: int,
    num_train_timesteps: int,
) -> torch.Tensor:
    alpha, alpha_prev = ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=num_levels,
        num_train_timesteps=num_train_timesteps,
    )
    alpha_t = torch.as_tensor(alpha, device=x_t.device, dtype=x_t.dtype)
    alpha_prev_t = torch.as_tensor(alpha_prev, device=x_t.device, dtype=x_t.dtype)
    beta = torch.clamp(1.0 - alpha_t, min=1e-6)
    score = -epsilon / torch.sqrt(beta)
    alpha_step = torch.clamp(alpha_t / torch.clamp(alpha_prev_t, min=1e-6), min=1e-6)
    return (x_t + beta * score) / torch.sqrt(alpha_step)


def _new_accumulator(num_levels: int) -> dict[str, list[list[float]]]:
    return {
        "epsilon_squared_error": [[] for _ in range(num_levels)],
        "epsilon_cosine": [[] for _ in range(num_levels)],
        "epsilon_target_rms": [[] for _ in range(num_levels)],
        "epsilon_pred_rms": [[] for _ in range(num_levels)],
        "teacher_forced_next_squared_error": [[] for _ in range(num_levels - 1)],
        "closed_loop_state_squared_error": [[] for _ in range(num_levels)],
        "closed_loop_next_squared_error": [[] for _ in range(num_levels - 1)],
    }


def _summarize(accumulator: dict, metadata: dict) -> dict:
    per_timestep = []
    num_levels = len(accumulator["epsilon_squared_error"])
    for iteration in range(num_levels):
        alpha, alpha_prev = ddim_iteration_alphas(
            iteration=iteration,
            num_iterations=num_levels,
            num_train_timesteps=int(metadata["ddim_num_train_timesteps"]),
        )
        eps_sq = np.concatenate(accumulator["epsilon_squared_error"][iteration])
        eps_cos = np.concatenate(accumulator["epsilon_cosine"][iteration])
        eps_target_rms = np.concatenate(accumulator["epsilon_target_rms"][iteration])
        eps_pred_rms = np.concatenate(accumulator["epsilon_pred_rms"][iteration])
        state_sq = np.concatenate(accumulator["closed_loop_state_squared_error"][iteration])
        row = {
            "iteration": iteration,
            "alpha_bar": alpha,
            "alpha_bar_prev": alpha_prev,
            "epsilon_mse": float(eps_sq.mean()),
            "epsilon_rmse": float(np.sqrt(eps_sq.mean())),
            "epsilon_cosine": float(eps_cos.mean()),
            "epsilon_target_rms": float(eps_target_rms.mean()),
            "epsilon_pred_rms": float(eps_pred_rms.mean()),
            "epsilon_pred_to_target_rms_ratio": float(
                eps_pred_rms.mean() / max(eps_target_rms.mean(), 1e-12)
            ),
            "closed_loop_state_rmse": float(np.sqrt(state_sq.mean())),
        }
        if iteration + 1 < num_levels:
            tf_next = np.concatenate(accumulator["teacher_forced_next_squared_error"][iteration])
            cl_next = np.concatenate(accumulator["closed_loop_next_squared_error"][iteration])
            row["teacher_forced_next_rmse"] = float(np.sqrt(tf_next.mean()))
            row["closed_loop_next_rmse"] = float(np.sqrt(cl_next.mean()))
        per_timestep.append(row)
    return {"per_timestep": per_timestep}


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    cache = _load_cache(args.cache_path)
    metadata = cache["metadata"]
    num_levels = int(metadata["labels_per_trajectory"])
    if num_levels != int(metadata["num_steps"]) + 1:
        raise ValueError("Expected num_steps + 1 cached scheduler levels.")

    config = training_config.get_config(args.config)
    if args.attention != "config":
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                bidirectional_attention=args.attention == "bidirectional",
                prediction_type="epsilon",
            ),
        )
    if config.model.prediction_type != "epsilon":
        raise ValueError("This diagnostic requires an epsilon-interpreted ref checkpoint.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ProxyScorePytorch(config.model).to(device)
    safetensors.torch.load_model(model, args.checkpoint_dir / "model.safetensors", device=str(device))
    model.eval()

    observation_cache_path = args.observation_cache_path or pathlib.Path(f"{args.cache_path}.observations")
    observations = _load_observation_arrays(observation_cache_path)
    split_observations = None
    if args.split != "all":
        if args.split_manifest is None:
            raise ValueError("--split-manifest is required for --split train/val.")
        split_payload = json.loads(args.split_manifest.read_text())
        split_observations = np.asarray(
            split_payload[f"{args.split}_indices"], dtype=np.int64
        )
    trajectory_indices, selected_observation_indices, num_selected_observations = (
        _select_observations(
            cache, args.max_observations, args.seed, selected_observations=split_observations
        )
    )

    # Sidecars store observations in first-appearance order, independently of K.
    if len(observations["states"]) <= int(selected_observation_indices.max()):
        raise ValueError("Observation sidecar is smaller than the selected unique observations.")

    accumulator = _new_accumulator(num_levels)
    num_train_timesteps = int(metadata["ddim_num_train_timesteps"])
    for start in range(0, len(trajectory_indices), args.batch_size):
        trajectory_batch = trajectory_indices[start : start + args.batch_size]
        observation_indices = selected_observation_indices[start : start + len(trajectory_batch)]
        observation = _observation_batch(observations, observation_indices, device)
        x_teacher = torch.from_numpy(cache["x_t"][trajectory_batch]).to(device)
        epsilon_teacher = torch.from_numpy(cache["epsilon"][trajectory_batch]).to(device)
        times = torch.from_numpy(cache["time"][trajectory_batch]).to(device)

        images, image_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, image_masks)
        flat_output = model._predict_model_output_from_prefix(
            state.repeat_interleave(num_levels, dim=0),
            prefix_embs.repeat_interleave(num_levels, dim=0),
            prefix_pad_masks.repeat_interleave(num_levels, dim=0),
            x_teacher.flatten(0, 1),
            times.flatten(),
            prefix_att_masks=prefix_att_masks.repeat_interleave(num_levels, dim=0),
        )
        epsilon_pred = flat_output.reshape_as(epsilon_teacher)

        for iteration in range(num_levels):
            diff = epsilon_pred[:, iteration] - epsilon_teacher[:, iteration]
            accumulator["epsilon_squared_error"][iteration].append(
                diff.square().flatten(1).mean(1).cpu().numpy()
            )
            accumulator["epsilon_cosine"][iteration].append(
                _cosine_rows(epsilon_pred[:, iteration], epsilon_teacher[:, iteration]).cpu().numpy()
            )
            accumulator["epsilon_target_rms"][iteration].append(
                epsilon_teacher[:, iteration].square().flatten(1).mean(1).sqrt().cpu().numpy()
            )
            accumulator["epsilon_pred_rms"][iteration].append(
                epsilon_pred[:, iteration].square().flatten(1).mean(1).sqrt().cpu().numpy()
            )
            if iteration + 1 < num_levels:
                next_pred = _mbd_score_step(
                    x_teacher[:, iteration],
                    epsilon_pred[:, iteration],
                    iteration=iteration,
                    num_levels=num_levels,
                    num_train_timesteps=num_train_timesteps,
                )
                next_diff = next_pred - x_teacher[:, iteration + 1]
                accumulator["teacher_forced_next_squared_error"][iteration].append(
                    next_diff.square().flatten(1).mean(1).cpu().numpy()
                )

        x_closed = x_teacher[:, 0].clone()
        for iteration in range(num_levels):
            state_diff = x_closed - x_teacher[:, iteration]
            accumulator["closed_loop_state_squared_error"][iteration].append(
                state_diff.square().flatten(1).mean(1).cpu().numpy()
            )
            output = model._predict_model_output_from_prefix(
                state,
                prefix_embs,
                prefix_pad_masks,
                x_closed,
                times[:, iteration],
                prefix_att_masks=prefix_att_masks,
            )
            if iteration + 1 < num_levels:
                x_closed = _mbd_score_step(
                    x_closed,
                    output,
                    iteration=iteration,
                    num_levels=num_levels,
                    num_train_timesteps=num_train_timesteps,
                )
                next_diff = x_closed - x_teacher[:, iteration + 1]
                accumulator["closed_loop_next_squared_error"][iteration].append(
                    next_diff.square().flatten(1).mean(1).cpu().numpy()
                )

    report = {
        "cache_path": str(args.cache_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "config": args.config,
        "attention": args.attention,
        "prediction_type": "epsilon",
        "num_observations": int(num_selected_observations),
        "num_trajectories": int(len(trajectory_indices)),
        "num_scheduler_levels": num_levels,
        "num_updates": num_levels - 1,
        **_summarize(accumulator, metadata),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
