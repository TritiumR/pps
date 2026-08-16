#!/usr/bin/env python3
"""Offline epsilon-field and sampled-action diagnostics for action-chunk ref BC."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import safetensors.torch
import torch

from diagnose_ref_cache import _cosine_rows, _load_observation_arrays, _observation_batch

from openpi.models_pytorch.proxy_score_pytorch import ProxyScorePytorch
from openpi.training import config as training_config


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=pathlib.Path, required=True)
    parser.add_argument("--observation-cache-path", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--split-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--config", default="score_ref_weight_demo_meanstd")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _mean(values: list[np.ndarray]) -> float:
    return float(np.concatenate(values).mean())


def _pairwise_rmse(chunks: torch.Tensor) -> torch.Tensor:
    count = chunks.shape[1]
    if count < 2:
        return torch.zeros(chunks.shape[0], device=chunks.device)
    rows, cols = torch.triu_indices(count, count, offset=1, device=chunks.device)
    diff = chunks[:, rows] - chunks[:, cols]
    return diff.square().flatten(2).mean(2).sqrt().mean(1)


@torch.inference_mode()
def main() -> None:
    args = _args()
    cache = np.load(args.cache_path, allow_pickle=False)
    metadata = json.loads(str(cache["metadata_json"].item()))
    num_observations = int(metadata["num_observations"])
    trajectories_per_observation = int(metadata["trajectories_per_observation"])
    num_levels = int(metadata["labels_per_trajectory"])
    action_horizon, action_dim = cache["x_t"].shape[1:]
    states = cache["x_t"].astype(np.float32).reshape(
        num_observations,
        trajectories_per_observation,
        num_levels,
        action_horizon,
        action_dim,
    )
    times = cache["time"].astype(np.float32).reshape(
        num_observations, trajectories_per_observation, num_levels
    )

    split_payload = json.loads(args.split_manifest.read_text())
    if args.split == "all":
        observation_indices = np.arange(num_observations, dtype=np.int64)
    else:
        observation_indices = np.asarray(
            split_payload[f"{args.split}_indices"], dtype=np.int64
        )

    config = training_config.get_config(args.config)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            prediction_type="epsilon",
            bidirectional_attention=True,
        ),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ProxyScorePytorch(config.model).to(device)
    safetensors.torch.load_model(
        model, args.checkpoint_dir / "model.safetensors", device=str(device)
    )
    model.eval()
    observations = _load_observation_arrays(args.observation_cache_path)

    timestep_stats = [
        {"squared_error": [], "cosine": [], "target_rms": [], "pred_rms": []}
        for _ in range(num_levels)
    ]
    sample_stats = {
        "nearest_teacher_rmse": [],
        "aligned_teacher_rmse": [],
        "predicted_rms": [],
        "teacher_rms": [],
        "predicted_pairwise_rmse": [],
        "teacher_pairwise_rmse": [],
    }

    for start in range(0, len(observation_indices), args.batch_size):
        obs_indices = observation_indices[start : start + args.batch_size]
        batch = len(obs_indices)
        clean = torch.from_numpy(states[obs_indices, :, -1]).to(device)
        noise = torch.from_numpy(states[obs_indices, :, 0]).to(device)
        observation = _observation_batch(observations, obs_indices, device)
        images, image_masks, state = model._preprocess_observation(observation, train=False)
        prefix, prefix_masks, prefix_att_masks = model.embed_prefix(images, image_masks)
        flat_clean = clean.flatten(0, 1)
        flat_noise = noise.flatten(0, 1)
        repeated_state = state.repeat_interleave(trajectories_per_observation, dim=0)
        repeated_prefix = prefix.repeat_interleave(trajectories_per_observation, dim=0)
        repeated_masks = prefix_masks.repeat_interleave(trajectories_per_observation, dim=0)
        repeated_att_masks = prefix_att_masks.repeat_interleave(
            trajectories_per_observation, dim=0
        )

        for level in range(num_levels):
            time_value = float(times[obs_indices[0], 0, level])
            time = torch.full(
                (batch * trajectories_per_observation,),
                time_value,
                device=device,
                dtype=flat_clean.dtype,
            )
            alpha = model._alpha_from_time(time, device, flat_clean.dtype)
            sqrt_alpha = alpha.sqrt()[:, None, None]
            sqrt_beta = (1.0 - alpha).clamp(min=1e-6).sqrt()[:, None, None]
            x_t = sqrt_alpha * flat_clean + sqrt_beta * flat_noise
            prediction = model._predict_model_output_from_prefix(
                repeated_state,
                repeated_prefix,
                repeated_masks,
                x_t,
                time,
                prefix_att_masks=repeated_att_masks,
            )
            diff = prediction - flat_noise
            timestep_stats[level]["squared_error"].append(
                diff.square().flatten(1).mean(1).cpu().numpy()
            )
            timestep_stats[level]["cosine"].append(
                _cosine_rows(prediction, flat_noise).cpu().numpy()
            )
            timestep_stats[level]["target_rms"].append(
                flat_noise.square().flatten(1).mean(1).sqrt().cpu().numpy()
            )
            timestep_stats[level]["pred_rms"].append(
                prediction.square().flatten(1).mean(1).sqrt().cpu().numpy()
            )

        repeated_obs_indices = np.repeat(obs_indices, trajectories_per_observation)
        sample_observation = _observation_batch(observations, repeated_obs_indices, device)
        sampled = model.sample_actions(
            device,
            sample_observation,
            noise=flat_noise,
            num_steps=args.num_steps,
        ).reshape(batch, trajectories_per_observation, action_horizon, action_dim)
        all_distances = (
            sampled[:, :, None] - clean[:, None, :]
        ).square().flatten(3).mean(3).sqrt()
        aligned = (sampled - clean).square().flatten(2).mean(2).sqrt()
        sample_stats["nearest_teacher_rmse"].append(
            all_distances.min(dim=2).values.flatten().cpu().numpy()
        )
        sample_stats["aligned_teacher_rmse"].append(aligned.flatten().cpu().numpy())
        sample_stats["predicted_rms"].append(
            sampled.square().flatten(2).mean(2).sqrt().flatten().cpu().numpy()
        )
        sample_stats["teacher_rms"].append(
            clean.square().flatten(2).mean(2).sqrt().flatten().cpu().numpy()
        )
        sample_stats["predicted_pairwise_rmse"].append(
            _pairwise_rmse(sampled).cpu().numpy()
        )
        sample_stats["teacher_pairwise_rmse"].append(
            _pairwise_rmse(clean).cpu().numpy()
        )

    per_timestep = []
    for level, stats in enumerate(timestep_stats):
        mse = _mean(stats["squared_error"])
        target_rms = _mean(stats["target_rms"])
        pred_rms = _mean(stats["pred_rms"])
        per_timestep.append(
            {
                "iteration": level,
                "time": float(times[0, 0, level]),
                "epsilon_mse": mse,
                "epsilon_rmse": float(np.sqrt(mse)),
                "epsilon_cosine": _mean(stats["cosine"]),
                "epsilon_target_rms": target_rms,
                "epsilon_pred_rms": pred_rms,
                "epsilon_pred_to_target_rms_ratio": pred_rms / max(target_rms, 1e-12),
            }
        )
    report = {
        "mode": "teacher_action_chunk_standard_epsilon",
        "split": args.split,
        "num_observations": int(len(observation_indices)),
        "num_teacher_chunks": int(
            len(observation_indices) * trajectories_per_observation
        ),
        "per_timestep": per_timestep,
        "sample_metrics": {key: _mean(value) for key, value in sample_stats.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
