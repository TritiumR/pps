#!/usr/bin/env python3
"""Analyze timestep behavior and K-path consistency of a cached MBD teacher."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_common.fk import FrankaFK
from sim_free_mpc.ddim import ddim_iteration_alphas
from vlm_dp.offline_context import _weight_stage, weight_episode_signals


STAGES = (
    "approach pear",
    "lift pear",
    "carry pear",
    "place pear",
    "approach apple",
    "lift apple",
    "carry apple",
    "place apple",
)


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _row_rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=tuple(range(1, values.ndim))))


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.reshape(len(left), -1).astype(np.float64)
    right = right.reshape(len(right), -1).astype(np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)


def _stage_ids(hdf5_path: pathlib.Path, demos: np.ndarray, steps: np.ndarray) -> np.ndarray:
    result = np.empty(len(demos), dtype=np.int64)
    with h5py.File(hdf5_path, "r") as source:
        signals = {}
        for index, (name, step) in enumerate(zip(demos, steps, strict=True)):
            name = str(name)
            if name not in signals:
                signals[name] = weight_episode_signals(source["data"][name])
            result[index] = min(int(_weight_stage(signals[name], int(step))), len(STAGES) - 1)
    return result


def _tcp_endpoints(current_q: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    targets = current_q[:, None, :] + decoded[:, :, -1, :7]
    fk = FrankaFK(device="cpu")
    endpoints = []
    with torch.inference_mode():
        flat = targets.reshape(-1, 7)
        for start in range(0, len(flat), 8192):
            position, _ = fk.fk(
                torch.from_numpy(flat[start : start + 8192]).to(torch.float32)
            )
            endpoints.append(position.detach().cpu().numpy())
    return np.concatenate(endpoints).reshape(*targets.shape[:2], 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--hdf5", type=pathlib.Path, required=True)
    parser.add_argument("--action-stats", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.cache, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        observations = int(metadata["num_observations"])
        k = int(metadata["trajectories_per_observation"])
        levels = int(metadata["labels_per_trajectory"])
        horizon, action_dim = cache["x_t"].shape[1:]
        shape = (observations, k, levels, horizon, action_dim)
        states = cache["x_t"].reshape(shape)
        epsilon = cache["epsilon"].reshape(shape)
        demos = cache["demo_name"].reshape(observations, k, levels)[:, 0, 0].astype(str)
        steps = cache["step_index"].reshape(observations, k, levels)[:, 0, 0]

        timestep = []
        updates = []
        for level in range(levels - 1):
            alpha, _ = ddim_iteration_alphas(
                iteration=level,
                num_iterations=levels,
                num_train_timesteps=int(metadata["ddim_num_train_timesteps"]),
            )
            eps = epsilon[:, :, level].reshape(-1, horizon, action_dim)
            numerator = -math.sqrt(max(1.0 - alpha, 1e-6)) * eps
            delta = (states[:, :, level + 1] - states[:, :, level]).reshape(
                -1, horizon, action_dim
            )
            updates.append(delta)
            timestep.append(
                {
                    "iteration": level,
                    "alpha_bar": alpha,
                    "epsilon_rms": _summary(_row_rms(eps)),
                    "beta_score_numerator_rms": _summary(_row_rms(numerator)),
                    "actual_update_rms": _summary(_row_rms(delta)),
                }
            )
        for level in range(len(updates) - 1):
            timestep[level]["cosine_to_next_update"] = _summary(
                _row_cosine(updates[level], updates[level + 1])
            )

        final_normalized = states[:, :, -1]
        stats = json.loads(args.action_stats.read_text())
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        decoded = final_normalized * std + mean

        pair_cos = np.zeros(observations, dtype=np.float64)
        pair_rmse = np.zeros(observations, dtype=np.float64)
        pair_count = 0
        for left, right in itertools.combinations(range(k), 2):
            pair_cos += _row_cosine(decoded[:, left], decoded[:, right])
            pair_rmse += _row_rms(decoded[:, left] - decoded[:, right])
            pair_count += 1
        pair_cos /= pair_count
        pair_rmse /= pair_count

        current_q = np.empty((observations, 7), dtype=np.float32)
        with h5py.File(args.hdf5, "r") as source:
            for index, (name, step) in enumerate(zip(demos, steps, strict=True)):
                current_q[index] = source["data"][str(name)]["obs/joint_pos"][int(step), :7]
        endpoints = _tcp_endpoints(current_q, decoded)
        endpoint_center = endpoints.mean(axis=1, keepdims=True)
        endpoint_spread = np.linalg.norm(endpoints - endpoint_center, axis=-1).mean(axis=1)
        endpoint_pair_distance = np.zeros(observations, dtype=np.float64)
        for left, right in itertools.combinations(range(k), 2):
            endpoint_pair_distance += np.linalg.norm(endpoints[:, left] - endpoints[:, right], axis=-1)
        endpoint_pair_distance /= pair_count

        final_gripper = decoded[:, :, -1, 7] > 0.5
        gripper_majority = np.maximum(final_gripper.mean(axis=1), 1.0 - final_gripper.mean(axis=1))
        gripper_unanimous = gripper_majority == 1.0
        stages = _stage_ids(args.hdf5, demos, steps)

    k_metrics = {
        "decoded_chunk_pairwise_cosine": pair_cos,
        "decoded_chunk_pairwise_rmse": pair_rmse,
        "tcp_endpoint_spread_m": endpoint_spread,
        "tcp_endpoint_pairwise_distance_m": endpoint_pair_distance,
        "gripper_majority_fraction": gripper_majority,
        "gripper_unanimous": gripper_unanimous.astype(np.float64),
    }
    report = {
        "cache": str(args.cache),
        "num_observations": observations,
        "trajectories_per_observation": k,
        "timestep_consistency": timestep,
        "k8_consistency": {
            "overall": {name: _summary(values) for name, values in k_metrics.items()},
            "by_stage": {
                STAGES[stage]: {
                    "num_observations": int(np.sum(stages == stage)),
                    **{
                        name: _summary(values[stages == stage])
                        for name, values in k_metrics.items()
                    },
                }
                for stage in range(len(STAGES))
            },
        },
    }
    (args.output_dir / "teacher_cache_consistency.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    iterations = [row["iteration"] for row in timestep]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for key, label in (
        ("epsilon_rms", "epsilon RMS"),
        ("beta_score_numerator_rms", "beta*score RMS"),
        ("actual_update_rms", "actual delta-x RMS"),
    ):
        axes[0].plot(iterations, [row[key]["mean"] for row in timestep], marker="o", label=label)
    axes[0].set_xlabel("reverse iteration")
    axes[0].set_ylabel("mean RMS")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    adjacent = timestep[:-1]
    axes[1].plot(
        [row["iteration"] for row in adjacent],
        [row["cosine_to_next_update"]["mean"] for row in adjacent],
        marker="o",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("reverse iteration")
    axes[1].set_ylabel("cosine(update[t], update[t+1])")
    axes[1].grid(alpha=0.3)
    fig.savefig(args.output_dir / "timestep_direction_scale.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    stage_names = list(STAGES)
    axes[0].bar(stage_names, [report["k8_consistency"]["by_stage"][s]["decoded_chunk_pairwise_cosine"]["mean"] for s in stage_names])
    axes[0].set_ylabel("mean pairwise cosine")
    axes[1].bar(stage_names, [report["k8_consistency"]["by_stage"][s]["tcp_endpoint_pairwise_distance_m"]["mean"] for s in stage_names])
    axes[1].set_ylabel("TCP pairwise distance (m)")
    axes[2].bar(stage_names, [report["k8_consistency"]["by_stage"][s]["gripper_unanimous"]["mean"] for s in stage_names])
    axes[2].set_ylabel("gripper unanimous fraction")
    for axis in axes:
        axis.tick_params(axis="x", rotation=55)
        axis.grid(axis="y", alpha=0.3)
    fig.savefig(args.output_dir / "k8_consistency_by_stage.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
