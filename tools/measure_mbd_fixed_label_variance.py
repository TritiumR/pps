#!/usr/bin/env python3
"""Repeat MBD teacher queries at fixed (observation, x_t, timestep) points."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENPI = ROOT / "openpi"
for path in (ROOT, OPENPI / "src", OPENPI / "scripts"):
    sys.path.insert(0, str(path))

import train_mpc_proxy_score_pytorch as teacher  # noqa: E402
from openpi.training import checkpoints as checkpoints  # noqa: E402
from openpi.training import config as training_config  # noqa: E402
from sim_free_mpc import SimFreeMPC, SimFreeMPCConfig  # noqa: E402
from sim_free_mpc.ddim import ddim_iteration_alphas  # noqa: E402
from vlm_dp.offline_context import (  # noqa: E402
    _weight_stage,
    attach_priority_cost,
    weight_episode_signals,
    weight_frame_context,
)


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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=pathlib.Path, required=True)
    parser.add_argument("--config", default="score_ref_weight_demo_meanstd")
    parser.add_argument("--base-config", default="score_ref_weight_demo_meanstd")
    parser.add_argument("--base-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--action-stats", type=pathlib.Path, required=True)
    parser.add_argument("--cost-config", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--observations-per-stage", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--probe-levels", default="0,5,10")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--obs-shard", type=int, default=0)
    parser.add_argument("--obs-num-shards", type=int, default=1)
    return parser.parse_args()


def _select(source: h5py.File, count: int, seed: int) -> list[tuple[str, int, int]]:
    by_stage = [[] for _ in STAGES]
    for name in sorted(source["data"]):
        demo = source["data"][name]
        signals = weight_episode_signals(demo)
        for step in range(int(demo.attrs["num_samples"])):
            stage = min(int(_weight_stage(signals, step)), len(STAGES) - 1)
            by_stage[stage].append((name, step, stage))
    rng = np.random.default_rng(seed)
    selected = []
    for stage, candidates in enumerate(by_stage):
        if len(candidates) < count:
            raise ValueError(f"Stage {stage} has only {len(candidates)} observations.")
        chosen = rng.choice(len(candidates), size=count, replace=False)
        selected.extend(candidates[int(index)] for index in chosen)
    return sorted(selected, key=lambda row: (int(row[0].rsplit("_", 1)[-1]), row[1]))


def _build_planner(args: argparse.Namespace):
    config = training_config.get_config(args.config)
    base_config = training_config.get_config(args.base_config)
    base_data = base_config.data.create(base_config.assets_dirs, base_config.model)
    norm_stats = checkpoints.load_norm_stats(args.base_checkpoint / "assets", base_data.asset_id)
    standalone = teacher.load_action_norm_stats(args.action_stats)
    embedded = norm_stats["actions"]
    for key in ("mean", "std"):
        if not np.allclose(
            np.asarray(getattr(embedded, key)), np.asarray(standalone[key]), rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"Action-stat mismatch for {key}.")
    score_data, input_transform = teacher._build_data_pipeline(config)
    if teacher._norm_stats_fingerprint(norm_stats) != teacher._norm_stats_fingerprint(
        score_data.norm_stats
    ):
        raise ValueError("Base and score normalization fingerprints differ.")
    decoder = teacher._ActionDecoder(norm_stats, use_quantile_norm=base_data.use_quantile_norm)
    planner = SimFreeMPC(
        decoder,
        SimFreeMPCConfig(
            task_name="weight",
            num_samples=4096,
            iterations=1,
            noise=0.4,
            temperature=0.1,
            beta_opt_iter=1.0,
            beta_horizon=1.0,
            action_dims=config.model.action_dim,
            cost_executable_actions=True,
            joint_delta_clip=0.05,
            logit_norm="raw",
            ancestral_eta=0.0,
            cost_style="priority",
            optimize_space="action",
            ddim_num_train_timesteps=config.model.ddim_num_train_timesteps,
            interpolate=True,
            control_frequency=40.0,
            interpolate_frequency=5.0,
        ),
    )
    with args.cost_config.open() as stream:
        attach_priority_cost(planner, yaml.safe_load(stream))
    return config, input_transform, planner


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def _summarize(labels: np.ndarray) -> dict[str, float]:
    mean = labels.mean(axis=0)
    residual = labels - mean
    mean_rms = float(np.sqrt(np.mean(np.square(mean, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    pair_cosines = []
    pair_rmse = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            pair_cosines.append(_cosine(labels[left], labels[right]))
            pair_rmse.append(float(np.sqrt(np.mean(np.square(labels[left] - labels[right])))))
    norms = np.sqrt(np.mean(np.square(labels, dtype=np.float64), axis=(1, 2)))
    return {
        "mean_label_rms": mean_rms,
        "repeat_noise_rms": noise_rms,
        "snr": mean_rms / max(noise_rms, 1e-12),
        "pairwise_cosine_mean": float(np.mean(pair_cosines)),
        "pairwise_cosine_min": float(np.min(pair_cosines)),
        "pairwise_rmse_mean": float(np.mean(pair_rmse)),
        "label_rms_cv": float(norms.std() / max(norms.mean(), 1e-12)),
    }


def main() -> None:
    args = _arguments()
    device = torch.device(args.device)
    probe_levels = {int(value) for value in args.probe_levels.split(",")}
    num_levels = args.num_steps + 1
    if not probe_levels.issubset(range(num_levels)):
        raise ValueError("Probe levels must lie in the cached 0..num_steps schedule.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config, input_transform, planner = _build_planner(args)

    records = []
    with h5py.File(args.hdf5, "r") as source:
        selected = _select(source, args.observations_per_stage, args.seed)
        selected = selected[args.obs_shard :: args.obs_num_shards]
        signal_cache = {}
        with torch.inference_mode():
            for observation_index, (name, step, stage) in enumerate(selected):
                demo = source["data"][name]
                sample = teacher._demo_sample(
                    demo,
                    step,
                    action_horizon=config.model.action_horizon,
                    prompt=teacher.DEFAULT_PROMPT,
                )
                model_inputs = teacher._torch_inputs(input_transform, sample, device)
                context = teacher._mpc_context(demo, step, task="weight", subtask_mode="heuristic")
                if name not in signal_cache:
                    signal_cache[name] = weight_episode_signals(demo)
                context = weight_frame_context(context, signal_cache[name], step)
                x_t = torch.randn(
                    (1, config.model.action_horizon, config.model.action_dim),
                    device=device,
                    dtype=torch.float32,
                )
                planner.begin_inference()
                for level in range(num_levels):
                    repeats = args.repeats if level in probe_levels else 1
                    labels = []
                    costs = []
                    first_score = None
                    for repeat in range(repeats):
                        score, diagnostics = planner.estimate_mbd_score_action_prox(
                            x_t,
                            model_inputs,
                            context,
                            iteration=level,
                            num_iterations=num_levels,
                        )
                        if first_score is None:
                            first_score = score
                        alpha, _ = ddim_iteration_alphas(
                            iteration=level,
                            num_iterations=num_levels,
                            num_train_timesteps=config.model.ddim_num_train_timesteps,
                        )
                        labels.append(
                            (-math.sqrt(max(1.0 - alpha, 1e-6)) * score)[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        costs.append(float(diagnostics["cost_min"]))
                    if level in probe_levels:
                        record = {
                            "demo": name,
                            "step": int(step),
                            "stage": int(stage),
                            "stage_name": STAGES[stage],
                            "iteration": level,
                            "repeats": repeats,
                            **_summarize(np.stack(labels)),
                            "cost_min_mean": float(np.mean(costs)),
                            "cost_min_std": float(np.std(costs)),
                        }
                        records.append(record)
                    if level + 1 < num_levels:
                        x_t = planner.step_from_score(
                            x_t,
                            first_score,
                            iteration=level,
                            num_iterations=num_levels,
                            update_mode="mbd_score",
                            active_dims=config.model.action_dim,
                        )
                if (observation_index + 1) % 8 == 0:
                    print(
                        f"fixed_label_variance {observation_index + 1}/{len(selected)} "
                        f"shard={args.obs_shard}",
                        flush=True,
                    )

    aggregate = {}
    for level in sorted(probe_levels):
        rows = [row for row in records if row["iteration"] == level]
        aggregate[str(level)] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "mean_label_rms",
                "repeat_noise_rms",
                "snr",
                "pairwise_cosine_mean",
                "pairwise_cosine_min",
                "pairwise_rmse_mean",
                "label_rms_cv",
                "cost_min_mean",
                "cost_min_std",
            )
        }
    report = {
        "settings": {
            "num_observations": len(selected),
            "observations_per_stage_before_sharding": args.observations_per_stage,
            "repeats": args.repeats,
            "probe_levels": sorted(probe_levels),
            "mpc_num_samples": 4096,
            "obs_shard": args.obs_shard,
            "obs_num_shards": args.obs_num_shards,
            "seed": args.seed,
        },
        "aggregate_by_iteration": aggregate,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    levels = sorted(probe_levels)
    axes[0].plot(levels, [aggregate[str(level)]["pairwise_cosine_mean"] for level in levels], marker="o")
    axes[0].set_ylabel("mean pairwise label cosine")
    axes[1].plot(levels, [aggregate[str(level)]["snr"] for level in levels], marker="o")
    axes[1].set_ylabel("label SNR")
    for axis in axes:
        axis.set_xlabel("reverse iteration")
        axis.grid(alpha=0.3)
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(json.dumps(report["settings"] | {"aggregate_by_iteration": aggregate}, indent=2))


if __name__ == "__main__":
    main()
