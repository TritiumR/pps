#!/usr/bin/env python3
"""Plot paired task-steering score and robot-trajectory diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_seed(records: list[dict], seed: int | None) -> list[dict]:
    if seed is None:
        return records
    return [record for record in records if record.get("seed") in (None, seed)]


def score_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        if record.get("event") != "inference":
            continue
        for denoise_step, item in enumerate(record.get("mpc_trace", [])):
            if "score_state_values" not in item:
                continue
            rows.append(
                {
                    "env_step": int(record["step"]),
                    "phase": record.get("phase", "unknown"),
                    "denoise_step": denoise_step,
                    **item,
                }
            )
    if not rows:
        raise ValueError(f"No score tensors found in debug trace with {len(records)} records.")
    return rows


def step_rows(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("event") == "step"]


def ddim_alphas_cumprod(num_train_timesteps: int) -> np.ndarray:
    alpha_cumprod = 1.0
    alphas = []
    for idx in range(num_train_timesteps):
        t1 = idx / float(num_train_timesteps)
        t2 = (idx + 1) / float(num_train_timesteps)
        alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2.0) ** 2
        alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2.0) ** 2
        beta = min(1.0 - alpha_bar_t2 / alpha_bar_t1, 0.999)
        alpha_cumprod *= 1.0 - beta
        alphas.append(alpha_cumprod)
    return np.asarray(alphas, dtype=np.float64)


def load_action_stats(path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload["norm_stats"]["actions"]
    return (
        np.asarray(stats["q01"][:action_dim], dtype=np.float32),
        np.asarray(stats["q99"][:action_dim], dtype=np.float32),
    )


def build_expert_action_bank(
    hdf5_path: Path,
    norm_stats_path: Path,
    *,
    horizon: int,
    action_dim: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    q01, q99 = load_action_stats(norm_stats_path, action_dim)
    chunks = []
    with h5py.File(hdf5_path, "r") as dataset:
        for demo_name in sorted(dataset["data"]):
            demo = dataset["data"][demo_name]
            actions = np.asarray(demo["obs/joint_actions"], dtype=np.float32)[..., :action_dim]
            states = np.asarray(demo["obs/joint_pos"], dtype=np.float32)
            length = actions.shape[0]
            offsets = np.arange(1, horizon + 1)
            for step in range(length):
                indices = np.minimum(step + offsets, length - 1)
                chunk = actions[indices].copy()
                joint_dims = min(7, action_dim)
                chunk[:, :joint_dims] -= states[step, :joint_dims]
                chunks.append(chunk)

    bank = np.stack(chunks)
    bank = (bank - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    if bank.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        bank = bank[rng.choice(bank.shape[0], size=max_samples, replace=False)]
    return bank.astype(np.float32, copy=False)


def expert_mixture_score(
    x_t: np.ndarray,
    action_bank: np.ndarray,
    alpha: float,
) -> np.ndarray:
    beta = max(1.0 - alpha, 1e-6)
    diff = math.sqrt(alpha) * action_bank - x_t[None]
    logits = -np.sum(diff.astype(np.float64) ** 2, axis=(1, 2)) / (2.0 * beta)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return np.sum(weights[:, None, None] * diff, axis=0) / beta


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs.reshape(-1).astype(np.float64)
    rhs = rhs.reshape(-1).astype(np.float64)
    return float(np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs) + 1e-12))


def add_expert_metrics(
    rows: list[dict],
    action_bank: np.ndarray,
    *,
    num_train_timesteps: int,
) -> None:
    alphas = ddim_alphas_cumprod(num_train_timesteps)
    for row in rows:
        x_t = np.asarray(row["score_state_values"], dtype=np.float32)[0]
        base = np.asarray(row["score_base_values"], dtype=np.float32)[0]
        task = np.asarray(row["score_task_values"], dtype=np.float32)[0]
        combined = np.asarray(row["score_combined_values"], dtype=np.float32)[0]
        time_cond = float(row["proxy_score_time"])
        timestep = int(round(np.clip(time_cond, 0.0, 1.0) * (num_train_timesteps - 1)))
        expert = expert_mixture_score(x_t, action_bank, float(alphas[timestep]))
        row["expert_score_norm"] = float(np.linalg.norm(expert))
        for name, score in (("base", base), ("task", task), ("combined", combined)):
            row[f"{name}_expert_rmse"] = float(np.sqrt(np.mean((score - expert) ** 2)))
            row[f"{name}_expert_cosine"] = cosine(score, expert)


def grouped_mean_std(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray(sorted({int(row["denoise_step"]) for row in rows}))
    means = []
    stds = []
    for step in steps:
        values = np.asarray([row[key] for row in rows if int(row["denoise_step"]) == step])
        means.append(values.mean())
        stds.append(values.std())
    return steps, np.asarray(means), np.asarray(stds)


def plot_score_diagnostics(rows: list[dict], output_path: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    line_styles = {
        "base": {
            "color": "tab:blue",
            "linestyle": "--",
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgewidth": 1.5,
            "linewidth": 2.0,
            "zorder": 5,
        },
        "task": {"color": "tab:orange", "marker": "o", "zorder": 3},
        "combined": {"color": "tab:green", "marker": "s", "zorder": 4},
        "expert": {"color": "tab:red", "marker": "o", "zorder": 2},
    }
    norm_keys = (
        ("score_base_proxy_norm", "Base", "base"),
        ("score_task_norm", "Task", "task"),
        ("score_combined_norm", "Combined", "combined"),
        ("expert_score_norm", "Expert-demo KDE", "expert"),
    )
    for key, label, style_name in norm_keys:
        steps, mean, std = grouped_mean_std(rows, key)
        style = line_styles[style_name]
        axes[0].plot(steps, mean, label=label, **style)
        axes[0].fill_between(
            steps,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=0.10,
            zorder=1,
        )
    axes[0].set(title="Mean score norm", xlabel="Denoising step", ylabel="L2 norm")
    axes[0].legend()

    distance_keys = (
        ("base_expert_rmse", "Base → expert", "base"),
        ("task_expert_rmse", "Task → expert", "task"),
        ("combined_expert_rmse", "Combined → expert", "combined"),
    )
    for key, label, style_name in distance_keys:
        steps, mean, std = grouped_mean_std(rows, key)
        style = line_styles[style_name]
        axes[1].plot(steps, mean, label=label, **style)
        axes[1].fill_between(
            steps,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=0.10,
            zorder=1,
        )
    axes[1].set(title="Distance to expert-demo score", xlabel="Denoising step", ylabel="RMSE ↓")
    axes[1].legend()

    steps, mean, std = grouped_mean_std(rows, "score_applied_residual_ratio")
    axes[2].plot(steps, 100.0 * mean, marker="o", color="tab:red")
    axes[2].fill_between(steps, 100.0 * (mean - std), 100.0 * (mean + std), alpha=0.15)
    axes[2].set(
        title="Applied residual / base",
        xlabel="Denoising step",
        ylabel="Relative magnitude (%)",
    )
    axes[2].axhline(0.0, color="black", linewidth=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    base_rmse = np.asarray([row["base_expert_rmse"] for row in rows])
    combined_rmse = np.asarray([row["combined_expert_rmse"] for row in rows])
    return {
        "mean_base_expert_rmse": float(base_rmse.mean()),
        "mean_combined_expert_rmse": float(combined_rmse.mean()),
        "mean_expert_rmse_improvement": float((base_rmse - combined_rmse).mean()),
        "fraction_combined_closer_to_expert": float(np.mean(combined_rmse < base_rmse)),
        "mean_applied_residual_ratio": float(
            np.mean([row["score_applied_residual_ratio"] for row in rows])
        ),
    }


def trace_array(rows: list[dict], key: str, dims: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    valid = [row for row in rows if key in row]
    steps = np.asarray([row["step"] for row in valid], dtype=np.int64)
    values = np.asarray([row[key] for row in valid], dtype=np.float32)
    if dims is not None:
        values = values[:, :dims]
    return steps, values


def plot_joint_trajectories(
    base_steps: list[dict],
    steer_steps: list[dict],
    output_path: Path,
    *,
    control_frequency: float,
) -> None:
    base_t, base_q = trace_array(base_steps, "joint_pos", dims=7)
    steer_t, steer_q = trace_array(steer_steps, "joint_pos", dims=7)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for joint in range(7):
        axis = axes.flat[joint]
        axis.plot(base_t / control_frequency, base_q[:, joint], label="Base", linewidth=1.2)
        axis.plot(steer_t / control_frequency, steer_q[:, joint], label="Steer λ=0.002", linewidth=1.2)
        axis.set_ylabel(f"q{joint + 1} (rad)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel("Rollout time (s)")
    fig.suptitle("Paired joint trajectories")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eef_trajectories(
    base_steps: list[dict],
    steer_steps: list[dict],
    output_path: Path,
    *,
    control_frequency: float,
) -> None:
    base_t, base_xyz = trace_array(base_steps, "eef_pos", dims=3)
    steer_t, steer_xyz = trace_array(steer_steps, "eef_pos", dims=3)
    fig = plt.figure(figsize=(14, 5))
    axis_3d = fig.add_subplot(1, 2, 1, projection="3d")
    axis_3d.plot(*base_xyz.T, label="Base")
    axis_3d.plot(*steer_xyz.T, label="Steer λ=0.002")
    axis_3d.scatter(*base_xyz[0], marker="o", s=35)
    axis_3d.scatter(*steer_xyz[0], marker="x", s=35)
    axis_3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="EEF path")
    axis_3d.legend()

    axis_time = fig.add_subplot(1, 2, 2)
    for dim, label in enumerate(("x", "y", "z")):
        axis_time.plot(base_t / control_frequency, base_xyz[:, dim], label=f"Base {label}")
        axis_time.plot(
            steer_t / control_frequency,
            steer_xyz[:, dim],
            linestyle="--",
            label=f"Steer {label}",
        )
    axis_time.set(xlabel="Rollout time (s)", ylabel="EEF position (m)", title="EEF coordinates")
    axis_time.legend(ncol=2)
    axis_time.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-debug", type=Path, required=True)
    parser.add_argument("--steer-debug", type=Path, required=True)
    parser.add_argument("--expert-hdf5", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-samples", type=int, default=512)
    parser.add_argument("--expert-seed", type=int, default=0)
    parser.add_argument("--num-train-timesteps", type=int, default=100)
    parser.add_argument("--control-frequency", type=float, default=15.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_records = select_seed(load_jsonl(args.base_debug), args.seed)
    steer_records = select_seed(load_jsonl(args.steer_debug), args.seed)
    steer_scores = score_rows(steer_records)
    sample_state = np.asarray(steer_scores[0]["score_state_values"])[0]
    action_bank = build_expert_action_bank(
        args.expert_hdf5,
        args.norm_stats,
        horizon=sample_state.shape[0],
        action_dim=sample_state.shape[1],
        max_samples=args.expert_samples,
        seed=args.expert_seed,
    )
    add_expert_metrics(
        steer_scores,
        action_bank,
        num_train_timesteps=args.num_train_timesteps,
    )

    summary = plot_score_diagnostics(
        steer_scores,
        args.output_dir / "02_score_and_expert_diagnostics.png",
    )
    base_steps = step_rows(base_records)
    steer_steps = step_rows(steer_records)
    plot_joint_trajectories(
        base_steps,
        steer_steps,
        args.output_dir / "03_joint_trajectories.png",
        control_frequency=args.control_frequency,
    )
    plot_eef_trajectories(
        base_steps,
        steer_steps,
        args.output_dir / "04_eef_trajectories.png",
        control_frequency=args.control_frequency,
    )
    summary.update(
        {
            "expert_action_samples": int(action_bank.shape[0]),
            "score_points": len(steer_scores),
            "base_rollout_steps": len(base_steps),
            "steer_rollout_steps": len(steer_steps),
        }
    )
    (args.output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
