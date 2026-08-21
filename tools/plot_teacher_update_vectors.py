#!/usr/bin/env python3
"""Plot MBD/DDIM and Pi0.5 flow updates in a shared physical action space."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


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


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def path_metrics(updates: np.ndarray) -> dict[str, object]:
    flat = updates.reshape(*updates.shape[:3], -1)
    norms = np.linalg.norm(flat, axis=-1)
    path_length = norms.sum(axis=-1)
    net = np.linalg.norm(flat.sum(axis=2), axis=-1)
    denominator = np.linalg.norm(flat[:, :, :-1], axis=-1) * np.linalg.norm(
        flat[:, :, 1:], axis=-1
    )
    turning = np.sum(flat[:, :, :-1] * flat[:, :, 1:], axis=-1) / np.maximum(
        denominator, 1e-12
    )
    return {
        "update_l2_by_step": [summary(norms[:, :, step]) for step in range(norms.shape[2])],
        "adjacent_update_cosine_by_step": [
            summary(turning[:, :, step]) for step in range(turning.shape[2])
        ],
        "path_length": summary(path_length),
        "net_displacement": summary(net),
        "path_efficiency": summary(net / np.maximum(path_length, 1e-12)),
    }


def positions(projected_updates: np.ndarray) -> np.ndarray:
    origin = np.zeros((*projected_updates.shape[:-2], 1, 2), dtype=np.float64)
    return np.concatenate([origin, np.cumsum(projected_updates, axis=-2)], axis=-2)


def draw_path(ax, path: np.ndarray, title: str, color_map, limits) -> None:
    delta = np.diff(path, axis=0)
    colors = color_map(np.linspace(0.05, 0.95, len(delta)))
    ax.quiver(
        path[:-1, 0],
        path[:-1, 1],
        delta[:, 0],
        delta[:, 1],
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        headwidth=4.0,
        headlength=5.0,
    )
    ax.scatter(path[0, 0], path[0, 1], marker="o", s=24, color="black", zorder=3)
    ax.scatter(path[-1, 0], path[-1, 1], marker="X", s=38, color="black", zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def plot_grid(paths_by_model, selections, output: pathlib.Path, descriptor: str, energy: float):
    fig, axes = plt.subplots(2, 8, figsize=(28, 7), sharex=True, sharey=True)
    all_positions = np.concatenate(
        [paths_by_model[model].reshape(-1, 2) for model in ("mbd", "pi05")], axis=0
    )
    low = all_positions.min(axis=0)
    high = all_positions.max(axis=0)
    padding = np.maximum((high - low) * 0.08, 1e-4)
    limits = ((low[0] - padding[0], high[0] + padding[0]), (low[1] - padding[1], high[1] + padding[1]))
    cmap = plt.get_cmap("viridis")
    for stage, stage_name in enumerate(STAGES):
        draw_path(axes[0, stage], paths_by_model["mbd"][stage], stage_name, cmap, limits)
        draw_path(axes[1, stage], paths_by_model["pi05"][stage], stage_name, cmap, limits)
        if selections is not None:
            axes[0, stage].text(
                0.02,
                0.02,
                selections[stage],
                transform=axes[0, stage].transAxes,
                fontsize=6,
                va="bottom",
            )
    axes[0, 0].set_ylabel("MBD score/DDIM\nprojection 2")
    axes[1, 0].set_ylabel("Pi0.5 velocity\nprojection 2")
    for ax in axes[1]:
        ax.set_xlabel("projection 1")
    fig.suptitle(
        f"Head-to-tail physical arm-action updates ({descriptor}); shared zero-origin SVD, "
        f"2D energy={energy:.1%}\nstart=o, end=X, color: step 0→9",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi05-samples", type=pathlib.Path, required=True)
    parser.add_argument("--mbd-cache", type=pathlib.Path, required=True)
    parser.add_argument("--mbd-stats", type=pathlib.Path, required=True)
    parser.add_argument("--pi05-stats", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.pi05_samples, allow_pickle=False) as pi_data:
        demos = pi_data["demo"].astype(str)
        steps = pi_data["step"].astype(np.int64)
        stages = pi_data["stage"].astype(np.int64)
        pi_states = pi_data["states"].astype(np.float64)

    with np.load(args.mbd_cache, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        observation_count = int(metadata["num_observations"])
        k = int(metadata["trajectories_per_observation"])
        levels = int(metadata["labels_per_trajectory"])
        horizon, action_dim = cache["x_t"].shape[1:]
        cache_shape = (observation_count, k, levels, horizon, action_dim)
        cache_demos = cache["demo_name"].reshape(observation_count, k, levels)[:, 0, 0].astype(str)
        cache_steps = cache["step_index"].reshape(observation_count, k, levels)[:, 0, 0]
        lookup = {(demo, int(step)): index for index, (demo, step) in enumerate(zip(cache_demos, cache_steps, strict=True))}
        indices = np.asarray([lookup[(demo, int(step))] for demo, step in zip(demos, steps, strict=True)])
        mbd_states = cache["x_t"].reshape(cache_shape)[indices].astype(np.float64)

    mbd_stats = json.loads(args.mbd_stats.read_text())
    mbd_mean = np.asarray(mbd_stats["mean"], dtype=np.float64)[:8]
    mbd_std = np.asarray(mbd_stats["std"], dtype=np.float64)[:8]
    mbd_physical = mbd_states[..., :8] * mbd_std + mbd_mean

    pi_stats = json.loads(args.pi05_stats.read_text())["norm_stats"]["actions"]
    q01 = np.asarray(pi_stats["q01"], dtype=np.float64)[:8]
    q99 = np.asarray(pi_stats["q99"], dtype=np.float64)[:8]
    pi_physical = (pi_states[..., :8] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    # First seven dimensions are physical joint deltas in radians for both policies.
    mbd_updates = np.diff(mbd_physical[..., :7], axis=2)
    pi_updates = np.diff(pi_physical[..., :7], axis=2)
    mbd_flat = mbd_updates.reshape(-1, 15 * 7)
    pi_flat = pi_updates.reshape(-1, 15 * 7)
    combined = np.concatenate([mbd_flat, pi_flat], axis=0)
    _, singular, right = np.linalg.svd(combined, full_matrices=False)
    components = right[:2]
    energy = float(np.sum(singular[:2] ** 2) / np.sum(singular**2))

    mbd_projected = (mbd_updates.reshape(*mbd_updates.shape[:3], -1) @ components.T)
    pi_projected = (pi_updates.reshape(*pi_updates.shape[:3], -1) @ components.T)

    stage_mean_paths = {"mbd": [], "pi05": []}
    for stage in range(len(STAGES)):
        mask = stages == stage
        stage_mean_paths["mbd"].append(positions(mbd_projected[mask].mean(axis=(0, 1))))
        stage_mean_paths["pi05"].append(positions(pi_projected[mask].mean(axis=(0, 1))))
    stage_mean_paths = {key: np.stack(value) for key, value in stage_mean_paths.items()}

    # Select one shared observation per stage whose k=0 efficiency is closest to the
    # median of the two-policy mean efficiency; this avoids cherry-picking either model.
    def efficiencies(updates):
        flat = updates.reshape(*updates.shape[:3], -1)
        return np.linalg.norm(flat.sum(axis=2), axis=-1) / np.maximum(
            np.linalg.norm(flat, axis=-1).sum(axis=2), 1e-12
        )

    mbd_eff = efficiencies(mbd_updates)
    pi_eff = efficiencies(pi_updates)
    representative_paths = {"mbd": [], "pi05": []}
    selection_labels = []
    selections = []
    for stage in range(len(STAGES)):
        candidates = np.flatnonzero(stages == stage)
        joint_eff = (mbd_eff[candidates, 0] + pi_eff[candidates, 0]) / 2.0
        chosen = candidates[np.argmin(np.abs(joint_eff - np.median(joint_eff)))]
        selections.append(int(chosen))
        selection_labels.append(f"{demos[chosen]}:{steps[chosen]}, k=0")
        representative_paths["mbd"].append(positions(mbd_projected[chosen, 0]))
        representative_paths["pi05"].append(positions(pi_projected[chosen, 0]))
    representative_paths = {key: np.stack(value) for key, value in representative_paths.items()}

    plot_grid(
        representative_paths,
        selection_labels,
        args.output_dir / "update_arrows_representative_physical.png",
        "shared representative observation per stage",
        energy,
    )
    plot_grid(
        stage_mean_paths,
        None,
        args.output_dir / "update_arrows_stage_mean_physical.png",
        "mean over 16 observations × K=8",
        energy,
    )

    report = {
        "space": "physical relative arm joint-action chunk; 15x7 radians; gripper excluded",
        "projection": "shared uncentered SVD of MBD and Pi0.5 update vectors",
        "projection_energy_fraction": energy,
        "representative_indices": selections,
        "representative_labels": selection_labels,
        "overall": {
            "mbd": path_metrics(mbd_updates),
            "pi05": path_metrics(pi_updates),
        },
        "by_stage": {
            STAGES[stage]: {
                "mbd": path_metrics(mbd_updates[stages == stage]),
                "pi05": path_metrics(pi_updates[stages == stage]),
            }
            for stage in range(len(STAGES))
        },
    }
    (args.output_dir / "update_vector_path_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
