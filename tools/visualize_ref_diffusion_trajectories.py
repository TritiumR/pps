#!/usr/bin/env python3
"""Plot actual K-path reverse-diffusion trajectories stored in a ref cache."""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm_dp.offline_context import _weight_stage, weight_episode_signals
from vlm_dp.sim_helpers import ROBOTIQ_GRASP_OFFSET
from sim_common.fk import FrankaFK, WorldFK


STAGES = (
    "approach pear",
    "lift pear",
    "carry pear",
    "place pear",
    "release pear / approach apple",
    "lift apple",
    "carry apple",
    "place apple",
    "release apple / done",
)


def demo_number(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def stage_midpoints(demo: h5py.Group) -> list[int]:
    length = int(demo.attrs["num_samples"])
    signals = weight_episode_signals(demo)
    stage_ids = np.asarray([_weight_stage(signals, step) for step in range(length)])
    result = []
    for stage in range(8):
        indices = np.flatnonzero(stage_ids == stage)
        if not len(indices):
            raise ValueError(f"Demo has no frames assigned to eval stage {stage}.")
        result.append(int(indices[len(indices) // 2]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--hdf5", type=pathlib.Path, required=True)
    parser.add_argument("--action-stats", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--projection", choices=("xz", "yz", "xyz"), default="xz")
    parser.add_argument(
        "--fixed-axis-radius",
        type=float,
        default=None,
        help="Use identical TCP-centered +/- radius (metres) in every trajectory panel.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizontal_dim = 0 if args.projection == "xz" else 1
    horizontal_label = "X" if args.projection == "xz" else "Y"

    stats = json.loads(args.action_stats.read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    base_fk = FrankaFK(device="cpu")

    with h5py.File(args.hdf5, "r") as source:
        names = sorted(source["data"], key=demo_number)
        chosen = [names[i] for i in np.rint(np.linspace(0, len(names) - 1, args.num_demos)).astype(int)]
        selected = {
            (name, step)
            for name in chosen
            for step in stage_midpoints(source["data"][name])
        }

        cache = np.load(args.cache, allow_pickle=True)
        cache_demo = cache["demo_name"]
        cache_step = cache["step_index"]
        cache_iteration = cache["iteration"]
        cache_trajectory = cache["trajectory_id"]
        active_stages = STAGES[:8]
        selected_rows = {
            pair: np.flatnonzero((cache_demo == pair[0]) & (cache_step == pair[1]))
            for pair in selected
        }
        missing = [pair for pair, rows in selected_rows.items() if len(rows) != 8 * 11]
        if missing:
            raise ValueError(f"Expected 88 cache rows per observation; mismatches: {missing}")
        x_t = cache["x_t"]

        manifest: dict[str, object] = {
            "cache": str(args.cache),
            "hdf5": str(args.hdf5),
            "selection": "stage midpoint observations from evenly spaced demos",
            "demos": chosen,
            "trajectories_per_observation": 8,
            "reverse_levels": 11,
            "displayed_stages": list(active_stages),
            "displayed_trajectories": "all K=8 teacher samples",
            "projection": args.projection,
            "fixed_axis_radius_m": args.fixed_axis_radius,
            "reversed_axes": ["x", "z"] if args.fixed_axis_radius is not None else [],
            "figures": [],
        }
        colors = plt.cm.tab10(np.arange(8))
        for name in chosen:
            demo = source["data"][name]
            steps = stage_midpoints(demo)[:8]
            if args.projection == "xyz":
                fig = plt.figure(figsize=(32, 8), constrained_layout=True)
                axes = np.empty((2, len(active_stages)), dtype=object)
                for column in range(len(active_stages)):
                    axes[0, column] = fig.add_subplot(2, len(active_stages), column + 1)
                    axes[1, column] = fig.add_subplot(
                        2, len(active_stages), len(active_stages) + column + 1,
                        projection="3d",
                    )
            else:
                fig, axes = plt.subplots(
                    2, len(active_stages), figsize=(32, 8), constrained_layout=True
                )
            for stage_idx, (stage, step) in enumerate(
                zip(active_stages, steps, strict=True)
            ):
                rows = selected_rows[(name, step)]
                traj_ids = np.unique(cache_trajectory[rows])
                if len(traj_ids) != 8:
                    raise ValueError(f"{name} frame {step}: expected K=8, got {len(traj_ids)}")

                image_ax = axes[0, stage_idx]
                xz_ax = axes[1, stage_idx]
                image_ax.imshow(np.asarray(demo["obs/table_cam"][step]))
                image_ax.set_title(f"{stage}\ncache observation: {name}, frame {step}")
                image_ax.axis("off")

                final_chunks = []
                for traj_id in traj_ids:
                    traj_rows = rows[cache_trajectory[rows] == traj_id]
                    order = np.argsort(cache_iteration[traj_rows])
                    levels = cache_iteration[traj_rows][order]
                    states = x_t[traj_rows][order]
                    if levels.tolist() != list(range(11)):
                        raise ValueError(f"Non-contiguous reverse levels for trajectory {traj_id}")
                    final_chunks.append(states[-1] * std[None] + mean[None])

                chunks = np.stack(final_chunks)
                current_q = np.asarray(demo["obs/joint_pos"][step, :7], dtype=np.float32)
                target_q = current_q[None, None, :] + chunks[:, :, :7]
                root_pose = np.asarray(
                    demo["states/articulation/robot/root_pose"][step], dtype=np.float32
                )
                world_fk = WorldFK(
                    base_fk,
                    root_pose[:3],
                    root_pose[3:7],
                    device="cpu",
                )
                xyz, _ = world_fk.grasp_point(
                    torch.from_numpy(target_q.reshape(-1, 7)),
                    ROBOTIQ_GRASP_OFFSET,
                )
                xyz = xyz.detach().cpu().numpy().reshape(8, chunks.shape[1], 3)
                current_xyz = np.asarray(demo["obs/eef_pos"][step, :3], dtype=np.float32)
                paths = np.concatenate(
                    [np.broadcast_to(current_xyz, (8, 1, 3)), xyz], axis=1
                )
                for trajectory_idx, path in enumerate(paths):
                    color = colors[trajectory_idx]
                    if args.projection == "xyz":
                        xz_ax.plot(
                            path[:, 0], path[:, 1], path[:, 2],
                            color=color, linewidth=1.8, alpha=0.85,
                            label=f"trajectory {trajectory_idx}",
                        )
                        xz_ax.scatter(
                            path[-1, 0], path[-1, 1], path[-1, 2],
                            color=color, marker="o", s=28,
                        )
                    else:
                        xz_ax.plot(
                            path[:, horizontal_dim], path[:, 2],
                            color=color, linewidth=1.8, alpha=0.85,
                            label=f"trajectory {trajectory_idx}",
                        )
                        xz_ax.scatter(
                            path[-1, horizontal_dim], path[-1, 2],
                            color=color, marker="o", s=28,
                        )
                if args.projection == "xyz":
                    xz_ax.scatter(
                        current_xyz[0], current_xyz[1], current_xyz[2],
                        color="black", marker="*", s=90, label="current TCP",
                    )
                    if args.fixed_axis_radius is not None:
                        radius = float(args.fixed_axis_radius)
                        xz_ax.set_xlim(
                            current_xyz[0] + radius, current_xyz[0] - radius
                        )
                        xz_ax.set_ylim(
                            current_xyz[1] - radius, current_xyz[1] + radius
                        )
                        xz_ax.set_zlim(
                            current_xyz[2] + radius, current_xyz[2] - radius
                        )
                    else:
                        mins = paths.reshape(-1, 3).min(axis=0)
                        maxs = paths.reshape(-1, 3).max(axis=0)
                        center = 0.5 * (mins + maxs)
                        radius = max(float((maxs - mins).max()) * 0.55, 0.03)
                        xz_ax.set_xlim(center[0] - radius, center[0] + radius)
                        xz_ax.set_ylim(center[1] - radius, center[1] + radius)
                        xz_ax.set_zlim(center[2] - radius, center[2] + radius)
                    xz_ax.set_box_aspect((1, 1, 1))
                    xz_ax.set_xlabel("world X (m)")
                    xz_ax.set_ylabel("world Y (m)")
                    xz_ax.set_zlabel("world Z (m)")
                else:
                    xz_ax.scatter(
                        current_xyz[horizontal_dim], current_xyz[2],
                        color="black", marker="*", s=90, label="current TCP",
                    )
                    if args.fixed_axis_radius is not None:
                        radius = float(args.fixed_axis_radius)
                        if horizontal_dim == 0:
                            xz_ax.set_xlim(
                                current_xyz[0] + radius, current_xyz[0] - radius
                            )
                        else:
                            xz_ax.set_xlim(
                                current_xyz[1] - radius, current_xyz[1] + radius
                            )
                        xz_ax.set_ylim(
                            current_xyz[2] + radius, current_xyz[2] - radius
                        )
                    else:
                        plane = paths[:, :, [horizontal_dim, 2]]
                        mins = plane.reshape(-1, 2).min(axis=0)
                        maxs = plane.reshape(-1, 2).max(axis=0)
                        center = 0.5 * (mins + maxs)
                        radius = max(float((maxs - mins).max()) * 0.55, 0.03)
                        xz_ax.set_xlim(center[0] - radius, center[0] + radius)
                        xz_ax.set_ylim(center[1] - radius, center[1] + radius)
                    xz_ax.set_aspect("equal", adjustable="box")
                    xz_ax.grid(alpha=0.3)
                    xz_ax.set_xlabel(f"world {horizontal_label} (m)")
                    if stage_idx == 0:
                        xz_ax.set_ylabel("world Z (m)")
                if stage_idx == 0:
                    xz_ax.legend(loc="best", fontsize=7)
                xz_ax.set_title("K=8 teacher-sampled action chunks")

            output = args.output_dir / f"{name}_cached_diffusion_trajectories.png"
            fig.savefig(output, dpi=130)
            plt.close(fig)
            manifest["figures"].append(str(output))

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
