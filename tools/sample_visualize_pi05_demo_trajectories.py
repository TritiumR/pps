#!/usr/bin/env python3
"""Sample K pi0.5 chunks on demo observations and visualize TCP trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpi.policies import policy_config
from openpi.training import config as training_config
from sim_common.fk import FrankaFK, WorldFK
from vlm_dp.offline_context import _weight_stage, weight_episode_signals
from vlm_dp.sim_helpers import ROBOTIQ_GRASP_OFFSET


STAGES = (
    "approach pear",
    "lift pear",
    "carry pear",
    "place pear",
    "release pear / approach apple",
    "lift apple",
    "carry apple",
    "place apple",
)


def demo_number(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def eval_stage_midpoints(demo: h5py.Group) -> list[int]:
    signals = weight_episode_signals(demo)
    length = int(demo.attrs["num_samples"])
    stage_ids = np.asarray([_weight_stage(signals, step) for step in range(length)])
    midpoints = []
    for stage in range(8):
        indices = np.flatnonzero(stage_ids == stage)
        if not len(indices):
            raise ValueError(f"Demo has no frames assigned to eval stage {stage}.")
        midpoints.append(int(indices[len(indices) // 2]))
    return midpoints


def observation(demo: h5py.Group, step: int, prompt: str) -> dict[str, object]:
    return {
        "observation/exterior_image_1_left": np.asarray(demo["obs/table_cam"][step]),
        "observation/wrist_image_left": np.asarray(demo["obs/wrist_cam"][step]),
        "observation/joint_position": np.asarray(
            demo["obs/joint_pos"][step, :7], dtype=np.float32
        ),
        "observation/gripper_position": np.asarray(
            demo["obs/gripper_pos"][step, :1], dtype=np.float32
        ),
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--demo-names", required=True)
    parser.add_argument("--config", default="pi05_droid_jointpos")
    parser.add_argument("--prompt", default="put pear and apple on the scale")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42050)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-axis-radius", type=float, default=0.18)
    parser.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Redraw from existing <demo>_pi05_samples.npz files without loading pi0.5.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = training_config.get_config(args.config)
    policy = None
    if not args.reuse_samples:
        policy = policy_config.create_trained_policy(
            config,
            args.checkpoint,
            sample_kwargs={"num_steps": args.num_steps},
            pytorch_device=args.device,
        )
    horizon = int(config.model.action_horizon)
    action_dim = int(config.model.action_dim)
    fk = FrankaFK(device="cpu")
    colors = plt.cm.tab10(np.arange(args.num_samples))
    requested = [name.strip() for name in args.demo_names.split(",") if name.strip()]
    manifest: dict[str, object] = {
        "model": "pi0.5",
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "hdf5": str(args.hdf5),
        "prompt": args.prompt,
        "num_samples_per_observation": args.num_samples,
        "num_steps": args.num_steps,
        "initial_state_distribution": "independent standard Gaussian",
        "stage_assignment": "eval weight stage 0-7",
        "fixed_axis_radius_m": args.fixed_axis_radius,
        "reversed_axes": ["x", "z"],
        "reused_saved_samples": args.reuse_samples,
        "demos": [],
    }

    with h5py.File(args.hdf5, "r") as source:
        for name in requested:
            demo = source["data"][name]
            steps = eval_stage_midpoints(demo)
            raw_path = args.output_dir / f"{name}_pi05_samples.npz"
            cached = np.load(raw_path) if args.reuse_samples else None
            if cached is not None:
                if cached["actions"].shape != (8, args.num_samples, horizon, 8):
                    raise ValueError(
                        f"Unexpected saved action shape in {raw_path}: "
                        f"{cached['actions'].shape}"
                    )
                if not np.array_equal(cached["step"], np.asarray(steps)):
                    raise ValueError(f"Saved stage steps do not match eval stages: {raw_path}")
            fig = plt.figure(figsize=(32, 8), constrained_layout=True)
            image_axes = [
                fig.add_subplot(2, len(STAGES), column + 1)
                for column in range(len(STAGES))
            ]
            xyz_axes = [
                fig.add_subplot(
                    2, len(STAGES), len(STAGES) + column + 1, projection="3d"
                )
                for column in range(len(STAGES))
            ]
            saved_actions = []
            saved_noise = []
            inference_ms = []

            for stage, (label, step) in enumerate(zip(STAGES, steps, strict=True)):
                if cached is not None:
                    actions = np.asarray(cached["actions"][stage], dtype=np.float32)
                    noise = np.asarray(cached["noise"][stage], dtype=np.float32)
                    inference_ms.append(float(cached["inference_ms"][stage]))
                else:
                    obs = observation(demo, step, args.prompt)
                    stage_seed = args.seed + 10_000 * demo_number(name) + stage
                    rng = np.random.default_rng(stage_seed)
                    noise = rng.standard_normal(
                        (args.num_samples, horizon, action_dim), dtype=np.float32
                    )
                    start = time.monotonic()
                    output = policy.infer_batch(
                        [copy.deepcopy(obs) for _ in range(args.num_samples)],
                        noise=noise,
                    )
                    inference_ms.append(1000.0 * (time.monotonic() - start))
                    actions = np.asarray(output["actions"], dtype=np.float32)
                if actions.shape != (args.num_samples, horizon, 8):
                    raise ValueError(
                        f"Expected decoded actions {(args.num_samples, horizon, 8)}, "
                        f"got {actions.shape}."
                    )
                if not np.all(np.isfinite(actions)):
                    raise ValueError(f"Non-finite pi0.5 actions at {name} stage {stage}.")
                saved_actions.append(actions)
                saved_noise.append(noise)

                image_ax = image_axes[stage]
                image_ax.imshow(np.asarray(demo["obs/table_cam"][step]))
                image_ax.set_title(f"{label}\n{name}, frame {step}")
                image_ax.axis("off")

                root_pose = np.asarray(
                    demo["states/articulation/robot/root_pose"][step], dtype=np.float32
                )
                world_fk = WorldFK(
                    fk, root_pose[:3], root_pose[3:7], device="cpu"
                )
                xyz, _ = world_fk.grasp_point(
                    torch.from_numpy(actions[:, :, :7].reshape(-1, 7)),
                    ROBOTIQ_GRASP_OFFSET,
                )
                xyz = xyz.detach().cpu().numpy().reshape(
                    args.num_samples, horizon, 3
                )
                current_xyz = np.asarray(
                    demo["obs/eef_pos"][step, :3], dtype=np.float32
                )
                paths = np.concatenate(
                    [
                        np.broadcast_to(
                            current_xyz, (args.num_samples, 1, 3)
                        ),
                        xyz,
                    ],
                    axis=1,
                )
                axis = xyz_axes[stage]
                for sample, path in enumerate(paths):
                    color = colors[sample]
                    axis.plot(
                        path[:, 0], path[:, 1], path[:, 2],
                        color=color, linewidth=1.8, alpha=0.85,
                        label=f"trajectory {sample}",
                    )
                    axis.scatter(
                        path[-1, 0], path[-1, 1], path[-1, 2],
                        color=color, marker="o", s=28,
                    )
                axis.scatter(
                    current_xyz[0], current_xyz[1], current_xyz[2],
                    color="black", marker="*", s=90, label="current TCP",
                )
                radius = float(args.fixed_axis_radius)
                axis.set_xlim(current_xyz[0] + radius, current_xyz[0] - radius)
                axis.set_ylim(current_xyz[1] - radius, current_xyz[1] + radius)
                axis.set_zlim(current_xyz[2] + radius, current_xyz[2] - radius)
                axis.set_box_aspect((1, 1, 1))
                axis.set_xlabel("world X (m)")
                axis.set_ylabel("world Y (m)")
                axis.set_zlabel("world Z (m)")
                axis.set_title("K=8 pi0.5 samples")
                if stage == 0:
                    axis.legend(loc="best", fontsize=7)

            figure_path = args.output_dir / f"{name}_pi05_xyz_trajectories.png"
            fig.savefig(figure_path, dpi=130)
            plt.close(fig)
            if not args.reuse_samples:
                np.savez_compressed(
                    raw_path,
                    demo_name=np.asarray(name),
                    stage=np.arange(8, dtype=np.int64),
                    step=np.asarray(steps, dtype=np.int64),
                    actions=np.stack(saved_actions).astype(np.float32),
                    noise=np.stack(saved_noise).astype(np.float32),
                    inference_ms=np.asarray(inference_ms, dtype=np.float64),
                )
            manifest["demos"].append(
                {
                    "name": name,
                    "steps": steps,
                    "figure": str(figure_path),
                    "samples": str(raw_path),
                    "mean_inference_ms_per_k8": float(np.mean(inference_ms)),
                }
            )

    manifest_path = args.output_dir / f"manifest_{requested[0]}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
