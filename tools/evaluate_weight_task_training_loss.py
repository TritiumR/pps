#!/usr/bin/env python3
"""Evaluate the weight task epsilon checkpoint on its source demonstrations.

This reconstructs the LeRobot samples produced by
``convert_isaaclab_data_to_lerobot.py`` directly from the IsaacLab HDF5 file,
then applies the checkpoint's normal training transforms.  It avoids creating
another ~24 GB dataset copy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import safetensors.torch
import torch

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models_pytorch.proxy_score_pytorch import ProxyScorePytorch
from openpi.shared import normalize
from openpi.training import config as config_lib


def resize_like_converter(image: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((320, 180), Image.Resampling.BICUBIC))


def episode_frames(h5: h5py.File) -> list[tuple[str, int]]:
    frames: list[tuple[str, int]] = []
    for name in sorted(h5["data"], key=lambda value: int(value.split("_")[-1])):
        length = int(h5["data"][name].attrs["num_samples"])
        frames.extend((name, step) for step in range(length))
    return frames


def raw_sample(h5: h5py.File, episode: str, step: int, horizon: int) -> dict:
    trajectory = h5["data"][episode]
    length = int(trajectory.attrs["num_samples"])
    action_indices = np.minimum(np.arange(step, step + horizon) + 1, length - 1)
    return {
        "exterior_image_1_left": resize_like_converter(trajectory["obs/table_cam"][step]),
        "wrist_image_left": resize_like_converter(trajectory["obs/wrist_cam"][step]),
        "joint_position": np.asarray(trajectory["obs/joint_pos"][step, :7], dtype=np.float32),
        "gripper_position": np.asarray(trajectory["obs/gripper_pos"][step, :1], dtype=np.float32),
        "actions": np.asarray(trajectory["obs/joint_actions"][action_indices], dtype=np.float32),
        "prompt": "put pear and apple on the scale",
    }


def collate(samples: list[dict], device: torch.device):
    output = torch.utils.data.default_collate(samples)
    output = torch.utils._pytree.tree_map(
        lambda value: value.to(device) if isinstance(value, torch.Tensor) else value,
        output,
    )
    actions = output.pop("actions").to(dtype=torch.float32)
    return model_lib.Observation.from_dict(output), actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", default="score_task_weight_demo_meanstd")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = config_lib.get_config(args.config)
    stats_path = args.checkpoint / "assets" / "cn356" / "isaaclab_weight"
    stats = normalize.load(stats_path)
    data_config = dataclasses.replace(
        config.data.create(config.assets_dirs, config.model), norm_stats=stats
    )
    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )

    model = ProxyScorePytorch(config.model).to(device)
    safetensors.torch.load_model(
        model, args.checkpoint / "model.safetensors", device=str(device)
    )
    model.eval()

    with h5py.File(args.data, "r") as h5:
        frames = episode_frames(h5)
        count = min(args.samples, len(frames))
        chosen = np.linspace(0, len(frames) - 1, count, dtype=np.int64)
        selected = [frames[index] for index in chosen]

        bidirectional_losses: list[np.ndarray] = []
        causal_losses: list[np.ndarray] = []
        timestep_indices: list[np.ndarray] = []
        generator = torch.Generator(device=device).manual_seed(args.seed)

        for batch_start in range(0, count, args.batch_size):
            batch_frames = selected[batch_start : batch_start + args.batch_size]
            samples = [
                transform(raw_sample(h5, episode, step, config.model.action_horizon))
                for episode, step in batch_frames
            ]
            observation, actions = collate(samples, device)
            batch = actions.shape[0]
            # Stratify over the same 100 discrete diffusion levels used in training.
            indices = torch.linspace(0, 99, batch, device=device).round().long()
            indices = (indices + batch_start * 17) % 100
            times = indices.to(torch.float32) / 99.0
            noise = torch.randn(actions.shape, generator=generator, device=device)

            # Reset augmentation RNG so causal and bidirectional passes see identical images.
            augmentation_seed = args.seed + batch_start
            with torch.no_grad():
                model.bidirectional_attention = True
                torch.manual_seed(augmentation_seed)
                bidir = model(observation, actions, noise=noise, time=times).mean((1, 2))
                model.bidirectional_attention = False
                torch.manual_seed(augmentation_seed)
                causal = model(observation, actions, noise=noise, time=times).mean((1, 2))
                model.bidirectional_attention = True

            bidirectional_losses.append(bidir.cpu().numpy())
            causal_losses.append(causal.cpu().numpy())
            timestep_indices.append(indices.cpu().numpy())
            print(
                f"scored {batch_start + batch}/{count}: "
                f"bidir={float(bidir.mean()):.4f} causal={float(causal.mean()):.4f}",
                flush=True,
            )

    bidir = np.concatenate(bidirectional_losses)
    causal = np.concatenate(causal_losses)
    indices = np.concatenate(timestep_indices)
    bins = {}
    for lower, upper in ((0, 24), (25, 49), (50, 74), (75, 99)):
        selected_bin = (indices >= lower) & (indices <= upper)
        bins[f"{lower:02d}-{upper:02d}"] = {
            "samples": int(selected_bin.sum()),
            "bidirectional_loss": float(bidir[selected_bin].mean()),
            "causal_loss": float(causal[selected_bin].mean()),
        }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "samples": int(len(bidir)),
        "seed": args.seed,
        "bidirectional_loss": float(bidir.mean()),
        "causal_loss": float(causal.mean()),
        "causal_over_bidirectional": float(causal.mean() / bidir.mean()),
        "zero_epsilon_baseline": 1.0,
        "timestep_bins": bins,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
