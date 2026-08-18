#!/usr/bin/env python3
"""Losslessly compact demo HDF5 images to the model's deterministic 224px input."""

from __future__ import annotations

import argparse
import os
import pathlib

import h5py
import numpy as np
from tqdm import tqdm

from openpi.shared import image_tools

CAMERA_KEYS = ("table_cam", "wrist_cam")


def _copy_attrs(source, target) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _copy_demo(source: h5py.Group, target: h5py.Group, *, height: int, width: int) -> None:
    _copy_attrs(source, target)
    for key in source.keys():
        if key != "obs":
            source.copy(key, target, name=key)

    source_obs = source["obs"]
    target_obs = target.create_group("obs")
    _copy_attrs(source_obs, target_obs)
    for key in source_obs.keys():
        if key not in CAMERA_KEYS:
            source_obs.copy(key, target_obs, name=key)

    for key in CAMERA_KEYS:
        source_camera = source_obs[key]
        shape = (len(source_camera), height, width, source_camera.shape[-1])
        target_camera = target_obs.create_dataset(
            key,
            shape=shape,
            dtype=np.uint8,
            chunks=(1, height, width, source_camera.shape[-1]),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        _copy_attrs(source_camera, target_camera)
        source_chunk = source_camera.chunks[0] if source_camera.chunks else 16
        batch_size = max(int(source_chunk), 16)
        for start in range(0, len(source_camera), batch_size):
            end = min(start + batch_size, len(source_camera))
            resized = image_tools.resize_with_pad(
                np.asarray(source_camera[start:end]), height, width
            )
            target_camera[start:end] = np.asarray(resized, dtype=np.uint8)


def build_shard(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as source, h5py.File(temporary, "w") as target:
        _copy_attrs(source, target)
        source_data = source["data"]
        target_data = target.create_group("data")
        _copy_attrs(source_data, target_data)
        demo_names = sorted(source_data.keys())[args.shard :: args.num_shards]
        total = 0
        for demo_name in tqdm(demo_names, desc=f"compact shard {args.shard}/{args.num_shards}"):
            source_demo = source_data[demo_name]
            target_demo = target_data.create_group(demo_name)
            _copy_demo(source_demo, target_demo, height=args.height, width=args.width)
            total += int(source_demo.attrs.get("num_samples", len(source_demo["obs/joint_actions"])))
        target_data.attrs["total"] = total
        target.attrs["compact_source"] = str(args.input.resolve())
        target.attrs["compact_resize"] = "openpi.shared.image_tools.resize_with_pad"
        target.attrs["compact_height"] = args.height
        target.attrs["compact_width"] = args.width
        target.attrs["compact_shard"] = args.shard
        target.attrs["compact_num_shards"] = args.num_shards
        target.flush()
    temporary.replace(output)


def build_master(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    total = 0
    with h5py.File(temporary, "w") as target:
        target_data = target.create_group("data")
        for input_index, input_path in enumerate(args.inputs):
            input_path = input_path.resolve()
            with h5py.File(input_path, "r") as source:
                if input_index == 0:
                    _copy_attrs(source, target)
                    _copy_attrs(source["data"], target_data)
                relative = os.path.relpath(input_path, output.parent)
                for demo_name in source["data"].keys():
                    if demo_name in seen:
                        raise ValueError(f"Duplicate demo across compact shards: {demo_name}")
                    seen.add(demo_name)
                    source_demo = source["data"][demo_name]
                    total += int(source_demo.attrs.get("num_samples", len(source_demo["obs/joint_actions"])))
                    target_data[demo_name] = h5py.ExternalLink(relative, f"/data/{demo_name}")
        target_data.attrs["total"] = total
        target.attrs["compact_master"] = True
        target.attrs["compact_num_demos"] = len(seen)
        target.attrs["compact_num_shards"] = len(args.inputs)
        target.flush()
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("shard")
    shard.add_argument("--input", type=pathlib.Path, required=True)
    shard.add_argument("--output", type=pathlib.Path, required=True)
    shard.add_argument("--num-shards", type=int, default=1)
    shard.add_argument("--shard", type=int, default=0)
    shard.add_argument("--height", type=int, default=224)
    shard.add_argument("--width", type=int, default=224)
    shard.set_defaults(func=build_shard)

    master = subparsers.add_parser("master")
    master.add_argument("--inputs", type=pathlib.Path, nargs="+", required=True)
    master.add_argument("--output", type=pathlib.Path, required=True)
    master.set_defaults(func=build_master)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "shard":
        if args.num_shards <= 0 or not 0 <= args.shard < args.num_shards:
            raise ValueError("Require num_shards > 0 and 0 <= shard < num_shards.")
    args.func(args)


if __name__ == "__main__":
    main()
