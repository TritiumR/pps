#!/usr/bin/env python3
"""Merge IsaacLab HDF5 datasets while renumbering demos contiguously."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py


def _demo_index(name: str) -> int:
    prefix = "demo_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected group {name!r}; expected names like demo_0")
    return int(name[len(prefix) :])


def _demo_names(data: h5py.Group) -> list[str]:
    return sorted(data.keys(), key=_demo_index)


def _validate_demo(demo: h5py.Group, source: Path) -> int:
    if "num_samples" not in demo.attrs:
        raise ValueError(f"{source}:{demo.name} has no num_samples attribute")
    num_samples = int(demo.attrs["num_samples"])
    for key in ("actions", "processed_actions"):
        if key not in demo:
            raise ValueError(f"{source}:{demo.name} has no {key} dataset")
        if not isinstance(demo[key], h5py.Dataset):
            raise ValueError(f"{source}:{demo.name}/{key} is not a dataset")
        if len(demo[key]) != num_samples:
            raise ValueError(
                f"{source}:{demo.name}/{key} has {len(demo[key])} rows, "
                f"expected {num_samples}"
            )
    return num_samples


def merge(inputs: list[Path], output: Path, expected_demos: int | None) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    demo_count = 0
    total_samples = 0
    expected_env_args = None
    try:
        with h5py.File(temporary, "w") as destination:
            output_data = destination.create_group("data")
            for input_path in inputs:
                with h5py.File(input_path, "r") as source:
                    if "data" not in source:
                        raise ValueError(f"{input_path} has no data group")
                    source_data = source["data"]
                    env_args = source_data.attrs.get("env_args")
                    if expected_env_args is None:
                        expected_env_args = env_args
                        for key, value in source_data.attrs.items():
                            output_data.attrs[key] = value
                    elif env_args != expected_env_args:
                        raise ValueError(f"env_args mismatch in {input_path}")

                    names = _demo_names(source_data)
                    print(f"Copying {len(names)} demos from {input_path}", flush=True)
                    for name in names:
                        demo = source_data[name]
                        total_samples += _validate_demo(demo, input_path)
                        source.copy(demo, output_data, name=f"demo_{demo_count}")
                        demo_count += 1

            if expected_demos is not None and demo_count != expected_demos:
                raise ValueError(f"Merged {demo_count} demos, expected {expected_demos}")
            output_data.attrs["total"] = total_samples
            destination.flush()
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"Wrote {demo_count} demos / {total_samples} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-demos", type=int)
    args = parser.parse_args()
    merge(args.input, args.output, args.expected_demos)


if __name__ == "__main__":
    main()
