#!/usr/bin/env python3
"""Restore <seed>_success.mp4 / <seed>_fail.mp4 names in consolidated evals."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+", type=pathlib.Path)
    args = parser.parse_args()

    for experiment in args.experiments:
        result_path = experiment / "results.json"
        payload = json.loads(result_path.read_text())
        expected_names = []
        for episode in payload["episodes"]:
            seed = int(episode["seed"])
            outcome = "success" if bool(episode.get("success")) else "fail"
            desired_name = f"{seed}_{outcome}.mp4"
            current_name = episode.get("video") or f"{seed}.mp4"
            current_path = experiment / current_name
            desired_path = experiment / desired_name
            if current_path != desired_path:
                if not current_path.is_file():
                    raise FileNotFoundError(current_path)
                if desired_path.exists():
                    raise FileExistsError(desired_path)
                os.replace(current_path, desired_path)
            episode["video"] = desired_name
            expected_names.append(desired_name)

        actual_names = sorted(path.name for path in experiment.glob("*.mp4"))
        if actual_names != sorted(expected_names):
            raise RuntimeError(f"video verification failed for {experiment}")
        with tempfile.NamedTemporaryFile("w", dir=experiment, delete=False) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temp_path = pathlib.Path(handle.name)
        os.replace(temp_path, result_path)
        print(f"restored suffixes: {experiment}")


if __name__ == "__main__":
    main()
