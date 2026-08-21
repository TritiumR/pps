#!/usr/bin/env python3
"""Check whether cached epsilon labels reproduce cached reverse states."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_free_mpc.ddim import ddim_iteration_alphas  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    target64 = target.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "target_rms": float(np.sqrt(np.mean(np.square(target64)))),
    }


def main() -> None:
    args = _args()
    with np.load(args.cache, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        levels = int(metadata["labels_per_trajectory"])
        trajectories = int(metadata["num_trajectories"])
        train_steps = int(metadata["ddim_num_train_timesteps"])
        x = cache["x_t"].reshape(trajectories, levels, *cache["x_t"].shape[1:])
        epsilon = cache["epsilon"].reshape(
            trajectories, levels, *cache["epsilon"].shape[1:]
        )

        results = []
        for iteration in range(levels - 1):
            alpha, alpha_prev = ddim_iteration_alphas(
                iteration=iteration,
                num_iterations=levels,
                num_train_timesteps=train_steps,
            )
            # Reproduce torch float32 arithmetic used by cache generation.
            alpha32 = np.float32(alpha)
            alpha_prev32 = np.float32(alpha_prev)
            beta32 = np.maximum(np.float32(1.0) - alpha32, np.float32(1e-6))
            sqrt_beta32 = np.sqrt(beta32, dtype=np.float32)
            sqrt_alpha32 = np.sqrt(np.maximum(alpha32, np.float32(1e-6)), dtype=np.float32)
            alpha_step32 = np.maximum(
                alpha32 / np.maximum(alpha_prev32, np.float32(1e-6)), np.float32(1e-6)
            )

            x_t = x[:, iteration]
            eps_t = epsilon[:, iteration]
            target = x[:, iteration + 1]

            # Exact cache update: score=-eps/sqrt(1-alpha), then mbd_score step.
            mbd_next = (x_t - sqrt_beta32 * eps_t) / np.sqrt(alpha_step32, dtype=np.float32)

            # Standard deterministic DDIM step used by ProxyScore.sample_actions.
            x0_hat = (x_t - sqrt_beta32 * eps_t) / sqrt_alpha32
            ddim_next = (
                np.sqrt(np.maximum(alpha_prev32, np.float32(0.0)), dtype=np.float32) * x0_hat
                + np.sqrt(
                    np.maximum(np.float32(1.0) - alpha_prev32, np.float32(0.0)),
                    dtype=np.float32,
                )
                * eps_t
            )

            row = {
                "iteration": iteration,
                "alpha_bar": alpha,
                "alpha_bar_prev": alpha_prev,
                "mbd_score_replay": _metrics(mbd_next, target),
                "standard_ddim_replay": _metrics(ddim_next, target),
                "mbd_vs_ddim": _metrics(ddim_next, mbd_next),
            }
            results.append(row)

        output = {
            "cache": str(args.cache),
            "num_trajectories": trajectories,
            "num_transitions": trajectories * (levels - 1),
            "labels_per_trajectory": levels,
            "trajectory_update_in_metadata": metadata.get("trajectory_update"),
            "target_transform_in_metadata": metadata.get("target_transform"),
            "per_iteration": results,
            "aggregate": {
                name: {
                    metric: float(
                        math.sqrt(np.mean([row[name][metric] ** 2 for row in results]))
                        if metric in {"rmse", "target_rms"}
                        else np.mean([row[name][metric] for row in results])
                    )
                    for metric in ("rmse", "mae", "max_abs", "target_rms")
                }
                for name in ("mbd_score_replay", "standard_ddim_replay", "mbd_vs_ddim")
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
