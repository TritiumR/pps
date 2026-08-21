#!/usr/bin/env python3
"""Plot matched K=8 teacher consistency for MBD and Pi0.5."""

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


def stage_metrics(report: dict, model: str, stage: str) -> dict:
    return report["k8_consistency"][model]["by_stage"][stage]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metric_specs = (
        ("joint_delta_chunk_pairwise_cosine", "Direction cosine ↑", 1.0),
        ("tcp_endpoint_pairwise_distance_m", "TCP pair distance (cm) ↓", 100.0),
        ("gripper_unanimous", "Gripper unanimous fraction ↑", 1.0),
    )
    x = np.arange(len(STAGES))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, (metric, title, scale) in zip(axes, metric_specs, strict=True):
        mbd = np.asarray(
            [stage_metrics(reports[0], "mbd", stage)[metric]["mean"] * scale for stage in STAGES]
        )
        pi_runs = np.asarray(
            [
                [stage_metrics(report, "pi05", stage)[metric]["mean"] * scale for stage in STAGES]
                for report in reports
            ]
        )
        pi_mean = pi_runs.mean(axis=0)
        pi_std = pi_runs.std(axis=0)
        ax.plot(x, mbd, "o-", label="MBD/DDIM", color="#4C78A8", linewidth=2)
        ax.plot(x, pi_mean, "o-", label="Pi0.5 flow", color="#F58518", linewidth=2)
        ax.fill_between(x, pi_mean - pi_std, pi_mean + pi_std, color="#F58518", alpha=0.18)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if metric != "tcp_endpoint_pairwise_distance_m":
            ax.set_ylim(0, 1.03)
    axes[0].set_xlim(-0.2, len(STAGES) - 0.8)
    axes[0].legend(loc="best")
    axes[-1].set_xticks(x, STAGES, rotation=25, ha="right")
    axes[-1].set_xlabel("Task progression")
    fig.suptitle(
        "K=8 teacher consistency on the same 128 observations (16 per stage)\n"
        "Pi0.5 band: mean ± std over three independent initial-noise seeds"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
