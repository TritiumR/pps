#!/usr/bin/env python3
"""Create compact summary figures for the ref-distillation debug experiments."""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _series(report: dict, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in report["per_timestep"]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=pathlib.Path, default=pathlib.Path("results/ref_distill_debug")
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("results/ref_distill_debug/figures")
    )
    args = parser.parse_args()
    root = args.results_dir
    reports = {
        "old": _load(root / "old_cache_closed_loop_full.json"),
        "broken": _load(root / "overfit36_a_step5000_train.json"),
        "fixed_train": _load(root / "overfit36_a_epsfix_step1000_train.json"),
        "fixed_val": _load(root / "overfit36_a_epsfix_step1000_val.json"),
        "b_train": _load(root / "overfit36_b_step5000_train.json"),
        "b_val": _load(root / "overfit36_b_step5000_val.json"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    colors = {
        "old": "#777777",
        "broken": "#d62728",
        "fixed_train": "#1f77b4",
        "fixed_val": "#2ca02c",
        "b_train": "#9467bd",
        "b_val": "#ff7f0e",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    for name, label in (
        ("old", "Old ref checkpoint"),
        ("broken", "A before target fix (train)"),
        ("fixed_train", "A fixed, step 1k (train)"),
        ("fixed_val", "A fixed, step 1k (val)"),
    ):
        values = _series(reports[name], "closed_loop_state_rmse")
        ax.plot(np.arange(1, len(values)), values[1:], marker="o", ms=3, label=label, color=colors[name])
    ax.set_yscale("log")
    ax.set_xlabel("Reverse update")
    ax.set_ylabel("Closed-loop state RMSE")
    ax.set_title("(a) Closed-loop rollout error")
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    for name, label in (
        ("broken", "A before target fix (train)"),
        ("fixed_train", "A fixed, step 1k (train)"),
        ("fixed_val", "A fixed, step 1k (val)"),
    ):
        ax.plot(
            _series(reports[name], "epsilon_cosine"),
            marker="o",
            ms=3,
            label=label,
            color=colors[name],
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Cached scheduler level")
    ax.set_ylabel("Epsilon cosine")
    ax.set_title("(b) Direct epsilon direction")
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    broken = reports["broken"]
    beta_sqrt = np.sqrt(1.0 - _series(broken, "alpha_bar"))
    ax.plot(
        _series(broken, "epsilon_pred_to_target_rms_ratio"),
        marker="o",
        ms=3,
        color=colors["broken"],
        label="Observed |eps_pred| / |eps_target|",
    )
    ax.plot(beta_sqrt, linestyle="--", color="black", label="sqrt(1 - alpha_bar)")
    ax.set_xlabel("Cached scheduler level")
    ax.set_ylabel("Magnitude ratio")
    ax.set_title("(c) Signature of the score/epsilon target bug")
    ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    for name, label in (("b_train", "B train"), ("b_val", "B val")):
        ax.plot(
            _series(reports[name], "epsilon_cosine"),
            marker="o",
            ms=3,
            label=label,
            color=colors[name],
        )
    ax.set_ylim(0.75, 1.01)
    ax.set_xlabel("Evaluation noise level")
    ax.set_ylabel("Epsilon cosine")
    ax.set_title("(d) Action-chunk B: standard online noising")
    ax.legend(fontsize=7.5)

    figure_path = args.output_dir / "ref_distill_points1_2_summary"
    fig.savefig(figure_path.with_suffix(".png"), dpi=220)
    fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)

    metrics = (
        "nearest_teacher_rmse",
        "aligned_teacher_rmse",
        "predicted_pairwise_rmse",
        "teacher_pairwise_rmse",
    )
    labels = ("Nearest teacher\nRMSE", "Aligned teacher\nRMSE", "Predicted\ndiversity", "Teacher\ndiversity")
    train = np.asarray([reports["b_train"]["sample_metrics"][key] for key in metrics])
    val = np.asarray([reports["b_val"]["sample_metrics"][key] for key in metrics])
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(x - width / 2, train, width, label="train", color=colors["b_train"])
    ax.bar(x + width / 2, val, width, label="val", color=colors["b_val"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized action-space RMSE")
    ax.set_title("Action-chunk B sampled-action fidelity and diversity")
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    action_path = args.output_dir / "ref_action_b_sampling_metrics"
    fig.savefig(action_path.with_suffix(".png"), dpi=220)
    fig.savefig(action_path.with_suffix(".pdf"))
    plt.close(fig)

    print(figure_path.with_suffix(".png"))
    print(action_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
