#!/usr/bin/env python3
"""Plot the verified B-Batch64 base/ref/task/full-steer evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt


LABELS = [
    "Base\nonly",
    "B Ref\nonly",
    "B Task\nonly",
    "Full\nλ=0.0",
    "Full\nλ=0.1",
    "Full\nλ=0.2",
    "Full\nλ=0.3",
    "Full\nλ=0.4",
    "Full\nλ=0.5",
    "Full\nλ=0.6",
    "Full\nλ=0.7",
    "Full\nλ=0.8",
    "Full\nλ=0.9",
    "Full\nλ=1.0",
]
SUCCESSES = [12, 0, 13, 14, 18, 8, 1, 2, 1, 2, 0, 0, 0, 0]
TOTAL = 20


def main() -> None:
    output_dir = Path("results/Isaac-Weight-Droid-Visuomotor-v0")
    output_dir.mkdir(parents=True, exist_ok=True)
    rates = [100.0 * value / TOTAL for value in SUCCESSES]
    x = list(range(len(LABELS)))

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 21,
        "xtick.labelsize": 15,
        "ytick.labelsize": 17,
    })
    fig, ax = plt.subplots(figsize=(21, 8))
    ax.plot(x, rates, color="#315A8A", linewidth=3.2, marker="o", markersize=11)
    ax.scatter(x[:3], rates[:3], s=150, color=["#377EB8", "#FF9F1C", "#2CA02C"], zorder=3)
    ax.scatter(x[3:], rates[3:], s=150, color="#D62728", zorder=3)

    for index, (successes, rate) in enumerate(zip(SUCCESSES, rates)):
        offset = 4 if rate < 92 else -8
        va = "bottom" if offset > 0 else "top"
        ax.annotate(
            f"{successes}/{TOTAL}",
            (index, rate),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=16,
            fontweight="bold",
        )

    ax.axvline(2.5, color="0.55", linestyle="--", linewidth=1.8)
    ax.text(1.0, 103, "Standalone policies", ha="center", fontsize=17, color="0.3")
    ax.text(8.0, 103, "Full steering: Base + λ(Task − Ref)", ha="center", fontsize=17, color="0.3")
    ax.set_title("B Batch64 Policy and Full-Steering Evaluation")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(-5, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    stem = output_dir / "b64_base_ref_task_fullsteer_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
