#!/usr/bin/env python
"""Plot bit-conditioned proxy steering success rates from evaluation_summary.json.

Reads every results/<task_id>/eval_<exp>_<task>/evaluation_summary.json that
matches the configured experiments and renders a grouped bar chart. Round 2 is
picked up automatically once its summaries exist, so this can be re-run as
results land.

    python plot_eval_results.py [-o results/hybrid_bit_eval_success_rates.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parent

# exp_name prefix -> label shown in the legend. Order defines series order.
EXPERIMENTS = [
    ("eval_hybrid_bit_20k", "Round 1 · gemma_12m, causal (33.4M)"),
    ("eval_hybrid_bit_40m_bidir_20k", "Round 2 · gemma_40m, bidirectional (59.9M)"),
]
TASKS = ["weight", "pot", "tea", "capsule"]
TASK_LABEL = {
    "weight": "weight",
    "pot": "pot",
    "tea": "tea",
    "capsule": "capsule",
}

# Validated palette (see dataviz references/palette.md). Light surface.
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - correct for proportions near 0/1 at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_results() -> dict[str, dict[str, dict]]:
    """{exp_prefix: {task: summary_dict}} for whatever exists on disk."""
    out: dict[str, dict[str, dict]] = {}
    for prefix, _ in EXPERIMENTS:
        found = {}
        for path in (ROOT / "results").glob(f"*/{prefix}_*/evaluation_summary.json"):
            task = path.parent.name[len(prefix) + 1 :]
            if task in TASKS:
                found[task] = json.loads(path.read_text())
        if found:
            out[prefix] = found
    return out


def rounded_bar(ax, x, width, y, height, color, radius_px=4.0):
    """Bar with a radius_px rounded far end, square at the baseline.

    The radius is converted separately for x and y so the corner stays circular
    on screen; using one data-space radius for both makes the bar render as a
    fat pill whenever the axes aspect is far from 1.
    """
    if width <= 0:
        return
    bbox = ax.get_window_extent()
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    rx = radius_px * abs(xhi - xlo) / max(bbox.width, 1e-6)
    ry = radius_px * abs(yhi - ylo) / max(bbox.height, 1e-6)
    rx = min(rx, width)
    ry = min(ry, height / 2)

    x1 = x + width
    verts = [
        (x, y),
        (x1 - rx, y),
        (x1, y),
        (x1, y + ry),
        (x1, y + height - ry),
        (x1, y + height),
        (x1 - rx, y + height),
        (x, y + height),
        (x, y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="results/hybrid_bit_eval_success_rates.png")
    args = ap.parse_args()

    data = load_results()
    if not data:
        print("No evaluation_summary.json files found; nothing to plot.")
        return 1

    present = [(p, lab) for p, lab in EXPERIMENTS if p in data]
    n_series = len(present)

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Order tasks by round-1 rate (descending) so the chart reads high -> low.
    ref = data[present[0][0]]
    ordered = sorted(TASKS, key=lambda t: -ref.get(t, {}).get("success_rate", -1))

    # Thin marks: keep well under half the category pitch so the bars read as
    # rules rather than blocks.
    group_h = 0.30 if n_series == 1 else 0.46
    bar_h = group_h / n_series
    gap = 0.02  # 2px-equivalent surface gap between adjacent bars

    # Bar geometry is computed against the axes bbox, so fix the limits first.
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    fig.canvas.draw()

    for si, (prefix, label) in enumerate(present):
        color = SERIES[si]
        for ti, task in enumerate(ordered):
            s = data[prefix].get(task)
            if not s:
                continue
            rate = s["success_rate"]
            n = s["num_episodes"]
            k = s["num_successes"]
            y = ti - group_h / 2 + si * bar_h + gap / 2
            h = bar_h - gap
            rounded_bar(ax, 0, rate, y, h, color)

            lo, hi = wilson_ci(k, n)
            yc = y + h / 2
            ax.plot([lo, hi], [yc, yc], color=INK_SECONDARY, lw=1.0, zorder=4,
                    solid_capstyle="butt")
            for b in (lo, hi):
                ax.plot([b, b], [yc - h * 0.18, yc + h * 0.18], color=INK_SECONDARY,
                        lw=1.0, zorder=4)

            ax.text(hi + 0.015, yc, f"{k}/{n} · {rate:.0%}", va="center", ha="left",
                    fontsize=9, color=INK_PRIMARY, zorder=5)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([TASK_LABEL[t] for t in ordered], fontsize=10, color=INK_PRIMARY)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=INK_MUTED)
    ax.set_xlabel("success rate over 50 seeds (Wilson 95% CI)", fontsize=9, color=INK_SECONDARY)

    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0)

    ax.text(0, 1.16, "Bit-conditioned proxy steering — IsaacLab success rate",
            transform=ax.transAxes, fontsize=12.5, color=INK_PRIMARY, va="bottom")
    ax.text(0, 1.055,
            "steer_scale 0.4 · seeds 1–50 · one checkpoint: bit 1 = task flow, bit 0 = reference flow",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED, va="bottom")

    if n_series >= 2:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SERIES[i], edgecolor="none")
                   for i in range(n_series)]
        ax.legend(handles, [lab for _, lab in present], loc="lower right", frameon=False,
                  fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")

    # Table view (the WCAG-clean twin of the chart).
    print()
    hdr = f"{'task':<9}" + "".join(f"{lab.split(' · ')[0]:>26}" for _, lab in present)
    print(hdr)
    for task in ordered:
        row = f"{task:<9}"
        for prefix, _ in present:
            s = data[prefix].get(task)
            if s:
                lo, hi = wilson_ci(s["num_successes"], s["num_episodes"])
                row += f"{s['num_successes']:>3}/{s['num_episodes']:<3} {s['success_rate']:>5.0%} [{lo:.0%}–{hi:.0%}]".rjust(26)
            else:
                row += f"{'—':>26}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
