"""Standalone rollout plots from an episode JSONL.

Ported in spirit from ~/hydrax/vlm_mpc/viz_mpc.py (fig_subgoal / fig_spaghetti / fig_traj3d),
retargeted at the records mujoco_eval already writes. Every quantity plotted here has been in the
logs all along with nothing to render it -- these figures need no new instrumentation.

    python -m mujoco_eval.viz_rollout results/stack/_rekep_fixed/101.jsonl -o results/viz/stack

Writes: subgoal.png (constraint residual + stage spans), cost.png (cost_min/weighted + ESS),
traj3d.png (executed EE path in 3D with object positions), spaghetti.png (per-axis EE traces).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

_STAGE_TINT = ("#eef4ff", "#fff3e6", "#eaf7ee", "#fdeaf0", "#f2eefb")


def load(path):
    """Return (replans, episode) from an episode JSONL."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    replans = [r for r in rows if r.get("kind") == "replan"]
    episode = next((r for r in rows if r.get("kind") == "episode"), None)
    if not replans:
        raise SystemExit(f"[viz] no replan records in {path}")
    return replans, episode


def _stage_spans(stages):
    """Contiguous [start, end, stage] runs, for shading the stage a metric was measured under."""
    spans, start = [], 0
    for i in range(1, len(stages) + 1):
        if i == len(stages) or stages[i] != stages[start]:
            spans.append((start, i - 1, stages[start]))
            start = i
    return spans


def _shade_stages(ax, stages):
    for lo, hi, stage in _stage_spans(stages):
        ax.axvspan(lo, hi + 1, color=_STAGE_TINT[stage % len(_STAGE_TINT)], zorder=0)


def fig_subgoal(replans, out):
    """ReKep constraint residual over the rollout, shaded by stage."""
    import matplotlib.pyplot as plt

    vals = [(i, r["rekep_subgoal_now"]) for i, r in enumerate(replans)
            if r.get("rekep_subgoal_now") is not None]
    if not vals:
        print("[viz] no rekep_subgoal_now in this log; skipping subgoal.png")
        return
    xs, ys = zip(*vals)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _shade_stages(ax, [r["stage_idx"] for r in replans])
    ax.plot(xs, ys, lw=1.8, color="#1a1a1a")
    ax.set_yscale("log")
    ax.set_xlabel("replan")
    ax.set_ylabel("subgoal residual (m, log)")
    ax.set_title(f"ReKep constraint residual   {ys[0]:.3f} to {ys[-1]:.4f}")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out / "subgoal.png", dpi=150)
    plt.close(fig)


def fig_cost(replans, out):
    """Planner cost and effective sample size -- is the sampler discriminating?"""
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    stages = [r["stage_idx"] for r in replans]
    for ax in (a1, a2):
        _shade_stages(ax, stages)
    a1.plot([r.get("cost_min", np.nan) for r in replans], lw=1.5, label="cost_min", color="#1a1a1a")
    a1.plot([r.get("cost_weighted", np.nan) for r in replans], lw=1.2, label="cost_weighted",
            color="#c2410c")
    a1.set_ylabel("cost")
    a1.legend(frameon=False, fontsize=8)
    a1.margins(x=0)
    a2.plot([r.get("weight_ess", np.nan) for r in replans], lw=1.5, color="#1d4ed8")
    a2.set_ylabel("weight ESS")
    a2.set_xlabel("replan")
    a2.margins(x=0)
    fig.tight_layout()
    fig.savefig(out / "cost.png", dpi=150)
    plt.close(fig)


def fig_traj3d(replans, out):
    """Executed end-effector path in 3D, with object start/end positions."""
    import matplotlib.pyplot as plt  # noqa: F401
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    import matplotlib.pyplot as plt

    tcp = np.array([r["tcp"] for r in replans])
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], lw=1.6, color="#1d4ed8", label="EE path")
    ax.scatter(*tcp[0], color="#16a34a", s=40, label="start")
    ax.scatter(*tcp[-1], color="#dc2626", s=40, label="end")
    for name in replans[0].get("objects", {}):
        first = np.asarray(replans[0]["objects"][name])
        last = np.asarray(replans[-1]["objects"][name])
        ax.scatter(*first, marker="s", s=30, alpha=0.5)
        ax.text(*last, f" {name}", fontsize=7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "traj3d.png", dpi=150)
    plt.close(fig)


def fig_spaghetti(replans, out):
    """Per-axis EE traces plus gripper aperture -- where motion stalls, and whether it was holding."""
    import matplotlib.pyplot as plt

    tcp = np.array([r["tcp"] for r in replans])
    grip = [r.get("gripper_read", np.nan) for r in replans]
    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    stages = [r["stage_idx"] for r in replans]
    for ax, series, label in zip(axes, [tcp[:, 0], tcp[:, 1], tcp[:, 2], grip],
                                 ["tcp x", "tcp y", "tcp z", "gripper"]):
        _shade_stages(ax, stages)
        ax.plot(series, lw=1.5, color="#1a1a1a")
        ax.set_ylabel(label)
        ax.margins(x=0)
    axes[-1].set_xlabel("replan")
    fig.tight_layout()
    fig.savefig(out / "spaghetti.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl")
    parser.add_argument("-o", "--out", default=None,
                        help="output directory (default: alongside the jsonl)")
    args = parser.parse_args()

    replans, episode = load(args.jsonl)
    out = pathlib.Path(args.out or pathlib.Path(args.jsonl).parent) / "figs"
    out.mkdir(parents=True, exist_ok=True)

    fig_subgoal(replans, out)
    fig_cost(replans, out)
    fig_traj3d(replans, out)
    fig_spaghetti(replans, out)
    tag = (f"success={episode['success']} stage_max={episode['stage_max']}"
           if episode else "(no episode record)")
    print(f"[viz] {len(replans)} replans, {tag}\n[viz] wrote {out}/", flush=True)


if __name__ == "__main__":
    main()
