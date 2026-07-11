# Scaffold-free grasp: cost + gripper-in-loop replaces HOVER/DESCEND/CLOSE/LIFT

**2026-06-30** — Resolved whether to keep accumulating grasp *primitives*. Answer: **no** — strip the
scaffold, run just the cost, and add back only what's principled. Companion to
[[2026-06-30-collaborator-mpc-comparison]] and the GOAL.md residual thesis.

## The arc
**1. Scaffold (baseline, `_droid_pear_grasp.py`).** Four scripted phases — HOVER → DESCEND → CLOSE →
LIFT — each a hand-chosen (target, gripper state, `tol`). The DIAL MPC only runs *inside*
HOVER/DESCEND/LIFT; the sequencing, gripper timing, and targets are all hard-coded. Earlier today the
`tol=0.02` default (despite a docstring claiming "no tuned tol") was found to be the grasp gate:
tight `tol=0.008` held **1/3** (seed 0 +14 cm), loose held **0/4** — systematic, not run-variance, and
the basin near the optimum is flat.

**2. Cost-only (`_droid_cost_only.py`).** ONE grasp cost, ONE continuous loop, gripper held open, no
phases / no `tol`. Across **4 seeds**: reaches the grasp pose reliably (min 0.36–0.88 cm), **never
disturbs the pear** (±0.2 cm), but the flat basin makes the pose **jitter** (finals scatter 0.6–3.7 cm),
and with the gripper open it never grasps. Key insight: its reach is **identical to the scaffold's
DESCEND** — the cost was never the scaffold's advantage.

**3. What the scaffold actually provided** (not better reaching): (a) the **CLOSE** — a discrete
contact event the kinematic cost structurally can't represent (the dominant reason cost-only can't
hold); (b) the **LIFT**; (c) **commitment** — the phase transition pins the pose against the flat basin
and triggers the close at a chosen moment.

**4. Cost + gripper-in-loop (`_droid_cost_gripper.py`).** Ported the collaborator's gripper-in-cost
(`desired_gripper = near`): the gripper closes by **proximity** (`d < grip_thresh`) + **latch** (commit,
so the flat-basin jitter can't chatter it open/closed — the failure mode flagged in her cost), then a
single coarse grasp→lift **stage switch** on the grasp event. Across **3 seeds**: the proximity close
fired reliably at **~1.1 cm every seed**, no chatter, lift triggered every time. Hold: **0/3** (seed 0
partially lifted +2.8 cm then slipped) — contact-limited, same regime as the scaffold's 1/3.

## Conclusion — the architecture isolates cleanly
- **reach + close-decision + commit → cost / state-driven, NO scaffold.** The scaffold-free pipeline
  reaches by cost, closes-and-commits from state (proximity + latch), and switches stage on the grasp
  event. No HOVER/DESCEND/CLOSE/LIFT, no `tol`.
- **contact hold → unchanged, the RESIDUAL's job.** The geometric kinematic cost cannot supply it; the
  slipping isn't a missing primitive, it's the edge of a contact-free cost. Adding "descend-to-contact"
  or any 5th primitive would be the wrong direction.

So: **no primitive library.** The geometric base is a trustworthy steerable base that hands off cleanly
at the grasp pose; the contact-rich hold is exactly the residual's domain (GOAL.md). The collaborator's
gripper-in-cost is the right mechanism for the *when-to-close* commitment — merging it onto our grounded
grasp cost gives a fully cost/state-driven grasp with zero scripted phases.

## Also this session
- **Cleaned `costs.py` / `sampler.py`** — Google-style docstrings/types, behavior-preserving
  (smoke-verified DIAL descends 14.3 → 4.68; all cost factories return correct shapes).
- **Hardcoded-value audit** — Tier 1 principled (calibrated Robotiq 0.1716, FK, limits); Tier 2 fixed
  cost-weight / DIAL knobs (one set, all tasks); **Tier 3 derivable-but-hardcoded** (object extent,
  floor height, voxel bounds — we have the perception/scene to compute these); Tier 4 scaffold knobs +
  the misleading "no tuned tol" docstring.

Files: `vlm_mpc/_droid_cost_only.py`, `vlm_mpc/_droid_cost_gripper.py` (scaffold `_droid_pear_grasp.py`
kept). Videos: `results/vlm_mpc/droid_cost_only_*`, `droid_cost_gripper_*`.
