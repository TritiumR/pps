# 2026-07-04 — sim_free_mbd refactor + the minimal VLM-DP base

Records the consolidation of the 07-02..04 debug harness into clean, merge-friendly modules, plus the
ablation→conclusion table for the experiments now archived under
`results/vlm_mpc/sim_free_mbd/archive/`.

## Arc: debug → fixes → minimal base

The debug (see `sim-free-mpc-smoothing-debug` memory) reproduced the collaborator's SimFreeMPC
jitter/collision in our IsaacLab weight task and found principled position-space fixes:

- **Twist (wrist wander)** — null-space wander excited by the score noise. Fixed by a **consistency term**
  (pull the plan toward the warm-started previous one) + a **B-spline knot basis** (approximating C2, no
  overshoot). Together they halve j4/j6 total variation, matching the CRN regime without CRN.
- **Collision** — her keepout is weak (0.1), hardcoded, and there is no table term. Fixed by **one general
  keepout** over all scene objects (extents from geometry, target/payload-excluded, faded near the target) +
  a **table floor plane**, evaluated at **several gripper points + the payload** rather than the TCP alone.
  Got scene disturbance to ~2.7 cm on the privileged pick-place demo.
- **Grasp** — not learned and not force-limited; it is **seating precision + close timing**. A general
  proximity gripper (close when the TCP is within ~3 cm of the object center) plus geometry-derived extents.
- **Minimal base** — the above consolidated into a **soft, general base** (a pi0.5 swap-in), deleting the
  task-specific gates/latches/transit. It grasps, lifts, and carries with five general terms; imperfect but
  steerable, which is the point for PPS.

## Module structure

The engine is entirely the collaborator's `sim_free_mpc`, imported unchanged. Our code is a thin driver plus
a small cost.

- `vlm_mpc/sim_free_core.py` — shared driver: `build_policy` (checkpoint-free decode via norm-stats),
  `build_mpc`, `apply_horizon_basis` (B-spline smoother), `guard_cost`, `policy_inputs`, `plan_chunk`;
  re-exports the decode and ddim helpers.
- `vlm_mpc/minimal_base_cost.py` — the minimal base cost. Reuses `_reach_cost`, `_regularization`,
  `_downward_orientation_cost` from `sim_free_mpc.costs`; adds `_consistency_cost`, `_straddle_cost`,
  `_general_collision_cost` as standalone functions. `usd_extents` reads per-object `(grip, keepout,
  half_height)` from the sim USD bounding boxes.
- `vlm_mpc/tasks/minimal_base.py` — `--task minimal_base`: general 2-phase target (grasp object → place
  location), general proximity gripper, geometry extents, no task-specific hacks.
- `vlm_mpc/tasks/sim_free_mbd.py` — slimmed faithful harness (1213 → 204 lines): her `PriorityStateCost` +
  the vetted recipe (B-spline + consistency + guard + joint-delta clip + phase latch), reading the env's
  `subtask_terms`.

Merge story: the collaborator already vendored our cost as `sim_free_mpc/costs_for_ref.py`, and the added
terms mirror her `_xxx_cost` signatures, so upstreaming is a paste into `costs.py`.

## Ablation → conclusion (archived experiments)

| phase | tested | conclusion |
|---|---|---|
| jitter-ablations | update/init/noise/param on her pipeline | delta_clip is the dominant calm knob; decode is not the cause |
| smoothing-basis | linear / cubic / B-spline / RBF knot interp | B-spline (approximating C2, no overshoot) is best |
| decode | real quantile stats vs identity; cumulative vs not | decode exonerated |
| delta-clip-react | joint_delta_clip × noise; exec-knot reactivity | delta_clip is the calm knob; reactivity needs warm-start |
| grasp-descend | approach-z, clearance gating, descend/settle | grasp = seating precision + close timing, not force |
| orientation | pin/no-orient/posture; CRN batches | twist is null-space wander; CRN or consistency calm it |
| cost-variants | consistency dose, clean-cost, collision-layer, transit | consistency + B-spline halve the twist; multi-point collision + payload needed for a clean carry |
| sdf | ReKep-style perception SDF collision | works but is the real-robot path; occlusion-limited in sim |
| minimal-base-iters | early minimal-base runs | needed geometry extents + a tight close gate |

## State and open items

- Refactor steps 1–5 done; this note is step 6.
- Open: multi-seed the refactored `minimal_base` against pre-refactor `minbase4` (one seed showed 83.8 cm
  disturbance, likely Isaac physics variance) to confirm the term composition preserved behavior.
- Milestone videos: `results/vlm_mpc/sim_free_mbd/{01,02,03}_*.mp4` and
  `results/vlm_mpc/minimal_base/05_minimal-base__general.mp4`.
