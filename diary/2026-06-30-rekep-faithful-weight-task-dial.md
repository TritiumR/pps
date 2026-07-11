# ReKep-faithful weight task, solved by our DIAL-MPC (fake VLM)

**2026-06-30** — Rebuilt the weight task so it's structured exactly like ReKep, solved by our DIAL-MPC,
with the VLM stubbed. This replaces the earlier simplified driver (`_droid_weight_task.py`: hardcoded
stages + geometric-centroid reach + TCP-proximity transitions), which was *not* faithful ReKep.

## Key prior finding
Faithful ReKep already exists in-repo: `rekep/run_rekep_rollout.py` runs the full stage loop (metadata →
per-stage subgoal/path constraints → `KeypointTracker` with movable keypoints → grasp/release) — but on
the Franka tasks with ReKep's **scipy `SubgoalSolver`**. The new build keeps that *structure* and swaps
the solver for **our DIAL** on the Droid weight task.

## What "the scaffold" vs "subgoal decomposition" are (resolved)
Different levels: subgoal decomposition = the task plan (stages + grasp/release + per-stage **keypoint
constraints**); the scaffold (HOVER/DESCEND/CLOSE/LIFT) = motion *within* one grasp (ReKep handles that
separately via a contact-aware primitive). Orthogonal.

## Built
- **`vlm_mpc/weight_fake_vlm.py`** — fake VLM: writes the *exact* `ConstraintGenerator` artifacts
  (`metadata.json` + `stage{i}_{subgoal,path}_constraints.txt`). Keypoint roles (pear/apple/scale)
  resolved from GT masks (the "selection"); placement offset derived from the object/scale point clouds.
  Constraints are faithful numpy (`‖keypoints[pear] − (keypoints[scale] + np.array([dx,dy,dz]))‖`) — the
  `np_shim` supports that vocabulary, so they look identical to real VLM output.
- **`vlm_mpc/_droid_weight_rekep.py`** — mirrors `run_rekep_rollout`'s loop on `DroidEnv` with DIAL: per
  stage, load the subgoal constraint → `make_rekep_cost`/`make_rekep_grasp_cost` → plan/execute/replan
  **until the subgoal constraint is satisfied on the tracked keypoints** (the ReKep transition) → grasp/
  release per the metadata flag.
- **`costs.py`** — two additions: (1) optional **smoothness term** `w_smooth·Σ‖q[t]−q[t−1]‖²` on
  `make_rekep_cost`/`make_rekep_grasp_cost` (default 0; the driver uses 0.3 to damp the flat-basin
  jitter — adopted from the collaborator's cost, the one regularizer ours lacked); (2) **movable-keypoint
  handling** in `make_rekep_cost` (`held_idx`/`held_offset`): a held object's keypoint is predicted as
  `candidate_TCP + offset` so a place constraint actually depends on the action. Without it the
  held-keypoint constraint is constant (zero gradient).

## Bug fixed
First run crashed silently in the place stage (no traceback — eaten by Isaac's hanging `close()`, same
as the earlier `env.scene` spin). Cause: the movable handling built keypoints as `[K,H,N,3]`, but ReKep
constraints index `keypoints[i]` expecting the **keypoint dim first** — so `keypoints[10]` indexed the
*candidate* dim. Fixed to `[N,K,H,3]` (keypoint dim first). Verified on a CPU unit test (held path →
`[K]`, finite, varies with the candidate) before re-running.

## Result (seed 0)
Structure works **end-to-end**: roles resolved, 4 stages flow, grasp subgoals satisfied via the
constraints (**1.17 cm / 1.09 cm**), movable handling functional (held `[8]`, `[11,12]`), transitions on
constraint-satisfaction, grasp/release from flags. Video: `results/vlm_mpc/weight_rekep/weight_rekep_s0.mp4`
(317 frames). **Place subgoals NOT satisfied** (capped 38/54 cm; objects unmoved, dz ≈ 0) — the
**contact carry-hold fails** (grasp closes on a *surface* keypoint, not the graspable center → marginal →
object not actually held → real subgoal never satisfies). Both are the known gaps: **perception
keypoint-placement (surface vs center)** + **contact hold = residual's job** — not structural bugs.

## Status
The weight task is now genuinely **ReKep-faithful + our DIAL + fake VLM**; the only plug-in left is the
real VLM (`ConstraintGenerator` → fills the same dir). Placement still needs (a) targeting the graspable
center, and/or (b) the contact residual — both out of the geometric base's scope, as scoped. Companion to
[[2026-06-30-scaffold-free-grasp-pipeline]]; reuses [[rekep-on-pps]] machinery + `np_shim`.
