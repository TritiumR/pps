# ReKep weight task on DIAL: the motion-quality arc (center grasp → clearance → sub-goal solve → transit)

**2026-06-30** — Everything tried to make the ReKep-faithful weight task (`vlm_mpc/_droid_weight_rekep.py`,
fake VLM) behave well, the findings, the tradeoffs, and the videos. Builds on
[[2026-06-30-rekep-faithful-weight-task-dial]]. All runs: receding-horizon DIAL, GT-mask grounding,
contact hold unaddressed (residual's job).

## What it is
ReKep structure (stages + grasp/release flags + relational keypoint constraints + `KeypointTracker` with
movable keypoints + constraint-satisfaction transitions) solved by **our DIAL-MPC**, fake VLM stub
(`weight_fake_vlm.py`). Per stage: build the DIAL cost, plan→execute→replan until the subgoal constraint
is satisfied on the tracked keypoints, then grasp/release per the metadata.

## The arc of fixes (each a cost term / config, no scaffold)
1. **Grasp = perception center, not the surface keypoint.** ReKep's keypoint *selects* the object; the
   grasp targets the object's **local masked-point centroid** (`weight_fake_vlm.local_centroid`) — the
   grasp-module half ReKep keeps and we'd skipped. Single-cam → still ~1 cm surface-biased.
2. **Clearance term** (`_clearance` in costs.py) — avoid the *other* objects' point clouds. Fixed the
   **scatter** (the earlier straight-line carry knocked the pear off the board).
3. **`exec_knot` 5→2→1.** Was executing the 5th knot/step (~open-loop, "insanely fast"); 1 = textbook
   receding horizon. Finding: this is NOT the jitter lever (the metric was confounded by step size).
4. **Consistency cost** (`w_consist` in sampler.py) = ReKep's "solution close to previous." Raised the
   smoothness metric (0.092→0.296) BUT on a *moving* place target it anchored a *bad* trajectory and
   drove the arm **out of frame**. Reverted (kept, `--w_consist 0`).
5. **Sub-goal solve** (`resolve_subgoal` in the driver) — resolve the relational place constraint into a
   **fixed TCP target** (gradient descent on the loaded constraint, held kp rides along), then reach
   that. A *stationary* problem → **in-frame, sensible** (vs the consistency run's out-of-frame). This
   is ReKep's sub-goal solver; the cost only ever chases a fixed point.
6. **Transit-clearance term** (`_transit_clearance` in costs.py) — keep the gripper above a
   **scene-derived** `z_clear` (tallest object + margin) while horizontally far from the target, descend
   near it → **lift → transit-high → descend emerges** (ReKep's path solver uses a table-clearance cost).
   Tuning that mattered: **linear** (not squared — squared was dwarfed by reach), `w_transit≈60`, and
   **no transit on the grasp** (it blocked the top-down descent; grasp uses orient-down instead).

## Key findings
- **The jitter IS the per-step re-sampling.** `plan_once` (plan one long trajectory, execute it) took
  smoothness 0.092→**0.878** — by far the smoothest. *Confirmed* the root cause.
- **But plan_once-dense under-optimizes.** DIAL can't solve a 30-step × 7-joint trajectory with 512
  samples → smooth but *doesn't reach* (caps). The fix is a **coarse-waypoint** parameterization (ReKep's
  path solver: few decision variables) — not built yet. Mode switch is in place (`--control_mode`).
- **The smoothness metric lies** (×3: consistency, plan_once both scored high while *failing the task*).
  Judge by behavior (reaches + in-frame + no scatter), not the metric.
- **PPS note:** plan-once + closed-loop tracking is the *best* steerable base (clean `v_vlm`, adapts to
  the steered state); per-step re-sampling is closed-loop but jittery; pure open-loop diverges under
  steering. The cost-term route (transit etc.) keeps the unified DIAL base → PPS-friendly; a separate
  path solver would be open-loop and awkward for PPS.

## Current best config (default)
`receding` mode, `exec_knot=1`, sub-goal solve on place, grasp = perception center (no transit),
place = clearance + transit (linear, `w_transit=60`), `w_consist=0`. Result: clean top-down grasp,
lift-then-transit carry, **in-frame, no scatter** — all from scene-derived cost terms.

## Remaining gaps (NOT motion — and not the geometric base's job)
- **Contact hold** — the rounded fruit slips; the gripper lifts *empty* (objects `dz≈0` every run). Force
  closure = the **residual**.
- **Surface-biased grasp center** — single-cam centroid sits ~1 cm high → marginal grip. Needs
  **multi-view / shape-fit** perception.
- **Residual jitter** — still present (re-sampling); fix = coarse-waypoint plan-once (deferred).

## Videos (`results/vlm_mpc/weight_rekep/`)
| video | what it shows |
|---|---|
| `weight_rekep_s0.mp4` | **First** ReKep-faithful run — surface-keypoint grasp; structure flows, place caps, hold fails. |
| `weight_rekep_center_s0.mp4` | **Center grasp** (perception centroid, not surface kp); reaches center, hold still fails. |
| `weight_rekep_clear_s0.mp4` | **+ clearance + exec_knot=2 + w_local** — scatter fixed (objects stay put), motion deliberate. |
| `weight_rekep_ek2.mp4` / `weight_rekep_ek1.mp4` | `exec_knot` 2 vs 1 jitter sweep (smoothness 0.258 vs 0.147 — confounded). |
| `weight_rekep_wc0.mp4` / `weight_rekep_wc5.mp4` | consistency cost 0 vs 5 — wc5 smoother metric (0.296) but **arm out of frame, scene scattered**. |
| `weight_rekep_subgoal_s0.mp4` | **Sub-goal solve** — place reaches a *fixed* target; **in-frame, sensible** (the fix for wc5's out-of-frame). |
| `weight_rekep_planonce_s0.mp4` | **plan_once** mode — very smooth (0.878) but under-optimized → **caps short** (smooth-but-wrong). |
| `weight_rekep_transit_s0.mp4` | **Transit v1** (squared, w=50) — rises but **diagonally/late**, still clips the cluster. |
| `weight_rekep_transit3_s0.mp4` | **Current best** — clean top-down grasp + **lift→transit→descend carry, no scatter, in-frame**. |

(transit v2 / w=100 was killed mid-run — too strong, blocked the grasp descent; no video.)
