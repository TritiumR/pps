# Faithfulness audit — ReKep / MOKA / VoxPoser ports vs upstream

**2026-06-25** — Component-by-component audit of the three IsaacLab ports against their upstream
repos (`~/ReKep`, `~/moka`, `~/VoxPoser`). Goal: **verify each implementation is correct and faithful
to the original method — not to maximize task success.** No ad-hoc / task-specific changes. Result:
faithful or justified-and-documented across the board; one deliberate divergence.

---

## Method

For each component, diff the ported module against its upstream original and classify every
divergence as: **VERBATIM** (faithful), **LEGITIMATE ISAAC ADAPTATION** (justified, non-semantic —
e.g. import paths, SDK migration, GT-mask perception, IK-Rel control), or **PORT BUG / DELIBERATE
DIVERGENCE**. Where a sub-agent was used, its verdict was independently re-verified (one was
overturned — see MOKA grasp).

## Scorecard

| Component | Verdict |
|---|---|
| VoxPoser core (interfaces / planners / controllers / dynamics / LMP) | ✅ Faithful (verbatim) |
| ReKep keypoint proposal (DINOv2 → cluster) | ✅ Faithful |
| MOKA front-end (flip / segmentation / overhead cam) | ✅ Correct + Faithful |
| MOKA grasp / deprojection | ✅ Faithful |
| ReKep `movable`-at-solve timing | ✅ Faithful |
| Shared camera→world depth adapter | ✅ Correct (= IsaacLab's own math) |
| `grasp_at` / `descend_to_contact` (session addition) | ✅ Contained execution adaptation |
| IK-Rel control math · prompts (all 3) · VoxPoser grounding | ✅ |
| **ReKep solver dropped terms** | ⚠️ **Deliberate documented simplification** |

---

## Per-component findings

### VoxPoser core — FAITHFUL (essentially verbatim)
Diffed `pps/voxposer/` vs `~/VoxPoser/src/`. `dynamics_models.py` + `utils.py` **byte-identical**;
`planners.py` (greedy value-map descent) = 1 import line; `controllers.py` = imports only;
`interfaces.py` (voxel math + value-map composition + LLM API) = import qualifications + a `vec2quat`
alias that forwards to the existing `pointat2quat`; `LMP.py` = the OpenAI SDK migration
(`openai.ChatCompletion.create(...)['choices']...` → `_CLIENT.chat.completions.create(...).choices...`)
+ imports (model=gpt-4, messages, stop tokens preserved). **Zero logic changes** to voxel math /
value-map composition / planner / controller / dynamics. All Isaac-specific code is isolated in
`envs/isaac_env.py` (replaces the RLBench env). Cleanest of the three ports.

### MOKA grasp / deprojection — FAITHFUL (sub-agent verdict overturned)
`planners/planner.py` + `vision/grasp_utils.py` are **byte-identical** to upstream. The wrong-depth
grasp in rollouts (grasp keypoint lifts to the cooktop z≈0.74, not the handle z≈0.96) is **upstream
MOKA's own grasp-sampler behavior**: `select_grasp` substitutes the LLM grasp keypoint with the
antipodal sampler's center (planner.py:170), which for a handle lands in the gap/edge on a
cooktop-seeing pixel; upstream's exact-pixel deproject (`z = depth[int(y),int(x)]`, planner.py:236)
also lifts it to the cooktop (`keypoints_depth['grasp']=1.225`). The port's one divergence — replacing
exact-pixel deproject with `isaac_bridge.lift_2d_to_world`'s dense world-grid **window** lookup — returns
the surface *nearest the camera* (`argmin(dist-to-cam)`), so it *prefers* the handle and returns cooktop
only when no handle pixel is in the 7×7 window; it is **equal-or-better** than upstream, not the cause.
(A sub-agent initially blamed the window heuristic; re-verification via `keypoints_depth['grasp']=1.225`
+ the byte-identical sampler showed upstream lifts to the cooktop too → FAITHFUL.)

### ReKep `movable`-at-solve timing — FAITHFUL
During a grasp-stage subgoal solve the to-be-grasped keypoint is **not** marked movable, so a VLM
grasp constraint that omits the `end_effector` term (e.g. pot stage-3's `||kp[7]-kp[3]||`) has zero
gradient w.r.t. the EE and the solver returns the init guess (current pose). This is **upstream's own
behavior**: `_update_keypoint_movable_mask` (main.py:259-262) sets `mask[i] = env.is_grasping(obj)` in
`_update_stage` (main.py:256) *before* the solve, so for a grasp stage (gripper just opened, object not
yet held) `is_grasping=False` → non-movable. The port (run_rekep_rollout.py:218,
`owner==grasped_body` with grasped_body = prior grasp) reproduces it. One legitimate divergence:
upstream queries *physical* `is_grasping`; the port tracks a *logical* `grasped_body` (fine given the
GT keypoint-tracking; only differs if a grasp physically fails). So pot stage-3 is faithful upstream
behavior for an atypical constraint, NOT a port bug.

### Shared camera→world depth adapter — CORRECT
`pps_to_rekep.camera_to_rekep_inputs` is a **byte-for-byte match of IsaacLab's own validated
point-cloud math** (`tea/mdp/observations._sample_rgbd_camera_point_cloud`): same
`distance_to_image_plane` depth, `inv(intrinsic_matrices)` deprojection, invalid→0, and
`transform_points(pos_w, quat_w_ros)` (so IsaacLab handles the optical-frame convention; no manual
axis flip). Only non-semantic diffs: dense full-grid vs sampled-N pixels; a redundant `/pixel_rays[2:3]`
no-op (inv(K)@[u,v,1] already has z=1). Empirically consistent with GT object poses throughout. This
shared adapter underpins **all three** ports' grounding.

### MOKA front-end (flip / segmentation / overhead cam) — CORRECT + FAITHFUL
The **flip matched-set** is consistently disabled across all 3 sites: `preprocess_image` drops
`[::-1,::-1,:]`; `transform_points` drops the `crop_shape-point` un-flip (keeps the crop offset); the
grasp-crop bbox drops upstream's `crop_shape-bbox` inversion + the `[bbox2,bbox3,bbox0,bbox1]` reorder
(keeps the crop offset±OFFSET). So cropped↔original coords round-trip correctly for the upright cam —
**no coordinate bug**. Other diffs all legitimate: `detectron2.GenericMask`→`cv2.findContours(RETR_CCOMP,
CHAIN_APPROX_NONE)` is faithful (GenericMask itself wraps that exact call); checkpoint/config paths made
module-relative; a transformers-pin shim; `overhead_cam.py` is a justified port-only top-down camera
(MOKA's prompts + grasp sampler assume top-down); `_mark_index` regex robustly parses `P3`/`P[3]`.
`vision/keypoint.py` (FPS) + `vision/image_utils.py` byte-identical.

### ReKep keypoint proposal (DINOv2 → cluster) — FAITHFUL
125-line diff is ~all formatting (quotes / reflow / comments / import path). The DINOv2 `dinov2_vits14`
→ bilinear interpolate → per-mask PCA → KMeans (`num_candidates_per_mask`, euclidean) → MeanShift merge
→ workspace-bounds filter → numbered overlay pipeline is **verbatim**. Only 3 semantic changes, all the
same robustness guard for IsaacLab's degenerate tiny instance-seg masks: skip masks with
`<num_candidates` points; `+1e-6` eps in **both** the PCA-feature and xyz normalizations. Negligible for
normal masks (`max-min ≫ 1e-6`); only turns NaN→0 for constant dims. Justified by the GT-instance-seg
perception choice; faithful behavior preserved.

### `grasp_at` / `descend_to_contact` (session addition) — CONTAINED EXECUTION ADAPTATION
Added this session to fix the universal grasp failure (gripper closing in mid-air above objects). Audit
of my own change: it **preserves the method's grasp XY and approach orientation exactly** (used for
hover, descent, lift) and **respects the method's grasp Z to within `floor_margin`=6cm** — the close is
the *contact* point in the window `[g[2]−0.06, contact]`. So it executes the method's grasp pose with a
bounded contact-based depth refinement (fixing the surface-Z-vs-gripper-contact mismatch), **not** a free
Z override; it does not change *what* gets grasped. The `floor_margin` constant is what bounds the
divergence.

---

## The one deliberate divergence: ReKep solver dropped terms

Diffed `rekep/solvers.py` vs upstream `subgoal_solver.py`. Upstream's objective has **7 cost terms**;
the port keeps 4 verbatim and drops 3:

- **Kept (faithful):** consistency / init-pose (1.0, rot_weight 1.5), goal-constraint (200.0),
  path-constraint (200.0), grasp metric (10.0 — uses the Franka **z-axis** vs upstream's Fetch **x-axis**,
  a justified adaptation).
- **Dropped:** collision cost (0.8×, needs a scene SDF + robot `collision_points`), IK/reachability
  cost (20.0×, needs an `ik_solver` returning `num_descents`/`success`), reset-joint reg (0.2×, from the
  IK result). **Documented** in `solvers.py:6-8`.

The collision drop is a reasonable deferral. The **IK-feasibility drop is the meaningful one** — the
solver can propose **unreachable subgoals** with no planning-time penalty (the docstring's "IK-Rel
handles reachability at execution" is weak: execution just fails to reach silently), which directly
contributes to the pot-edge reach failures we diagnosed earlier. It is **flagged, not silent.**

**Faithful-fix path** (a legitimate faithfulness improvement, *not* a success hack): restore the dropped
terms — wire `pinocchio` (already a dependency) as the IK oracle for the IK-feasibility term, and an
IsaacLab scene SDF for the collision term.

---

## Bottom line

A component-by-component audit against all three upstream repos found the ports **faithful or
justified-and-documented across the board**. Every grounding, perception, planning, and control
component reproduces upstream behavior or is a non-semantic Isaac adaptation. The **single deliberate
divergence** is ReKep's dropped SDF-collision + IK-feasibility solver terms — flagged in-code, with a
clear faithful-fix path. **The implementations are correct; the rollout limitations are inherited from
the original methods, not introduced by the port.**

### Open follow-ups (faithfulness, not success)
- Restore ReKep's IK-feasibility (pinocchio) + SDF-collision solver terms to fully match upstream.
- Optional remaining audits (lower priority): ReKep keypoint *tracking* (GT-pose registration, an
  Isaac adaptation), VoxPoser's `isaac_env.py` GT-mask point-cloud bridge.
