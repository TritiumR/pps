# ReKep implementation outline — faithful core, Isaac adaptations, potential additions

**2026-06-28** — A map of our ReKep port (`pps/rekep/`): what is **faithful** (unchanged from
upstream `~/ReKep`), what is **changed** to run inside Isaac Sim / IsaacLab, and what **potential
additions** address limitations we've since hit (most concretely, the single-camera
handle-visibility vs. grasp-pose trade-off → a multi-view rig). Complements the deeper
per-line diffs in [2026-06-25-faithfulness-audit.md](2026-06-25-faithfulness-audit.md); this entry
is the structural overview rather than the audit.

ReKep = "Relational Keypoint Constraints." Pipeline: propose keypoints from an image → a VLM writes
per-stage relational constraint functions over those keypoints → a solver turns each stage's
constraints into an end-effector subgoal → the robot is driven there, grasping/releasing per stage.

---

## 1. Faithful — unchanged (or cosmetically-only changed) from upstream

These carry ReKep's method verbatim; divergences are formatting/imports or a documented robustness
guard, never semantics.

- **Keypoint proposal** — `keypoint_proposal.py`. DINOv2 (`dinov2_vits14`) patch features → bilinear
  upsample → per-mask PCA → KMeans (`num_candidates_per_mask`) → MeanShift merge in **3D cartesian**
  → workspace-bounds filter → numbered overlay. Verbatim except `+1e-6` eps guards for degenerate
  tiny masks (turns NaN→0 only on constant dims; negligible otherwise).
- **Constraint generation** — `constraint_generation.py`. GPT writes the
  `stage{i}_{subgoal|path}_constraint{j}(end_effector, keypoints)` functions + `num_stages` /
  `grasp_keypoints` / `release_keypoints` metadata. The prompt, chat structure, parsing, and metadata
  format are upstream's. (Changes: prompt loaded from `prompts/prompt_template.txt`, explicit
  `task_dir`, current OpenAI SDK — see §2.)
- **Prompt** — `prompts/prompt_template.txt` is the upstream instruction text verbatim (stage
  decomposition rules, the `(end_effector(3,), keypoints(K,3)) -> cost<=0` contract,
  `get_grasping_cost_by_keypoint_idx`).
- **Constraint sandbox** — `utils.load_functions_from_txt` execs the VLM code with only `np` +
  `get_grasping_cost_by_keypoint_idx` in scope, exactly as upstream. `get_callable_grasping_cost_fn`
  (0 if keypoint held else 1) is upstream's.
- **`movable`-at-solve timing** — a grasp-stage's target keypoint is **not** movable during its own
  solve (object not yet held). This is upstream's behavior, reproduced faithfully (audit has the
  detail). Legit divergence: we track a *logical* `grasped_body` instead of querying *physical*
  `is_grasping` — equivalent given GT tracking.

## 2. Changed — implementation adaptations to match Isaac Sim

Same method, different substrate. Each change is non-semantic or a documented, justified divergence.

- **Perception: SAM → GT instance segmentation.** Upstream segments RGB with SAM; we use IsaacLab's
  `instance_id_segmentation_fast` + `id_to_prim` (GT masks). This is how ReKep itself runs in sim
  (privileged masks), so it's faithful-in-spirit. Consequence handled in `grounding.py`: IsaacLab
  segments the **whole room** (200+ prims), so we restrict masks to the **workspace bbox**
  (`workspace_bounds_from_scene`) and to the scene's **rigid-object prims** (`task_object_ids`) before
  clustering — otherwise the proposer clusters hundreds of background masks.
- **Camera → world points.** `pps_to_rekep.camera_to_rekep_inputs` builds the dense `(H,W,3)` world
  point map from IsaacLab depth + intrinsics/extrinsics. Verified to equal IsaacLab's own
  unprojection math (audit: "shared camera→world depth adapter — correct").
- **Keypoint tracking: visual tracker → privileged GT pose.** `keypoint_tracking.py` registers each
  proposed keypoint to the nearest rigid body (offset in body frame) and recomputes its world
  position from the body's **GT pose** each step. Replaces a real-robot visual point-tracker; GOAL.md
  sanctions reading privileged state **in sim**. *Note:* this is **tracking only** — it does not
  affect which keypoints exist (the camera + DINOv2 still decide that) nor what the VLM sees.
- **Control: motion planner → IK-Rel.** Upstream reaches an absolute EE pose with a collision-free
  planner (`EndEffectorPoseViaPlanning`); we drive there with IK-Rel base-frame deltas
  (`drive_to_pose`) + a contact-aware `grasp_at` primitive (hover → descend-to-contact → close →
  lift) shared with MOKA. Same target, straight-line execution, no collision avoidance.
- **Subgoal solver: dropped terms.** `solvers.py` keeps ReKep's subgoal/path constraint objective but
  **drops the SDF-collision and IK-feasibility terms** — IsaacLab's IK handles reachability and our
  tasks are uncluttered. This is the one **deliberate documented simplification** (not a port bug).
- **OpenAI SDK migration.** Legacy `openai.ChatCompletion` → current `client.chat.completions`; model
  / messages / stop tokens unchanged.
- **Camera source.** Upstream's dedicated `vlm_camera` → each PPS task's existing `table_cam`
  (augmented with depth+seg via `grounding.augment_table_cam_with_depth_and_seg`).

## 3. Reuse beyond the rollout — ReKep cost inside DIAL-MPC (VLM-DP rungs)

Separate from the faithful ReKep rollout, the **same** ReKep constraint *form* now also feeds the
PyTorch sampling-MPC controller (`vlm_mpc/`), per GOAL.md's VLM-DP direction:
- The GPT constraint (`‖end_effector − keypoints[i]‖`) is loaded through an **np→torch shim**
  (`vlm_mpc/np_shim.py`) so it evaluates **batched** over `[K,H,3]` candidate trajectories, and
  becomes the task term of `make_rekep_cost`, which **DIAL-MPC** minimizes. Validated rung-3 on the
  mug: camera keypoints → DINOv2 → GPT-4o (correctly selected the handle, kp4) → shim → DIAL **reached
  the handle (3 cm)**. The physical lift did **not** hold (`held=False`) — see §4.
- This is an *addition*, not a change to ReKep: the front-end (proposal + VLM + tracking) is the
  faithful ReKep port; only the **back-end controller** (DIAL instead of ReKep's scipy subgoal solver)
  differs.

## 4. Potential additions — addressing limitations met later

Forward-looking; **not yet implemented**. Each is scoped against a concrete limitation we hit.

- **Multi-view keypoint rig** *(top priority — limitation: single-camera visibility ⇄ grasp pose).*
  With one camera, a keypoint exists only on surfaces that camera sees, so **handle-visible** and
  **handle-toward-robot (easy grasp)** are mutually exclusive — the direct cause of rung-3's
  `held=False` (handle had to face the camera, i.e. away from the robot). Fix: place **N cameras**
  around the workspace, run DINOv2+cluster per view, **pool the 3D candidates** and run the existing
  MeanShift merge once (the merge is already in 3D world space, so this is the same algorithm on a
  bigger pool — no new math). For the VLM's single annotated image, pick the **primary view = the one
  that sees the most merged keypoints** and project all keypoints onto it. Lets the handle face the
  robot *and* be a keypoint. Plan sketched; deferred.
- **Visual keypoint tracker** *(limitation: GT tracking is sim-only).* Replace the privileged-GT
  tracker with a real point tracker (the upstream path) for real-robot / no-privilege fidelity.
- **Grasp-yaw alignment in the cost** *(limitation: thin-handle grasp slips).* The geometric cost
  aligns the approach axis down but not the gripper **yaw**; on a thin handle loop the fingers don't
  reliably straddle the bar (contributing to `held=False`). Add a yaw term (closing axis ⟂ handle
  bar), or leave it to a learned residual.
- **Finer keypoints on thin features** *(limitation: DINOv2 clusters broad surfaces).* The handle
  **bar** never got its own keypoint — only the handle **top** (kp4). Proposal favors large surface
  clusters over thin loops. Could add a thin-structure proposal pass or lower the mask granularity.
- **Restore solver collision / IK-feasibility terms** *(limitation: cluttered scenes).* The dropped
  SDF-collision and IK-feasibility terms would matter once tasks have obstacles; re-add them for
  cluttered settings.
- **Multi-image VLM prompt** *(fallback if a single annotated view is insufficient).* Show the VLM one
  annotated image **per camera** instead of a single primary view — more faithful coverage at the cost
  of diverging from ReKep's single-image prompt.

---

**Bottom line.** The ReKep *method* (proposal → VLM constraints → relational cost) is ported
faithfully; the changes are all substrate adaptations (GT masks, GT tracking, IK-Rel control, SDK) or
one documented solver simplification. The limitations we've hit are about **perception coverage**
(single view, thin features) and **contact-rich grasping** (yaw, hold) — not about the relational-cost
idea, which holds up. The multi-view rig is the highest-value next addition.
