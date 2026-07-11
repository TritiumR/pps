# ReKep package refactor (for a shareable release)

**Goal.** Clean up `pps/rekep/` so it can be committed and shared: a lean, self-contained,
consistently-documented package with no scattered hardcoded paths and no cross-package coupling.
Purely a hygiene/structure pass — **no algorithm logic changed anywhere** (the faithful ports stay
behavior-identical; verified by container imports + logic-token checks).

---

## Passes (in order)

**1. Survey — used vs. unused.** Three kinds of files were mixed at the package root: the importable
library (`utils`, `video`, `keypoint_tracking`, `rekep_viz`, `grounding`, `keypoint_proposal`,
`constraint_generation`, plus the camera adapter), standalone CLI demos (`run_rekep`,
`run_rekep_rollout`, `visualize_keypoints`), and a scratch smoke test (`_smoke_frontend`). `solvers.py`
was alive only through the rollout demo.

**2. Script consolidation.** `run_rekep` (grounding only) was a strict subset of `run_rekep_rollout`
(grounding + subgoal solve + motion-planned rollout) — and the latter already had `--use_cached` to
skip grounding, so they were two halves of one pipeline. **Merged into one `rekep/scripts/run_rekep.py`**
with a `--plan-only` flag (stop after grounding) vs. the default full solve+rollout; deleted both
originals (including `run_rekep`'s **stale duplicate** scene helpers). Moved `visualize_keypoints.py`
into `scripts/`; moved `_smoke_frontend.py` out to `agent_tests/`. Fixed import depth for the moved
scripts (`_REKEP_DIR`/`_REPO_DIR` derived one level deeper).

**3. Visualization dedup.** The "read camera → project keypoints → draw dots + text → BGR frame"
pattern was **triplicated** (`vlm_mpc/overlay.py`, `run_rekep`, and inline in `visualize`). Hoisted one
`camera_overlay_frame(camera, positions, text_lines)` into `rekep_viz.py` — the right layer, so the
`rekep/` scripts use it **without importing `vlm_mpc`**. `overlay.camera_overlay_frame` became a one-line
wrapper (the ~10 vlm_mpc task call sites are untouched). `visualize` also dropped its local
`_write_video_h264` copy (uses `rekep.video`). Kept `video.py` standalone — 24 importers made moving it
pure churn for no gain.

**4. Split `grounding`/`pps_to_rekep` by dependency weight.** The camera adapter is **light**
(`isaaclab.utils.math` + numpy) and used *independently* by MOKA and VoxPoser (they have their own
front-ends, never touch ReKep's proposer). `grounding` is **heavy** (pulls `KeypointProposer` →
DINOv2/KMeans/sklearn). Merging them naively would force MOKA/VoxPoser to import the clustering stack
just to read a camera. So created **`isaaclab_helpers.py`** = light glue (camera adapter + scene/workspace
helpers), leaving **`grounding.py`** as just `propose_keypoints`. Deleted `pps_to_rekep.py`; updated its 5
external importers.

**5. Removed the `moka.grasp_at` cross-dependency.** `run_rekep`'s grasp reached into `moka` for a
contact-aware descend — the only `rekep → moka` edge. Replaced with a local `grasp()` helper
(hover → descend → close → lift) built on the script's own `drive_to_pose`/`hold_gripper`. Now **`rekep/`
has zero cross-package deps**, and the grasp is *more* faithful to upstream ReKep (no contact sensing;
the contact-aware variant stays in moka for the PPS pipeline).

**6. Fixed a layering inversion.** `quat_wxyz_to_matrix` lived in `rekep_viz` (rendering) but was imported
by `keypoint_tracking` (core) — core depending on viz. Moved it to `utils` (geometry); both pull it from
there now.

**7. Config resolution centralized (the big one).** `config.yaml` was hardcoded as
`os.path.join(_REPO, "rekep", "config.yaml")` in **~10 files across vlm_mpc + agent_tests** — vlm_mpc
reaching into rekep's directory. Moved `config.yaml → configs/default.yaml`; added `default_config_path()`
+ `load_default_config()` in `utils` as the **one** place that knows where rekep's config lives; added a
`--config` flag to the scripts. Consumers now call `load_default_config()` — **no cross-package path
reaching**. So any future config move is a one-line change in the resolver, not a 10-file hunt.

**8. Config parameter completeness.** Moved three hardcoded knobs into `configs/default.yaml`:
`subgoal_solver` (`sampling_maxfun`/`maxiter`, was hardcoded in `run_rekep`; upstream ReKep has this
section), `keypoint_proposer.dino_model` (config-selectable DINOv2 backbone), and `margin` (workspace
padding). Left genuine constants hardcoded (`patch_size=14`, mask specks, overlay cosmetics) to avoid
config bloat.

**9. Comment/formatting pass, every file.** Comments made short/concise and **self-explanatory in
isolation** — dropped lineage/discussion references (hydrax, "GOAL.md sanctions…", "faithful port…",
"our discussion" framings), collapsed Google `Args:/Returns:` blocks on small helpers, fixed a few stale
docstrings (`utils`, `keypoint_proposal`). Imports consolidated: the scattered `pot_scene_fix` import
(buried inside `_run`) hoisted into the post-`AppLauncher` block. The **two-phase import structure is
required** (IsaacLab: `AppLauncher` must launch before `torch`/`isaaclab_tasks` import) — kept it with a
one-line note. The two pristine ports (`keypoint_proposal`, `constraint_generation`) had comments refined
with **logic byte-identical**.

---

## Final structure

```
rekep/
  configs/default.yaml        # keypoint_proposer + constraint_generator + subgoal_solver + bounds/margin
  prompts/prompt_template.txt
  isaaclab_helpers.py         # light glue: camera->rekep-inputs + scene/workspace helpers
  grounding.py                # propose_keypoints (the heavy DINOv2 orchestration)
  keypoint_proposal.py        # port: DINOv2 -> KMeans -> MeanShift -> overlay
  constraint_generation.py    # port: GPT-4o -> constraint code + metadata
  solvers.py                  # adapted port: 6-DoF subgoal optimizer
  keypoint_tracking.py        # register keypoints to rigid bodies, track via GT poses
  rekep_viz.py                # project + draw + camera_overlay_frame
  video.py                    # H.264 mp4 writer
  utils.py                    # config resolver + geometry + constraint sandbox
  scripts/
    run_rekep.py              # ground (--plan-only) OR full solve+rollout
    visualize_keypoints.py    # keypoint-tracking video
```

## Principles established (reusable)
- **Each package owns its config resolution.** Consumers ask rekep (`load_default_config()`); never
  hardcode a cross-package path.
- **Dependency direction is `vlm_mpc → rekep`, never the reverse.** The shared viz primitive lives in
  `rekep_viz`, not `vlm_mpc/overlay`, so rekep scripts don't import vlm_mpc.
- **Light glue separated from heavy orchestration** (`isaaclab_helpers` vs `grounding`) so light consumers
  (MOKA, VoxPoser) don't pull DINOv2/clustering.
- **`rekep/` is self-contained** — zero cross-package deps.
- **Faithful ports:** refine comments if you must, but keep logic byte-identical (diverges from upstream's
  comments — acceptable for a shared release, not for upstream-sync).

## Verification
No sim runs needed for a hygiene pass: `ast.parse` for syntax, **container imports** of the IsaacLab-free
modules (they import without booting the sim — `rekep_viz`/`keypoint_tracking`/`utils`/`solvers`/
`keypoint_proposal`/`constraint_generation`, and `isaaclab_helpers`/`grounding` with the IsaacLab source
paths on `sys.path` but no `AppLauncher`), logic-token greps for the ports, and `grep` for stale
references (no `pps_to_rekep`, no hardcoded `rekep/config.yaml` left in live code).

## Left open
- **README for `rekep/`** — the one deferred item (module map + how to run the demos + the faithful-port
  note). Highest-value remaining thing for a receiver.
- The dropped solver terms (SDF collision, IK feasibility) remain the meaningful faithfulness gap for the
  rollout demo — see [[rekep-on-pps]]; unchanged here.
