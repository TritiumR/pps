# Plan (rough): Run ReKep's front-end on PPS's tasks

**Status:** draft / scoping. Companion to [GOAL.md](GOAL.md).

## 1. Goal & scope

Run **only ReKep's grounding front-end** — keypoint proposal + VLM constraint generation — on the
**tasks/environments used by PPS** (IsaacLab), to produce, per task: a set of 3D keypoints and the
GPT-4o-written relational constraints (sub-goal + path).

This is "Approach A". Explicitly **out of scope** (for now):

- ReKep's own controller/solvers (`subgoal_solver`, `path_solver`, `ik_solver`, `environment.py`
  exposed functions) — i.e. running ReKep closed-loop in IsaacLab. That's "Approach B" and is
  discarded anyway under VLM-DP (DIAL-MPC replaces the solver).
- The PPS steering algorithm.
- Porting scenes to MuJoCo/MJX and running DIAL-MPC (a separate, later lift).

**Why do this:** (a) validate that ReKep grounds PPS's task semantics well, and (b) produce the
cost spec (keypoints + constraints) that VLM-DP would consume, **on the same task suite PPS uses** —
the alignment needed to eventually steer VLM-DP with PPS.

## 2. Background (one-liners)

- **ReKep front-end is sim-agnostic.** [keypoint_proposal.py](keypoint_proposal.py) `get_keypoints(rgb, points, masks)`
  (DINOv2) and [constraint_generation.py](constraint_generation.py) `generate(img, instruction, metadata)`
  (GPT-4o) take plain numpy arrays + a text instruction. Nothing OmniGibson-specific is required to
  *generate* the cost; the OmniGibson coupling lives in `environment.py`, which we are NOT using here.
- **PPS provides everything ReKep needs.** IsaacLab/PhysX, Franka + Robotiq gripper. RGB-D cameras
  with intrinsics (table + wrist), an instance-segmentation point-cloud variant, and ground-truth
  object poses. Evaluated tasks: `pot` (lid+egg), `weight` (fruit on scale), `tea` (pour teapot→cup),
  `capsule` (coffee-maker pod); ~33 in the suite (incl. `pick_place`, `stack`, `lift`, `pen`).

## 3. What maps cleanly vs. what we build

| ReKep front-end needs | PPS provides | Work |
|---|---|---|
| `rgb` (H,W,3) | table `exterior_image_1_left` (640×360) | none |
| per-pixel world `points` (H,W,3) | depth + camera intrinsics + camera prim pose | **build** — lift depth→world (reuse [og_utils.py](og_utils.py) `pixel_to_3d_points`) |
| `masks` (H,W semantic) | `instance_id_segmentation_fast` (pointcloud cfg) | **enable + remap** instance ids → ReKep mask format |
| `instruction` | `task_prompts.json` | none |
| workspace `bounds_min/max` | — | **set** for the PPS table (replaces pen-scene bounds in `configs/default.yaml`) |
| keypoint→object association (for later tracking) | GT object poses + instance ids | **easy** (easier than ReKep's mesh-sampling tracker) |

## 4. Where it runs

Two options; **recommend (b) for the first pass:**

- **(a) In-process**: run ReKep's proposer/VLM inside the IsaacLab env. Cleanest long-term, but
  entangles ReKep deps with the IsaacLab/Isaac-4.x stack.
- **(b) Decoupled / offline (recommended first):** dump a single observation frame
  (`rgb`, `depth`, `instance_seg`, camera `intrinsics`, camera `extrinsics`, object poses) from the
  IsaacLab task at reset, then run ReKep's front-end **offline** on that frame — in the working ReKep
  Docker we already built (DINOv2 + GPT-4o confirmed working there). Avoids the Isaac-version
  entanglement entirely and reuses known-good infra.

## 5. Steps (incremental)

0. **Pick one task: `tea`** (pour teapot → cup). It overlaps the `teapot_bowl` work already done, so
   we have a reference for sensible keypoints/constraints.
1. **Obs dump from IsaacLab.** In the `tea` env config, enable `distance_to_image_plane` (depth) and
   `instance_id_segmentation` on the table `CameraCfg` (cameras already defined). At reset, save one
   frame: rgb, depth, instance-seg, intrinsic matrix, camera world pose, and the GT object poses +
   their instance ids. (One config edit + a small dump script.)
2. **Obs adapter** (`pps_to_rekep.py`, ~50 lines): build per-pixel world `points` from
   depth+intrinsics+extrinsics (mirror [og_utils.py](og_utils.py) `pixel_to_3d_points`, incl. the
   camera-convention axis flip), and remap instance-seg → ReKep's `masks`. Output `(rgb, points, masks)`.
3. **Keypoint proposal.** Run `KeypointProposer.get_keypoints` on the adapter output. Tune
   `bounds_min/max`, `min_dist_bt_keypoints`, `max_mask_ratio` for the PPS table. Inspect the numbered
   keypoint image.
4. **Constraint generation.** Run `ConstraintGenerator.generate` with the `tea` instruction → stages +
   sub-goal/path constraint `.txt` files + `metadata.json`. (Already working in our Docker.)
5. **Sanity check + keypoint→body map.** Confirm keypoints land on the right parts (teapot grasp,
   spout, cup opening) and the constraints reference the right indices. Record which IsaacLab object
   body each keypoint sits on (instance id + nearest GT pose) for later state-readout.
6. **(optional) Breadth.** Repeat 1–5 for `pot`, `weight`, `capsule` → open-vocab sanity (GOAL.md
   Direction 3) on PPS's suite.

## 6. Deliverables

- `pps_to_rekep.py` adapter (PPS obs → ReKep `get_keypoints` inputs).
- Per task: keypoint image, constraint `.txt` files, `metadata.json`, keypoint→body mapping.
- Short writeup: did ReKep ground each task? keypoint quality, constraint coherence, failures.

## 7. Risks / open questions

- **Workspace bounds**: PPS table geometry differs from the pen scene; keypoint bounds-filtering must
  be re-set or good keypoints get dropped (we saw this on `apple_plate`).
- **Mask granularity**: are task-relevant *parts* (pot lid, teapot spout, cup opening) separable in the
  instance seg, or only whole objects? Object-level masks may under-segment the parts ReKep wants;
  may need finer masks or rely on DINOv2 clustering within an object mask.
- **Camera extrinsics / frame conventions**: get the IsaacLab camera prim world pose and the
  depth→world convention right (og_utils applies an axis flip `T_mod`). Easy to get a mirrored/2× cloud.
- **Resolution**: 640×360 → DINOv2 resizes to a patch multiple internally; fine, but verify.
- **VLM**: use `gpt-4o` (config already switched; key configured). `chatgpt-4o-latest` is not accessible.
- **Tracking is deferred**: we only need initial keypoints for cost *generation*. Live keypoint readout
  during a rollout is a VLM-DP/MJX concern, not this experiment — but the keypoint→body map we record
  is what that later step will use.

## 8. Success criteria

- **Minimum**: for `tea`, ReKep yields a sensible keypoint set (covers teapot grasp + spout + cup
  opening) and GPT-4o writes coherent multi-stage constraints referencing the right keypoints.
- **Stretch**: all 4 evaluated PPS tasks grounded with no code changes between tasks (only the
  instruction changes) — demonstrating open-vocab breadth on the PPS suite.

## 9. Heads-up beyond this experiment

PPS lives in **IsaacLab/PhysX with a trained openpi base**; VLM-DP is meant to live in **hydrax/MJX
with a weightless DIAL-MPC base** (GOAL.md). This experiment yields the *cost* for PPS's tasks, but
running *VLM-DP* on them still needs those scenes rebuilt in **MJX** — tracked separately, not blocking.
