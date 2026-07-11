# VLM manipulation front-ends on PPS/IsaacLab — progress snapshot

**2026-06-23** — First diary entry. State of the program to port ReKep, MOKA, and VoxPoser
into the PPS/IsaacLab environment and run them on the 4 tasks (tea, pot, weight, capsule).

---

## 1. Progress

**All three front-ends are ported and run end-to-end** inside the Isaac Sim 5.0 Docker
(`docker/`), reusing a shared IsaacLab adapter layer (camera → world points + GT instance
masks, IK-Rel control, H.264 video).

- **ReKep** (`rekep/`) — DINOv2 + GPT-4o keypoint proposal + constraint generation + subgoal
  solver. Front-end (keypoints/constraints/overlays) validated on all 4 tasks. Closed-loop
  solver rollout drives the Franka via IK-Rel.
- **MOKA** (`moka/`) — GroundedSAM (GroundingDINO + SAM) mark-based visual prompting + GPT-4o.
  Front-end (bbox/grid/marks, 3D lift) validated; phase-machine rollout runs.
- **VoxPoser** (`voxposer/`) — LLM-composed dense 3D voxel value maps + greedy planner.
  Both milestones run: plan-only value maps (cost field is dense + monotonic — the headline
  finding) and `--execute` IK-Rel rollout. Multi-camera rig + GT-mask point clouds.

**Pot scene "floating lid" bug — diagnosed and fixed.** The pot/cover wrap *existing nested
kitchen prims* (`usd_path=""`), so their PhysX bodies spawn ~0.52m off the room-shifted USD
geometry; the kinematic pot rendered at its mesh while the dynamic lid was driven to its
(wrong) physics pose. Reset-time pose writes are silently dropped for these nested bodies, but
mid-episode writes stick — so the fix is `pot_scene_fix.seat_pot_lid(env, hold)`, called after
the post-reset settle in all 6 front-end entry points (no-op on non-pot tasks); `init_state`
added to `pot_env_cfg.py` to declare the target. Validated: lid↔pot XY separation 0.52m → 0.066m,
lid seated on the pot. Kept to the front-ends — the shared task wasn't changed.

**Distractor disambiguation fix.** In the pot scene `"pot" in prim` matched `model_potted_plant1`
(1355 instance ids smeared across the kitchen). Now both VoxPoser (`name2ids`) and ReKep
(`grounding.task_object_ids`) match each object's *exact* prim subtree.

**All pot deliverables regenerated with the seated lid** across the three ports: VoxPoser value
maps (4-cam) + `--execute` rollout video; ReKep keypoints + keypoint-tracking video + rollout;
MOKA front-end marks + rollout. Outputs under `results/{voxposer,rekep,moka}/pot/`.

**Spot-checked tea / weight / capsule** (per-object physics-link vs rendered-centroid gap +
eyeballed renders): all clean (gaps <0.07m, nothing floating). The lid bug is **pot-only** —
the other tasks' objects are file-spawned, so `init_state` applies at spawn and physics matches
the visual.

---

## 2. Roadblocks

**The rollouts run but do not complete the task.** All three pot rollouts end with the lid still
on the pot and the egg unplaced. Investigation (final frames + per-step debug JSONs + logs)
found several compounding causes — these are **pre-existing limitations, not regressions from
the lid fix**:

1. **IK-Rel descent/reach stalls.** The position-primary controller (`moka/isaac_control.drive_to_pose`)
   clips per-axis deltas and doesn't prioritize Z, so reaching the pot (far in −x) makes the EE
   *rise* instead of descend — it ends ~15cm too high. ReKep stage-1 grasp: z 1.07→1.13 vs target
   0.96, err 0.19m. MOKA `reach_grasp`: moved only 0.035m, err 0.20m. Free-space moves reach fine.
2. **MOKA perception 3D-lift error.** The lid-handle grasp pixel deprojects to z=0.743 (cooktop
   level) instead of ~0.96 (where the function keypoint correctly lands) — grasps empty cooktop.
3. **ReKep subgoal solver non-convergence.** Stage 3 (grasp egg) returned the current EE pose as
   the target (cost 371; target `[7.65,…,1.13]` vs egg `[8.40,…,0.79]`) — the consistency term +
   a hard constraint landscape keep it from reaching the egg. The arm never goes to the egg.
4. **Object physics on the nested prims.** `modify_mass_properties` fails on `E_pot1_1`,
   `E_cover_2`, and `egg` — grasp-and-hold may misbehave even when a target is reached (the
   originally-documented "grasp doesn't hold").

   VoxPoser is the exception on reaching — it descends its cost-field waypoints well
   (dist-to-target → 0.02) — but the grasp/lift still doesn't take, so the task is incomplete.

**VoxPoser `--execute` is slow.** IsaacLab renders every scene camera each step, so the 4 rig
cameras over the ~220-waypoint multi-subtask pot rollout time out (>30 min). Workaround:
`--num_cams 1` (~4s/waypoint) for the video, then a 4-cam plan-only run to restore the
full-fidelity value maps. The video quality is unaffected (records table_cam + the value-map panel).

---

## 3. To-Dos

In rough priority order (manipulation correctness first):

- [ ] **Fix the IK-Rel descent control** (roadblock #1) — blocks both ReKep and MOKA. Likely
      tunable: descend-then-extend sequencing, Z-priority in the delta, or a larger step budget.
      First confirm the workspace-edge hypothesis with a small reach test (drive the EE straight
      down at the pot's x,y and log where it stalls).
- [ ] **Fix MOKA's grasp deprojection** (roadblock #2) — the grasp keypoint should lift to the
      handle, not the cooktop behind it.
- [ ] **Fix the ReKep subgoal solver fallback** (roadblock #3) — don't return the init guess
      when a constraint is grossly violated; reach the actual target keypoint.
- [ ] **Confirm the lid/egg are graspable dynamic bodies** (roadblock #4) — investigate the
      `modify_mass_properties` failures on the nested prims.
- [ ] **Validate rollouts on tea / weight / capsule** — the reach/solver issues likely affect
      them too; only the pot rollouts have been examined in depth.
- [ ] **Decide on committing the work** — there's a large uncommitted pile on `main` (the
      `voxposer/`, `rekep/`, `moka/`, `docker/` ports + the lid fix). Would branch first and
      decide what belongs in the commit(s).

**Open question:** how much of the "doesn't complete the task" gap is control/perception
(fixable here) vs. a fundamental limit of these front-ends' open-loop reaching on contact-rich
grasps in this sim (the research framing — front-ends propose, but closing the grasp is hard).
