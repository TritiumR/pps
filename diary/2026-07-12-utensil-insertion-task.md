# Utensil insertion task: scene build, spatula sourcing, and a motion-planning demo

**Goal.** Add a fifth PPS/IsaacLab manipulation task — insert a utensil into a holder — styled after
the existing four (pot / tea / ice / weight): manager-based env, `config/droid` variants, gym
registration, `__file__`-relative assets. Then produce a simple scripted motion-planning demo of the
spatula going into the holder.

---

## 1. The task package

New package `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/utensil/`
(cloned from `holder`, renamed throughout):

```
utensil_env_cfg.py                              # scene, events, terminations (base)
config/droid/utensil_joint_pos_visuomotor_env_cfg.py   # robot, cameras, positions, randomize, rotations
config/droid/utensil_ik_rel_visuomotor_env_cfg.py      # IK-Rel variant (drives the demo)
config/droid/utensil_{joint_pos,ik_rel}_pointcloud_env_cfg.py
config/droid/__init__.py                        # gym.register(...)
mdp/terminations.py                             # task_done_utensil_inserted
mdp/utensil_events.py                           # deactivate_prim, convex-decomp, etc.
```

Registered task ids:
`Isaac-Utensil-Droid-Visuomotor-v0`, `-Visuomotor-IK-Rel-v0`, `-PointCloud-v0`, `-PointCloud-IK-Rel-v0`.

**Success termination** (`mdp/terminations.py::task_done_utensil_inserted`): utensil xy over the holder
opening + within a height band + gripper released. Thresholds are still **placeholder** (carried over
from the holder-scene scale) — retune to the pen-holder opening + utensil geometry before using success
as a metric.

---

## 2. Assets

| Object | Asset | Notes |
|---|---|---|
| Table  | `assets/ArtVIP/Interactive_scene/kitchen/kitchen.usd` | reused from the **capsule** task's `interactive_kitchen` counter |
| Holder | `assets/pen holder001/model_pen holder001_0.usd` | orange pen-holder crock, scale (1.5,1.5,1.2) |
| Knife  | `assets/knife/knife.usd` | existing ArtVIP asset, scale 1.5 |
| Spatula| `assets/spatula/spatula_physics.usd` | **CC0 Kenney, sourced + converted + grafted** (below) |

**Spatula sourcing.** Probed the Isaac nucleus library first — no built-in spatula (YCB has scissors
only). Pulled a free **CC0 Kenney spatula** (https://poly.pizza/m/VMluJiBnkx); provenance recorded in
`assets/spatula/source/ATTRIBUTION.txt` (CC0 needs none, but kept it). GLB→USD via
`agent_tests/_gltf_to_usd.py` (`omni.kit.asset_converter`) → `spatula.usd`.

### The converted-asset spawn gotcha (the hard part)

The converted `spatula.usd` **would not spawn as an IsaacLab `RigidObject`** — only inside the full
`InteractiveScene`:

```
RuntimeError: Failed to find a rigid body when resolving '/World/envs/env_.*/spatula'.
[Warning] Could not perform 'modify_rigid_body_properties' on any prims under '/World/envs/env_0/spatula'
```

The trap: it's provably fine in isolation. Reproducing IsaacLab's exact spawn path
(`isaacsim.core.utils.prims.create_prim` + `schemas.modify_*`) on a bare stage composes `RigidBodyAPI`
onto the spawn prim and succeeds — identical to the working knife. Flattening, clearing `instanceable`,
adding both `UsdPhysics.*` and `PhysxSchema.*` APIs, and clearing `kind='component'` on the root **all
failed**; the failure reproduces only in the full scene (cloner / robot / physics context) and I could
not root-cause it from the IsaacLab source. (~30 Isaac boots sunk here — don't repeat that.)

**Fix that works — graft geometry onto a proven skeleton** (`agent_tests/_fix_spatula.py`): `knife.usd`
spawns correctly in the very same scene, so copy it and swap only the mesh's
`points`/`faceVertexCounts`/`faceVertexIndices` for the spatula's; drop the source primvars/normals
(wrong element counts), reset extent, rename the root `/kitchen_knife`→`/Spatula`, re-point the default
prim. You inherit the working asset's exact authoring. Two follow-on fixes in the same script:

- **Zero the skeleton's `geometry`/`mesh` xformOps** after grafting — the knife bakes an orientation
  there that the grafted geometry otherwise inherits (stood the spatula upright at z=3.665).
- **Bake orientation into the points**, not an init-rotation: the converted mesh is Y-up (thin axis =
  Y), so rotate points +90° about X → thin axis world +Z (flat). An init-rot would be clobbered by the
  scene's yaw-randomize-about-Z reset; baking keeps it flat like the knife. Also unbind the knife
  material and set a neutral grey `displayColor` (the grafted mesh renders black otherwise — the knife
  material samples through the dropped UVs).

Final scale `(0.004, 0.004, 0.004)` → ~27 cm length. `spatula.usd` (source) + `spatula_physics.usd`
(used) both kept.

---

## 3. Scene: kitchen-counter layout + utensil orientation

Iterated through **kitchen table → plain SeattleLabTable → kitchen counter**. The plain SeattleLabTable
is a `Props/Mounts/` bench with an ugly built-in equipment post, so per user feedback we reused the
**capsule task's kitchen counter** (clean granite): `interactive_kitchen` at `ROOM_INIT_POS=[-4.3,-0.8,
-0.6]`, capsule machine + can not spawned, oven deactivated. Robot mounts at the counter at
`(3.0, 1.9, 0.2)` facing **−y** (`ROBOT_INIT_ROT=(0.7071,0,0,-0.7071)`), mirroring the capsule geometry.

Object layout on the counter (surface z≈0.20), relative to the robot: **holder** center `(2.9,1.35)`,
**knife** right `(2.7,1.35)`, **spatula** left `(3.1,1.35)`.

**Utensil orientation** (rotate both 90° so handle tips point back at the robot, +y). The handle end of
each was found **deterministically from mesh geometry** (`agent_tests/_handle_dir.py`: handle = the
narrow cross-section end), not guessed from renders — they're on opposite local-X ends:

- knife handle at local −X → yaw **−90°** points it toward +y.
- spatula handle at local +X → yaw **+90°**.

Set via the per-utensil `*_RANDOMIZE_POSE_RANGE` yaw ranges in the droid config.

Verify render: `agent_tests/_scene_video.py --task Isaac-Utensil-Droid-Visuomotor-v0 --exp_name <x>`
→ `results/vlm_mpc/scene/<x>.mp4`.

---

## 4. The motion-planning demo

`agent_tests/_utensil_insert_demo.py` — GT-driven, scripted IK on the **IK-Rel** task, no VLM
front-end. Reuses `moka/isaac_control.py` (`drive_to_pose`, `descend_to_contact`, `hold_gripper`).
Videos → `results/utensil_demo/`. Best run: `spatula_insert8.mp4`.

**Recipe (phases):**

1. **Grasp** — read GT spatula pose; grasp the **solid handle** (`grasp_handle_off` toward the handle),
   fingers closing **across** it (natural rod grasp). Hover → `descend_to_contact` (open) → close &
   settle (90 steps) → **gentle stepped lift** (8 small increments).
2. **Reorient** vertical, handle-down: rotate the held pose −90° about world X (spatula length +y → −z).
3. **Carry** over the holder, correcting for the reorient's horizontal tip offset
   (`hold_x − handle_dir_x·d_tip`) so the tip centres on the crock opening.
4. **Insert** — `descend_to_contact` (closed) lowers gently until the tip meets resistance.
5. **Release + retract** (gentle stepped lift while open).

**Hard-won recipe lessons** (each was a failure mode, ~9 render iterations):

- **Grasp the solid handle, not the slotted head** — top-down fingers fall straight through the head's
  slots. (User's insight.)
- **Gentle stepped lift**, not one fast lift — a jerky lift shakes the thin flat handle out of the
  fingers. (User's insight.) This is what finally made the grasp *hold*.
- **Wrap the grasp yaw to [−90°, 90°]** (parallel gripper is 180°-symmetric) — a raw 253° command
  exceeded the panda wrist limit and the IK tilted the gripper.
- **`descend_to_contact` for the insert**, never a forced `drive_to_pose` to a fixed z — the latter
  slams and flings the light spatula/holder off the counter.
- **Utensil mass 10 g → 50 g** (`utensil_env_cfg.py`) — at 10 g the spatula fled the gripper on
  contact. 50 g is also more realistic.

Parametrized for tuning: `--grasp_handle_off --grasp_yaw_off --lift_steps --lift_h --reorient_z
--hover_z --insert_z`.

---

## 5. Remaining gaps (paused here, by agreement)

The pipeline works **through insertion** — `spatula_insert8` grasps, reorients, carries, and lowers the
handle so the tip enters the crock. What's not solved:

- **Release + retract** carries the spatula back up with the open fingers instead of leaving it
  standing in the holder.
- **Run-to-run variance** from the per-reset randomization: some runs catch the crock rim and fling the
  spatula off the counter.

Likely fixes when revisited: fix the initial pose for the demo (deterministic), insert a bit deeper so
the crock walls support the spatula before releasing, and/or clear the gripper sideways before lifting.
The grasp of a thin, flat object flush on a table is the fundamental limitation (top-down jaws can only
pinch thin edges; you can't get a finger under a flush object) — consistent with the documented ReKep/
MOKA "grasp doesn't hold" experience. Also still open: retune the success thresholds; only the knife is
wired as a grasp/place target so far.
