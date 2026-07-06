# vlm_mpc — DIAL-MPC controller + task drivers

Sampling-MPC (DIAL) over geometric/affordance costs, run on IsaacLab manipulation tasks. The library
is shared and stable; task drivers are minimal modules dispatched through a single entry point.

## Running

```bash
# from the repo root, inside the pps Docker (headless EGL):
docker compose exec -T -e DISPLAY= pps /isaac-sim/python.sh vlm_mpc/main.py --task <name> [task args]
```

`main.py --task <name>` boots Isaac once, then runs `tasks/<name>.py`. List of tasks = the `TASKS`
registry in `main.py`. Examples:

```bash
vlm_mpc/main.py --task droid_weight --vlm real --exp_name weight_realvlm_s0   # ReKep weight task on DIAL
vlm_mpc/main.py --task droid_weight_free                                       # scaffold-free 4-stage weight task
vlm_mpc/main.py --task droid_grasp --mode cost_only --object pear              # single-object grasp experiments
vlm_mpc/main.py --task droid_grasp --mode {cost_only,cost_gripper,collab,scaffold}
vlm_mpc/main.py --task franka_lift --mode pick                                 # early Franka lift-cube rungs
vlm_mpc/main.py --task rekep_pick --ground vlm                                 # ReKep mug grasp (gt|vlm)
vlm_mpc/main.py --task fk_sanity --n 64                                        # FK consistency diagnostic
```

## Layout

```
# ── shared library (stable; import, don't fork) ──
costs.py            geometric cost terms (make_grasp_cost, make_rekep_cost, fixed_reach, ...)
sampler.py          the DIAL accel sampler (make_accel_sampler)
fk.py               FrankaFK
np_shim.py          torch<->numpy shim for ReKep constraint code
voxposer_bridge.py  VoxPoser affordance field -> DIAL cost
weight_fake_vlm.py  GT-mask "fake VLM" + perception helpers (local_centroid, object_for_keypoint)
droid_env.py        DroidEnv (panda + Robotiq, weight kitchen task)  + WorldFK
isaac_env.py        LiftEnv (Franka lift cube/mug)
lift_mug_task.py    registers Isaac-Lift-Mug-Franka-v0

# ── shared driver runtime (the boilerplate every driver used to repeat) ──
runtime.py          bootstrap_syspath / add_launcher_args / boot (AppLauncher)
overlay.py          camera_overlay_frame, plain_frame (frame capture)
control.py          receding(), hold_pose(), tt() (execution-loop helpers)

# ── task drivers (minimal: add_args(parser) + run(args)) ──
main.py             --task dispatch (the single entry point)
tasks/
  droid_weight.py        ReKep weight task on DIAL (--vlm fake|real)
  droid_weight_free.py   scaffold-free 4-stage weight task (GT grounding)
  droid_grasp.py         single-object grasp experiments (--mode cost_only|cost_gripper|collab|scaffold)
  franka_lift.py         Franka lift-cube (--mode reach|pick)
  fk_sanity.py           FrankaFK vs IsaacLab pose consistency check (diagnostic)
  rekep_pick.py          ReKep-cost mug grasp (--ground gt|vlm)
  rekep_frontend.py      ReKep front-end only (keypoints + GPT constraints; plan-only, no video)
  voxposer_pick.py       VoxPoser affordance field -> DIAL cost, mug pick (+ cost-field side panel)
  voxposer_frontend.py   VoxPoser LMP front-end only (affordance maps; plan-only)

archive/            the 14 original standalone "_" drivers, kept for reference/comparison (never deleted)
agent_tests/        staging for NEW throwaway "_" experiments before they're refactored into tasks/
```

## Conventions

- **A task module** has only light top-level imports (stdlib); everything that touches IsaacLab/torch
  goes inside `run(args)`, which `main.py` calls *after* the Isaac app boots. CLI args are declared in
  `add_args(parser)`. To add a task: write `tasks/<name>.py` + register it in `main.py`'s `TASKS`.
- **New exploratory scripts** start as `_`-prefixed throwaways in `agent_tests/` (package-scoped probes;
  the top-level `agent_tests/` is for cross-cutting ones). Once an experiment is confirmed worth keeping,
  refactor it into a proper `tasks/<name>.py` driving the shared library — don't leave standalone `_`
  scripts in the package root.
- **Superseded drivers** move to `archive/` (kept on disk so A/B comparisons and revisits stay trivial),
  never deleted.

See `results/vlm_mpc/README.md` for the output videos and `diary/` for the findings behind each task.
