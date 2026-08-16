# sim_common

Shared IsaacLab infrastructure for the base (`vlm_base`) and native (`dial_mpc`) stacks. It depends on
neither (imports only rekep / isaaclab / torch), so both build on it without coupling: env wrappers,
forward kinematics, scene geometry, the launch harness, frame overlay, the ReKep constraint bridge, and the
grounding layer. A library — imported, not run directly.

## Layout

```
sim_common/
├── runtime.py          # launch harness: sys.path bootstrap + AppLauncher boot + run_standalone wrapper
├── envs/
│   ├── base.py         # IsaacLabEnv: shared single-env wrapper (reset/step, FK, gripper, pose reads)
│   ├── droid.py        # DroidEnv: Droid-Visuomotor tasks (Franka + Robotiq, kitchen scenes)
│   ├── lift.py         # LiftEnv: Franka lift-cube/mug
│   └── lift_mug.py     # registers Isaac-Lift-Mug-Franka-v0
├── grounding/
│   ├── __init__.py     # the Grounding contract + get_source registry
│   ├── gt.py           # ground-truth pick-and-place grounding
│   ├── rekep.py        # ReKep relational-keypoint grounding
│   ├── fake_vlm.py     # per-task canned VLM output (no GPT-4o call)
│   └── masks.py        # GT-mask perception: keypoint / object name -> masked depth points
├── fk.py               # batched PyTorch Franka FK (FrankaFK + world-frame WorldFK)
├── geometry.py         # quaternion -> matrix, USD extents, DEFAULT_EXTENT
├── overlay.py          # camera-frame overlays for rollout videos
├── constraints.py      # numpy -> torch bridge for ReKep constraint code over sampler candidates
└── assets/             # bundled robot assets (Franka URDF)
```

## Grounding

The decoupling point between front-ends and controllers. A `GroundingSource` turns an environment into a
`Grounding` — scene obstacles plus an ordered list of `Stage`s (each: a live target, a gripper intent, and
a task-progress advance flag). Drivers (`vlm_base.base_driver`, `dial_mpc`) consume only this contract, so
front-ends vary independently of control.

| `get_source(name)` | Front-end |
|---|---|
| `gt` | ground-truth two-stage pick-and-place from simulator poses |
| `rekep_fake` | ReKep keypoints + constraints from a canned VLM (no LLM call) |
| `rekep_real` | ReKep keypoints + constraints from GPT-4o |
