# `vlm_dp` — VLM-DP integration package (design)

Combine **our ReKep VLM geometric cost** + **the MBD sampling-MPC** + **Yixuan's score-space steering**
into one reproducible pipeline. This doc is the *target shape* to eyeball **before any code moves** — no code
has been moved yet.

---

## 1. The one architectural fact

Both entry points already sit on **one frozen engine, `sim_free_mpc`**. `eval_steering.py` imports *zero*
`sim_common` modules — it is a fully parallel shell (4038 lines, no `main()`, boots IsaacLab at import) that
duplicates env/rollout/success/video and shares *only* the engine. So integration is **not "merge two
engines"** — it is: **run our ReKep cost + her steering through the same engine, and consolidate the glue.**

The lever: the base score both harnesses steer is produced by *whatever cost object is on the planner*
(`_cost_active_samples -> self.cost`, `planner.py:349`). Put **our `CompositeCost` on the planner** → her
`combine_scores` steering works over our ReKep cost with **zero new steering code**.

---

## 2. Principles

1. **`vlm_dp` OWNS the integration domain** (cost, task representation, world model, the bridge glue) — the
   stuff currently smeared across `sim_common/` + `vlm_base/`.
2. **`vlm_dp` DEPENDS on the stable cores** (`rekep`, `sim_free_mpc`, steering, perception) — absorbing those
   would *re-create* the sprawl.
3. **Move + repoint, never copy.** `sim_common`/`vlm_base` also feed the kept legacy `vlm_base/main.py`. Moving
   code into `vlm_dp` + repointing the legacy imports = **no duplication, nothing deleted** (legacy runs on as
   a thin caller of `vlm_dp`).
4. **Phase-sequenced.** Phase-1 ships a bridge importing from *current* locations; Phase-2 does the
   move-and-repoint after the seam is proven. Never refactor blind.

---

## 3. Target file tree (annotated)

`[new]` = written fresh · `[move]` = relocated from an existing file (source noted) · `[dep]` = external, unchanged

```
vlm_dp/
  __init__.py                 [new]
  README.md                   [new]   one-command repro + what this is
  DESIGN.md                   [new]   this doc

  # ── the integration domain (OWNED) ──────────────────────────────────────────
  cost/
    __init__.py               [new]
    composite.py              [move]  CompositeCost + CostInputs        <- vlm_base/base_cost.py:56/68
    terms.py                  [move]  registered TERMS (rekep_subgoal/  <- vlm_base/cost_terms.py
                                      rekep_path/tip_z/yaw/center_region/aperture/straddle/close_gripper/
                                      carry_hold/place_descent)
    constraints.py            [move]  make_torch_constraint shim        <- sim_common/constraints.py:122
                                      (rekep constraint-as-cost, TorchNumpyShim)

  grounding/
    __init__.py               [move]  Grounding/Stage/SceneObject +     <- sim_common/grounding/__init__.py:73,76
                                      get_source registry
    gt.py                     [move]  GTGrounding (privileged)          <- sim_common/grounding/gt.py:39
    rekep.py                  [move]  RekepGrounding (keypoints +       <- sim_common/grounding/rekep.py:43
                                      constraint stages); depends on rekep/
    masks.py                  [move]  mask helpers (rekep_real only)    <- sim_common/grounding/masks.py

  world.py                    [move]  WorldModel / GTWorld / SensedWorld <- sim_common/world.py
  env.py                      [move]  EnvView.attach(raw_env): read     <- sim_common/envs/droid.py (accessor half)
                                      accessors q0()/tcp()/object_pose()/fk/rgb/cam over eval_steering's
                                      RAW IsaacLab env (eval_steering feeds full 8-dim actions to env.step,
                                      so this is read-mostly; no apply_arm path)

  # ── the bridge (THE seam, the only genuinely new logic) ─────────────────────
  bridge.py                   [new]   VlmDpBridge: attach_cost(mpc) + reset(env) + advance(flags) + context()
                                      — orchestrates the three below
  context.py                  [new]   build_context(env, obs, grounding, stage) -> cost-context dict.
                                      *THE crux*: unifies _ctx_from_stage (base_driver.py:53, world-frame +
                                      body_pos_w) and eval_steering.build_mpc_context (env-origin-relative),
                                      committing to ONE frame convention.
  stage.py                    [new]   advance_stage(idx, stage, flags): flag-based phase advance, reusing the
                                      _should_advance semantics (base_driver.py:73-77 already advances on
                                      stage.done_flag) — NO coarsen, NO release-advance.

  # ── reproducibility + entry (Phase-2 niceties; add when handoff/2nd-entry warrants) ──
  manifest.py                 [new]   typed Manifest + check() preflight (base ckpt, task/ref proxies,
                                      dinov3, IsaacLab assets, task_prompts entry, transformers/gcsfs)
  config.py                   [new]   typed run-config from one YAML
  run.py                      [new]   `python -m vlm_dp.run --config configs/weight_gt.yaml`
  configs/
    weight_gt.yaml            [new]   Increment-1: weight, ground=gt, cost=composite, base-only
    weight_rekep_steer.yaml   [new]   full: rekep front-end + task/full steering

  tests/
    test_context.py           [new]   CPU: frame convention on a known GT scene (the crux risk)
    test_cost_contract.py     [new]   CPU: CompositeCost returns [N] 1-D; cost_style routing assertion
    test_steering_identity.py [new]   CPU: combine_scores steer_scale=0 -> bit-identical base
```

### Depends on — unchanged, external cores
```
rekep/                        [dep]  VLM front-end: keypoint proposer/tracker, GPT-4o constraint generation
sim_free_mpc/                 [dep]  FROZEN engine: SimFreeMPC, DIAL, decode_model_action_chunks, FK,
                                     step_/estimate_ methods, score_steering.combine_scores
openpi (proxy_score, ...)     [dep]  the DINOv3-image score proxies + policy_config.create_trained_policy
sim_common/perception*        [dep]  perception.py / visual_tracker / grasp_sensor (real-perception only;
                                     needed for rekep_real / --state real). Kept separable, absorb later if ever.
IsaacLab                      [dep]  env task defs, AppLauncher
```

### Legacy — repointed, NOT deleted
```
vlm_base/main.py, base_driver.py, sim_free_core.py, grasp_flow_cost.py
   -> keep as the old rollout path; change their imports to `from vlm_dp.cost import CompositeCost`,
      `from vlm_dp.grounding import get_source`, `from vlm_dp.world import GTWorld`, etc.
   -> after the repoint, vlm_dp depends on NOTHING in vlm_base.
```

### The `eval_steering.py` seam
```
Phase-1: ~15 lines — add --vlm_cost {none,gt,rekep_fake,rekep_real}; when set, build the bridge at
         env.reset and overwrite mpc.cost + supply bridge.context() at the 2 mpc_context sites.
Phase-2: give eval_steering a main(args) / run_rollout() boundary so vlm_dp/run.py can call it
         (extraction only — behavior-preserving; still not moved into vlm_dp).
```

---

## 4. Dependency graph (target)

```
                 ┌──────────── vlm_dp ────────────┐
                 │ cost/  grounding/  world  env  │  (owns the integration domain)
                 │ bridge  context  stage         │
                 │ manifest config run tests      │
                 └───┬─────────┬─────────┬────────┘
       depends on →  │         │         │
            ┌────────▼──┐  ┌───▼──────┐  ▼ ┌──────────────┐
            │  rekep/   │  │sim_free_ │    │ openpi        │
            │ (VLM      │  │mpc/      │    │ proxy_score + │
            │ front-end)│  │(FROZEN   │    │ score_steering│
            └───────────┘  │ engine)  │    └──────────────┘
                           └──────────┘
   eval_steering.py  ──(harness: env boot, rollout, success, video)──> uses vlm_dp via ~15-line seam
   vlm_base/main.py  ──(legacy rollout)──> imports cost/grounding/world FROM vlm_dp  (repointed)
```
No `vlm_dp -> vlm_base` edge. No `vlm_dp` / `sim_common` duplication (sim_common's reusable parts *become*
vlm_dp; only `perception*` stays in sim_common as a dependency).

---

## 5. Migration table (source → destination → who repoints)

| source (today) | → vlm_dp destination | repoint these callers |
|---|---|---|
| `vlm_base/base_cost.py` (`CompositeCost`, `CostInputs`) | `vlm_dp/cost/composite.py` | `vlm_base/main.py`, `vlm_base/grasp_flow_cost.py` |
| `vlm_base/cost_terms.py` (`TERMS`, `@register`) | `vlm_dp/cost/terms.py` | `vlm_base/base_cost.py` importers |
| `sim_common/constraints.py` | `vlm_dp/cost/constraints.py` | `cost_terms` (rekep_subgoal/path) |
| `sim_common/grounding/__init__.py` (`Grounding`,`Stage`,`get_source`) | `vlm_dp/grounding/__init__.py` | `vlm_base/main.py`, `base_driver.py` |
| `sim_common/grounding/gt.py` (`GTGrounding`) | `vlm_dp/grounding/gt.py` | `get_source` registry |
| `sim_common/grounding/rekep.py` (`RekepGrounding`) | `vlm_dp/grounding/rekep.py` | `get_source` registry |
| `sim_common/world.py` (`GTWorld`,`SensedWorld`) | `vlm_dp/world.py` | `vlm_base/main.py`, `base_driver.py`, grounding |
| `sim_common/envs/droid.py` (accessors) | `vlm_dp/env.py` (`EnvView`) | `vlm_base/main.py` (legacy keeps `DroidEnv` too, or aliases) |
| `vlm_base/base_driver.py` helpers `_ctx_from_stage`/`_should_advance`/`_capture_held` | `vlm_dp/context.py` + `vlm_dp/stage.py` | `base_driver.run_base` imports them back |
| `vlm_base/sim_free_core.py::guard_cost` | `vlm_dp/cost/__init__.py` (or `bridge.py`) | `base_driver`, `sim_free_core` |

**Not moved (stay put):** `rekep/*`, `sim_free_mpc/*`, `openpi/*`, `sim_common/perception.py`,
`sim_common/geometry.py`, `sim_common/runtime.py`, `sim_common/grasp_sensor.py`, `sim_common/visual_tracker.py`.

---

## 6. Key interfaces (contracts the bridge honors)

- **Cost contract** (unchanged): `cost(*, real_actions:[N,H,8], ee_pos:[N,H,3], ee_quat:[N,H,4], context:dict)
  -> Tensor[N]` (lower=better, **1-D**). Routed by `cost_style` (`planner.py:347-349`): our `CompositeCost`
  needs **`cost_style='priority'`** (the `ee_pos=` convention). A `grasp_flow`-family value silently calls it
  with `tcp_pos=` → breaks.
- **Bridge API:** `VlmDpBridge(task, ground, state, cost_cfg)`; `reset(env) -> builds world+grounding at
  env.reset`; `context(env, obs) -> dict`; `advance(flags) -> maybe bump stage`; `attach_cost(mpc)`.
- **Context / frame convention (the crux):** one builder, one frame. Decide **world-frame + `body_pos_w`**
  (our `_ctx_from_stage`) *or* **env-origin-relative + ee-frame-sensor root** (eval_steering) — the cost
  compares FK-ee to object positions, so a mismatch mis-optimizes silently. Unit-tested first.
- **Stage advance:** `_should_advance` (`base_driver.py:73-77`) already returns `flags[stage.done_flag]` — so
  advance on the env `subtask_terms`. No `coarsen_grounding`, no `released`.
- **Steering (rides free):** base score from `estimate_mbd_score_action_prox_terms` (`planner.py:885`) over
  *our* cost; `combine_scores(base, task, mode, steer_scale, ref)` (`score_steering.py:11`) unchanged.
  Requires `--mpc_update mbd_score_action_prox` (not the default `score_space`).

---

## 7. Risks (carried from the review — the things that make this real, not slop)

1. **Frame mismatch (top).** world-frame vs env-origin-relative context. Resolve in `context.py`; unit-test
   before wiring the rollout.
2. **`cost_style` trap.** Must be `'priority'`; assert it in `attach_cost`.
3. **`transformers` version conflict.** `rekep_real` (GroundedSAM) needs `4.44.2`; `ProxyScorePytorch`
   hard-gates `4.53.2`. → ship steering on `--vlm_cost gt` first; split envs (or resolve) for
   real-perception + steering.
4. **Steering needs `mbd_score_action_prox`**, not `score_space`.
5. **Keypoint reproducibility** (rekep path): deterministic proposer (re-seed + `cudnn.deterministic` +
   `MeanShift n_jobs=1`) or VLM constraints reference the wrong points.

---

## 8. Phase sequencing

- **Phase-1 (ship Increment-1, ~1 new file + ~15-line seam):** `vlm_dp/bridge.py` (+ `context.py`) importing
  cost/grounding/world **from their current locations**. Command:
  `eval_steering --task weight --vlm_base --vlm_cost gt --mpc_cost priority --mpc_update score_space`.
  Tests: `test_context.py` (CPU) + a 1-seed GPU smoke. **This also fixes our base competence** (flag-based
  advance replaces the coarsen under-reach).
- **Phase-1b:** `--vlm_cost rekep_fake` → `rekep_real` (front-end).
- **Phase-1c:** `--task_steer --mpc_update mbd_score_action_prox` (steering over our cost, zero new code).
- **Phase-2 (consolidate):** execute the migration table (move + repoint), add `manifest`/`config`/`run` +
  the `eval_steering.main()` boundary. Destination shape = this tree.

---

## 9. Eval + reproducibility hooks (see the eval-matrix)

- **Paired, N≥30, mean±std** (the N=20 lesson): `reset_to(same initial_state)` so base vs steer see identical
  scenes; per-seed rescue/break table; report fixed-scene and randomized-scene separately; decompose PhysX vs
  sampler variance; pin `PYTHONHASHSEED`/`cudnn.deterministic`/per-seed seeds.
- **`manifest.check()`** fails loud on any missing artifact (we hit those walls all session).
- **CPU regression tests** in `tests/` gate the contracts (cost shape, `cost_style` routing, steering-off
  identity).

---

## 10. Open questions

- Frame convention: adopt eval_steering's (env-relative) so the harness stays untouched, or world-frame (our
  cost's native) and translate in `env.py`? (Decide in `test_context.py`.)
- `env.py`: absorb `DroidEnv` wholesale, or a slim read-only `EnvView.attach` over the raw env? (Lean =
  `EnvView`.)
- `perception*`: keep in `sim_common` (dep) or eventually fold into `vlm_dp/perception/`? (Defer.)
- Does the legacy `vlm_base/main.py` path stay maintained, or become a thin deprecation shim after Phase-2?

---

## 11. Migration map (APPROVED 2026-07-21 — execute once rekep_fake is proven in the harness)

Trigger: `--vlm_cost rekep_fake` smoke passes in eval_steering. Then move-and-repoint (git mv; nothing
copied, nothing deleted) so the package structurally IS the integration:

| moves into vlm_dp/                | from                                            | notes |
|-----------------------------------|--------------------------------------------------|-------|
| `cost/base_cost.py`               | `vlm_base/base_cost.py`                          | CompositeCost |
| `cost/terms.py`                   | `vlm_base/cost_terms.py`                         | incl. `gripper_smooth` |
| `grounding/` (pkg)                | `sim_common/grounding/`                          | gt / rekep sources, Stage/Grounding, fake_vlm, masks |
| `world.py`                        | `sim_common/world.py`                            | GTWorld / SensedWorld |
| `perception.py`                   | `sim_common/perception.py`                       | sam_vlm / GroundedSAM |
| `grasp_sensor.py`                 | `sim_common/grasp_sensor.py`                     | aperture stall |
| `visual_tracker.py`               | `sim_common/visual_tracker.py`                   | CoTracker |

Stays external (imported): `rekep/` (front-end), `sim_free_mpc/` (FROZEN engine), `eval_steering.py`
(harness; only the gated `--vlm_cost` seam + camera augmentation touch it), `sim_common/envs/`+`fk.py`+
`geometry.py`+`constraints.py` (shared env/FK utilities used by both vlm_dp and vlm_base).

Repoint: `vlm_base/*` and `sim_common/grounding` importers switch to `vlm_dp.*` paths; vlm_base stays the
standalone 15 Hz twin (its driver machinery is NOT part of vlm_dp — scope contract). After the move, run
`vlm_dp.tests` + one vlm_base smoke + one `--vlm_cost gt` seed to confirm the repoint broke nothing.
