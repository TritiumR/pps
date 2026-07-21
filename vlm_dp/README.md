# vlm_dp

A VLM-grounded cost running inside the `eval_steering.py` sampling-MPC harness.

The package combines two existing systems without modifying either:

- **The ReKep-style VLM front-end** (`grounding/`): turns a language instruction into scene
  objects, task stages, tracked keypoints, and per-stage relational constraints.
- **The `sim_free_mpc` sampling planner** (frozen, driven by `eval_steering.py`): optimizes an
  action chunk against a cost at every replan and supports score-space steering.

vlm_dp supplies the cost — a config-weighted sum of geometric terms built from the grounding —
and the glue that feeds it live scene state each replan. The planner, the rollout loop, and the
steering machinery are untouched; with `--vlm_cost none` (the default) the harness behaves
exactly as before.

## How it works

```
instruction ──► grounding/ ──► stages, keypoints, constraints
                                      │
eval_steering rollout ──► bridge.py ──┤ per replan:
  (env, planner, steering)            │   context.py  world-frame scene context
                                      │   cost/       CompositeCost scores the samples
                                      └── stage machine: advance on progress, regress on a drop
```

`bridge.py` attaches the cost once, re-grounds each episode, and per replan builds the context
and advances the stage. All positions in the context share the simulator world frame (see
`context.py`; pinned by `tests/test_context.py`).

## Layout

| path | what |
|---|---|
| `grounding/` | instruction → objects, stages, keypoints, constraints (`gt`, `rekep_fake`, `rekep_real`) |
| `cost/` | `CompositeCost` + the term library (`terms.py`) |
| `bridge.py`, `context.py`, `stage.py` | planner attachment, per-replan context, stage predicates |
| `world.py` | object-state seam: simulator (`GTWorld`) or sensors (`SensedWorld`) |
| `perception.py`, `grasp_sensor.py`, `visual_tracker.py` | the sensed-state stack |
| `configs/base.yaml` | term weights + gripper/collision geometry |
| `tests/` | CPU-only regression tests |
| `DESIGN.md` | rationale, migration history, open risks |

External dependencies: `rekep/` (keypoint proposal + constraint generation), `sim_free_mpc/`
(the planner; frozen), `sim_common/` (env views, FK, geometry). The standalone `vlm_base/`
driver imports from vlm_dp, never the reverse.

## Running

From the repo root, inside the Isaac Sim container:

```bash
python eval_steering.py \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --vlm_base --no_steer \
  --vlm_cost gt \
  --mpc_update mbd_score_action_prox --mpc_cost priority --mpc_optimize_space action \
  --mpc_num_samples 4096 --mpc_iterations 1 --mpc_noise 0.8 --mpc_temperature 0.1 \
  --mpc_joint_delta_clip 0.15 --num_steps 10 --mpc_ddim_train_timesteps 100 \
  --task_num_steps 800 --steps_per_inference 4 --interpolate --headless \
  --exp_name my_run --seed_start 1 --seed_end 11
```

Notes:

- `--vlm_cost` requires a planner mode (`--vlm_base`, `--task_steer`, or `--full_steer`) and
  `--mpc_cost priority` (asserted at startup).
- `rekep_fake` / `rekep_real` add depth + segmentation to the table camera automatically.
  `rekep_real` calls GPT-4o for constraint generation (needs `OPENAI_API_KEY`).
- Results land in `results/<task>/<exp_name>/`: per-seed videos (H.264), `results.json`,
  `mpc_debug.jsonl`.

### Grounding modes

| `--vlm_cost` | keypoints | constraints | use |
|---|---|---|---|
| `gt` | none | none (plain stage targets) | controller upper bound / debugging |
| `rekep_fake` | proposed from vision | canned per-task files | the pipeline without VLM calls |
| `rekep_real` | proposed from vision | written by GPT-4o | the full VLM stack |

Object state is currently read from the simulator (`GTWorld`); the sensed stack
(`SensedWorld` + aperture grasp sensor + visual tracking) is in the package and wiring it into
the bridge is planned.

## Tests

```bash
python -m vlm_dp.tests.test_context   # CPU only, no simulator
```

## Status

The integration is validated end-to-end for `gt` and `rekep_fake` (grasping works; the default
harness path is regression-checked). Known open problem: carry stability — the object slips
during lift/transit, so task success is currently 0/10 on the weight task versus 8/20 for the
harness's built-in grasp-flow cost. Steering over this base has not been run yet.
