# vlm_dp

A VLM-grounded cost that plugs into the `eval_steering.py` sampling-MPC harness. It combines a
ReKep-style VLM front-end (`grounding/`, instruction → objects, stages, keypoints, constraints,
completion predicates) with the frozen `sim_free_mpc` sampling planner: vlm_dp supplies the **cost**
(a config-weighted sum of geometric terms built from the grounding), the **stage machine** that
executes the plan, and the glue that feeds both live scene state each replan. The planner, rollout
loop, and steering machinery are untouched; with `--vlm_cost none` (the default) the harness behaves
exactly as before.

The division of labour, enforced throughout:

> **The VLM plan owns task semantics and task-specific geometry. The runtime owns sensing,
> temporal filtering, and calibrated physical thresholds.**

## Dependencies

Assumes an environment that already runs the base `eval_steering.py` (Isaac Sim + IsaacLab + openpi,
with the `pi05_droid_jointpos` decode surface and assets under `IsaacLab/assets/`). vlm_dp adds a VLM
front-end and vision backends on top; install into that same environment (`pip`, or `uv pip`):

```bash
pip install openai kmeans-pytorch parse pyyaml scikit-learn opencv-python \
    segment-anything supervision addict yapf timm pycocotools easydict termcolor
```

- **transformers is pinned by openpi, not by vlm_dp**: `transformers==4.53.2` plus the file overlay
  from `openpi/src/openpi/models_pytorch/transformers_replace/` (asserted at runtime). Do not use the
  4.44.2 pin that older revisions of this README suggested — it breaks openpi.
- **GroundingDINO** (needed for `--vlm_segment groundedsam`, the default) builds a CUDA op from source
  against your torch — install from https://github.com/IDEA-Research/GroundingDINO. On torch 2.x,
  patch `value.type()` → `value.scalar_type()` in `ms_deform_attn_cuda.cu` if the build fails.
- **DINOv2** and **CoTracker3** load from the local `torch.hub` cache; a first run with network
  populates it, then runs are offline.
- **`OPENAI_API_KEY`** — only for `rekep_real*` and `--vlm_derive_vocab` (GPT-4o).

## Running (current reference stack)

From the repo root. This is the configuration behind the current best result (weight 16/20):

```bash
python eval_steering.py \
  `# --- vlm_dp: grounding + cost ---` \
  --vlm_base --no_steer --base_decode_only \
  --vlm_cost rekep_fake_vlm \
  --vlm_state real \
  --vlm_track visual \
  --vlm_segment groundedsam \
  --vlm_cost_config vlm_dp/configs/test_configs/simple_auth.yaml \
  `# --- decode space (opt-in; see "Action space" note below) ---` \
  --base_action_space demo_delta \
  --base_action_stats data/chuanruo_stats/weight_action_norm_stats.json \
  `# --- standard eval_steering.py arguments ---` \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --mpc_update mbd_score_action_prox \
  --mpc_cost priority \
  --mpc_optimize_space action \
  --mpc_num_samples 4096 --mpc_iterations 1 \
  --mpc_noise 0.4 --mpc_temperature 0.1 \
  --mpc_joint_delta_clip 0.05 \
  --num_steps 10 --mpc_ddim_train_timesteps 100 \
  --cost_executable_actions \
  --task_num_steps 1600 --steps_per_inference 4 \
  --interpolate --headless \
  --exp_name my_run --seed_start 1 --seed_end 11
```

Three flags matter more than any weight in the config:

- `--cost_executable_actions` — candidates are costed **after** the execution clamp. Without it the
  clamp saturates and the scored plan is not the executed one (measured: 1/20 → 10/20 on weight).
- `--base_action_space demo_delta` — decode the planner's proposals through zero-mean, demo-scaled
  action statistics instead of the DROID quantile band (measured: 11/20 → 14/20 on the hand-written
  cost). Per-task stats live in `data/chuanruo_stats/`. The default (`policy`) keeps the checkpoint's
  own normalization — required for score-space steering compatibility (see below).
- `advance.plan_authoritative: true` (in the cost config) — stage transitions are decided by the
  plan's own sub-goal constraints and completion predicates, never by bridge-side geometric
  heuristics. Without it, plans whose constraint text doesn't match the legacy heuristics stall
  silently; the advanceability preflight refuses such plans loudly.

Results land in `results/<task>/<exp_name>/`: per-seed videos, `results.json`, `mpc_debug.jsonl`
(per-step state, per-term costs, predicate components, grounded keypoints).

### Grounding modes (`--vlm_cost`)

| mode | keypoints | constraints | stage list | use |
|---|---|---|---|---|
| `gt` | none | none | hand-coded | controller upper bound / debugging |
| `rekep_fake` | vision | canned per-task `raw.txt` | **template** (injects a lift, drops markerless stages, discards constraints on grasp/lift stages) | legacy comparisons |
| `rekep_fake_vlm` | vision | canned per-task `raw.txt` | **as written by the plan** (every stage keeps its constraints and predicate) | **the current default choice** |
| `rekep_real` / `rekep_real_vlm` | vision | written live by GPT-4o | as above | the full VLM stack (non-deterministic) |

The canned plans live in `grounding/gt_vlm_output/<task>/raw.txt` — one file per task, in the exact
format `rekep/prompts/prompt_template.txt` specifies, parsed by the same splitter as a live GPT-4o
response. Editing a plan is editing one text file.

### Completion predicates and the primitive API

Each stage carries, besides its sub-goal/path constraints, a **completion predicate**: a boolean
"has the event this stage names actually happened", evaluated on sensed state and short history —
not a `cost < eps` test. Predicates are written in a seven-name trusted vocabulary the runtime
injects (`grounding/predicates.py :: PredicateRuntime`):

- events: `grasped(kp, name, "acquire"|"maintain")`, `released(kp, name)`, `stationary(kp, name)`,
  `sustained(cost_fn, name)` — the runtime owns the contact band, co-motion tests, history windows
  and thresholds, and auto-populates diagnostic `components`;
- tolerances: `grasp_tolerance(i)`, `target_region_radius(j)`, `clearance_margin(i)` — resolved
  from the **grounded grasp-feature geometry** (a declared 10 mm rim is 10 mm, never the owning
  body's 320 mm extent; `clearance_margin` refuses fixture-owned features loudly).

The rule: **custom geometry is allowed; custom sensor semantics are not.** Any novel geometric
relation is plain NumPy wrapped in `sustained(...)`; a new primitive is added only for a genuinely
new sensor or temporal concept.

### Preflights

Two guards run at grounding time and refuse in seconds what used to fail as an inexplicable
50-minute rollout:

- **grounding preflight** — roles distinct and on their objects, masks within physical extents,
  every grasp-intended object actually pinchable at the radius the cost terms will use;
- **advanceability preflight** — every stage's *real* advance test (the same functions the rollout
  calls, including the real grasp sensor and hold latch) can fire from a state satisfying that
  stage's own constraint.

### Ablation rungs

`--vlm_state {gt,real}` swaps privileged sim poses for perception; `--vlm_track {fk,reperceive,visual}`
sets the between-look tracker. Every rung shares one code path; the flags change only *where
information comes from*. When comparing costs use `rekep_fake*` (a live VLM re-grounds differently
each rollout), change one axis at a time, and report the rung with every number — differences under
~20 points at N=10–20 are noise.

### Current status (2026-08-09)

| task | status |
|---|---|
| weight | **16/20** on the reference stack (the hand-written cost tops out at 14/20 with demo-delta) |
| capsule | grasp machinery verified end-to-end; lid-opening strategy under active iteration |
| tea | grounds cleanly; handle is now a declared 10mm grasp feature (GPU rollout validation pending) |
| pot | plan + predicates ready and offline-validated; blocked on the `kitchen_with_parlor` asset |

`configs/test_configs/simple_auth.yaml` is the reference config. The `parity*` ladder is the
historical experimental record (one change per file) and is superseded for new work.

## Steering (for collaborators)

The base above (`--vlm_base`) is the vlm_dp MBD cost — no learned network (`--base_decode_only`
loads only the checkpoint's transforms and normalization stats). Steering adds learned score
proxies in score space: `--task_steer` (base + `steer_scale · (task − base)`) or `--full_steer`
(base + `steer_scale · (task − ref)`). Point `--task_checkpoint_dir` / `--ref_checkpoint_dir` at
`PROXY_SCORE` checkpoints.

Three validity requirements, all checked or measurable before spending rollouts:

1. **One action space.** Base and proxies must share normalization stats (asserted at startup).
   This also means `--base_action_space demo_delta` and score steering are mutually exclusive until
   the proxy side is re-exported in the same space — a joint decision, not a default.
2. **The proxy must run under the attention pattern it was trained with.** Checkpoints trained
   bidirectionally (`bidirectional_attention: true` in their metadata) produce a near-negated score
   under a causal mask. Verify with the token-dependency test (perturb a late action row; earlier
   rows' scores must respond) before any sweep.
3. **The control arm is `--task_steer --steer_scale 0.0`**, not `--no_steer` — the two paths are
   not bitwise-identical.

What a proxy can and cannot move: it shapes the arm trajectory within a stage; it cannot change
which stage is active or when a stage advances — transitions live in the bridge and the plan's
predicates, outside the score entirely, protecting the VLM-authored plan from a learned proxy.

## Code structure

```
instruction ──► grounding/ ──► stages, keypoints, constraints, completion predicates
                                      │
eval_steering rollout ──► bridge.py ──┤ per replan:
  (env, planner, steering)            │   context.py  world-frame scene context
                                      │   cost/       CompositeCost scores the samples
                                      └── stage machine: plan-authoritative transitions,
                                          grip-loss invariants, bounded recoveries
```

| path | what |
|---|---|
| `grounding/` | instruction → objects, stages, keypoints, constraints (`gt`, `rekep_fake[_vlm]`, `rekep_real[_vlm]`, `capsule`) |
| `grounding/gt_vlm_output/` | the canned per-task plans (`raw.txt`), single source of truth |
| `grounding/predicates.py` | completion-predicate loader + the seven-primitive `PredicateRuntime` |
| `cost/` | `CompositeCost` + the term library (`terms.py`) |
| `bridge.py`, `context.py`, `stage.py` | planner attachment, per-replan context, stage machine |
| `world.py` | object-state seam: simulator (`GTWorld`) or sensors (`SensedWorld`); holds resolve at grasp points |
| `perception.py`, `grasp_sensor.py`, `visual_tracker.py`, `grasp_recovery.py` | the sensed-state stack (size-prior mask assignment, calibrated aperture sensing) |
| `configs/test_configs/simple_auth.yaml` | the **reference** config |
| `configs/`, `configs/test_configs/parity*` | working configs + the historical ablation ladder |
| `tests/` | CPU-only regression tests (grasp geometry, predicate primitives, gating) |

External deps: `rekep/` (keypoint proposal + constraint generation + the prompt template, which now
documents the primitive vocabulary for live VLM use), `sim_free_mpc/` (the planner, frozen; includes
the opt-in demo-delta decode surface), `sim_common/` (env views, FK, geometry).
