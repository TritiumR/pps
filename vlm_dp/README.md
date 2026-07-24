# vlm_dp

A VLM-grounded cost that plugs into the `eval_steering.py` sampling-MPC harness. It combines a
ReKep-style VLM front-end (`grounding/`, instruction → objects, stages, keypoints, constraints) with the
frozen `sim_free_mpc` sampling planner: vlm_dp supplies the **cost** (a config-weighted sum of geometric
terms built from the grounding) and the glue that feeds it live scene state each replan. The planner,
rollout loop, and steering machinery are untouched; with `--vlm_cost none` (the default) the harness
behaves exactly as before.

## Dependencies

Assumes an environment that already runs the base `eval_steering.py` (Isaac Sim + IsaacLab + openpi, with
a base-policy checkpoint such as `pi05_droid_jointpos` and assets under `IsaacLab/assets/`). vlm_dp adds a
VLM front-end and vision backends on top; install these into that same environment (`pip`, or `uv pip`):

```bash
pip install openai kmeans-pytorch parse pyyaml scikit-learn opencv-python \
    segment-anything supervision addict yapf timm pycocotools easydict transformers==4.44.2
```

- **GroundingDINO** (needed for `--vlm_segment groundedsam`, the default) builds a CUDA op from source
  against your torch — install from https://github.com/IDEA-Research/GroundingDINO.
- **DINOv2** and **CoTracker3** load from the local `torch.hub` cache; a first run with network populates
  it, then runs are offline.
- **`OPENAI_API_KEY`** — only for `rekep_real` and `--vlm_derive_vocab` (GPT-4o):

  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

## Running

From the repo root. Each argument is on its own line; the new vlm_dp arguments are grouped and marked so
they stand out from the standard `eval_steering.py` harness.

```bash
python eval_steering.py \
  `# --- new vlm_dp arguments ---` \
  --vlm_base \
  --vlm_cost rekep_fake \
  --vlm_state real \
  --vlm_track visual \
  --vlm_cost_config vlm_dp/configs/parity13_rekep_legacy_deadzone.yaml \
  `# --- standard eval_steering.py arguments ---` \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --no_steer \
  --mpc_update mbd_score_action_prox \
  --mpc_cost priority \
  --mpc_optimize_space action \
  --mpc_num_samples 4096 \
  --mpc_iterations 1 \
  --mpc_noise 0.8 \
  --mpc_temperature 0.1 \
  --mpc_joint_delta_clip 0.15 \
  --num_steps 10 \
  --mpc_ddim_train_timesteps 100 \
  --task_num_steps 800 \
  --steps_per_inference 4 \
  --interpolate \
  --headless \
  --exp_name my_run \
  --seed_start 1 \
  --seed_end 11
```

`--vlm_cost` picks the grounding mode (below); `--vlm_state {gt,real}` swaps privileged sim poses for
perception; `--vlm_track {fk,reperceive,visual}` sets the between-look tracker. Results land in
`results/<task>/<exp_name>/`: per-seed videos, `results.json`, `mpc_debug.jsonl`.

### Grounding modes (`--vlm_cost`)

| mode | keypoints | constraints | use |
|---|---|---|---|
| `gt` | none | none (plain stage targets) | controller upper bound / debugging |
| `rekep_fake` | proposed from vision | canned per-task files | the pipeline without VLM calls (use for cost comparisons) |
| `rekep_real` | proposed from vision | written by GPT-4o | the full VLM stack (non-deterministic) |

### Ablation rungs (change one axis at a time)

Every rung shares one code path; the flags change only *where information comes from*. Use them to
localise a failure before touching a cost. When comparing costs, use `rekep_fake` (a live VLM re-grounds
differently each rollout), change one axis at a time, and report the rung with every number — differences
under ~20 points at N=10-20 are noise.

## Steering (for collaborators)

The base above (`--vlm_base`) is the vlm_dp MBD cost — no learned network. Steering adds two proxies in
score space, PPS-style: `score = base + steer_scale · (task − ref)`, where the **task proxy** is trained
on demonstrations and the **reference proxy** is distilled from the base. Both proxies condition on
**geometry** (object positions, keypoints, phase), not images, because the base is geometry-driven.

**Where the proxies go**: each is an openpi checkpoint directory (e.g. under `openpi/checkpoints/`). Point
`--task_checkpoint_dir` / `--ref_checkpoint_dir` at them, or set them per-task in `task_prompts.json`. The
base cost (`--vlm_cost` / `--vlm_cost_config`) is unchanged from the base rollout.

The score mode is one mutually-exclusive flag:

- `--vlm_base` — base only, no proxies (the base rollout above).
- `--full_steer` — base + `steer_scale · (task − ref)`; loads both proxies. The PPS path.
- `--task_steer` — base + `steer_scale · (task − base)`, ref-free (CFG-style); `--gamma_base` scales the base.

To steer, take the base command and swap `--vlm_base --no_steer` for `--full_steer` plus the proxies and
`--steer_scale`:

```bash
python eval_steering.py \
  `# --- base cost (identical to the base rollout) ---` \
  --vlm_cost rekep_fake \
  --vlm_state real \
  --vlm_track visual \
  --vlm_cost_config vlm_dp/configs/parity13_rekep_legacy_deadzone.yaml \
  `# --- score-space steering ---` \
  --full_steer \
  --task_checkpoint_dir openpi/checkpoints/<task_proxy> \
  --ref_checkpoint_dir openpi/checkpoints/<ref_proxy> \
  --steer_scale 0.4 \
  `# --- standard eval_steering.py arguments ---` \
  --task Isaac-Weight-Droid-Visuomotor-v0 \
  --mpc_update mbd_score_action_prox \
  --mpc_cost priority \
  --mpc_optimize_space action \
  --mpc_num_samples 4096 \
  --mpc_iterations 1 \
  --mpc_noise 0.8 \
  --mpc_temperature 0.1 \
  --mpc_joint_delta_clip 0.15 \
  --num_steps 10 \
  --mpc_ddim_train_timesteps 100 \
  --task_num_steps 800 \
  --steps_per_inference 4 \
  --interpolate \
  --headless \
  --exp_name my_steer_run \
  --seed_start 1 \
  --seed_end 11
```

What a proxy can and cannot move:

- **Can move**: the arm trajectory within a substage; where along a graspable span the grasp lands
  (`grasp_region`); when in the horizon the gripper closes (`grasp_commit`). Both are opt-in.
- **Cannot move**: which substage is active, and when a stage advances — stage transitions live in
  `bridge`, outside the score entirely, protecting the VLM-authored plan from a learned proxy.

## Code structure

```
instruction ──► grounding/ ──► stages, keypoints, constraints
                                      │
eval_steering rollout ──► bridge.py ──┤ per replan:
  (env, planner, steering)            │   context.py  world-frame scene context
                                      │   cost/       CompositeCost scores the samples
                                      └── stage machine: advance on progress, regress on a drop
```

`bridge.py` attaches the cost once, re-grounds each episode, and per replan builds the context and
advances the stage. All positions share the simulator world frame (`context.py`).

| path | what |
|---|---|
| `grounding/` | instruction → objects, stages, keypoints, constraints (`gt`, `rekep_fake`, `rekep_real`, `capsule`) |
| `cost/` | `CompositeCost` + the term library (`terms.py`) |
| `bridge.py`, `context.py`, `stage.py` | planner attachment, per-replan context, stage predicates |
| `world.py` | object-state seam: simulator (`GTWorld`) or sensors (`SensedWorld`) |
| `perception.py`, `grasp_sensor.py`, `visual_tracker.py`, `grasp_recovery.py` | the sensed-state stack |
| `configs/` | term weights + gripper/collision geometry (`base.yaml` and task/ablation variants) |
| `tests/` | CPU-only regression tests |

External deps: `rekep/` (keypoint proposal + constraint generation), `sim_free_mpc/` (the planner,
frozen), `sim_common/` (env views, FK, geometry). The standalone `vlm_base/` driver imports from vlm_dp,
never the reverse.
