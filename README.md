# Proxy Policy Steering

Project page: https://proxy-policy-steering.github.io/

## Prerequisites

- NVIDIA Isaac Sim + IsaacLab dependencies and a GPU (evaluation launches a sim app).
- The bundled IsaacLab is **code only** — place scene/object assets under
  `IsaacLab/assets/` before running a task.
- A trained base policy checkpoint, e.g. `pi05_droid_jointpos` at
  `openpi/checkpoints/pytorch/pi05_droid_jointpos`.

## Training

The proxy is trained in **two stages**, both from `openpi/`. `<train-config>` is the
proxy training config (e.g. `proxy_isaaclab_droid_pot_pi05_jointpos` for the pot task,
`proxy_isaaclab_droid_tea_pi05_jointpos` for tea); the dataset is defined by the config.

**Stage 1 — reference proxy** from the base policy (the "on-the-fly" distillation):

```bash
cd openpi
uv run scripts/distill_pytorch.py <train-config> \
  --exp_name reference \
  --teacher_checkpoint_dir checkpoints/pytorch/pi05_droid_jointpos
# -> checkpoints/<train-config>/reference/20000
```

**Stage 2 — task proxy**:

```bash
uv run scripts/train_pytorch.py <train-config> \
  --exp_name task \
  --pytorch_weight_path checkpoints/<train-config>/reference/20000
# -> checkpoints/<train-config>/task/{8000,16000,24000}
```

### Bidirectional score-task proxies

Weight, tea, capsule, and pot use one shared clean score-training path. The task
wrappers fix epsilon prediction, bidirectional attention, the OpenPI Gemma
replacement, and no legacy input scaling:

```bash
tools/train_score_task_weight.sh  1 32 8
tools/train_score_task_tea.sh     1 32 8
tools/train_score_task_capsule.sh 1 32 8
tools/train_score_task_pot.sh     1 32 8
```

The arguments are GPU count, global batch size, and cache-build workers. The
default experiment name is `task_eps_bidir_openpi`; compatible checkpoints resume
automatically. Set `EXP_NAME` for a separate run, `DATA_FILE` to override the task
HDF5 path, or `TRAIN_MODE=--overwrite` to explicitly restart that experiment.
Legacy `task_eps_bidir` checkpoints are evaluation-only and are rejected by the
resume semantic check.

## Evaluation in simulation

Run from the repo root. Steering combines a base policy with a `reference` and a
`task` proxy (using the Stage-1 reference checkpoint and the Stage-2 task checkpoint).

```bash
TASK="Isaac-Pot-Droid-Visuomotor-v0"
PROMPT="remove the lid of the pot and put egg in it"

# Evaluate full score steering
python eval_steering.py \
  --task "$TASK" \
  --full_steer \
  --exp_name eval_pps \
  --base_checkpoint_dir  openpi/checkpoints/pytorch/pi05_droid_jointpos \
  --task_checkpoint_dir openpi/checkpoints/<train-config>/task/24000 \
  --ref_checkpoint_dir openpi/checkpoints/<train-config>/reference/20000 \
  --prompt "$PROMPT" \
  --steer_scale 0.4 \
  --task_num_steps 1200

# Evaluate the sim-free MPC base (Pi0.5 norm stats only; model weights are skipped)
python eval_steering.py \
  --task "$TASK" \
  --vlm_base \
  --exp_name eval_mpc_base \
  --base_checkpoint_dir  openpi/checkpoints/pytorch/pi05_droid_jointpos \
  --prompt "$PROMPT" \
  --task_num_steps 1200
```

Each eval run is written to `results/<task>/<exp_name>/<run-id>_<config>/`, with
`results.json` and short episode names such as `<seed>_success.mp4` or
`<seed>_fail.mp4`.
Choose exactly one mode: `--task_only`, `--vlm_base`, `--task_steer`, or
`--full_steer`. `--steer_scale` controls task score steering strength.

## Evaluation in the real world

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=checkpoints/pi05_droid
```
