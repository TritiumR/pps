# PPS

Steering pi0 / pi0.5 VLA policies and evaluating them in IsaacLab manipulation
simulations. See `CLAUDE.md` for repo layout and conventions.

## Prerequisites

- NVIDIA Isaac Sim + IsaacLab dependencies and a GPU (evaluation launches a sim app).
- The bundled IsaacLab is **code only** — place scene/object assets under
  `IsaacLab/assets/` before running a task.
- A trained base policy checkpoint, e.g. `pi05_droid_jointpos` at
  `openpi/checkpoints/pytorch/pi05_droid_jointpos`.

## Train a proxy (steering) model

The proxy is trained in **two stages**, both from `openpi/`. `<train-config>` is the
proxy training config (e.g. `proxy_isaaclab_droid_pot_pi05_jointpos` for the pot task,
`proxy_isaaclab_droid_tea_pi05_jointpos` for tea); the dataset is defined by the config.

**Stage 1 — distill the mimic** from the base policy (the "on-the-fly" distillation):

```bash
cd openpi
uv run scripts/distill_pytorch.py <train-config> \
  --exp_name reference \
  --teacher_checkpoint_dir checkpoints/pytorch/pi05_droid_jointpos
# -> checkpoints/<train-config>/reference/20000
```

The teacher config name is inferred from `--teacher_checkpoint_dir`.

**Stage 2 — train the steer model from the mimic**:

```bash
uv run scripts/train_pytorch.py <train-config> \
  --exp_name task \
  --pytorch_weight_path checkpoints/<train-config>/reference/20000
# -> checkpoints/<train-config>/task/{8000,16000,24000}
```

## Evaluate with steering (`eval_steering.py`)

Run from the repo root. Steering combines a base policy with a `steer` and a
`mimic` proxy (here both are the same proxy config, using the Stage-2 steer
checkpoint and the Stage-1 mimic checkpoint). The config names are read from the
checkpoint dir paths, so they are not passed explicitly; cameras and headless
mode are on by default, and steering is applied over the whole denoise:

```bash
TASK="Isaac-Pot-Droid-Visuomotor-v0"
PROMPT="remove the lid of the pot and put egg in it"

python eval_steering.py \
  --task "$TASK" \
  --exp_name eval \
  --base_checkpoint_dir  openpi/checkpoints/pytorch/pi05_droid_jointpos \
  --steer_checkpoint_dir openpi/checkpoints/<train-config>/task/24000 \
  --mimic_checkpoint_dir openpi/checkpoints/<train-config>/reference/20000 \
  --prompt "$PROMPT" \
  --steer_scale 0.4 \
  --task_num_steps 1200
```

Rollout videos are written to `results/<task>/<exp_name>/<seed>_{success,fail}.mp4`.
`--steer_scale` controls steering strength (0.4–0.8 typical); `--only_steer` uses
the steer velocity alone.

## Serve pi-0.5 on droid

```bash
cd openpi
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=checkpoints/pi05_droid
```
