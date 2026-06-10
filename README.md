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

The proxy is trained in **two stages**, both from `openpi/`. Example below is the
`pot` task with 150 demonstration episodes (`proxy_isaaclab_droid_pot_pi05_jointpos`).

**Stage 1 — distill the mimic** from the base policy (the "on-the-fly" distillation):

```bash
cd openpi
uv run scripts/distill_pytorch.py proxy_isaaclab_droid_pot_pi05_jointpos \
  --exp_name distill_on_the_fly_150 \
  --teacher_config_name pi05_droid_jointpos \
  --teacher_checkpoint_dir checkpoints/pytorch/pi05_droid_jointpos \
  --num_distill_steps 10 \
  --num_train_steps 20001 \
  --save_interval 20000 \
  --batch_size 8 \
  --data.num_episodes 150 \
  --overwrite
# -> checkpoints/proxy_isaaclab_droid_pot_pi05_jointpos/distill_on_the_fly_150/20000
```

**Stage 2 — train the steer model from the mimic**:

```bash
uv run scripts/train_pytorch.py proxy_isaaclab_droid_pot_pi05_jointpos \
  --exp_name steer_from_mimic_150 \
  --pytorch_weight_path checkpoints/proxy_isaaclab_droid_pot_pi05_jointpos/distill_on_the_fly_150/20000 \
  --num_train_steps 24001 \
  --save_interval 8000 \
  --batch_size 64 \
  --data.num_episodes 150 \
  --overwrite
# -> checkpoints/proxy_isaaclab_droid_pot_pi05_jointpos/steer_from_mimic_150/{8000,16000,24000}
```

For another task, swap the config name (e.g. `proxy_isaaclab_droid_tea_pi05_jointpos`),
the task id, and the prompt. The dataset is defined by the config.

## Evaluate with steering (`eval_steering.py`)

Run from the repo root. Steering combines a base policy with a `steer` and a
`mimic` proxy (here both are the same proxy config, using the Stage-2 steer
checkpoint and the Stage-1 mimic checkpoint):

```bash
TASK="Isaac-Pot-Droid-Visuomotor-v0"
PROMPT="remove the lid of the pot and put egg in it"
STEER_MODEL_NAME="proxy_isaaclab_droid_pot_pi05_jointpos"

python eval_steering.py \
  --task "$TASK" \
  --exp_name steering_pot_pi05_jointpos_steer150_24000_mimic150_20000 \
  --enable_cameras \
  --base_model_name  pi05_droid_jointpos \
  --steer_model_name "$STEER_MODEL_NAME" \
  --mimic_model_name "$STEER_MODEL_NAME" \
  --base_checkpoint_dir  openpi/checkpoints/pytorch/pi05_droid_jointpos \
  --steer_checkpoint_dir openpi/checkpoints/$STEER_MODEL_NAME/steer_from_mimic_150/24000 \
  --mimic_checkpoint_dir openpi/checkpoints/$STEER_MODEL_NAME/distill_on_the_fly_150/20000 \
  --headless \
  --prompt "$PROMPT" \
  --steer_scale 0.4 \
  --steer_step 0.0 \
  --seed_start 1 \
  --seed_end 31 \
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
