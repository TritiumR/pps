# RoboLab Spoon data and keypose training

This directory contains the portable pipeline used for the 50-demo RoboLab
spaghetti-spoon dataset and its H21 keypose/waypoint proxy. Large datasets and
checkpoints are intentionally not stored in Git.

The collector is a randomized scripted Rev11 expert. It is **not** MimicGen;
MimicGen-style generation can be added as a separate, explicitly evaluated
source without changing the conversion or training contracts here.

## Environment

- PPS: this repository's `mujoco-eval` branch.
- RoboLab scenes: `jsiburian/RoboLab`, branch
  `feature/new-benchmark-scenes`, commit
  `6a55c764980dd4118e8c3678db9d465b2e6329b9`.
- Clone RoboLab with Git LFS enabled (`git lfs install && git lfs pull`).
- Isaac Sim / IsaacLab must expose `/isaac-sim/python.sh`, or set
  `PYTHON_BIN` for training. The generator itself must run with Isaac's Python.
- The pi0.5 DROID base checkpoint and its `assets/droid/norm_stats.json` must be
  staged at the normal OpenPI checkpoint location before training.

Keep artifacts on a shared data volume. A complete transfer comprises the
converted HDF5, its manifest/validation receipt, and (when resuming or
evaluating) the checkpoint directory including `model.safetensors`,
`action_norm_stats.json`, and training metadata.

## 1. Generate native RoboLab episodes

From the PPS repository root:

```bash
/isaac-sim/python.sh -m robolab_eval.data_generation.collect_spoon \
  --out /data/robolab_spoon_single50 \
  --target-successes 50 --max-attempts 100 --seed-base 53000 \
  --max-steps 1100 --headless --device cuda:0
```

The command is resumable: attempted episodes are indexed monotonically in
`source.hdf5` and `attempt_results.jsonl`. Only episodes passing the task
predicate, one-insertion/one-release checks, finite-action check, and distractor
negative control are accepted. The contact-sensor diet retains all six contact
signals consumed by generation, labels, and success checks.

## 2. Convert, validate, and inspect

```bash
python -m robolab_eval.data_generation.convert_spoon \
  --src /data/robolab_spoon_single50/source.hdf5 \
  --results /data/robolab_spoon_single50/attempt_results.jsonl \
  --out /data/robolab_spoon_single50/demo_224.hdf5 --count 50

python -m robolab_eval.data_generation.validate_spoon \
  --hdf5 /data/robolab_spoon_single50/demo_224.hdf5 \
  --receipt /data/robolab_spoon_single50/validation_receipt.json --expected 50

python -m robolab_eval.data_generation.preview_spoon \
  --hdf5 /data/robolab_spoon_single50/demo_224.hdf5 \
  --out /data/robolab_spoon_single50/previews
```

The converted action is the achieved next arm joint position (7) plus the
original continuous gripper command (1). Images are the 224x224 training
front camera and wrist camera.

## 3. Train the H21 proxy

```bash
export SPOON_DATASET=/data/robolab_spoon_single50/demo_224.hdf5
export SPOON_CHECKPOINT_DIR=/data/robolab_spoon_single50/checkpoints
robolab_eval/data_generation/train_spoon_keypose.sh
```

The fixed contract is H21 = 15 executable action rows + 5 AWE waypoint rows +
1 keypose row, continuous gripper, pooled demo-delta normalization, x0
prediction, and a bidirectional suffix. The shared keypose labeler preserves
legacy binary-gripper behavior exactly while treating a continuous close/open
ramp as one semantic phase transition.

Override `TRAIN_STEPS`, `BATCH_SIZE`, `NUM_WORKERS`, `EXP_NAME`,
`WANDB_PROJECT`, or `WANDB_RUN_NAME` through environment variables. The
launcher never deletes an existing checkpoint directory; use the trainer's
normal resume/overwrite workflow deliberately if needed.
