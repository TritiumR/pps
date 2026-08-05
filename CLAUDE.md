# CLAUDE.md

Guidance for working in this repository (published as `TritiumR/pps`).

## What this repo is

A self-contained research codebase for **steering pi0 / pi0.5 VLA policies** and
evaluating them in **IsaacLab** manipulation simulations. It bundles three things:

- **`openpi/`** — the policy/model code and training scripts (a fork of Physical
  Intelligence's openpi). Defines the pi0 family and the PROXY steering models.
- **`eval_steering.py`** — the main evaluation harness for *steered* rollouts.
- **`eval_pi.py`** — plain (un-steered) pi0/pi0.5 evaluation, with OOD options.
- **`IsaacLab/`** — a bundled copy of IsaacLab, **code only** (assets excluded).

The repo was deliberately trimmed to these pieces; it is meant to be a lean,
publishable release rather than the full development tree.

## Layout

```
eval_steering.py            # steered rollout eval -> results/<task>/<exp>/<seed>_recording.mp4
eval_pi.py                  # plain pi0/pi0.5 eval (supports --ood_mode light/camera)
openpi/
  src/openpi/
    models/                 # JAX/Flax models: pi0, pi05, pi0_fast, gemma; ModelType enum
    models_pytorch/         # PyTorch model implementations
    training/config.py      # config registry -> get_config(<name>) e.g. "pi05_droid"
    policies/               # policy_config.create_trained_policy(config, checkpoint_dir)
    serving/ , shared/ , transforms.py
  scripts/                  # train*.py, distill*.py, serve_policy*.py (entry points)
IsaacLab/
  source/isaaclab*/         # bundled IsaacLab packages (isaaclab, isaaclab_tasks, isaaclab_mimic, ...)
  source/isaaclab_tasks/.../manipulation/{pot,tea,ice,can,drink,...}/  # custom tasks
```

`ModelType` (in `openpi/src/openpi/models/model.py`): `PI0`, `PI0_FAST`, `PI05`,
and the steering proxies `PROXY`, `PROXY_SCORE`, `PROXY_POINTCLOUD`, `PROXY_DP3`,
`PROXY_SOUND`.

## How steering works (eval_steering.py)

Evaluation has four explicit modes: `--task_only`, `--vlm_base`, `--task_steer`,
and `--full_steer`. Score-space MPC loads only the Pi0/Pi0.5 normalization stats
for action encoding/decoding; it does not load the base model weights. Task and
reference steering checkpoints are `PROXY_SCORE` policies, combined as geometric
base score plus the configured task residual (and minus the reference score in
full steering).

The rollout writes the basic RGB trajectory video as `<seed>_{success|fail}.mp4`.
Sound/thermal video variants and projected-axis/recovery debug overlays are not
part of the eval path.

## Self-containment conventions (read before editing)

- **In-repo IsaacLab imports.** `eval_steering.py` / `eval_pi.py` prepend
  `IsaacLab/source/<pkg>` (anchored to the script dir) to `sys.path` so
  `import isaaclab*` resolves from the bundled copy. **Do not** reintroduce a
  `sys.path.append("../IsaacLab")` pointing at an external checkout.
- **Asset paths.** IsaacLab task configs build asset paths as
  `os.path.join(os.path.dirname(__file__), "../../../../../../assets/...")`, which
  resolves to **`IsaacLab/assets/`**. Because tasks load from the in-repo IsaacLab,
  this stays inside the repo. **Assets themselves are not committed** (large,
  gitignored) — place them under `IsaacLab/assets/` (e.g. `ArtVIP/...`) before
  running tasks. Keep asset references `__file__`-relative, never absolute.
- **No machine-specific paths.** Avoid hardcoding `/home/...` or `/share/...`.
  Checkpoints are passed via flags or fetched remotely (see below).

## Running

Evaluation requires **NVIDIA Isaac Sim + IsaacLab dependencies and a GPU**; the
scripts launch a sim app via `AppLauncher`. On the development setup this runs
under the conda env `env_isaaclab` (Python 3.11) where the IsaacLab packages and
`isaacsim` are installed.

Steered eval:
```bash
python eval_steering.py \
  --task <IsaacLab-task-id> \
  --base_model_name pi05_droid   --base_checkpoint_dir  <dir> \
  --steer_model_name <proxy_cfg> --steer_checkpoint_dir <dir> \
  --mimic_model_name <proxy_cfg> --mimic_checkpoint_dir <dir> \
  --prompt "<instruction>" --exp_name demo \
  --steer_scale 0.4 --num_steps 10 --seed_start 42 --seed_end 43
```

Plain pi eval:
```bash
python eval_pi.py --task <task> --model_name pi05_droid --prompt "<instruction>" --name demo
```

Outputs land in `results/<task>/<exp_or_name>/`. Checkpoint dirs are optional —
if omitted, weights are downloaded (e.g. `gs://openpi-assets-simeval/<model_name>`).
Model configs are resolved by name through `openpi.training.config.get_config()`.

Training / serving live under `openpi/` (run from that directory), e.g.
`python scripts/train_pytorch.py ...`, `scripts/distill_pytorch.py ...`, or
`uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid ...`.

## Gotchas

- `README.md` references `test_isaac.sh` / `test_steer_separate.sh`; those wrapper
  scripts are not in this trimmed repo. Invoke `eval_steering.py` / `eval_pi.py`
  directly (see above).
- `IsaacLab/assets/` must be populated before any task that loads scene/object
  USDs will run.
- `.gitignore` excludes `data/`, `results/`, `videos/`, `*.mp4`, checkpoints, and
  the `openpi/third_party/{aloha,libero}` submodule paths — don't commit those.
- `.gitmodules` lists aloha/libero submodules but no gitlinks are included, so it
  is inert (a recursive submodule update finds nothing).
