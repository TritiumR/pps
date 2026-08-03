# mujoco_eval

A CPU-friendly MuJoCo/robosuite evaluation harness for the **MBD sampling-MPC base** on MimicGen tasks.

`mujoco_eval` mirrors the IsaacLab `eval_steering.py` harness while reusing the same planner
(`sim_free_mpc`), cost (`vlm_dp`), and control structure. It currently supports 12 tasks and runs CPU
episodes in seconds rather than GPU episodes in minutes.

The base uses no policy checkpoint. An identity decoder converts planner outputs directly into joint
deltas, so the cost function determines every action. This isolates whether a proposed mechanism improves
the base planner itself.

Base rollouts require one conda environment and no GPU or container. Proxy-driven runs (`expert` and
steering modes) and proxy training additionally require the OpenPI environment and a GPU.

---

## Installation

### 1. Create the environment

```bash
conda create -n mg python=3.10 pip -y -c conda-forge --override-channels
conda activate mg
```

### 2. Install the pinned simulator and I/O dependencies

Use the benchmark-compatible versions:

- `robosuite==1.4.1`, as used by MimicGen
- `mujoco==2.3.2`, as expected by robosuite's DM bindings

```bash
pip install \
    "mujoco==2.3.2" \
    "robosuite==1.4.1" \
    "imageio[ffmpeg]" \
    h5py \
    pyyaml \
    opencv-python

pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the pinned robomimic commit

Do not use the PyPI release. Version 0.3.0 hard-imports `mujoco_py` and is incompatible with robosuite
1.4.

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git ~/robomimic
git -C ~/robomimic checkout d0b37cf214bd24fb590d182edb6384333f67b661
pip install -e ~/robomimic
```

### 4. Install MimicGen

MimicGen registers the `*_D0` environments.

```bash
git clone https://github.com/NVlabs/mimicgen.git ~/mimicgen
pip install -e ~/mimicgen
```

Alternatively, `setup/01_create_env.sh` performs steps 1–4 and writes a resolved environment lock file.

---

## Data setup

### 5. Configure the data and results directories

Both paths are configurable and otherwise default to locations inside the repository.

```bash
export MUJOCO_EVAL_DATA=~/mg_data          # <task>/demo.hdf5
export MUJOCO_EVAL_RESULTS=~/mg_results    # jsonl + mp4
```

### 6. Download MimicGen datasets

Pass any subset of the supported MimicGen task names.

```bash
bash setup/04_download_data.sh stack_d0 square_d0 stack_three_d0
```

### 7. Download the robomimic `lift` and `can` datasets

These tasks use robomimic `ph` datasets rather than MimicGen datasets. The `kitchen` directory also omits
the `_d0` suffix.

```bash
python ~/robomimic/robomimic/scripts/download_datasets.py \
    --tasks lift can \
    --dataset_types ph \
    --hdf5_types raw \
    --download_dir $MUJOCO_EVAL_DATA
```

Rendering uses headless EGL. When DRM device access is restricted, select the software renderer:

```bash
export MUJOCO_EGL_DEVICE_ID=3
```

Device `3` is the default. Set the variable to a GPU index to use GPU rendering.

Run commands from the repository root so the sibling `sim_free_mpc`, `vlm_dp`, and `sim_common` packages
remain importable.

---

## Quick start

Each episode produces:

- `<results>/<task>/<exp>/<seed>.jsonl`
- `<results>/<task>/<exp>/<seed>.mp4`

Each JSONL file contains:

- one `replan` record per inference
- one `stage` record per transition
- one final `episode` summary

### 1. Run the base

Run one episode:

```bash
python -m mujoco_eval.eval \
    --task stack \
    --exp smoke \
    --seed 42 \
    --candidates 512
```

Run a 50-seed screen with the shipped base settings. Arguments after `--` are forwarded to the evaluation
CLI.

```bash
python -m mujoco_eval.parallel \
    --task stack \
    --exp base50 \
    --seeds 1-50 \
    --workers 8 \
    -- \
    --candidates 4096
```

`parallel.py` lowers `OMP_NUM_THREADS` per worker to prevent every process from creating a full BLAS
thread pool.

A seed that exceeds `--timeout` is killed and reported as `rc=-9`; this denotes a timeout rather than a
task failure.

### 2. Run the expert policy by itself

With `--steer expert`, the BC proxy controls the episode directly and executes its own action chunk. The
planner and cost are bypassed.

Treat this as the expert-only baseline that a steering method must outperform.

```bash
CKPT=$MUJOCO_EVAL_DATA/mg_stack/checkpoints/score_task_stack/task_bc/30000

python -m mujoco_eval.parallel \
    --task stack \
    --exp expert50 \
    --seeds 1-50 \
    --workers 4 \
    -- \
    --candidates 4096 \
    --steer expert \
    --proxy_checkpoint $CKPT \
    --proxy_prediction_mode x0 \
    --proxy_device cuda:0
```

### 3. Run the base with proxy steering

#### Candidate injection

Candidate injection samples a fraction `rho` of the MBD population around the proxy's clean action,
matching the mechanism in `eval_steering`.

```bash
python -m mujoco_eval.parallel \
    --task stack \
    --exp steer_inject \
    --seeds 1-50 \
    --workers 4 \
    -- \
    --candidates 4096 \
    --steer inject \
    --inject_rho 0.15 \
    --inject_schedule flat \
    --proxy_checkpoint $CKPT \
    --proxy_prediction_mode x0 \
    --proxy_device cuda:0
```

#### Additive score blending

Additive steering blends the proxy score field into the planner's per-level Monte Carlo score.

```bash
python -m mujoco_eval.parallel \
    --task stack \
    --exp steer_add \
    --seeds 1-50 \
    --workers 4 \
    -- \
    --candidates 4096 \
    --steer additive \
    --steer_gamma 0.4 \
    --steer_ref base \
    --proxy_score_at xt \
    --proxy_checkpoint $CKPT \
    --proxy_prediction_mode x0 \
    --proxy_device cuda:0 \
    --proxy_kv_cache on
```

`--proxy_score_at xt` evaluates the proxy at the current iterate, following PPS Algorithm 2. The default
`candidates` mode instead scores the full population; on off-manifold inputs it measured 2.60 s/replan
versus 0.53 s/replan.

**`--steer_ref`.** PPS Eq. (4) is `v_PPS = v_base + gamma*(v_task - v_ref)`. Because no reference proxy
is trained, this flag selects the substitute for `v_ref`:

| `--steer_ref` | `v_ref :=` | score | gamma=0 | gamma=1 |
|---|---|---|---|---|
| `none` | `0` | `s_base + gamma*s_task` | base | base **+** task |
| `base` | `v_base` | `(1-gamma)*s_base + gamma*s_task` | base | task |

The `-gamma*s_base` term is the entire distinction. `none` forms an unbounded sum and equals neither
policy at gamma=1. `base` implements Eq. (3), `pi_base^(1-gamma) * pi_task^gamma`; it is bounded for
gamma in [0, 1].

The identity check is at **gamma=0**, where the addend is exactly zero and the run must reproduce
`--steer off` on the same seed. gamma=1 does **not** reproduce `--steer expert`: that mode bypasses
the planner and executes the proxy chunk directly (`cost_min` is NaN), while additive still runs the
full sampler -- finite candidate set, `delta_clip`, its own noise schedule -- with only the score
replaced. The two agree only if the sampler exactly inverts the score field, which a Monte Carlo
cost-weighted mean does not.

Both substitutes are weaker than the full method; the paper reports a 64% -> 55% drop when ablating to
`v_ref := v_base`.

**Both forms require a shared action representation.** The planner uses one standard deviation per action
dimension, whereas a `demo_delta` proxy uses a per-(row, dimension) standard deviation that grows by
about 13x across the chunk. The resulting bridge slope spans 0.36–5.01, pushing mapped `x_t` values off
the proxy manifold. At gamma 0.4, the addend reaches NaN by level 4; with aligned representations it
remains O(1). Train with `--action_norm_pooled` to fix the mismatch. `--align_proxy_norm on` also aligns
the spaces, but reduced stage-0 base clearance from 100% to 47%.

#### The other modes

| `--steer` | what it does | notes |
|---|---|---|
| `expert` | the proxy drives; planner and cost unused | the bar a steered run must beat |
| `proxy_only` | every candidate from the proxy (`rho = 1`) | control for `inject` |
| `select` | proxy ranks M base plans, closest executed | `--select_m`; costs M denoise chains |
| `verify` | expert proposes, base vetoes | `--verify_rank`, `--verify_gate` |
| `tilt` | Gaussian tilt of the softmax toward the proxy | `--tilt_lambda`; reweights the base's own candidates |

`--verify_rank` selects the score bucket used by the base. The default, `feasibility`, contains only
keepout terms and showed no discrimination: across 820 chunks, the median expert-minus-base gap was
0.0000, so the gate never fired and behavior collapsed to the expert. `task` uses phase attractors and
therefore makes handoff implicitly phase-adaptive. `task_no_nh` removes `not_hold`, which the sampler
needs but which over-penalizes demonstration-scale motion when used as a judge.

`--kp` composes with `select`, `verify`, `additive`, and `tilt`, but not with `inject` or `proxy_only`;
those modes replace whole candidates and would overwrite the waypoint rows.

Set `--proxy_device` explicitly. CPU serving is about 5x slower per replan, and checkpoints must live
under the path mounted by `container.py`. Proxy runs typically use four workers; each worker launches one
proxy service and consumes roughly 5 GB of GPU memory.

### 4. Train a proxy

#### Render the training dataset

Render both cameras at 224 px using this repository's dataset schema.

After every environment reset, the converter restores `geomgroup[0] = 0`. Otherwise, training frames
include collision geometry absent during evaluation, creating a visual-domain mismatch that shifts
rollout behavior.

```bash
python mujoco_eval/setup/convert_mimicgen.py \
    --task stack_d0 \
    --size 224 \
    --demo_end 200
```

#### Train a BC proxy

Run this command from `openpi/` in the OpenPI environment.

```bash
python scripts/train_mpc_proxy_score_pytorch.py train-bc \
    --config score_task_stack_bc \
    --hdf5_path $MUJOCO_EVAL_DATA/stack_d0/demo_224.hdf5 \
    --prompt "stack the red block on the green block" \
    --exp_name task_bc \
    --checkpoint_base_dir <ckpt dir> \
    --train_steps 30000 \
    --batch_size 32 \
    --num_workers 6 \
    --ema_decay 0.999 \
    --aug_shift_px 4 \
    --val_demos 10
```

#### Train a score proxy

First generate an MPC score-label cache with the same cost used during evaluation, keeping training
labels and evaluation scores aligned under one objective.

```bash
python scripts/train_mpc_proxy_score_pytorch.py generate-cache \
    --config score_task_stack \
    --hdf5_path $MUJOCO_EVAL_DATA/stack_d0/demo_224.hdf5 \
    --cache_path <cache>.npz \
    --task stack \
    --task_module mujoco_eval/tasks/offline_labels.py \
    --vlm_cost_config mujoco_eval/configs/core.yaml \
    --prompt "stack the red block on the green block" \
    --subtask_mode empty \
    --mpc_cost priority \
    --mpc_num_samples 4096 \
    --mpc_iterations 1 \
    --mpc_noise 0.8 \
    --mpc_temperature 0.1 \
    --mpc_joint_delta_clip 0.3 \
    --control_frequency 20 \
    --stride 4 \
    --num_steps 10
```

Then train the proxy on the cache:

```bash
python scripts/train_mpc_proxy_score_pytorch.py train \
    --config score_task_stack \
    --hdf5_path $MUJOCO_EVAL_DATA/stack_d0/demo_224.hdf5 \
    --cache_path <cache>.npz \
    --prompt "stack the red block on the green block" \
    --exp_name task_score \
    --checkpoint_base_dir <ckpt dir> \
    --batch_size 32
```

---

## Tasks and base results

The following results use `base.yaml`, 4096 candidates, and 50 seeds.

| task | base | task | base |
|---|---:|---|---:|
| `lift` | 50/50 | `threading` | 0/50 |
| `stack` | 47/50 | `coffee` | 0/50 |
| `can` | 34/50 | `mug_cleanup` | 0/50 |
| `stack_three` | 4/50 | `three_piece_assembly` | 0/50 |
| `square` | 5/50* | `hammer_cleanup` | 0/50 |
|  |  | `kitchen`, `coffee_prep` | 0/50 |

\* See the `square` table below; `base` scored 0 on the 34 seeds that completed before the screen was
stopped.

A zero result does not necessarily indicate a broken task. The harder tasks remain useful stress tests
for mechanisms intended to improve the base.

### `square`, seeds 101–150

Cost design mattered more than any steering mechanism tested. The `not_hold` sigma was the largest single
lever, but the term remained load-bearing: removing it entirely performed worse than either calibrated
setting.

| arm | cost | result |
|---|---|---:|
| base | `insert` (26 terms, sigma 0.05) | 0/50 |
| base | `insert_nhcal` (26 terms, sigma 0.0085) | 5/50 |
| base | `insert_min` (16 terms, sigma 0.0085) | **7/50** |
| base | `insert_nh0` (`not_hold` deleted) | 0/25 |
| expert | `--steer expert --horizon 15 --spi 15` | **19/50** |
| expert | `--steer expert --spi 8` | 12/50 |

`--spi` had a larger effect than any measured mechanism: executing the full 15-row chunk improved the
expert from 12/50 to 19/50. The base behaved differently; it required `--spi 2` and scored 1/50 at `--spi
8`.

No steering arm outperformed the expert. `verify` was monotonic in its gate: the base overrode 90.6%,
49.9%, and 35.8% of chunks at gates 0.0, 1.0, and 3.0, producing 1/50, 3/25, and 6/25. Success therefore
rose as intervention decreased, with the expert alone remaining the best point. `tilt` was harmful:
stage-0 clearance fell from 96% to 62% to 24% at lambda 0, 0.25, and 1.0.

---

## Configuration files

A configuration may define:

- cost terms, weights, and geometry
- an `advance:` block for stage transitions
- an optional `planner:` block that overrides CLI settings

Pass either a configuration name or a file path.

| config | description |
|---|---|
| `base` | Default configuration: core cost plus B-spline action interpolation |
| `core` | Same cost without interpolation; used as the ablation and for benchmark and label scoring |
| `full` | Enables every cost term, including terms omitted by `core`; intended as an upper bound rather than a stronger default |
| `insert` | Adds an insertion corridor that funnels toward the peg axis and gates descent; used for `square` and `threading` |
| `insert_nhcal` | `insert` with `not_hold_sigma` 0.05 -> 0.0085; 0/50 -> 5/50 on `square` |
| `insert_min` | `insert_nhcal` minus the 10-term pose-shaping stack; the strongest `square` base at 7/50 |
| `insert_nh0` | `insert_min` with `not_hold` deleted; 0/25, i.e. the term is load-bearing |
| `churn` | Adds a regrasp penalty and hold-latch hysteresis; reduced churn from 17.4% to 0.2% |
| `artic` | Adds articulated-fixture terms such as `hook_pull` and `press_axis`; used for kitchens and drawers |
| `rekep` | Adds ReKep constraint costs; pair with `--ground rekep` |
| `rekep_vlm` | Lets the VLM control the place stage and release gate; pair with `--ground rekep_vlm` |
| `keypose` | Uses the keypose sampler with `(chunk, waypoints)` candidates; pair with `--kp` |

The `base`, `insert`, `churn`, and `artic` configurations include interpolation. The remaining
configurations do not and should therefore be compared with `core`.

B-spline interpolation produced the largest single base improvement. Representing an 8-row action chunk
with four B-spline knots reduced the search dimension by 4x. At the same sample budget, success increased
from 11/50 to 47/50 on `stack` and from 12/50 to 34/50 on `can`.

---

## Evaluation throughput

Evaluation throughput motivated this harness: the IsaacLab workflow could not reliably screen a
hypothesis within one day.

| metric | IsaacLab | MuJoCo harness | factor |
|---|---:|---:|---:|
| Per episode, single process | ~6 min | 0.7–1.9 min | ~3–4× |
| Concurrent episodes per machine | 1–2, up to 6 at 2 per GPU | 8+ and CPU-bound | ~4–8× |
| 20-seed screen, wall time | 2–6 h | ~18 min | **~10–20×** |
| 12 tasks × 50 seeds | ~2.5 GPU-weeks | 4 h 08 m | Overnight |

Four factors drive the throughput gain:

1. **Simulator overhead:** MuJoCo avoids PhysX and RTX stepping, a 62-second policy load, and a 20-second scene reset for each episode.
2. **Parallel execution:** Episodes are independent and CPU-bound, so eight workers can saturate one machine. IsaacLab is typically limited to about two episodes per GPU.
3. **IsaacLab optimizations:** The `--base_decode_only` and `--fast_gt` flags still improve IsaacLab performance by about 1.4× per episode and 3.5× during startup, but they are not the main source of the throughput gain.
4. **Proxy RPC optimization:** The serving path previously rebuilt its DDIM alpha table 22 times per request. Memoizing the table reduced one chain from 7.64 seconds to 0.209 seconds, a 36× improvement with bitwise-identical output. As a result, steered runs now cost about the same as base runs.

Interpret the comparison with two caveats:

- Per-episode speed improves by only about 3–4x; the main benefit is aggregate throughput.
- Because this harness uses a different simulator, it is intended for hypothesis screening rather than final confirmation. Confirm final results in IsaacLab.

---

## Repository layout

```text
eval.py       CLI + per-task defaults        parallel.py   N seeds x W workers
runner.py     the rollout loop               selection.py  best-of-M + sub-goal verifier
record.py     jsonl schema + video           paths.py      roots, derived or env-overridden
container.py  host <-> container paths
env/          MuJoCoEnv, MGWorld, the JOINT_POSITION delta shim
grounding/    gt.py · rekep.py · propose.py
tasks/        registry.py (ladders, extents, success predicates) · offline_labels.py
bench/        bench.py (G1 demo-compatibility) · fk_fit.py · fk_fits/
sampling/     keypose.py · beam.py           steering/  proxy.py
perturb/      protocol.py                    configs/  setup/  tests/
```

New mechanisms usually integrate through `runner.py`, which exposes four extension points that are
disabled by default:

- keypose
- beam
- steering
- perturbation

Use `tests/identity_gate.py` to confirm that a change preserves baseline behavior. Use
`tests/demo_domain.py` to confirm that rendered training data matches the evaluation renderer.
