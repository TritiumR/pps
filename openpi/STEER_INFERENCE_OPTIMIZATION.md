# 3-Model Steering Inference Optimization

## Problem

Running the 3-model steering pipeline (base PI0 3B + steer proxy 12M + mimic proxy 12M) in eager PyTorch mode took ~253ms per inference. The base model alone runs at ~20ms when using `torch.compile` via `sample_actions`, but the manual decomposition into `infer_actions` bypassed compilation entirely.

## Root Cause

`infer_actions` calls individual model methods (`denoise_step`, `embed_suffix`, `expert_model.forward`) in a Python `while` loop. Each call dispatches separate CUDA kernels with Python overhead in between. Meanwhile, `pi0_pytorch.py` compiles its entire `sample_actions` (prefix + denoising loop) into a single fused graph via `torch.compile(mode="max-autotune")`, achieving 20ms.

## Solution

Extracted the full GPU computation (prefix encoding + 10-step denoising loop) into a single `torch.compile`-friendly function `_steer_forward_all`, then compiled it with `torch.compile(mode="max-autotune")`.

### Key design decisions

- **Unconditional steering**: Removed the `if denoise_time >= args.steer_step:` branch to make the loop `torch.compile`-friendly (no data-dependent control flow).
- **No visualization in compiled path**: Removed all `.cpu().numpy()` transfers from the compiled function. Visualization data cannot be collected in the compiled path.
- **Prefix included in compiled region**: The initial version only compiled the denoising loop (108ms). Including prefix computation (PaliGemma vision/language encoder, DINO encoder, KV cache construction) brought it down to 65ms, matching how `pi0_pytorch.py` compiles `sample_actions`.
- **Explicit tensor arguments**: Images are passed as `images_0, images_1` instead of a Python list, since `torch.compile` traces tensor arguments more reliably than container types.

## Files Changed

### `openpi/src/openpi/serving/websocket_policy_server.py`

- **`_steer_forward_all()`** (new): Pure-tensor function containing all GPU work: base prefix encoding, proxy DINO prefix encoding, KV cache construction, and the 10-step denoising loop with steering applied unconditionally at every step.
- **`_get_compiled_steer_forward()`** (new): Lazy initializer that compiles `_steer_forward_all` with `torch.compile(mode="max-autotune")`. Can be disabled via `OPENPI_DISABLE_TORCH_COMPILE=1`.
- **`infer_actions_compiled()`** (new): Entry point that runs `obs_to_input` (CPU/NumPy transforms) in eager mode, delegates all GPU work to the compiled function, then runs `output_to_actions`.
- **`WebsocketSteerServer._handler`**: Default serving path changed from `infer_actions` to `infer_actions_compiled`.
- **Earlier changes** (from benchmarking phase): `skip_viz` flag on `infer_actions`/`infer_actions_fast`, `concurrent_proxies` flag on `infer_actions_fast`, bug fix in `compare_mode`.

### `openpi/scripts/benchmark_steer.py` (new)

Standalone benchmarking script that:
- Loads all three models from checkpoints
- Loads real observations from a LeRobot dataset
- Runs per-component breakdown using CUDA events
- Benchmarks multiple inference variants (eager, KV-cached, compiled)
- Verifies numerical equivalence between variants
- Optionally emits `torch.profiler` traces for TensorBoard

## Benchmark Results

Tested on cluster GPU with `pi0_droid_jointpos` (base) + `proxy_real_droid_spoon_jointpos` (steer/mimic):

| Variant | Avg (ms) | Min (ms) | Speedup |
|---|---|---|---|
| `infer_actions` (eager) | 252.8 | 235.8 | 1.00x |
| `infer_actions_compiled` (prefix + denoise) | **65.7** | **64.8** | **3.85x** |

For reference, base model only with `torch.compile` achieves ~20ms.

### Per-component breakdown (eager, single denoising step)

| Component | Avg (ms) |
|---|---|
| Base prefix setup (PaliGemma) | 47.5 |
| Proxy prefix setup (DINO) | 21.6 |
| Base denoise step | 6.3 |
| Steer/mimic embed suffix | 0.5 |
| Steer/mimic expert forward | 7.2 |
| Apply steering | 0.1 |
| **Total** | **83.1** |

## Usage

### Run the server with compiled inference (default)

```bash
cd openpi && uv run scripts/steer_policy.py \
    --base_model_name pi0_droid_jointpos \
    --steer_model_name proxy_real_droid_spoon_jointpos \
    --mimic_model_name proxy_real_droid_spoon_jointpos \
    --base_checkpoint_dir checkpoints/pytorch/pi0_droid_jointpos \
    --steer_checkpoint_dir checkpoints/proxy_real_droid_spoon_jointpos/steer_from_mimic/8000 \
    --mimic_checkpoint_dir checkpoints/proxy_real_droid_spoon_jointpos/distill_on_the_fly/40000 \
    --steer_step 0.0 --steer_scale 0.4
```

### Run the benchmark

```bash
cd openpi && uv run scripts/benchmark_steer.py --test_compile --skip_comparison --num_warmup 10 --num_samples 20
```

### Disable compilation (for debugging)

```bash
OPENPI_DISABLE_TORCH_COMPILE=1 uv run scripts/steer_policy.py ...
```

## Compilation Caching

PyTorch 2.7+ has `fx_graph_cache` and `autotune_local_cache` enabled by default. The autotuning (which takes several minutes) only runs on the very first invocation ever. Subsequent process restarts load compiled graphs and Triton kernel choices from `~/.cache/torch/inductor/`. The cache is invalidated automatically if model architecture, input shapes, PyTorch version, or GPU change.
