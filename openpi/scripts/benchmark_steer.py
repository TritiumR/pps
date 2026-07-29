"""Standalone benchmark for 3-model steering inference latency.

Loads base/steer/mimic models and runs inference on dataset observations,
measuring end-to-end and per-component timings. Compares infer_actions (slow)
vs infer_actions_fast (KV-cached proxies).

Usage:
    uv run scripts/benchmark_steer.py \
        --base_model_name pi0_droid_jointpos \
        --steer_model_name proxy_real_droid_spoon_jointpos \
        --mimic_model_name proxy_real_droid_spoon_jointpos \
        --base_checkpoint_dir checkpoints/pytorch/pi0_droid_jointpos \
        --steer_checkpoint_dir checkpoints/proxy_real_droid_spoon_jointpos/steer_from_mimic/8000 \
        --mimic_checkpoint_dir checkpoints/proxy_real_droid_spoon_jointpos/distill_on_the_fly/40000

    Pass --profile to emit a torch.profiler trace for TensorBoard.
"""

import dataclasses
import logging
import statistics
import time

import numpy as np
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import (
    infer_actions,
    infer_actions_fast,
    infer_actions_compiled,
)
from openpi.models_pytorch.proxy_pytorch import build_proxy_expert_masks
from openpi.training import config as _config


@dataclasses.dataclass
class BenchmarkArgs:
    base_model_name: str = "pi0_droid_jointpos"
    steer_model_name: str = "proxy_real_droid_spoon_jointpos"
    mimic_model_name: str = "proxy_real_droid_spoon_jointpos"
    base_checkpoint_dir: str = "checkpoints/pytorch/pi0_droid_jointpos"
    steer_checkpoint_dir: str = "checkpoints/proxy_real_droid_spoon_jointpos/steer_from_mimic/8000"
    mimic_checkpoint_dir: str = "checkpoints/proxy_real_droid_spoon_jointpos/distill_on_the_fly/40000"

    default_prompt: str = "pick up the spoon"

    steer_step: float = 0.0
    steer_scale: float = 0.4
    use_decreasing_steer_scale: bool = False
    use_increasing_steer_scale: bool = False
    num_steps: int = 10
    only_steer: bool = False
    steer_chunk_target: str = "all"

    num_warmup: int = 3
    num_samples: int = 20

    repo_id: str = "cn356/spoon"
    episode_index: int = 0

    profile: bool = False
    profile_dir: str = "benchmark_traces"

    skip_comparison: bool = False
    test_compile: bool = False


def load_policies(args: BenchmarkArgs):
    logging.info("Loading steer policy...")
    steer_policy = _policy_config.create_trained_policy(
        _config.get_config(args.steer_model_name),
        args.steer_checkpoint_dir,
    )
    logging.info("Loading mimic policy...")
    mimic_policy = _policy_config.create_trained_policy(
        _config.get_config(args.mimic_model_name),
        args.mimic_checkpoint_dir,
    )
    logging.info("Loading base policy...")
    base_policy = _policy_config.create_trained_policy(
        _config.get_config(args.base_model_name),
        args.base_checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    return base_policy, steer_policy, mimic_policy


def load_observation_from_dataset(repo_id: str, episode_index: int) -> dict:
    """Load a single observation from a LeRobot dataset and format it
    the same way the robot client would send it over websocket."""
    import einops
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    logging.info("Loading dataset %s, episode %d...", repo_id, episode_index)
    dataset = LeRobotDataset(repo_id, episodes=[episode_index])
    sample = dataset[0]

    ext_img = np.array(sample["exterior_image_1_left"])
    wrist_img = np.array(sample["wrist_image_left"])
    if ext_img.shape[0] == 3:
        ext_img = einops.rearrange(ext_img, "c h w -> h w c")
    if wrist_img.shape[0] == 3:
        wrist_img = einops.rearrange(wrist_img, "c h w -> h w c")
    if np.issubdtype(ext_img.dtype, np.floating):
        ext_img = (255 * ext_img).astype(np.uint8)
    if np.issubdtype(wrist_img.dtype, np.floating):
        wrist_img = (255 * wrist_img).astype(np.uint8)

    from PIL import Image
    ext_img = np.array(Image.fromarray(ext_img).resize((224, 224)))
    wrist_img = np.array(Image.fromarray(wrist_img).resize((224, 224)))

    gripper = np.array(sample["gripper_position"])
    if gripper.ndim == 0:
        gripper = gripper[np.newaxis]

    obs = {
        "observation/exterior_image_1_left": ext_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": np.array(sample["joint_position"]),
        "observation/gripper_position": gripper,
    }

    if "task" in sample:
        obs["prompt"] = sample["task"]

    return obs


def make_synthetic_observation() -> dict:
    """Fallback: create a random observation matching DROID format."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7).astype(np.float32),
        "observation/gripper_position": np.random.rand(1).astype(np.float32),
        "prompt": "pick up the spoon",
    }


class SteerArgs:
    """Mimics the Args dataclass from steer_policy.py for passing to infer_actions."""
    def __init__(self, bench_args: BenchmarkArgs):
        self.steer_step = bench_args.steer_step
        self.steer_scale = bench_args.steer_scale
        self.use_decreasing_steer_scale = bench_args.use_decreasing_steer_scale
        self.use_increasing_steer_scale = bench_args.use_increasing_steer_scale
        self.num_steps = bench_args.num_steps
        self.only_steer = bench_args.only_steer
        self.steer_chunk_target = bench_args.steer_chunk_target


def timed_infer(fn, base_policy, steer_policy, mimic_policy, obs, steer_args, noise=None, **kwargs):
    """Run an inference function with CUDA-synchronized wall-clock timing."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(base_policy, steer_policy, mimic_policy, obs, steer_args, noise=noise, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def benchmark_variant(
    name, fn, base_policy, steer_policy, mimic_policy, obs, steer_args,
    num_warmup, num_samples, **kwargs,
):
    """Run warmup + timed samples for a single variant, return list of ms."""
    logging.info("Benchmarking: %s (%d warmup, %d samples)", name, num_warmup, num_samples)
    for _ in range(num_warmup):
        with torch.no_grad():
            fn(base_policy, steer_policy, mimic_policy, obs, steer_args, **kwargs)

    times = []
    for i in range(num_samples):
        with torch.no_grad():
            _, ms = timed_infer(fn, base_policy, steer_policy, mimic_policy, obs, steer_args, **kwargs)
        times.append(ms)
        if (i + 1) % 5 == 0:
            logging.info("  sample %d/%d: %.1f ms", i + 1, num_samples, ms)
    return times


def run_component_breakdown(base_policy, steer_policy, mimic_policy, obs, steer_args, num_runs=10):
    """Measure per-component latency inside the denoising loop using CUDA events."""
    logging.info("Running per-component breakdown (%d runs)...", num_runs)

    obs_processed, inputs = base_policy.obs_to_input(obs)
    bsize = obs_processed.state.shape[0]
    device = obs_processed.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model
    mimic_model = mimic_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs_processed, train=False)
    )

    component_times = {
        "base_prefix_setup": [],
        "proxy_prefix_setup": [],
        "base_denoise_step": [],
        "steer_embed_suffix": [],
        "mimic_embed_suffix": [],
        "steer_expert_forward": [],
        "mimic_expert_forward": [],
        "apply_steering": [],
    }

    for run_idx in range(num_runs):
        with torch.no_grad():
            # --- Base prefix (one-time) ---
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

            base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
                base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            )
            base_prefix_att_2d_masks = make_att_2d_masks(base_prefix_pad_masks, base_prefix_att_masks)
            base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1
            base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(base_prefix_att_2d_masks)
            base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
            _, base_past_key_values = base_model.paligemma_with_expert.forward(
                attention_mask=base_prefix_att_2d_masks_4d,
                position_ids=base_prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[base_prefix_embs, None],
                use_cache=True,
            )

            ev_end.record()
            torch.cuda.synchronize()
            component_times["base_prefix_setup"].append(ev_start.elapsed_time(ev_end))

            # --- Proxy prefix (one-time) ---
            ev_start.record()

            steer_prefix_embs, steer_prefix_pad_masks, steer_prefix_att_masks = steer_model.embed_prefix(images, img_masks)
            if (
                steer_model.config.freeze_dino_encoder
                and mimic_model.config.freeze_dino_encoder
                and steer_model.config.dino_model_name == mimic_model.config.dino_model_name
            ):
                mimic_prefix_embs = steer_prefix_embs
                mimic_prefix_pad_masks = steer_prefix_pad_masks
                mimic_prefix_att_masks = steer_prefix_att_masks
            else:
                mimic_prefix_embs, mimic_prefix_pad_masks, mimic_prefix_att_masks = mimic_model.embed_prefix(images, img_masks)

            ev_end.record()
            torch.cuda.synchronize()
            component_times["proxy_prefix_setup"].append(ev_start.elapsed_time(ev_end))

            # --- Denoising loop (single step measurement) ---
            actions_shape = (bsize, base_model.config.action_horizon, base_action_dim)
            noise = base_model.sample_noise(actions_shape, device)
            x_t = noise
            denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)
            expanded_time = denoise_time.expand(bsize)

            # base_denoise_step
            ev_start.record()
            base_v_t = base_model.denoise_step(
                state, base_prefix_pad_masks, base_past_key_values, x_t, expanded_time,
            )
            ev_end.record()
            torch.cuda.synchronize()
            component_times["base_denoise_step"].append(ev_start.elapsed_time(ev_end))

            # steer embed_suffix
            ev_start.record()
            steer_suffix_embs, steer_suffix_pad_masks, steer_suffix_att_masks, steer_adarms_cond = (
                steer_model.embed_suffix(state[:, :proxy_action_dim], x_t[:, :, :proxy_action_dim], expanded_time)
            )
            ev_end.record()
            torch.cuda.synchronize()
            component_times["steer_embed_suffix"].append(ev_start.elapsed_time(ev_end))

            # mimic embed_suffix
            ev_start.record()
            mimic_suffix_embs, mimic_suffix_pad_masks, mimic_suffix_att_masks, mimic_adarms_cond = (
                mimic_model.embed_suffix(state[:, :proxy_action_dim], x_t[:, :, :proxy_action_dim], expanded_time)
            )
            ev_end.record()
            torch.cuda.synchronize()
            component_times["mimic_embed_suffix"].append(ev_start.elapsed_time(ev_end))

            # steer expert forward
            steer_embs = torch.cat([steer_prefix_embs, steer_suffix_embs], dim=1)
            steer_attention_mask, steer_pad_masks = build_proxy_expert_masks(
                steer_model, steer_prefix_pad_masks, steer_prefix_att_masks,
                steer_suffix_pad_masks, steer_suffix_att_masks,
            )
            steer_position_ids = torch.cumsum(steer_pad_masks, dim=1).to(dtype=torch.long) - 1

            ev_start.record()
            steer_hidden_states, _ = steer_model.expert_model.forward(
                attention_mask=steer_attention_mask,
                position_ids=steer_position_ids,
                past_key_values=None,
                inputs_embeds=steer_embs,
                use_cache=False,
                adarms_cond=steer_adarms_cond,
            )
            ev_end.record()
            torch.cuda.synchronize()
            component_times["steer_expert_forward"].append(ev_start.elapsed_time(ev_end))

            # mimic expert forward
            mimic_embs = torch.cat([mimic_prefix_embs, mimic_suffix_embs], dim=1)
            mimic_attention_mask, mimic_pad_masks = build_proxy_expert_masks(
                mimic_model, mimic_prefix_pad_masks, mimic_prefix_att_masks,
                mimic_suffix_pad_masks, mimic_suffix_att_masks,
            )
            mimic_position_ids = torch.cumsum(mimic_pad_masks, dim=1).to(dtype=torch.long) - 1

            ev_start.record()
            mimic_hidden_states, _ = mimic_model.expert_model.forward(
                attention_mask=mimic_attention_mask,
                position_ids=mimic_position_ids,
                past_key_values=None,
                inputs_embeds=mimic_embs,
                use_cache=False,
                adarms_cond=mimic_adarms_cond,
            )
            ev_end.record()
            torch.cuda.synchronize()
            component_times["mimic_expert_forward"].append(ev_start.elapsed_time(ev_end))

            # apply steering
            steer_suffix_out = steer_hidden_states[:, -steer_model.config.action_horizon:].to(dtype=torch.float32)
            mimic_suffix_out = mimic_hidden_states[:, -mimic_model.config.action_horizon:].to(dtype=torch.float32)
            steer_v_t = steer_model.action_out_proj(steer_suffix_out)
            mimic_v_t = mimic_model.action_out_proj(mimic_suffix_out)

            ev_start.record()
            from openpi.serving.websocket_policy_server import _apply_proxy_steering
            v_t, _, _ = _apply_proxy_steering(
                base_v_t=base_v_t,
                steer_v_t=steer_v_t,
                mimic_v_t=mimic_v_t,
                proxy_action_dim=proxy_action_dim,
                steer_scale=steer_args.steer_scale,
                steer_target_idx=None,
                only_steer=steer_args.only_steer,
            )
            ev_end.record()
            torch.cuda.synchronize()
            component_times["apply_steering"].append(ev_start.elapsed_time(ev_end))

    return component_times


def run_profiler_trace(fn, base_policy, steer_policy, mimic_policy, obs, steer_args, profile_dir):
    """Capture a torch.profiler trace for TensorBoard analysis."""
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    logging.info("Capturing profiler trace to %s ...", profile_dir)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(6):
            with torch.no_grad():
                fn(base_policy, steer_policy, mimic_policy, obs, steer_args)
            prof.step()

    logging.info("Trace saved to %s. View with: tensorboard --logdir %s", profile_dir, profile_dir)


def print_results_table(results: dict[str, list[float]]):
    baseline_avg = statistics.mean(results[next(iter(results))]) if results else 1.0

    header = f"{'Variant':<30} {'Avg (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Speedup':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, times in results.items():
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        speedup = baseline_avg / avg if avg > 0 else float("inf")
        print(f"{name:<30} {avg:>10.1f} {std:>10.1f} {min(times):>10.1f} {max(times):>10.1f} {speedup:>9.2f}x")
    print("=" * len(header) + "\n")


def print_component_table(component_times: dict[str, list[float]]):
    header = f"{'Component':<30} {'Avg (ms)':>10} {'Std (ms)':>10}"
    print("\n" + "=" * len(header))
    print("Per-Component Breakdown (single denoising step)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    total = 0.0
    for name, times in component_times.items():
        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        total += avg
        print(f"{name:<30} {avg:>10.2f} {std:>10.2f}")
    print("-" * len(header))
    print(f"{'TOTAL':<30} {total:>10.2f}")
    print("=" * len(header) + "\n")


def main(args: BenchmarkArgs) -> None:
    base_policy, steer_policy, mimic_policy = load_policies(args)

    try:
        obs = load_observation_from_dataset(args.repo_id, args.episode_index)
    except Exception as e:
        logging.warning("Failed to load dataset (%s), using synthetic observation", e)
        obs = make_synthetic_observation()

    if "prompt" not in obs:
        obs["prompt"] = args.default_prompt

    steer_args = SteerArgs(args)

    # --- Per-component breakdown ---
    component_times = run_component_breakdown(
        base_policy, steer_policy, mimic_policy, obs, steer_args, num_runs=args.num_samples,
    )
    print_component_table(component_times)

    # --- End-to-end benchmarks ---
    results = {}

    # 1. Baseline: infer_actions (slow, no proxy KV cache, with viz)
    results["infer_actions (slow)"] = benchmark_variant(
        "infer_actions (slow)", infer_actions,
        base_policy, steer_policy, mimic_policy, obs, steer_args,
        args.num_warmup, args.num_samples,
    )

    if not args.skip_comparison:
        # 2. Fast path: proxy KV cache
        results["fast"] = benchmark_variant(
            "fast", infer_actions_fast,
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            args.num_warmup, args.num_samples,
        )

        # 3. Fast + skip viz
        results["fast + skip_viz"] = benchmark_variant(
            "fast + skip_viz", infer_actions_fast,
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            args.num_warmup, args.num_samples, skip_viz=True,
        )

        # 4. Fast + skip viz + concurrent proxy streams
        results["fast + skip_viz + concurrent"] = benchmark_variant(
            "fast + skip_viz + concurrent", infer_actions_fast,
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            args.num_warmup, args.num_samples, skip_viz=True, concurrent_proxies=True,
        )

        # --- Verify equivalence ---
        with torch.no_grad():
            base_model = base_policy._model
            obs_processed, _ = base_policy.obs_to_input(obs)
            actions_shape = (
                obs_processed.state.shape[0],
                base_model.config.action_horizon,
                base_model.config.action_dim,
            )
            shared_noise = base_model.sample_noise(actions_shape, obs_processed.state.device)
            slow_result = infer_actions(
                base_policy, steer_policy, mimic_policy, obs, steer_args, noise=shared_noise.clone(),
            )
            fast_result = infer_actions_fast(
                base_policy, steer_policy, mimic_policy, obs, steer_args, noise=shared_noise.clone(),
            )
            concurrent_result = infer_actions_fast(
                base_policy, steer_policy, mimic_policy, obs, steer_args,
                noise=shared_noise.clone(), concurrent_proxies=True,
            )
        slow_actions = slow_result["actions"]
        fast_actions = fast_result["actions"]
        concurrent_actions = concurrent_result["actions"]
        logging.info("Max action diff (slow vs fast): %.6e", np.abs(slow_actions - fast_actions).max())
        logging.info("Max action diff (slow vs concurrent): %.6e", np.abs(slow_actions - concurrent_actions).max())

    # --- 5. Compiled full denoising loop ---
    if args.test_compile:
        results["compiled loop"] = benchmark_variant(
            "compiled loop", infer_actions_compiled,
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            args.num_warmup + 5, args.num_samples,
        )

        # Verify compiled vs eager equivalence
        with torch.no_grad():
            base_model = base_policy._model
            obs_processed, _ = base_policy.obs_to_input(obs)
            actions_shape = (
                obs_processed.state.shape[0],
                base_model.config.action_horizon,
                base_model.config.action_dim,
            )
            shared_noise = base_model.sample_noise(actions_shape, obs_processed.state.device)
            eager_result = infer_actions(
                base_policy, steer_policy, mimic_policy, obs, steer_args, noise=shared_noise.clone(),
            )
            compiled_result = infer_actions_compiled(
                base_policy, steer_policy, mimic_policy, obs, steer_args, noise=shared_noise.clone(),
            )
        eager_actions = eager_result["actions"]
        compiled_actions = compiled_result["actions"]
        logging.info("Max action diff (eager vs compiled): %.6e", np.abs(eager_actions - compiled_actions).max())

    print_results_table(results)

    # --- Optional profiler trace ---
    if args.profile:
        run_profiler_trace(
            infer_actions_fast if not args.skip_comparison else infer_actions,
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            args.profile_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(BenchmarkArgs))
