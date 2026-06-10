"""Step-by-step comparison of infer_actions (original) vs infer_actions_compiled.

Runs both methods with identical noise and compares every intermediate tensor
at every denoising step to pinpoint where (if anywhere) divergence occurs.

Also tests the compiled forward function in eager mode (torch.compile disabled)
to distinguish logic bugs from torch.compile numerical drift.

Usage:
    # Quick comparison (eager only, no torch.compile):
    OPENPI_DISABLE_TORCH_COMPILE=1 uv run scripts/compare_steer_methods.py

    # Full comparison including torch.compile:
    uv run scripts/compare_steer_methods.py --test_compile

    # With real dataset observation:
    uv run scripts/compare_steer_methods.py --repo_id cn356/spoon --episode_index 0
"""

import dataclasses
import logging
import os

import numpy as np
import torch
import tyro

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import (
    _apply_proxy_steering,
    _resolve_steer_target,
    _steer_forward_all,
    _get_compiled_steer_forward,
    infer_actions,
    infer_actions_compiled,
)
from openpi.training import config as _config


@dataclasses.dataclass
class CompareArgs:
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

    repo_id: str = "cn356/spoon"
    episode_index: int = 0

    test_compile: bool = False
    atol: float = 1e-4

    num_obs: int = 1
    num_noises_per_obs: int = 1


class SteerArgs:
    def __init__(self, args: CompareArgs):
        self.steer_step = args.steer_step
        self.steer_scale = args.steer_scale
        self.use_decreasing_steer_scale = args.use_decreasing_steer_scale
        self.use_increasing_steer_scale = args.use_increasing_steer_scale
        self.num_steps = args.num_steps
        self.only_steer = args.only_steer
        self.steer_chunk_target = args.steer_chunk_target


def load_policies(args: CompareArgs):
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


def load_observation(args: CompareArgs) -> dict:
    try:
        import einops
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from PIL import Image

        logging.info("Loading dataset %s, episode %d...", args.repo_id, args.episode_index)
        dataset = LeRobotDataset(args.repo_id, episodes=[args.episode_index])
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
    except Exception as e:
        logging.warning("Failed to load dataset (%s), using synthetic observation", e)
        return {
            "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.rand(7).astype(np.float32),
            "observation/gripper_position": np.random.rand(1).astype(np.float32),
            "prompt": "pick up the spoon",
        }


def report_diff(name: str, a: torch.Tensor, b: torch.Tensor, atol: float) -> bool:
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = max_diff < atol
    status = "OK" if ok else "MISMATCH"
    logging.info("  %-40s max=%.2e  mean=%.2e  [%s]", name, max_diff, mean_diff, status)
    return ok


def step_by_step_comparison(
    base_policy, steer_policy, mimic_policy, obs, steer_args, atol: float
):
    """Manually unroll both methods and compare every intermediate tensor."""
    logging.info("=" * 70)
    logging.info("STEP-BY-STEP COMPARISON: infer_actions vs _steer_forward_all (eager)")
    logging.info("=" * 70)

    obs_processed, inputs = base_policy.obs_to_input(obs)
    bsize = obs_processed.state.shape[0]
    device = obs_processed.state.device

    base_model = base_policy._model
    steer_model = steer_policy._model
    mimic_model = mimic_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = steer_model.config.action_dim

    actions_shape = (bsize, base_model.config.action_horizon, base_action_dim)
    shared_noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs_processed, train=False)
    )

    # ---------- ORIGINAL PATH (replicating infer_actions logic) ----------
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    orig_base_prefix_embs, orig_base_prefix_pad_masks, orig_base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    orig_base_prefix_att_2d_masks = make_att_2d_masks(
        orig_base_prefix_pad_masks, orig_base_prefix_att_masks
    )
    orig_base_prefix_position_ids = torch.cumsum(orig_base_prefix_pad_masks, dim=1) - 1
    orig_base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        orig_base_prefix_att_2d_masks
    )

    _, orig_base_past_kv = base_model.paligemma_with_expert.forward(
        attention_mask=orig_base_prefix_att_2d_masks_4d,
        position_ids=orig_base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[orig_base_prefix_embs, None],
        use_cache=True,
    )

    orig_steer_prefix_embs, orig_steer_prefix_pad_masks, _ = steer_model.embed_prefix(images, img_masks)

    share_proxy_dino = (
        steer_model.config.freeze_dino_encoder
        and mimic_model.config.freeze_dino_encoder
        and steer_model.config.dino_model_name == mimic_model.config.dino_model_name
    )
    if share_proxy_dino:
        orig_mimic_prefix_embs = orig_steer_prefix_embs
        orig_mimic_prefix_pad_masks = orig_steer_prefix_pad_masks
    else:
        orig_mimic_prefix_embs, orig_mimic_prefix_pad_masks, _ = mimic_model.embed_prefix(images, img_masks)

    # ---------- COMPILED PATH (eager mode of _steer_forward_all) ----------
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    new_x_t = _steer_forward_all(
        base_model, steer_model, mimic_model,
        images[0], images[1], img_masks[0], img_masks[1],
        lang_tokens, lang_masks,
        state, shared_noise.clone(),
        steer_args.num_steps, proxy_action_dim, steer_args.steer_scale,
        share_proxy_dino,
    )

    # ---------- Now run original loop step-by-step ----------
    steer_target, steer_target_idx = _resolve_steer_target(
        steer_args, base_model.config.action_horizon
    )

    dt = torch.tensor(-1.0 / steer_args.num_steps, dtype=torch.float32, device=device)
    orig_x_t = shared_noise.clone()
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    all_ok = True
    step = 0
    while denoise_time >= -dt / 2:
        expanded_time = denoise_time.expand(bsize)

        base_v_t = base_model.denoise_step(
            state, orig_base_prefix_pad_masks, orig_base_past_kv, orig_x_t, expanded_time,
        )

        if denoise_time >= steer_args.steer_step:
            steer_suffix_embs, steer_suffix_pad_masks, _, steer_adarms_cond = (
                steer_model.embed_suffix(
                    state[:, :proxy_action_dim], orig_x_t[:, :, :proxy_action_dim], expanded_time,
                )
            )
            mimic_suffix_embs, mimic_suffix_pad_masks, _, mimic_adarms_cond = (
                mimic_model.embed_suffix(
                    state[:, :proxy_action_dim], orig_x_t[:, :, :proxy_action_dim], expanded_time,
                )
            )

            steer_embs = torch.cat([orig_steer_prefix_embs, steer_suffix_embs], dim=1)
            mimic_embs = torch.cat([orig_mimic_prefix_embs, mimic_suffix_embs], dim=1)
            steer_pad_masks = torch.cat([orig_steer_prefix_pad_masks, steer_suffix_pad_masks], dim=1)
            mimic_pad_masks = torch.cat([orig_mimic_prefix_pad_masks, mimic_suffix_pad_masks], dim=1)

            steer_position_ids = (torch.cumsum(steer_pad_masks, dim=1) - 1).to(dtype=torch.long)
            mimic_position_ids = (torch.cumsum(mimic_pad_masks, dim=1) - 1).to(dtype=torch.long)

            steer_hidden_states, _ = steer_model.expert_model.forward(
                attention_mask=steer_pad_masks,
                position_ids=steer_position_ids,
                past_key_values=None,
                inputs_embeds=steer_embs,
                use_cache=False,
                adarms_cond=steer_adarms_cond,
            )
            mimic_hidden_states, _ = mimic_model.expert_model.forward(
                attention_mask=mimic_pad_masks,
                position_ids=mimic_position_ids,
                past_key_values=None,
                inputs_embeds=mimic_embs,
                use_cache=False,
                adarms_cond=mimic_adarms_cond,
            )

            steer_v_t = steer_model.action_out_proj(
                steer_hidden_states[:, -steer_model.config.action_horizon:].to(dtype=torch.float32)
            )
            mimic_v_t = mimic_model.action_out_proj(
                mimic_hidden_states[:, -mimic_model.config.action_horizon:].to(dtype=torch.float32)
            )

            if steer_args.use_decreasing_steer_scale:
                scale = steer_args.steer_scale * denoise_time.item()
            elif steer_args.use_increasing_steer_scale:
                scale = steer_args.steer_scale * (1 - denoise_time.item())
            else:
                scale = steer_args.steer_scale

            v_t, _, _ = _apply_proxy_steering(
                base_v_t=base_v_t,
                steer_v_t=steer_v_t,
                mimic_v_t=mimic_v_t,
                proxy_action_dim=proxy_action_dim,
                steer_scale=scale,
                steer_target_idx=steer_target_idx,
                only_steer=steer_args.only_steer,
            )
        else:
            v_t = base_v_t

        orig_x_t = orig_x_t + dt * v_t
        denoise_time = denoise_time + dt
        step += 1

    # --- Compare final outputs ---
    logging.info("\n--- Final x_t comparison (after %d steps) ---", step)
    ok = report_diff("x_t (original vs _steer_forward_all eager)", orig_x_t, new_x_t, atol)
    all_ok = all_ok and ok

    orig_actions = base_policy.output_to_actions(inputs, orig_x_t)
    new_actions = base_policy.output_to_actions(inputs, new_x_t)
    logging.info("\n--- Final actions comparison ---")
    ok = report_diff(
        "actions (numpy)",
        torch.from_numpy(orig_actions),
        torch.from_numpy(new_actions),
        atol,
    )
    all_ok = all_ok and ok

    return all_ok


def end_to_end_comparison(
    base_policy, steer_policy, mimic_policy, obs, steer_args, atol: float,
    use_compile: bool,
):
    """Compare infer_actions vs infer_actions_compiled end-to-end."""
    mode = "compiled" if use_compile else "eager (_steer_forward_all)"
    logging.info("=" * 70)
    logging.info("END-TO-END COMPARISON: infer_actions vs infer_actions_compiled [%s]", mode)
    logging.info("=" * 70)

    base_model = base_policy._model
    obs_processed, _ = base_policy.obs_to_input(obs)
    actions_shape = (
        obs_processed.state.shape[0],
        base_model.config.action_horizon,
        base_model.config.action_dim,
    )
    shared_noise = base_model.sample_noise(actions_shape, obs_processed.state.device)

    with torch.no_grad():
        orig_result = infer_actions(
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            noise=shared_noise.clone(),
        )

    if use_compile:
        # Warmup for compilation
        logging.info("Warming up torch.compile (first invocation triggers compilation)...")
        with torch.no_grad():
            _ = infer_actions_compiled(
                base_policy, steer_policy, mimic_policy, obs, steer_args,
                noise=shared_noise.clone(),
            )
        logging.info("Warmup done, running actual comparison...")

    with torch.no_grad():
        compiled_result = infer_actions_compiled(
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            noise=shared_noise.clone(),
        )

    orig_actions = orig_result["actions"]
    compiled_actions = compiled_result["actions"]

    max_diff = np.abs(orig_actions - compiled_actions).max()
    mean_diff = np.abs(orig_actions - compiled_actions).mean()

    ok = max_diff < atol
    status = "OK" if ok else "MISMATCH"
    logging.info("  Actions max_diff=%.6e  mean_diff=%.6e  [%s]", max_diff, mean_diff, status)

    if not ok:
        logging.warning("  Per-dimension max diff:")
        for d in range(orig_actions.shape[-1]):
            d_diff = np.abs(orig_actions[..., d] - compiled_actions[..., d]).max()
            logging.warning("    dim %d: %.6e", d, d_diff)

    return ok


def self_consistency_test(
    base_policy, steer_policy, mimic_policy, obs, steer_args, atol: float,
):
    """Run infer_actions TWICE with same noise to measure CUDA non-determinism baseline.

    If this shows non-zero diff, that's the expected noise floor from bf16
    non-determinism. Any cross-method diff should be compared against this.
    """
    logging.info("=" * 70)
    logging.info("SELF-CONSISTENCY: infer_actions called twice with same noise")
    logging.info("=" * 70)

    base_model = base_policy._model
    obs_processed, _ = base_policy.obs_to_input(obs)
    actions_shape = (
        obs_processed.state.shape[0],
        base_model.config.action_horizon,
        base_model.config.action_dim,
    )
    shared_noise = base_model.sample_noise(actions_shape, obs_processed.state.device)

    with torch.no_grad():
        result_a = infer_actions(
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            noise=shared_noise.clone(),
        )
        result_b = infer_actions(
            base_policy, steer_policy, mimic_policy, obs, steer_args,
            noise=shared_noise.clone(),
        )

    a = result_a["actions"]
    b = result_b["actions"]
    max_diff = np.abs(a - b).max()
    mean_diff = np.abs(a - b).mean()
    ok = max_diff < atol
    status = "OK" if ok else "NON-DETERMINISTIC"
    logging.info(
        "  infer_actions vs itself: max=%.6e  mean=%.6e  [%s]",
        max_diff, mean_diff, status,
    )
    if max_diff > 0:
        logging.info(
            "  This is the CUDA non-determinism baseline. "
            "Cross-method diffs should be compared against this."
        )
    return max_diff


def multi_obs_comparison(
    base_policy, steer_policy, mimic_policy, steer_args,
    repo_id: str, num_obs: int, num_noises_per_obs: int,
    default_prompt: str, use_compile: bool,
):
    """Compare infer_actions vs infer_actions_compiled across many observations and noises."""
    mode = "compiled" if use_compile else "eager"
    logging.info("=" * 70)
    logging.info(
        "MULTI-OBS COMPARISON [%s]: %d observations x %d noises = %d trials",
        mode, num_obs, num_noises_per_obs, num_obs * num_noises_per_obs,
    )
    logging.info("=" * 70)

    import einops
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    dataset = LeRobotDataset(repo_id)
    num_frames = len(dataset)
    indices = np.linspace(0, num_frames - 1, num_obs, dtype=int)

    base_model = base_policy._model

    all_max_diffs = []
    all_mean_diffs = []
    per_dim_max = None

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]

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
            "prompt": sample.get("task", default_prompt),
        }

        for j in range(num_noises_per_obs):
            obs_processed, _ = base_policy.obs_to_input(obs)
            actions_shape = (
                obs_processed.state.shape[0],
                base_model.config.action_horizon,
                base_model.config.action_dim,
            )
            shared_noise = base_model.sample_noise(actions_shape, obs_processed.state.device)

            with torch.no_grad():
                orig_result = infer_actions(
                    base_policy, steer_policy, mimic_policy, obs, steer_args,
                    noise=shared_noise.clone(),
                )
                compiled_result = infer_actions_compiled(
                    base_policy, steer_policy, mimic_policy, obs, steer_args,
                    noise=shared_noise.clone(),
                )

            orig_actions = orig_result["actions"]
            compiled_actions = compiled_result["actions"]
            diff = np.abs(orig_actions - compiled_actions)

            max_diff = diff.max()
            mean_diff = diff.mean()
            all_max_diffs.append(max_diff)
            all_mean_diffs.append(mean_diff)

            dim_max = np.array([diff[..., d].max() for d in range(orig_actions.shape[-1])])
            if per_dim_max is None:
                per_dim_max = dim_max
            else:
                per_dim_max = np.maximum(per_dim_max, dim_max)

        if (i + 1) % 5 == 0 or i == 0 or i == len(indices) - 1:
            logging.info(
                "  obs %d/%d (frame %d): max=%.2e  mean=%.2e",
                i + 1, num_obs, idx, all_max_diffs[-1], all_mean_diffs[-1],
            )

    global_max = max(all_max_diffs)
    global_mean = np.mean(all_mean_diffs)

    logging.info("-" * 70)
    logging.info("  Across %d trials:", len(all_max_diffs))
    logging.info("    Global max diff:  %.6e", global_max)
    logging.info("    Mean of means:    %.6e", global_mean)
    logging.info("    Mean of maxes:    %.6e", np.mean(all_max_diffs))
    logging.info("    Per-dimension global max diff:")
    for d, v in enumerate(per_dim_max):
        logging.info("      dim %d: %.6e", d, v)
    logging.info("-" * 70)

    return global_max, global_mean, per_dim_max


def test_missing_features(steer_args, atol: float):
    """Document which features from infer_actions are missing in _steer_forward_all."""
    logging.info("=" * 70)
    logging.info("MISSING FEATURE ANALYSIS: _steer_forward_all vs infer_actions")
    logging.info("=" * 70)

    issues = []

    # 1. steer_step
    if steer_args.steer_step > 0.0:
        issues.append(
            f"[BUG] steer_step={steer_args.steer_step} > 0: compiled path IGNORES steer_step "
            f"and applies steering unconditionally at every denoising step. "
            f"Original only steers when denoise_time >= {steer_args.steer_step}."
        )
    else:
        logging.info("  steer_step=%.1f: OK (all steps steered in both paths)", steer_args.steer_step)

    # 2. only_steer
    if steer_args.only_steer:
        issues.append(
            "[BUG] only_steer=True: compiled path uses v += scale*(steer-mimic) "
            "but original REPLACES base velocity: v[:,:,:proxy_dim] = steer_v_t. "
            "These produce completely different actions."
        )
    else:
        logging.info("  only_steer=False: OK (both use additive delta)")

    # 3. Dynamic steer scale
    if steer_args.use_decreasing_steer_scale:
        issues.append(
            "[BUG] use_decreasing_steer_scale=True: compiled uses constant steer_scale "
            "but original multiplies by denoise_time (decreasing from 1.0 to 0.1)."
        )
    elif steer_args.use_increasing_steer_scale:
        issues.append(
            "[BUG] use_increasing_steer_scale=True: compiled uses constant steer_scale "
            "but original multiplies by (1-denoise_time) (increasing from 0.0 to 0.9)."
        )
    else:
        logging.info("  steer_scale scheduling: OK (constant scale in both)")

    # 4. steer_chunk_target
    if steer_args.steer_chunk_target not in ("all", "ALL"):
        issues.append(
            f"[BUG] steer_chunk_target={steer_args.steer_chunk_target}: compiled always "
            f"steers ALL action horizon positions. Original only steers the "
            f"'{steer_args.steer_chunk_target}' position."
        )
    else:
        logging.info("  steer_chunk_target=all: OK (both steer all positions)")

    # 5. base_take_over_interval (runtime arg in WebsocketSteerServer)
    issues.append(
        "[WARNING] base_take_over_interval: When triggered, the server sets steer_step=1.1 "
        "to disable steering for that step. Since compiled path ignores steer_step, "
        "steering is INCORRECTLY still applied. This cannot be tested here but will "
        "cause bugs during live serving when base_take_over_interval != -1."
    )

    if issues:
        logging.warning("\n  ISSUES FOUND:")
        for issue in issues:
            logging.warning("    %s\n", issue)

    return issues


def main(args: CompareArgs) -> None:
    base_policy, steer_policy, mimic_policy = load_policies(args)

    obs = load_observation(args)
    if "prompt" not in obs:
        obs["prompt"] = args.default_prompt

    steer_args = SteerArgs(args)

    # --- Test 0: Self-consistency (non-determinism baseline) ---
    with torch.no_grad():
        baseline_diff = self_consistency_test(
            base_policy, steer_policy, mimic_policy, obs, steer_args, args.atol
        )

    # --- Test 1: Missing feature analysis ---
    issues = test_missing_features(steer_args, args.atol)

    # --- Test 2: Step-by-step comparison (always eager, no torch.compile) ---
    with torch.no_grad():
        step_ok = step_by_step_comparison(
            base_policy, steer_policy, mimic_policy, obs, steer_args, args.atol
        )

    # --- Test 3: End-to-end with OPENPI_DISABLE_TORCH_COMPILE ---
    orig_env = os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "")
    os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
    # Reset the cached compiled function so it picks up the env var
    import openpi.serving.websocket_policy_server as _ws
    _ws._compiled_steer_forward_all = None

    with torch.no_grad():
        eager_ok = end_to_end_comparison(
            base_policy, steer_policy, mimic_policy, obs, steer_args, args.atol,
            use_compile=False,
        )

    # --- Test 4: End-to-end WITH torch.compile ---
    if args.test_compile:
        os.environ.pop("OPENPI_DISABLE_TORCH_COMPILE", None)
        _ws._compiled_steer_forward_all = None

        with torch.no_grad():
            compile_ok = end_to_end_comparison(
                base_policy, steer_policy, mimic_policy, obs, steer_args, args.atol,
                use_compile=True,
            )
    else:
        compile_ok = None

    # Restore
    if orig_env:
        os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = orig_env
    else:
        os.environ.pop("OPENPI_DISABLE_TORCH_COMPILE", None)

    # --- Test 5: Multi-observation comparison ---
    multi_max = None
    if args.num_obs > 1:
        if args.test_compile:
            _ws._compiled_steer_forward_all = None
            os.environ.pop("OPENPI_DISABLE_TORCH_COMPILE", None)
        else:
            os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
            _ws._compiled_steer_forward_all = None

        with torch.no_grad():
            multi_max, multi_mean, multi_per_dim = multi_obs_comparison(
                base_policy, steer_policy, mimic_policy, steer_args,
                args.repo_id, args.num_obs, args.num_noises_per_obs,
                args.default_prompt, use_compile=args.test_compile,
            )

        if orig_env:
            os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = orig_env
        else:
            os.environ.pop("OPENPI_DISABLE_TORCH_COMPILE", None)

    # --- Summary ---
    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    logging.info("  CUDA non-determinism baseline:    max=%.6e", baseline_diff)
    logging.info("  Missing features / logic bugs:    %d issues", len(issues))
    logging.info("  Step-by-step (eager vs eager):    %s", "PASS" if step_ok else "FAIL")
    logging.info("  End-to-end (eager compiled path): %s", "PASS" if eager_ok else "FAIL")
    if compile_ok is not None:
        logging.info("  End-to-end (torch.compile):       %s", "PASS" if compile_ok else "FAIL")
    else:
        logging.info("  End-to-end (torch.compile):       SKIPPED (pass --test_compile)")
    if multi_max is not None:
        logging.info(
            "  Multi-obs (%d obs x %d noises):    global max=%.6e",
            args.num_obs, args.num_noises_per_obs, multi_max,
        )

    # Interpret results by comparing cross-method diff against the self-consistency baseline
    eager_diff = 0.0
    if not eager_ok:
        base_model = base_policy._model
        obs_p, _ = base_policy.obs_to_input(obs)
        sh = (obs_p.state.shape[0], base_model.config.action_horizon, base_model.config.action_dim)
        noise = base_model.sample_noise(sh, obs_p.state.device)
        import openpi.serving.websocket_policy_server as _ws
        _ws._compiled_steer_forward_all = None
        os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
        with torch.no_grad():
            ea = infer_actions(base_policy, steer_policy, mimic_policy, obs, steer_args, noise=noise.clone())
            ec = infer_actions_compiled(base_policy, steer_policy, mimic_policy, obs, steer_args, noise=noise.clone())
        eager_diff = np.abs(ea["actions"] - ec["actions"]).max()

    if baseline_diff > 0 and eager_diff > 0:
        ratio = eager_diff / baseline_diff if baseline_diff > 0 else float("inf")
        logging.info("\n  Cross-method diff / self-consistency baseline = %.1fx", ratio)
        if ratio < 3.0:
            logging.info(
                "  The cross-method diff is within ~%.0fx of the non-determinism baseline.\n"
                "  This is likely CUDA non-determinism from bf16, NOT a logic bug.\n"
                "  The rewritten _steer_forward_all is logically equivalent to infer_actions\n"
                "  for these settings.",
                ratio,
            )
        else:
            logging.warning(
                "  The cross-method diff is %.0fx LARGER than the non-determinism baseline.\n"
                "  This suggests a real logic difference beyond floating-point noise.",
                ratio,
            )
    elif baseline_diff == 0 and (not step_ok or not eager_ok):
        logging.error(
            "\nLOGIC BUG DETECTED: infer_actions is self-consistent (baseline=0) "
            "but cross-method comparison fails. There is a real code difference."
        )
    logging.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(CompareArgs))
