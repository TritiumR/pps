"""Diagnose exactly where infer_actions and _steer_forward_all diverge.

Shares the SAME prefix computation and compares per-step intermediates
to pinpoint the exact source of the difference.

Usage:
    OPENPI_DISABLE_TORCH_COMPILE=1 uv run scripts/diagnose_steer_bug.py
"""

import dataclasses
import logging

import numpy as np
import torch
import tyro

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import (
    _apply_proxy_steering,
    _resolve_steer_target,
    _steer_forward_all,
    infer_actions,
)
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    base_model_name: str = "pi0_droid_jointpos"
    steer_model_name: str = "proxy_real_droid_spoon_jointpos"
    mimic_model_name: str = "proxy_real_droid_spoon_jointpos"
    base_checkpoint_dir: str = "checkpoints/pytorch/pi0_droid_jointpos"
    steer_checkpoint_dir: str = "checkpoints/proxy_real_droid_spoon_jointpos/steer_from_mimic/8000"
    mimic_checkpoint_dir: str = "checkpoints/proxy_real_droid_spoon_jointpos/distill_on_the_fly/40000"
    default_prompt: str = "pick up the spoon"
    steer_step: float = 0.0
    steer_scale: float = 0.4
    num_steps: int = 10
    repo_id: str = "cn356/spoon"
    episode_index: int = 0


class SteerArgs:
    def __init__(self, a):
        self.steer_step = a.steer_step
        self.steer_scale = a.steer_scale
        self.use_decreasing_steer_scale = False
        self.use_increasing_steer_scale = False
        self.num_steps = a.num_steps
        self.only_steer = False
        self.steer_chunk_target = "all"


def load_obs(args):
    try:
        import einops
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from PIL import Image
        ds = LeRobotDataset(args.repo_id, episodes=[args.episode_index])
        s = ds[0]
        ext = np.array(s["exterior_image_1_left"])
        wrist = np.array(s["wrist_image_left"])
        if ext.shape[0] == 3: ext = einops.rearrange(ext, "c h w -> h w c")
        if wrist.shape[0] == 3: wrist = einops.rearrange(wrist, "c h w -> h w c")
        if np.issubdtype(ext.dtype, np.floating): ext = (255 * ext).astype(np.uint8)
        if np.issubdtype(wrist.dtype, np.floating): wrist = (255 * wrist).astype(np.uint8)
        ext = np.array(Image.fromarray(ext).resize((224, 224)))
        wrist = np.array(Image.fromarray(wrist).resize((224, 224)))
        gripper = np.array(s["gripper_position"])
        if gripper.ndim == 0: gripper = gripper[np.newaxis]
        obs = {
            "observation/exterior_image_1_left": ext,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": np.array(s["joint_position"]),
            "observation/gripper_position": gripper,
        }
        if "task" in s: obs["prompt"] = s["task"]
        return obs
    except Exception:
        return {
            "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.rand(7).astype(np.float32),
            "observation/gripper_position": np.random.rand(1).astype(np.float32),
            "prompt": "pick up the spoon",
        }


def rdiff(name, a, b):
    d = (a - b).abs()
    mx = d.max().item()
    mn = d.mean().item()
    tag = "OK" if mx == 0 else f"DIFF max={mx:.3e} mean={mn:.3e}"
    logging.info("  %-50s %s", name, tag)
    return mx


def main(args: Args):
    logging.info("Loading policies...")
    steer_pol = _policy_config.create_trained_policy(_config.get_config(args.steer_model_name), args.steer_checkpoint_dir)
    mimic_pol = _policy_config.create_trained_policy(_config.get_config(args.mimic_model_name), args.mimic_checkpoint_dir)
    base_pol = _policy_config.create_trained_policy(_config.get_config(args.base_model_name), args.base_checkpoint_dir, default_prompt=args.default_prompt)

    obs = load_obs(args)
    if "prompt" not in obs:
        obs["prompt"] = args.default_prompt
    steer_args = SteerArgs(args)

    obs_processed, inputs = base_pol.obs_to_input(obs)
    bsize = obs_processed.state.shape[0]
    device = obs_processed.state.device

    base_model = base_pol._model
    steer_model = steer_pol._model
    mimic_model = mimic_pol._model

    proxy_action_dim = steer_model.config.action_dim
    base_action_dim = base_model.config.action_dim

    actions_shape = (bsize, base_model.config.action_horizon, base_action_dim)
    noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(obs_processed, train=False)
    )

    # ========== SHARED prefix computation (compute ONCE, use for BOTH paths) ==========
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(base_prefix_pad_masks, base_prefix_att_masks)
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1
    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(base_prefix_att_2d_masks)

    _, base_past_kv = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    steer_prefix_embs, steer_prefix_pad_masks, _ = steer_model.embed_prefix(images, img_masks)

    share_proxy_dino = (
        steer_model.config.freeze_dino_encoder
        and mimic_model.config.freeze_dino_encoder
        and steer_model.config.dino_model_name == mimic_model.config.dino_model_name
    )
    if share_proxy_dino:
        mimic_prefix_embs = steer_prefix_embs
        mimic_prefix_pad_masks = steer_prefix_pad_masks
    else:
        mimic_prefix_embs, mimic_prefix_pad_masks, _ = mimic_model.embed_prefix(images, img_masks)

    logging.info("share_proxy_dino = %s", share_proxy_dino)

    # ========== PATH A: Manual denoising loop (infer_actions logic) ==========
    dt = torch.tensor(-1.0 / args.num_steps, dtype=torch.float32, device=device)
    x_a = noise.clone()
    t_a = torch.tensor(1.0, dtype=torch.float32, device=device)

    steer_target, steer_target_idx = _resolve_steer_target(steer_args, base_model.config.action_horizon)

    a_base_vts = []
    a_steer_vts = []
    a_mimic_vts = []
    a_vts = []
    a_xts = [x_a.clone()]

    for step in range(args.num_steps):
        et = t_a.expand(bsize)
        base_v = base_model.denoise_step(state, base_prefix_pad_masks, base_past_kv, x_a, et)
        a_base_vts.append(base_v.clone())

        s_suf, s_pm, _, s_ac = steer_model.embed_suffix(state[:, :proxy_action_dim], x_a[:, :, :proxy_action_dim], et)
        m_suf, m_pm, _, m_ac = mimic_model.embed_suffix(state[:, :proxy_action_dim], x_a[:, :, :proxy_action_dim], et)

        s_embs = torch.cat([steer_prefix_embs, s_suf], dim=1)
        m_embs = torch.cat([mimic_prefix_embs, m_suf], dim=1)
        s_masks = torch.cat([steer_prefix_pad_masks, s_pm], dim=1)
        m_masks = torch.cat([mimic_prefix_pad_masks, m_pm], dim=1)

        s_pos = (torch.cumsum(s_masks, dim=1) - 1).to(dtype=torch.long)
        m_pos = (torch.cumsum(m_masks, dim=1) - 1).to(dtype=torch.long)

        s_hs, _ = steer_model.expert_model.forward(
            attention_mask=s_masks, position_ids=s_pos, past_key_values=None,
            inputs_embeds=s_embs, use_cache=False, adarms_cond=s_ac,
        )
        m_hs, _ = mimic_model.expert_model.forward(
            attention_mask=m_masks, position_ids=m_pos, past_key_values=None,
            inputs_embeds=m_embs, use_cache=False, adarms_cond=m_ac,
        )

        s_vt = steer_model.action_out_proj(s_hs[:, -steer_model.config.action_horizon:].to(torch.float32))
        m_vt = mimic_model.action_out_proj(m_hs[:, -mimic_model.config.action_horizon:].to(torch.float32))
        a_steer_vts.append(s_vt.clone())
        a_mimic_vts.append(m_vt.clone())

        v = base_v.clone()
        v[:, :, :proxy_action_dim] += args.steer_scale * (s_vt - m_vt)
        a_vts.append(v.clone())

        x_a = x_a + dt * v
        t_a = t_a + dt
        a_xts.append(x_a.clone())

    # ========== PATH B: _steer_forward_all with SAME prefix ==========
    # We can't easily share the prefix because _steer_forward_all computes it internally.
    # So instead, let's run _steer_forward_all from scratch and compare.
    x_b = _steer_forward_all(
        base_model, steer_model, mimic_model,
        images[0], images[1], img_masks[0], img_masks[1],
        lang_tokens, lang_masks,
        state, noise.clone(),
        args.num_steps, proxy_action_dim, args.steer_scale,
        share_proxy_dino,
    )

    # ========== Compare ==========
    logging.info("\n=== FINAL OUTPUT ===")
    rdiff("x_t final (path A manual vs path B _steer_forward_all)", a_xts[-1], x_b)

    # ========== Test: Does running path A FIRST affect path B? ==========
    # Run _steer_forward_all in a FRESH context (first thing after prefix)
    logging.info("\n=== ISOLATION TEST: run _steer_forward_all FIRST (no prior loop) ===")

    # Re-compute prefix fresh
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    base_prefix_embs2, base_prefix_pad_masks2, base_prefix_att_masks2 = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    rdiff("base_prefix_embs (1st vs 2nd compute)", base_prefix_embs, base_prefix_embs2)

    steer_prefix_embs2, steer_prefix_pad_masks2, _ = steer_model.embed_prefix(images, img_masks)
    rdiff("steer_prefix_embs (1st vs 2nd compute)", steer_prefix_embs, steer_prefix_embs2)

    if not share_proxy_dino:
        mimic_prefix_embs2, mimic_prefix_pad_masks2, _ = mimic_model.embed_prefix(images, img_masks)
        rdiff("mimic_prefix_embs (1st vs 2nd compute)", mimic_prefix_embs, mimic_prefix_embs2)

    # Run _steer_forward_all again (3rd time for the prefix, 2nd time for the loop)
    x_c = _steer_forward_all(
        base_model, steer_model, mimic_model,
        images[0], images[1], img_masks[0], img_masks[1],
        lang_tokens, lang_masks,
        state, noise.clone(),
        args.num_steps, proxy_action_dim, args.steer_scale,
        share_proxy_dino,
    )
    rdiff("_steer_forward_all: 1st call vs 2nd call", x_b, x_c)

    # ========== Test: Are the prefix embeddings different between _steer_forward_all calls? ==========
    logging.info("\n=== KEY QUESTION: Does _steer_forward_all compute different prefixes? ===")
    logging.info("  If embed_prefix is non-deterministic, the prefixes differ between")
    logging.info("  infer_actions and _steer_forward_all since each computes its own.")

    # Test if infer_actions produces different results from manual loop with shared prefix
    logging.info("\n=== TEST: infer_actions vs manual loop (both compute own prefix) ===")
    sa = SteerArgs(args)
    with torch.no_grad():
        ia_result = infer_actions(base_pol, steer_pol, mimic_pol, obs, sa, noise=noise.clone())
    ia_actions = ia_result["actions"]
    manual_actions = base_pol.output_to_actions(inputs, a_xts[-1])
    rdiff("infer_actions vs manual loop",
          torch.from_numpy(ia_actions), torch.from_numpy(manual_actions))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    with torch.no_grad():
        main(tyro.cli(Args))
