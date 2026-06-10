#!/usr/bin/env python3
import argparse
import copy
import json
import os
import pathlib
import random

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

import einops
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace JAX vs PyTorch pi05 denoising step-by-step for a single seed."
    )
    parser.add_argument("--config-name", default="pi05_droid_jointpos")
    parser.add_argument(
        "--jax-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pi05_droid_jointpos"),
    )
    parser.add_argument(
        "--torch-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pytorch/pi05_droid_jointpos"),
    )
    parser.add_argument("--seed", type=int, default=20260334)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument(
        "--pytorch-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--dump-json", type=pathlib.Path, default=None)
    return parser.parse_args()


def max_abs_diff(a, b) -> float:
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    if a_np.dtype == np.bool_ or b_np.dtype == np.bool_:
        return float(np.max(np.not_equal(a_np, b_np).astype(np.float32)))
    return float(np.max(np.abs(a_np - b_np)))


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_random_obs_and_noise(
    seed: int,
    *,
    prompt: str,
    action_horizon: int,
    action_dim: int,
) -> tuple[dict[str, np.ndarray | str], np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = {
        "observation/exterior_image_1_left": rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": rng.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/joint_position": rng.random(7, dtype=np.float32),
        "observation/gripper_position": rng.random(1, dtype=np.float32),
        "prompt": prompt,
    }
    noise = rng.normal(size=(action_horizon, action_dim)).astype(np.float32)
    return obs, noise


def load_policy(
    config_name: str, checkpoint_dir: pathlib.Path, *, pytorch_device: str | None = None
):
    config = _config.get_config(config_name)
    return policy_config.create_trained_policy(
        config,
        checkpoint_dir.resolve(),
        pytorch_device=pytorch_device,
    )


def prepare_observations(jax_policy, torch_policy, obs, noise, pytorch_device):
    jax_inputs = jax.tree.map(lambda x: x, copy.deepcopy(obs))
    jax_inputs = jax_policy._input_transform(jax_inputs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], jax_inputs)
    jax_observation = _model.Observation.from_dict(jax_inputs)
    jax_observation = _model.preprocess_observation(None, jax_observation, train=False)

    torch_inputs = jax.tree.map(lambda x: x, copy.deepcopy(obs))
    torch_inputs = torch_policy._input_transform(torch_inputs)
    torch_inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(pytorch_device)[None, ...],
        torch_inputs,
    )
    torch_observation = _model.Observation.from_dict(torch_inputs)

    torch_model = torch_policy._model
    images, img_masks, lang_tokens, lang_masks, state = torch_model._preprocess_observation(
        torch_observation, train=False
    )
    state = torch.nn.functional.pad(
        state,
        (0, torch_model.config.action_dim - state.shape[1]),
        mode="constant",
        value=0,
    )

    noise_jax = jnp.asarray(noise)[None, ...]
    noise_torch = torch.from_numpy(noise).to(pytorch_device)[None, ...]

    return (
        jax_observation,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise_jax,
        noise_torch,
    )


def trace(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)

    config = _config.get_config(args.config_name)
    obs, noise = make_random_obs_and_noise(
        args.seed,
        prompt=args.prompt,
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
    )

    jax_policy = load_policy(args.config_name, args.jax_checkpoint)
    torch_policy = load_policy(
        args.config_name,
        args.torch_checkpoint,
        pytorch_device=args.pytorch_device,
    )
    jax_model = jax_policy._model
    torch_model = torch_policy._model

    (
        jax_observation,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        torch_state,
        x_t_jax,
        x_t_torch,
    ) = prepare_observations(
        jax_policy,
        torch_policy,
        obs,
        noise,
        args.pytorch_device,
    )

    summary: dict[str, object] = {
        "seed": args.seed,
        "config_name": args.config_name,
        "num_steps": args.num_steps,
        "preprocess": {
            "state_max_abs_diff": max_abs_diff(
                np.asarray(jax_observation.state),
                to_numpy(torch_state),
            ),
            "tokenized_prompt_max_abs_diff": max_abs_diff(
                np.asarray(jax_observation.tokenized_prompt),
                to_numpy(lang_tokens),
            ),
            "tokenized_prompt_mask_max_abs_diff": max_abs_diff(
                np.asarray(jax_observation.tokenized_prompt_mask),
                to_numpy(lang_masks),
            ),
        },
    }

    jax_prefix_tokens, jax_prefix_mask, jax_prefix_ar_mask = jax_model.embed_prefix(
        jax_observation
    )
    torch_prefix_embs, torch_prefix_pad_masks, torch_prefix_att_masks = (
        torch_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    summary["prefix"] = {
        "tokens_max_abs_diff": max_abs_diff(
            np.asarray(jax_prefix_tokens),
            to_numpy(torch_prefix_embs),
        ),
        "pad_mask_max_abs_diff": max_abs_diff(
            np.asarray(jax_prefix_mask),
            to_numpy(torch_prefix_pad_masks),
        ),
        "att_mask_max_abs_diff": max_abs_diff(
            np.broadcast_to(
                np.asarray(jax_prefix_ar_mask),
                torch_prefix_att_masks.shape,
            ),
            to_numpy(torch_prefix_att_masks),
        ),
    }

    jax_prefix_attn_mask = make_attn_mask(jax_prefix_mask, jax_prefix_ar_mask)
    jax_positions = jnp.cumsum(jax_prefix_mask, axis=1) - 1
    _, jax_kv_cache = jax_model.PaliGemma.llm(
        [jax_prefix_tokens, None],
        mask=jax_prefix_attn_mask,
        positions=jax_positions,
    )

    torch_prefix_att_2d = make_att_2d_masks(
        torch_prefix_pad_masks,
        torch_prefix_att_masks,
    )
    torch_prefix_position_ids = torch.cumsum(torch_prefix_pad_masks, dim=1) - 1
    torch_prefix_att_2d_masks_4d = torch_model._prepare_attention_masks_4d(
        torch_prefix_att_2d
    )
    torch_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, torch_past_key_values = torch_model.paligemma_with_expert.forward(
        attention_mask=torch_prefix_att_2d_masks_4d,
        position_ids=torch_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[torch_prefix_embs, None],
        use_cache=True,
    )

    step_records = []
    dt = -1.0 / args.num_steps
    batch_size = x_t_jax.shape[0]
    time_value = 1.0
    for step_index in range(args.num_steps):
        time_jax = jnp.broadcast_to(jnp.asarray(time_value, dtype=jnp.float32), batch_size)
        time_torch = torch.full(
            (batch_size,),
            float(time_value),
            dtype=torch.float32,
            device=args.pytorch_device,
        )

        jax_suffix_tokens, jax_suffix_mask, jax_suffix_ar_mask, jax_adarms_cond = (
            jax_model.embed_suffix(jax_observation, x_t_jax, time_jax)
        )
        jax_suffix_attn_mask = make_attn_mask(jax_suffix_mask, jax_suffix_ar_mask)
        jax_prefix_attn = einops.repeat(
            jax_prefix_mask,
            "b p -> b s p",
            s=jax_suffix_tokens.shape[1],
        )
        jax_full_attn_mask = jnp.concatenate([jax_prefix_attn, jax_suffix_attn_mask], axis=-1)
        jax_step_positions = (
            jnp.sum(jax_prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(jax_suffix_mask, axis=-1)
            - 1
        )
        (_, jax_suffix_out), _ = jax_model.PaliGemma.llm(
            [None, jax_suffix_tokens],
            mask=jax_full_attn_mask,
            positions=jax_step_positions,
            kv_cache=jax_kv_cache,
            adarms_cond=[None, jax_adarms_cond],
        )
        jax_suffix_out = jax_suffix_out[:, -jax_model.action_horizon :]
        jax_v_t = jax_model.action_out_proj(jax_suffix_out)

        torch_suffix_embs, torch_suffix_pad_masks, torch_suffix_att_masks, torch_adarms_cond = (
            torch_model.embed_suffix(torch_state, x_t_torch, time_torch)
        )
        suffix_len = torch_suffix_pad_masks.shape[1]
        prefix_len = torch_prefix_pad_masks.shape[1]
        torch_prefix_pad_2d = torch_prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        torch_suffix_att_2d = make_att_2d_masks(
            torch_suffix_pad_masks, torch_suffix_att_masks
        )
        torch_full_att_2d = torch.cat([torch_prefix_pad_2d, torch_suffix_att_2d], dim=2)
        torch_prefix_offsets = torch.sum(torch_prefix_pad_masks, dim=-1)[:, None]
        torch_positions = (
            torch_prefix_offsets + torch.cumsum(torch_suffix_pad_masks, dim=1) - 1
        )
        torch_full_att_4d = torch_model._prepare_attention_masks_4d(torch_full_att_2d)
        torch_model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        torch_outputs_embeds, _ = torch_model.paligemma_with_expert.forward(
            attention_mask=torch_full_att_4d,
            position_ids=torch_positions,
            past_key_values=torch_past_key_values,
            inputs_embeds=[None, torch_suffix_embs],
            use_cache=False,
            adarms_cond=[None, torch_adarms_cond],
        )
        torch_suffix_out = torch_outputs_embeds[1][:, -torch_model.config.action_horizon :]
        torch_suffix_out = torch_suffix_out.to(dtype=torch.float32)
        torch_v_t = torch_model.action_out_proj(torch_suffix_out)

        step_record = {
            "step": step_index,
            "time": time_value,
            "suffix_tokens_max_abs_diff": max_abs_diff(
                np.asarray(jax_suffix_tokens),
                to_numpy(torch_suffix_embs),
            ),
            "suffix_mask_max_abs_diff": max_abs_diff(
                np.asarray(jax_suffix_mask),
                to_numpy(torch_suffix_pad_masks),
            ),
            "adarms_cond_max_abs_diff": max_abs_diff(
                np.asarray(jax_adarms_cond),
                to_numpy(torch_adarms_cond),
            ),
            "suffix_out_max_abs_diff": max_abs_diff(
                np.asarray(jax_suffix_out),
                to_numpy(torch_suffix_out),
            ),
            "v_t_max_abs_diff": max_abs_diff(
                np.asarray(jax_v_t),
                to_numpy(torch_v_t),
            ),
            "v_t_dim7_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(jax_v_t)[0, :, 7]
                        - to_numpy(torch_v_t)[0, :, 7]
                    )
                )
            ),
            "x_t_before_dim7_max_abs_diff": float(
                np.max(
                    np.abs(
                        np.asarray(x_t_jax)[0, :, 7]
                        - to_numpy(x_t_torch)[0, :, 7]
                    )
                )
            ),
            "jax_v_t_dim7_head": np.asarray(jax_v_t)[0, :5, 7].tolist(),
            "torch_v_t_dim7_head": to_numpy(torch_v_t)[0, :5, 7].tolist(),
        }

        x_t_jax = x_t_jax + dt * jax_v_t
        x_t_torch = x_t_torch + dt * torch_v_t

        step_record["x_t_after_max_abs_diff"] = max_abs_diff(
            np.asarray(x_t_jax),
            to_numpy(x_t_torch),
        )
        step_record["x_t_after_dim7_max_abs_diff"] = float(
            np.max(
                np.abs(
                    np.asarray(x_t_jax)[0, :, 7]
                    - to_numpy(x_t_torch)[0, :, 7]
                )
            )
        )
        step_records.append(step_record)
        time_value += dt

    summary["steps"] = step_records
    summary["final"] = {
        "raw_x_t_max_abs_diff": max_abs_diff(
            np.asarray(x_t_jax),
            to_numpy(x_t_torch),
        ),
        "raw_x_t_per_dim_max_abs_diff": np.max(
            np.abs(np.asarray(x_t_jax) - to_numpy(x_t_torch)),
            axis=(0, 1),
        ).tolist(),
    }

    with torch.no_grad():
        jax_actions = np.asarray(jax_policy.infer(copy.deepcopy(obs), noise=noise)["actions"])
        torch_actions = np.asarray(torch_policy.infer(copy.deepcopy(obs), noise=noise)["actions"])
    summary["postprocess"] = {
        "actions_max_abs_diff": max_abs_diff(jax_actions, torch_actions),
        "actions_per_dim_max_abs_diff": np.max(np.abs(jax_actions - torch_actions), axis=0).tolist(),
    }

    return summary


def main() -> None:
    args = parse_args()
    summary = trace(args)
    print(json.dumps(summary, indent=2))
    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote trace JSON to: {args.dump_json}")


if __name__ == "__main__":
    main()
