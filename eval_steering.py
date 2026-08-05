import sys
import os
from typing import Any
import subprocess

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_ISAACLAB_DIR = os.path.join(_REPO_DIR, "IsaacLab")
for _isaaclab_pkg in (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_rl",
    "isaaclab_mimic",
):
    _isaaclab_pkg_src = os.path.join(_ISAACLAB_DIR, "source", _isaaclab_pkg)
    if _isaaclab_pkg_src not in sys.path:
        sys.path.insert(0, _isaaclab_pkg_src)
from isaaclab.app import AppLauncher
import pinocchio

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

import time

# from openpi.policies import libero_policy

import os
import time
import cv2
import h5py
import json
import numpy as np
import torch
import torch.nn.functional as F

# import dill
# import hydra
import argparse
import copy
import dataclasses
import re
from tqdm import tqdm

# from botocore.exceptions import NoCredentialsError

# # from diffusion_policy.common.precise_sleep import precise_wait
# from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from diffusion_policy.real_world.real_inference_util import (
#     get_real_obs_resolution,
#     get_real_obs_dict_droid,
# )

# import scipy.spatial.transform as R
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.proxy_pytorch import build_proxy_expert_masks


_SOUND_VIDEO_SCALE = None
_SOUND_VIDEO_MAX_DISTANCE_M = 0.05
_SOUND_AUDIO_CACHE = None
_SOUND_AUDIO_SAMPLE_RATE = 48_000
_SOUND_AUDIO_ATTENUATION_POWER = 2.0
_SOUND_AUDIO_REFERENCE_DISTANCE = 1.0
_SOUND_AUDIO_MIN_DISTANCE = 1e-3


def _phone_ringtone_path():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/phone/ringtone.wav",
        )
    )


def _to_numpy_unbatched(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _find_first_present(data, keys):
    for key in keys:
        if key in data:
            return key
    return None


def _policy_uses_thermal_inputs(policy) -> bool:
    transforms = getattr(policy._input_transform, "transforms", ())
    return any(type(transform).__name__ == "_OverlayThermalDroidImages" for transform in transforms)


def _validate_policy_environment_inputs(policy, raw_obs: dict, role: str) -> None:
    model_type = policy._model.config.model_type

    has_rgb = (
        "observation/exterior_image_1_left" in raw_obs
        and "observation/wrist_image_left" in raw_obs
    )
    has_sound = "observation/sound" in raw_obs or (
        "observation/mic1_log_mel" in raw_obs
        and "observation/mic2_log_mel" in raw_obs
    )
    has_thermal = (
        "observation/thermal_exterior_image_1_left" in raw_obs
        and "observation/thermal_wrist_image_left" in raw_obs
    )
    has_pointcloud = "observation/pointcloud" in raw_obs or (
        "observation/pointcloud_coord" in raw_obs
        and "observation/pointcloud_color" in raw_obs
    )

    if model_type in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
        _model.ModelType.PROXY,
        _model.ModelType.PROXY_SOUND,
        _model.ModelType.RESIDUAL,
    ) and not has_rgb:
        raise ValueError(
            f"{role} model ({model_type.value}) requires RGB observations, but the environment does not provide "
            "'observation/exterior_image_1_left' and 'observation/wrist_image_left'."
        )

    if model_type == _model.ModelType.PROXY_SOUND and not has_sound:
        raise ValueError(
            f"{role} model ({model_type.value}) requires sound observations, but the environment does not provide "
            "'observation/sound' or both 'observation/mic1_log_mel' and 'observation/mic2_log_mel'."
        )

    if _policy_uses_thermal_inputs(policy) and not has_thermal:
        raise ValueError(
            f"{role} model ({model_type.value}) requires thermal observations, but the environment does not provide "
            "'observation/thermal_exterior_image_1_left' and 'observation/thermal_wrist_image_left'."
        )

    if model_type in (
        _model.ModelType.PROXY_POINTCLOUD,
        _model.ModelType.PROXY_DP3,
    ) and not has_pointcloud:
        raise ValueError(
            f"{role} model ({model_type.value}) requires pointcloud observations, but the environment does not provide "
            "'observation/pointcloud' or both 'observation/pointcloud_coord' and 'observation/pointcloud_color'."
        )


def _obs_to_input_checked(policy, raw_obs: dict, role: str):
    _validate_policy_environment_inputs(policy, raw_obs, role)
    try:
        return policy.obs_to_input(raw_obs)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Failed to build inputs for {role} model ({policy._model.config.model_type.value}): {exc}"
        ) from exc


def _policy_input_norm_stats(policy):
    transforms = getattr(policy._input_transform, "transforms", ())
    for transform in transforms:
        if type(transform).__name__ == "Normalize":
            return transform.norm_stats, transform.use_quantiles
    return None, None


def _stat_values_for_compare(stats, use_quantiles: bool):
    if stats is None:
        return ()
    if use_quantiles and stats.q01 is not None and stats.q99 is not None:
        return (np.asarray(stats.q01), np.asarray(stats.q99))
    return (np.asarray(stats.mean), np.asarray(stats.std))


def _warn_if_norm_mismatch(
    base_policy,
    task_policy,
    ref_policy,
    *,
    action_dim: int,
):
    policies = {
        "base": base_policy,
        "task": task_policy,
        "ref": ref_policy,
    }
    norm_info = {
        role: _policy_input_norm_stats(policy)
        for role, policy in policies.items()
    }

    for role, (norm_stats, _) in norm_info.items():
        missing = [
            key
            for key in ("state", "actions")
            if norm_stats is None or key not in norm_stats
        ]
        if missing:
            print(f"WARNING: {role} policy is missing normalization stats for {missing}.")

    base_stats, base_use_quantiles = norm_info["base"]
    if base_stats is None:
        return

    for role in ("task", "ref"):
        role_stats, role_use_quantiles = norm_info[role]
        if role_stats is None:
            continue
        if base_use_quantiles != role_use_quantiles:
            print(
                "WARNING: normalization mode differs between base and "
                f"{role}: base use_quantiles={base_use_quantiles}, "
                f"{role} use_quantiles={role_use_quantiles}."
            )
        for key in ("state", "actions"):
            if key not in base_stats or key not in role_stats:
                continue
            compare_dim = action_dim if key == "actions" else min(action_dim, 8)
            base_values = _stat_values_for_compare(base_stats[key], base_use_quantiles)
            role_values = _stat_values_for_compare(role_stats[key], role_use_quantiles)
            for base_value, role_value in zip(base_values, role_values, strict=False):
                dims = min(compare_dim, base_value.shape[-1], role_value.shape[-1])
                if not np.allclose(
                    base_value[..., :dims],
                    role_value[..., :dims],
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    print(
                        "WARNING: normalized steering space may be mismatched: "
                        f"base and {role} {key} stats differ in the first {dims} dims."
                    )
                    break


def _run_sequence_proxy_expert(
    model,
    prefix_embs,
    prefix_pad_masks,
    suffix_embs,
    suffix_pad_masks,
    adarms_cond,
    prefix_att_masks=None,
    suffix_att_masks=None,
):
    if hasattr(model, "_run_action_expert"):
        return model._run_action_expert(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            adarms_cond,
        )

    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    attention_mask, pad_masks = build_proxy_expert_masks(
        model, prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
    )
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    position_ids = position_ids.to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -model.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def _prepare_proxy_steering(model, observation):
    model_type = model.config.model_type

    if model_type == _model.ModelType.PROXY:
        images, img_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks
        )
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
        }

    if model_type == _model.ModelType.PROXY_SOUND:
        images, img_masks, sound, state = model._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, sound
        )
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
        }

    if model_type == _model.ModelType.PROXY_POINTCLOUD:
        pointcloud, point_mask, state = model._preprocess_observation(
            observation, train=False
        )
        if hasattr(model, "_log_ptv3_token_counts"):
            prefix_embs, prefix_pad_masks, _, point_token_counts = model.embed_prefix(
                pointcloud,
                point_mask,
                train=False,
                return_token_counts=True,
            )
            model._log_ptv3_token_counts(
                point_token_counts,
                train=False,
                loaded_point_count=pointcloud.shape[1],
            )
        else:
            prefix_embs, prefix_pad_masks, _ = model.embed_prefix(pointcloud, point_mask)
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
        }

    if model_type == _model.ModelType.PROXY_DP3:
        pointcloud, point_mask, state = model._preprocess_observation(
            observation, train=False
        )
        obs_features = model.encode_observation(pointcloud, point_mask, state, train=False)
        return {
            "kind": "dp3",
            "obs_features": obs_features,
        }

    raise ValueError(f"Unsupported steer model type: {model_type}")


def _predict_proxy_flow(prepared_proxy, model, x_t_path, time_cond):
    action_dim = model.config.action_dim
    x_t_model = x_t_path[:, :, :action_dim]

    if prepared_proxy["kind"] == "dp3":
        return model._run_dp3(x_t_model, time_cond, prepared_proxy["obs_features"])

    suffix_args = (prepared_proxy["state"], x_t_model, time_cond)
    if "action_expert_bit" in prepared_proxy:
        suffix_args += (prepared_proxy["action_expert_bit"],)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
        model.embed_suffix(*suffix_args)
    )
    return _run_sequence_proxy_expert(
        model,
        prepared_proxy["prefix_embs"],
        prepared_proxy["prefix_pad_masks"],
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
        prefix_att_masks=prepared_proxy.get("prefix_att_masks"),
        suffix_att_masks=suffix_att_masks,
    )


def _sequence_proxy_flow_from_prefix(
    model,
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t_path,
    time_cond,
    action_dim: int,
    prefix_att_masks=None,
    action_expert_bit=None,
):
    x_t_model = x_t_path[:, :, :action_dim]
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
        state,
        x_t_model,
        time_cond,
        action_expert_bit,
    )
    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    attention_mask, pad_masks = build_proxy_expert_masks(
        model, prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
    )
    position_ids = (torch.cumsum(pad_masks, dim=1) - 1).to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=embs,
        use_cache=False,
        adarms_cond=adarms_cond,
    )
    suffix_out = hidden_states[:, -model.config.action_horizon :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def _eval_steer_forward_all(
    base_model,
    task_model,
    ref_model,
    base_images,
    base_img_masks,
    lang_tokens,
    lang_masks,
    base_state,
    task_images,
    task_img_masks,
    task_state,
    ref_images,
    ref_img_masks,
    ref_state,
    task_action_expert_bit,
    ref_action_expert_bit,
    x_t,
    num_steps: int,
    proxy_action_dim: int,
    steer_step: torch.Tensor,
    steer_scale: torch.Tensor,
    share_proxy_dino: bool,
    only_steer: bool,
):
    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(base_images, base_img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1
    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    task_prefix_embs, task_prefix_pad_masks, task_prefix_att_masks = (
        task_model.embed_prefix(task_images, task_img_masks)
    )
    if share_proxy_dino:
        ref_prefix_embs = task_prefix_embs
        ref_prefix_pad_masks = task_prefix_pad_masks
        ref_prefix_att_masks = task_prefix_att_masks
    else:
        ref_prefix_embs, ref_prefix_pad_masks, ref_prefix_att_masks = (
            ref_model.embed_prefix(ref_images, ref_img_masks)
        )

    bsize = x_t.shape[0]
    device = x_t.device
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    for _ in range(num_steps):
        expanded_time = denoise_time.expand(bsize)
        base_v_t = base_model.denoise_step(
            base_state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )
        task_v_t = _sequence_proxy_flow_from_prefix(
            task_model,
            task_state,
            task_prefix_embs,
            task_prefix_pad_masks,
            x_t,
            expanded_time,
            proxy_action_dim,
            prefix_att_masks=task_prefix_att_masks,
            action_expert_bit=task_action_expert_bit,
        )
        ref_v_t = _sequence_proxy_flow_from_prefix(
            ref_model,
            ref_state,
            ref_prefix_embs,
            ref_prefix_pad_masks,
            x_t,
            expanded_time,
            proxy_action_dim,
            prefix_att_masks=ref_prefix_att_masks,
            action_expert_bit=ref_action_expert_bit,
        )

        steer_mask = (denoise_time >= steer_step).to(dtype=base_v_t.dtype)
        if only_steer:
            steered_v_t = base_v_t.clone()
            steered_v_t[:, :, :proxy_action_dim] = task_v_t
            v_t = torch.where(steer_mask.to(dtype=torch.bool), steered_v_t, base_v_t)
        else:
            v_t = base_v_t.clone()
            v_t[:, :, :proxy_action_dim] += (
                steer_mask * steer_scale * (task_v_t - ref_v_t)
            )

        x_t = x_t + dt * v_t
        denoise_time = denoise_time + dt

    return x_t


_compiled_eval_steer_forward_all = None
_compiled_eval_steer_failed = False


def _uses_bit_conditioned_steering(args) -> bool:
    return getattr(args, "bit_conditioned_checkpoint_dir", None) is not None


def _make_action_expert_bit(batch_size: int, value: int, device):
    return torch.full(
        (batch_size,),
        value,
        dtype=torch.long,
        device=device,
    )


def _get_compiled_eval_steer_forward():
    global _compiled_eval_steer_forward_all
    if _compiled_eval_steer_forward_all is None:
        if os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            _compiled_eval_steer_forward_all = _eval_steer_forward_all
            print("eval steer forward: using eager (torch.compile disabled)")
        else:
            _compiled_eval_steer_forward_all = torch.compile(
                _eval_steer_forward_all,
                mode="max-autotune",
            )
            print("eval steer forward: compiled with max-autotune")
    return _compiled_eval_steer_forward_all


def _can_use_compiled_infer(base_policy, task_policy, ref_policy, args) -> bool:
    if args.compare_difference:
        return False
    if base_policy._model.config.model_type not in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    ):
        return False
    if task_policy._model.config.model_type != _model.ModelType.PROXY:
        return False
    if ref_policy._model.config.model_type != _model.ModelType.PROXY:
        return False
    return True


def _infer_actions_compiled(base_policy, task_policy, ref_policy, raw_obs, args):
    base_obs, base_inputs = _obs_to_input_checked(base_policy, raw_obs, "base")
    task_obs, _ = _obs_to_input_checked(task_policy, raw_obs, "task")
    if _uses_bit_conditioned_steering(args):
        ref_obs = task_obs
    else:
        ref_obs, _ = _obs_to_input_checked(ref_policy, raw_obs, "ref")

    bsize = base_obs.state.shape[0]
    device = base_obs.state.device

    base_model = base_policy._model
    task_model = task_policy._model
    ref_model = ref_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = task_model.config.action_dim
    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    base_images, base_img_masks, lang_tokens, lang_masks, base_state = (
        base_model._preprocess_observation(base_obs, train=False)
    )
    task_images, task_img_masks, task_state = task_model._preprocess_observation(
        task_obs, train=False
    )
    if _uses_bit_conditioned_steering(args):
        ref_images = task_images
        ref_img_masks = task_img_masks
        ref_state = task_state
        task_action_expert_bit = _make_action_expert_bit(bsize, 1, device)
        ref_action_expert_bit = _make_action_expert_bit(bsize, 0, device)
    else:
        ref_images, ref_img_masks, ref_state = ref_model._preprocess_observation(
            ref_obs, train=False
        )
        task_action_expert_bit = None
        ref_action_expert_bit = None

    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )
    share_proxy_dino = (
        _uses_bit_conditioned_steering(args)
        or (
            getattr(task_model.config, "freeze_dino_encoder", False)
            and getattr(ref_model.config, "freeze_dino_encoder", False)
            and getattr(task_model.config, "dino_model_name", None)
            == getattr(ref_model.config, "dino_model_name", None)
        )
    )

    forward_fn = _get_compiled_eval_steer_forward()
    x_t = forward_fn(
        base_model,
        task_model,
        ref_model,
        base_images,
        base_img_masks,
        lang_tokens,
        lang_masks,
        base_state,
        task_images,
        task_img_masks,
        task_state,
        ref_images,
        ref_img_masks,
        ref_state,
        task_action_expert_bit,
        ref_action_expert_bit,
        noise,
        args.num_steps,
        proxy_action_dim,
        torch.as_tensor(0.0, dtype=torch.float32, device=device),
        torch.as_tensor(args.steer_scale, dtype=torch.float32, device=device),
        share_proxy_dino,
        args.only_steer,
    )

    actions = base_policy.output_to_actions(base_inputs, x_t)
    return actions, {}


def infer_actions(base_policy, task_policy, ref_policy, raw_obs, args):
    global _compiled_eval_steer_failed
    if (
        not _compiled_eval_steer_failed
        and _can_use_compiled_infer(base_policy, task_policy, ref_policy, args)
    ):
        try:
            return _infer_actions_compiled(
                base_policy, task_policy, ref_policy, raw_obs, args
            )
        except Exception as exc:
            _compiled_eval_steer_failed = True
            print(
                f"Compiled eval inference failed once; falling back to eager infer_actions. Error: {exc}"
            )
    return _infer_actions_eager(base_policy, task_policy, ref_policy, raw_obs, args)


def _infer_actions_eager(base_policy, task_policy, ref_policy, raw_obs, args):
    base_obs, base_inputs = _obs_to_input_checked(base_policy, raw_obs, "base")
    task_obs, _ = _obs_to_input_checked(task_policy, raw_obs, "task")
    if _uses_bit_conditioned_steering(args):
        ref_obs = task_obs
    else:
        ref_obs, _ = _obs_to_input_checked(ref_policy, raw_obs, "ref")

    bsize = base_obs.state.shape[0]
    device = base_obs.state.device
    need_compare = args.compare_difference

    base_model = base_policy._model
    task_model = task_policy._model
    ref_model = ref_policy._model

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = task_model.config.action_dim
    compare_action_dim = proxy_action_dim - 1

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = (
        base_model._preprocess_observation(base_obs, train=False)
    )

    base_prefix_embs, base_prefix_pad_masks, base_prefix_att_masks = (
        base_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    )
    base_prefix_att_2d_masks = make_att_2d_masks(
        base_prefix_pad_masks, base_prefix_att_masks
    )
    base_prefix_position_ids = torch.cumsum(base_prefix_pad_masks, dim=1) - 1

    base_prefix_att_2d_masks_4d = base_model._prepare_attention_masks_4d(
        base_prefix_att_2d_masks
    )
    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )

    _, base_past_key_values = base_model.paligemma_with_expert.forward(
        attention_mask=base_prefix_att_2d_masks_4d,
        position_ids=base_prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[base_prefix_embs, None],
        use_cache=True,
    )

    prepared_task = _prepare_proxy_steering(task_model, task_obs)
    if _uses_bit_conditioned_steering(args):
        prepared_ref = prepared_task.copy()
        prepared_task["action_expert_bit"] = _make_action_expert_bit(
            bsize, 1, device
        )
        prepared_ref["action_expert_bit"] = _make_action_expert_bit(
            bsize, 0, device
        )
    else:
        prepared_ref = _prepare_proxy_steering(ref_model, ref_obs)

    # if the DINO encoder is frozen and the model names are the same, use the same prefix embeddings for mimic
    if (
        getattr(task_model.config, "freeze_dino_encoder", False)
        and getattr(ref_model.config, "freeze_dino_encoder", False)
        and getattr(task_model.config, "dino_model_name", None)
        == getattr(ref_model.config, "dino_model_name", None)
        and prepared_task["kind"] == "sequence"
        and prepared_ref["kind"] == "sequence"
    ):
        prepared_ref["prefix_embs"] = prepared_task["prefix_embs"]
        prepared_ref["prefix_pad_masks"] = prepared_task["prefix_pad_masks"]

    dt = -1.0 / args.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    teacher_path_x_t = noise.clone() if need_compare else None
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

    if need_compare:
        shared_compare_stats = {
            "ref_minus_base": None,
            "task_minus_ref": None,
        }
        teacher_compare_stats = {
            "ref_minus_base": None,
            "task_minus_ref": None,
        }

    while denoise_time >= -dt / 2:
        expanded_time = denoise_time.expand(bsize)

        base_v_t = base_model.denoise_step(
            state,
            base_prefix_pad_masks,
            base_past_key_values,
            x_t,
            expanded_time,
        )
        if need_compare:
            teacher_base_v_t = base_model.denoise_step(
                state,
                base_prefix_pad_masks,
                base_past_key_values,
                teacher_path_x_t,
                expanded_time,
            )

        if denoise_time >= 0.0:
            task_v_t = _predict_proxy_flow(
                prepared_task, task_model, x_t, expanded_time
            )
            ref_v_t = _predict_proxy_flow(
                prepared_ref, ref_model, x_t, expanded_time
            )
            if need_compare:
                teacher_task_v_t = _predict_proxy_flow(
                    prepared_task, task_model, teacher_path_x_t, expanded_time
                )
                teacher_ref_v_t = _predict_proxy_flow(
                    prepared_ref, ref_model, teacher_path_x_t, expanded_time
                )

            if need_compare:
                shared_base = base_v_t[:, :, :compare_action_dim]
                shared_task = task_v_t[:, :, :compare_action_dim]
                shared_ref = ref_v_t[:, :, :compare_action_dim]
                teacher_base = teacher_base_v_t[:, :, :compare_action_dim]
                teacher_task = teacher_task_v_t[:, :, :compare_action_dim]
                teacher_ref = teacher_ref_v_t[:, :, :compare_action_dim]

                shared_compare_stats["ref_minus_base"] = accumulate_stats(
                    shared_compare_stats["ref_minus_base"],
                    compute_batch_metrics(shared_base, shared_ref),
                )
                shared_compare_stats["task_minus_ref"] = accumulate_stats(
                    shared_compare_stats["task_minus_ref"],
                    compute_batch_metrics(shared_ref, shared_task),
                )
                teacher_compare_stats["ref_minus_base"] = accumulate_stats(
                    teacher_compare_stats["ref_minus_base"],
                    compute_batch_metrics(teacher_base, teacher_ref),
                )
                teacher_compare_stats["task_minus_ref"] = accumulate_stats(
                    teacher_compare_stats["task_minus_ref"],
                    compute_batch_metrics(teacher_ref, teacher_task),
                )

            v_t = base_v_t.clone()
            v_t[:, :, :proxy_action_dim] += args.steer_scale * (task_v_t - ref_v_t)
            if args.only_steer:  # testing the code for steering only
                v_t[:, :, :proxy_action_dim] = task_v_t
        else:
            v_t = base_v_t

        x_t = x_t + dt * v_t
        if need_compare:
            teacher_path_x_t = teacher_path_x_t + dt * teacher_base_v_t
        denoise_time += dt

    actions = base_policy.output_to_actions(base_inputs, x_t)
    compare_stats = {}
    if need_compare:
        compare_stats["shared_flow_path"] = {
            key: value
            for key, value in shared_compare_stats.items()
            if value is not None
        }
        compare_stats["teacher_denoise_path"] = {
            key: value
            for key, value in teacher_compare_stats.items()
            if value is not None
        }
    return actions, compare_stats


def get_pi_observation(env_obs_dict):
    obs = dict()
    joint_pos = _to_numpy_unbatched(env_obs_dict["joint_pos"])
    obs["observation/joint_position"] = joint_pos[:7]
    if joint_pos.shape[0] > 7:
        obs["observation/gripper_position"] = joint_pos[7:8]
    else:
        gripper_key = _find_first_present(
            env_obs_dict, ("gripper_pos", "gripper_position")
        )
        if gripper_key is None:
            raise KeyError("Missing gripper position in IsaacLab observation.")
        obs["observation/gripper_position"] = _to_numpy_unbatched(
            env_obs_dict[gripper_key]
        )[:1]

    table_cam_key = _find_first_present(env_obs_dict, ("table_cam",))
    if table_cam_key is not None:
        obs["observation/exterior_image_1_left"] = _to_numpy_unbatched(
            env_obs_dict[table_cam_key]
        )

    wrist_cam_key = _find_first_present(env_obs_dict, ("wrist_cam",))
    if wrist_cam_key is not None:
        obs["observation/wrist_image_left"] = _to_numpy_unbatched(
            env_obs_dict[wrist_cam_key]
        )

    thermal_table_cam_key = _find_first_present(
        env_obs_dict,
        ("thermal_table_cam", "thermal_exterior_image_1_left"),
    )
    if thermal_table_cam_key is not None:
        obs["observation/thermal_exterior_image_1_left"] = _to_numpy_unbatched(
            env_obs_dict[thermal_table_cam_key]
        )

    thermal_wrist_cam_key = _find_first_present(
        env_obs_dict,
        ("thermal_wrist_cam", "thermal_wrist_image_left"),
    )
    if thermal_wrist_cam_key is not None:
        obs["observation/thermal_wrist_image_left"] = _to_numpy_unbatched(
            env_obs_dict[thermal_wrist_cam_key]
        )

    point_coord_key = _find_first_present(
        env_obs_dict,
        ("pointcloud_coord", "point_positions", "point_position"),
    )
    point_color_key = _find_first_present(
        env_obs_dict,
        ("pointcloud_color", "point_color"),
    )
    if point_coord_key is not None and point_color_key is not None:
        obs["observation/pointcloud_coord"] = _to_numpy_unbatched(
            env_obs_dict[point_coord_key]
        )
        obs["observation/pointcloud_color"] = _to_numpy_unbatched(
            env_obs_dict[point_color_key]
        )
    else:
        pointcloud_key = _find_first_present(env_obs_dict, ("pointcloud", "point_cloud"))
        if pointcloud_key is not None:
            obs["observation/pointcloud"] = _to_numpy_unbatched(env_obs_dict[pointcloud_key])

    mic1_key = _find_first_present(env_obs_dict, ("mic1_log_mel",))
    mic2_key = _find_first_present(env_obs_dict, ("mic2_log_mel",))
    if mic1_key is not None and mic2_key is not None:
        obs["observation/mic1_log_mel"] = _to_numpy_unbatched(env_obs_dict[mic1_key])
        obs["observation/mic2_log_mel"] = _to_numpy_unbatched(env_obs_dict[mic2_key])
    else:
        sound_key = _find_first_present(env_obs_dict, ("sound",))
        if sound_key is not None:
            obs["observation/sound"] = _to_numpy_unbatched(env_obs_dict[sound_key])

    return obs


def _to_uint8_image(image):
    image = _to_numpy_unbatched(image)
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.size > 0 and np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _has_thermal_observation(obs):
    return (
        "observation/thermal_exterior_image_1_left" in obs
        and "observation/thermal_wrist_image_left" in obs
    )


def _has_sound_observation(obs):
    return "observation/sound" in obs or (
        "observation/mic1_log_mel" in obs and "observation/mic2_log_mel" in obs
    )


def _get_sound_spectrograms(obs):
    if "observation/mic1_log_mel" in obs and "observation/mic2_log_mel" in obs:
        return obs["observation/mic1_log_mel"], obs["observation/mic2_log_mel"]

    sound = _to_numpy_unbatched(obs["observation/sound"])
    sound = np.asarray(sound)
    if sound.ndim == 3 and sound.shape[0] == 2:
        return sound[0], sound[1]
    if sound.ndim == 3 and sound.shape[-1] == 2:
        return sound[..., 0], sound[..., 1]
    raise ValueError(f"Expected sound shape [2, F, T] or [F, T, 2], got {sound.shape}.")


def _to_numpy_spectrogram(spectrogram):
    spectrogram = _to_numpy_unbatched(spectrogram)
    spectrogram = np.asarray(spectrogram)
    while spectrogram.ndim > 2 and spectrogram.shape[0] == 1:
        spectrogram = spectrogram[0]
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected spectrogram shape [F, T], got {spectrogram.shape}.")
    return spectrogram.astype(np.float32)


def _build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    hz_to_mel = lambda freq_hz: 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    mel_fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for mel_idx in range(n_mels):
        left = bin_indices[mel_idx]
        center = bin_indices[mel_idx + 1]
        right = bin_indices[mel_idx + 2]
        if center > left:
            mel_fb[mel_idx, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            mel_fb[mel_idx, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return mel_fb


def _get_sound_video_scale():
    global _SOUND_VIDEO_SCALE
    if _SOUND_VIDEO_SCALE is not None:
        return _SOUND_VIDEO_SCALE

    sample_rate = 48_000
    n_fft = 2048
    n_mels = 80
    f_min = 50.0
    eps = 1e-8
    reference_distance = 1.0
    attenuation_power = _SOUND_AUDIO_ATTENUATION_POWER

    try:
        import soundfile as sf
        from scipy import signal

        audio, sr = sf.read(_phone_ringtone_path(), always_2d=True)
        audio = audio.mean(axis=1).astype(np.float64)
        if sr != sample_rate:
            gcd = np.gcd(sr, sample_rate)
            audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

        peak = np.max(np.abs(audio))
        if peak > 0.0:
            audio = audio / peak

        win_length = int(round(25.0 / 1000.0 * sample_rate))
        hop_length = int(round(10.0 / 1000.0 * sample_rate))
        _, _, zxx = signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        mel_fb = _build_mel_filterbank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=sample_rate / 2,
        )
        base_log_mel = np.log(mel_fb @ (np.abs(zxx) ** 2) + eps)
        gain_at_max = 2.0 * attenuation_power * np.log(reference_distance / _SOUND_VIDEO_MAX_DISTANCE_M)
        vmax = float(np.percentile(base_log_mel, 99) + gain_at_max)
    except Exception as exc:
        print(f"[WARN] Failed to compute global sound video scale: {exc}")
        vmax = 2.0

    vmin = float(np.log(eps))
    if vmax <= vmin:
        vmax = vmin + 1.0
    _SOUND_VIDEO_SCALE = (vmin, vmax)
    return _SOUND_VIDEO_SCALE


def _load_phone_ringtone_audio(sample_rate=_SOUND_AUDIO_SAMPLE_RATE):
    global _SOUND_AUDIO_CACHE
    if _SOUND_AUDIO_CACHE is not None and _SOUND_AUDIO_CACHE["sample_rate"] == sample_rate:
        return _SOUND_AUDIO_CACHE["audio"]

    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(_phone_ringtone_path(), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float64)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        audio = signal.resample_poly(audio, up=sample_rate // gcd, down=sr // gcd)

    peak = np.max(np.abs(audio))
    if peak > 0.0:
        audio = audio / peak
    if len(audio) == 0:
        raise ValueError("Phone ringtone WAV is empty.")

    _SOUND_AUDIO_CACHE = {"sample_rate": sample_rate, "audio": audio.astype(np.float32)}
    return _SOUND_AUDIO_CACHE["audio"]


def _quat_apply_wxyz(quat, vec):
    quat_xyz = quat[..., 1:]
    quat_w = quat[..., :1]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


def _get_gripper_mic_distances(env, mic_spacing=0.25, mic_axis=(0.0, 1.0, 0.0)):
    try:
        ee_frame = env.scene["ee_frame"]
        phone = env.scene["phone_1"]
    except KeyError:
        return None

    ee_pos_w = ee_frame.data.target_pos_w[:1, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:1, 0, :]
    axis_local = torch.tensor(mic_axis, dtype=ee_pos_w.dtype, device=ee_pos_w.device).view(1, 3)
    axis_local = axis_local / torch.linalg.vector_norm(axis_local, dim=1, keepdim=True).clamp_min(1e-6)
    axis_w = _quat_apply_wxyz(ee_quat_w, axis_local)

    mic_offset_w = 0.5 * mic_spacing * axis_w
    mic1_pos_w = ee_pos_w + mic_offset_w
    mic2_pos_w = ee_pos_w - mic_offset_w
    phone_pos_w = phone.data.root_pos_w[:1]

    distance_mic1 = torch.linalg.vector_norm(mic1_pos_w - phone_pos_w, dim=1).clamp_min(
        _SOUND_AUDIO_MIN_DISTANCE
    )
    distance_mic2 = torch.linalg.vector_norm(mic2_pos_w - phone_pos_w, dim=1).clamp_min(
        _SOUND_AUDIO_MIN_DISTANCE
    )
    return float(distance_mic1.item()), float(distance_mic2.item())


def _build_stereo_sound_audio_frame(
    env,
    frame_index,
    fps=15,
    sample_rate=_SOUND_AUDIO_SAMPLE_RATE,
):
    distances = _get_gripper_mic_distances(env)
    if distances is None:
        return None

    source_audio = _load_phone_ringtone_audio(sample_rate=sample_rate)
    start_sample = int(round(frame_index * sample_rate / fps))
    end_sample = int(round((frame_index + 1) * sample_rate / fps))
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    mono = source_audio[np.mod(sample_indices, len(source_audio))]

    distance_mic1, distance_mic2 = distances
    gain_mic1 = (_SOUND_AUDIO_REFERENCE_DISTANCE / distance_mic1) ** _SOUND_AUDIO_ATTENUATION_POWER
    gain_mic2 = (_SOUND_AUDIO_REFERENCE_DISTANCE / distance_mic2) ** _SOUND_AUDIO_ATTENUATION_POWER
    return np.stack((mono * gain_mic1, mono * gain_mic2), axis=1).astype(np.float32)


def _write_stereo_sound_audio(audio_path, audio_frames, sample_rate=_SOUND_AUDIO_SAMPLE_RATE):
    if not audio_frames:
        return None

    import soundfile as sf

    audio = np.concatenate(audio_frames, axis=0)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.98
    sf.write(audio_path, audio, sample_rate)
    return audio_path


def _visualize_log_mel_spectrogram(
    spectrogram,
    target_height,
    target_width,
    label,
    vmin=None,
    vmax=None,
):
    spectrogram = _to_numpy_spectrogram(spectrogram)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)

    if vmin is None:
        vmin = float(np.percentile(spectrogram, 1))
    if vmax is None:
        vmax = float(np.percentile(spectrogram, 99))
    if vmax <= vmin:
        normalized = np.zeros_like(spectrogram, dtype=np.float32)
    else:
        normalized = (spectrogram - vmin) / (vmax - vmin)

    image = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    image = np.flipud(image)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return image


def _mux_audio_into_video(tmp_video_path, final_video_path, audio_path, label):
    if audio_path is None or not os.path.exists(audio_path):
        os.replace(tmp_video_path, final_video_path)
        return

    mux_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        tmp_video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        final_video_path,
    ]
    try:
        subprocess.run(mux_cmd, check=True)
        os.remove(tmp_video_path)
        os.remove(audio_path)
        print(f"{label} video saved with stereo sound")
    except Exception as exc:
        print(f"[WARN] Failed to mux {label} audio into video: {exc}")
        try:
            os.remove(audio_path)
        except OSError:
            pass
        os.replace(tmp_video_path, final_video_path)


def _overlay_thermal_on_rgb(rgb_image, thermal_image, alpha=0.45):
    rgb_image = _to_uint8_image(rgb_image)
    thermal_image = _to_uint8_image(thermal_image)

    if rgb_image.shape[:2] != thermal_image.shape[:2]:
        thermal_image = cv2.resize(
            thermal_image,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.addWeighted(rgb_image, 1.0 - alpha, thermal_image, alpha, 0.0)


def _build_rollout_frame(obs, use_thermal_overlay=False):
    table_image = _to_uint8_image(obs["observation/exterior_image_1_left"])
    wrist_image = _to_uint8_image(obs["observation/wrist_image_left"])

    if use_thermal_overlay and _has_thermal_observation(obs):
        table_image = _overlay_thermal_on_rgb(
            table_image, obs["observation/thermal_exterior_image_1_left"]
        )
        wrist_image = _overlay_thermal_on_rgb(
            wrist_image, obs["observation/thermal_wrist_image_left"]
        )

    frame = np.concatenate((table_image, wrist_image), axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if _has_sound_observation(obs):
        table_only = cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)
        spectrogram_width = table_only.shape[1]
        mic1_spectrogram, mic2_spectrogram = _get_sound_spectrograms(obs)
        mic_vmin, mic_vmax = _get_sound_video_scale()
        mic1_image = _visualize_log_mel_spectrogram(
            mic1_spectrogram,
            target_height=table_only.shape[0],
            target_width=spectrogram_width,
            label="mic1_log_mel",
            vmin=mic_vmin,
            vmax=mic_vmax,
        )
        mic2_image = _visualize_log_mel_spectrogram(
            mic2_spectrogram,
            target_height=table_only.shape[0],
            target_width=spectrogram_width,
            label="mic2_log_mel",
            vmin=mic_vmin,
            vmax=mic_vmax,
        )
        return np.concatenate((mic2_image, table_only, mic1_image), axis=1)

    return frame


def _episode_video_name(seed, success, suffix=""):
    video_name = f"{seed}{suffix}"
    if success:
        return video_name + "_success.mp4"
    return video_name + "_fail.mp4"


def _episode_sort_key(name):
    match = re.search(r"(\d+)$", name)
    if match is None:
        return name
    return int(match.group(1))


def _load_hdf5_state(group, device):
    state = dict()
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            state[key] = torch.from_numpy(value[:]).to(device=device)
        else:
            state[key] = _load_hdf5_state(value, device)
    return state


def compute_batch_metrics(
    teacher_flows: torch.Tensor,
    student_flows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    diff = student_flows - teacher_flows

    teacher_vec = teacher_flows.reshape(teacher_flows.shape[0], teacher_flows.shape[1], -1)
    student_vec = student_flows.reshape(student_flows.shape[0], student_flows.shape[1], -1)
    cosine = torch.nn.functional.cosine_similarity(student_vec, teacher_vec, dim=-1)

    return {
        "sq_error_sum": diff.pow(2).sum(dtype=torch.float64),
        "abs_error_sum": diff.abs().sum(dtype=torch.float64),
        "teacher_sq_sum": teacher_flows.pow(2).sum(dtype=torch.float64),
        "student_sq_sum": student_flows.pow(2).sum(dtype=torch.float64),
        "cosine_sum": cosine.sum(dtype=torch.float64),
        "element_count": torch.tensor(diff.numel(), dtype=torch.float64, device=diff.device),
        "vector_count": torch.tensor(cosine.numel(), dtype=torch.float64, device=diff.device),
        "batch_count": torch.tensor(teacher_flows.shape[0], dtype=torch.float64, device=diff.device),
    }


def accumulate_stats(
    stats: dict[str, torch.Tensor] | None,
    batch_stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if stats is None:
        return {key: value.clone() for key, value in batch_stats.items()}
    for key, value in batch_stats.items():
        stats[key] += value
    return stats


def stats_to_serializable(stats: dict[str, torch.Tensor] | None) -> dict[str, float]:
    if stats is None:
        return {}
    return {key: float(value.item()) for key, value in stats.items()}


def average_stats_per_step(
    stats: dict[str, torch.Tensor] | None,
    n_steps: int,
) -> dict[str, float]:
    if stats is None or n_steps <= 0:
        return {}
    return {key: float((value / n_steps).item()) for key, value in stats.items()}


def summarize_metrics(stats: dict[str, torch.Tensor] | None) -> dict[str, float]:
    if stats is None:
        return {}

    eps = torch.finfo(torch.float64).eps
    element_count = torch.clamp(stats["element_count"], min=eps)
    vector_count = torch.clamp(stats["vector_count"], min=eps)
    teacher_sq_sum = torch.clamp(stats["teacher_sq_sum"], min=eps)

    return {
        "mse": float((stats["sq_error_sum"] / element_count).item()),
        "mae": float((stats["abs_error_sum"] / element_count).item()),
        "cosine": float((stats["cosine_sum"] / vector_count).item()),
        "teacher_rms": float(torch.sqrt(stats["teacher_sq_sum"] / element_count).item()),
        "student_rms": float(torch.sqrt(stats["student_sq_sum"] / element_count).item()),
        "rel_l2": float(torch.sqrt(stats["sq_error_sum"] / teacher_sq_sum).item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the model on the real droid robot."
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="water the plant with the cup")
    parser.add_argument("--exp_name", type=str, default="eval")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to the output directory."
    )
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--seed_end", type=int, default=51)
    parser.add_argument(
        "--load_init_from_dataset",
        type=str,
        default=None,
        help="Optional HDF5 dataset path. If provided, rollout i loads initial_state from the i-th episode instead of using randomized reset state.",
    )
    parser.add_argument(
        "--initial_action_after_reset",
        action="store_true",
        help=(
            "Optionally step one hold action immediately after reset. Disabled by default "
            "so the first policy observation matches the reset or dataset initial state."
        ),
    )
    parser.add_argument(
        "--base_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory. Required; the training config name is derived from this path.",
    )
    parser.add_argument(
        "--task_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory. Required; the training config name is derived from this path.",
    )
    parser.add_argument(
        "--ref_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory. Required; the training config name is derived from this path.",
    )
    parser.add_argument(
        "--bit_conditioned_checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Use one bit-conditioned proxy checkpoint for both steering roles: "
            "bit 1 is the task flow and bit 0 is the reference flow. When set, "
            "--task_checkpoint_dir and --ref_checkpoint_dir are ignored."
        ),
    )
    parser.add_argument("--steer_scale", type=float, default=0.4)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--task_num_steps", type=int, default=1200)
    parser.add_argument("--only_steer", action="store_true")
    parser.add_argument("--compare_difference", action="store_true")
    return parser


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
# Default the IsaacLab app to headless rendering with cameras enabled so these
# flags are not needed on the inference command line.
parser.set_defaults(enable_cameras=True, headless=True)

args = parser.parse_args()

# output path
output_path = os.path.join("results", f"{args.task}/{args.exp_name}")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Make the robot env
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import asyncio
import gymnasium as gym
import inspect
import random

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401
import isaaclab.utils.math as math_utils

import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401


# Setup output paths and get env name
output_dir = os.path.join("results", f"{args.task}/{args.exp_name}")
output_file_name = "eval.hdf5"
task_name = args.task
if task_name:
    task_name = args.task.split(":")[-1]
env_name = task_name

print(f"Environment name: {env_name}")

# Configure environment
env_cfg = parse_env_cfg(env_name, device=args.device, num_envs=1)

env_cfg.env_name = env_name

# Extract success checking function
success_term = None
if hasattr(env_cfg.terminations, "success"):
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
else:
    raise NotImplementedError(
        "No success termination term was found in the environment."
    )

# Configure for data generation
# env_cfg.terminations = None
# env_cfg.observations.policy.concatenate_terms = False

# Create environment
env = gym.make(env_name, cfg=env_cfg).unwrapped

# Derive each training-config name from its checkpoint dir so the model names do
# not need to be passed on the command line. Checkpoints follow the layout
# ".../checkpoints/<config_name>/<exp_name>/<step>" (proxies) or
# ".../checkpoints/pytorch/<config_name>" (base policy).
def _config_name_from_checkpoint_dir(checkpoint_dir):
    if checkpoint_dir is None:
        raise ValueError(
            "A checkpoint dir is required so the training config name can be derived from it."
        )
    parts = [p for p in os.path.normpath(checkpoint_dir).split(os.sep) if p]
    if "checkpoints" in parts:
        idx = parts.index("checkpoints") + 1
        # skip an optional framework wrapper segment (e.g. "pytorch")
        if idx < len(parts) and parts[idx] == "pytorch":
            idx += 1
        if idx < len(parts):
            return parts[idx]
    raise ValueError(
        f"Could not derive a training config name from checkpoint dir: {checkpoint_dir!r}. "
        "Expected '.../checkpoints/<config_name>/<exp_name>/<step>' or "
        "'.../checkpoints/pytorch/<config_name>'."
    )


# load checkpoint
base_checkpoint_dir = args.base_checkpoint_dir
task_checkpoint_dir = args.task_checkpoint_dir
ref_checkpoint_dir = args.ref_checkpoint_dir
base_config = _config.get_config(_config_name_from_checkpoint_dir(base_checkpoint_dir))
base_policy = policy_config.create_trained_policy(base_config, base_checkpoint_dir)

if _uses_bit_conditioned_steering(args):
    bit_checkpoint_dir = args.bit_conditioned_checkpoint_dir
    task_config = _config.get_config(
        _config_name_from_checkpoint_dir(bit_checkpoint_dir)
    )
    if task_config.model.model_type != _model.ModelType.PROXY:
        raise ValueError(
            "--bit_conditioned_checkpoint_dir must point to a ProxyConfig model, "
            f"got {task_config.model.model_type}."
        )
    if not getattr(task_config.model, "use_action_expert_bit", False):
        task_config = dataclasses.replace(
            task_config,
            model=dataclasses.replace(
                task_config.model,
                use_action_expert_bit=True,
            ),
        )
        print(
            "Enabled use_action_expert_bit for the shared steering checkpoint "
            "configuration"
        )
    task_policy = policy_config.create_trained_policy(
        task_config, bit_checkpoint_dir
    )
    ref_config = task_config
    ref_policy = task_policy
    task_checkpoint_dir = bit_checkpoint_dir
    ref_checkpoint_dir = bit_checkpoint_dir
    print(
        "Using one bit-conditioned proxy for steering: bit 1 = task, "
        "bit 0 = reference"
    )
else:
    task_config = _config.get_config(
        _config_name_from_checkpoint_dir(task_checkpoint_dir)
    )
    ref_config = _config.get_config(
        _config_name_from_checkpoint_dir(ref_checkpoint_dir)
    )
    task_policy = policy_config.create_trained_policy(
        task_config, task_checkpoint_dir
    )
    ref_policy = policy_config.create_trained_policy(ref_config, ref_checkpoint_dir)

dataset_file = None
dataset_demo_names = None
if args.load_init_from_dataset is not None:
    dataset_file = h5py.File(args.load_init_from_dataset, "r")
    dataset_demo_names = sorted(dataset_file["data"].keys(), key=_episode_sort_key)
    num_rollouts = args.seed_end - args.seed_start
    if num_rollouts > len(dataset_demo_names):
        raise ValueError(
            f"Dataset {args.load_init_from_dataset} only has {len(dataset_demo_names)} episodes, "
            f"but {num_rollouts} rollouts were requested."
        )

assert (
    base_policy._model.config.action_horizon
    == task_policy._model.config.action_horizon
    == ref_policy._model.config.action_horizon
), "Action horizon must be the same for all models"

assert (
    task_policy._model.config.action_dim == ref_policy._model.config.action_dim
), "Action dimension must be the same for proxy models"

_warn_if_norm_mismatch(
    base_policy,
    task_policy,
    ref_policy,
    action_dim=task_policy._model.config.action_dim,
)

steps_per_inference = 8
CONTROL_FREQUENCY = 15

print("Warming up policy inference")
env_obs_dict, _ = env.reset()
obs = get_pi_observation(env_obs_dict["policy"])
obs["prompt"] = args.prompt
with torch.no_grad():
    _, _ = infer_actions(
        base_policy, task_policy, ref_policy, copy.deepcopy(obs), args
    )

print("Ready!")
comparison_metadata = {
    "shared_flow_path": {
        "description": "Compare on the rollout's shared denoise path.",
        "comparisons": {
            "ref_minus_base": {
                "teacher_model": "base_model",
                "student_model": "ref_model",
            },
            "task_minus_ref": {
                "teacher_model": "ref_model",
                "student_model": "task_model",
            },
        },
    },
    "teacher_denoise_path": {
        "description": "Compare on the base model's own denoise path.",
        "comparisons": {
            "ref_minus_base": {
                "teacher_model": "base_model",
                "student_model": "ref_model",
            },
            "task_minus_ref": {
                "teacher_model": "ref_model",
                "student_model": "task_model",
            },
        },
    },
}
overall_comparison_stats = {
    path_name: {
        comparison_name: None
        for comparison_name in path_metadata["comparisons"]
    }
    for path_name, path_metadata in comparison_metadata.items()
}
episode_comparison_summaries = []
episode_results = []
total_comparison_observation_steps = 0
total_inference_time_s = 0.0
total_inference_calls = 0
for rollout_idx, seed in enumerate(range(args.seed_start, args.seed_end)):
    success = None
    episode_comparison_stats = {
        path_name: {
            comparison_name: None
            for comparison_name in path_metadata["comparisons"]
        }
        for path_name, path_metadata in comparison_metadata.items()
    }
    episode_comparison_observation_steps = 0
    episode_inference_time_s = 0.0
    episode_inference_calls = 0

    # Set seed for generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reset before starting
    if dataset_file is not None:
        initial_state = _load_hdf5_state(
            dataset_file["data"][dataset_demo_names[rollout_idx]]["initial_state"],
            env.device,
        )
        env_obs_dict, _ = env.reset_to(initial_state, env_ids=None, is_relative=True)
    else:
        env_obs_dict, _ = env.reset()

    if args.initial_action_after_reset:
        # Optional hold step for environments that need an action-buffer flush. Keep
        # the current gripper command too, otherwise dataset-initialized eval is shifted.
        initial_joint_pos = _to_numpy_unbatched(env_obs_dict["policy"]["joint_pos"])
        initial_action = initial_joint_pos[:8]
        env_obs_dict, _, _, _, _ = env.step(
            torch.as_tensor(
                initial_action[None],
                dtype=torch.float32,
                device=env.device,
            )
        )

    excute_frames = []
    thermal_overlay_frames = []
    sound_audio_frames = []
    obs = get_pi_observation(env_obs_dict["policy"])
    obs["prompt"] = args.prompt

    # ========== policy control loop ==============
    step_idx = 0
    success = False
    for step_idx in tqdm(range(args.task_num_steps), desc="Policy Control Loop"):
        try:
            if step_idx % steps_per_inference == 0:
                # print('predict_action')
                # run inference
                with torch.no_grad():
                    infer_start = time.perf_counter()
                    actions, compare_stats = infer_actions(
                        base_policy, task_policy, ref_policy, copy.deepcopy(obs), args
                    )
                    infer_elapsed = time.perf_counter() - infer_start
                    episode_inference_time_s += infer_elapsed
                    episode_inference_calls += 1
                    total_inference_time_s += infer_elapsed
                    total_inference_calls += 1
                    if args.compare_difference and compare_stats:
                        episode_comparison_observation_steps += 1
                        for path_name, path_stats in compare_stats.items():
                            for comparison_name, batch_stats in path_stats.items():
                                episode_comparison_stats[path_name][
                                    comparison_name
                                ] = accumulate_stats(
                                    episode_comparison_stats[path_name][
                                        comparison_name
                                    ],
                                    batch_stats,
                                )

                # execute actions
                start_idx = 0
                end_idx = start_idx + steps_per_inference
                actions = actions[start_idx:end_idx]

            action_step = actions[step_idx % steps_per_inference]

            # perform step
            env_obs_dict, rewards, terminated, truncated, extras = env.step(
                torch.as_tensor(
                    action_step[None],
                    dtype=torch.float32,
                    device=env.device,
                )
            )

            obs = get_pi_observation(env_obs_dict["policy"])
            obs["prompt"] = args.prompt

            # Check for task success using success_term
            task_success = bool(success_term.func(env, **success_term.params)[0])

            # save visualization
            vis_image = _build_rollout_frame(obs, use_thermal_overlay=False)
            excute_frames.append(vis_image)
            if _has_sound_observation(obs):
                sound_audio_frame = _build_stereo_sound_audio_frame(
                    env,
                    frame_index=len(sound_audio_frames),
                    fps=CONTROL_FREQUENCY,
                )
                if sound_audio_frame is not None:
                    sound_audio_frames.append(sound_audio_frame)
            if _has_thermal_observation(obs):
                thermal_overlay_frames.append(
                    _build_rollout_frame(obs, use_thermal_overlay=True)
                )

            step_idx += 1

            if terminated or truncated or task_success:
                print("terminated or truncated or task completed")
                success = task_success
                break

        except KeyboardInterrupt:
            print("Interrupted!")
            break

    # save excute_frames as video
    video_name = _episode_video_name(seed, success)
    if success:
        print("success")
    else:
        print("fail")

    video_path = os.path.join(output_path, video_name)
    audio_path = _write_stereo_sound_audio(
        os.path.join(output_path, f"{seed}_recording.wav"),
        sound_audio_frames,
    )
    video_write_path = (
        os.path.join(output_path, f"{seed}_recording.mp4")
        if audio_path is not None
        else video_path
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        video_write_path, fourcc, 15, (excute_frames[0].shape[1], excute_frames[0].shape[0])
    )
    for frame in excute_frames:
        out.write(frame)
    out.release()
    if audio_path is not None:
        _mux_audio_into_video(video_write_path, video_path, audio_path, "rollout")

    if thermal_overlay_frames:
        thermal_video_name = _episode_video_name(seed, success, "_thermal_overlay")
        thermal_video_path = os.path.join(output_path, thermal_video_name)
        thermal_out = cv2.VideoWriter(
            thermal_video_path,
            fourcc,
            15,
            (
                thermal_overlay_frames[0].shape[1],
                thermal_overlay_frames[0].shape[0],
            ),
        )
        for frame in thermal_overlay_frames:
            thermal_out.write(frame)
        thermal_out.release()

    print("video saved")
    if episode_inference_calls:
        avg_infer_ms = 1000.0 * episode_inference_time_s / episode_inference_calls
        print(
            f"seed {seed} infer_actions avg latency: {avg_infer_ms:.2f} ms "
            f"over {episode_inference_calls} calls"
        )

    if args.compare_difference:
        episode_summary = {
            "rollout_index": rollout_idx,
            "seed": seed,
            "success": bool(success),
            "n_observation_steps": episode_comparison_observation_steps,
            "paths": {},
        }
        for path_name, path_metadata in comparison_metadata.items():
            path_summary = {
                "description": path_metadata["description"],
                "comparisons": {},
            }
            for comparison_name, metadata in path_metadata["comparisons"].items():
                episode_stats = episode_comparison_stats[path_name][comparison_name]
                if episode_stats is None:
                    continue
                overall_comparison_stats[path_name][comparison_name] = accumulate_stats(
                    overall_comparison_stats[path_name][comparison_name],
                    episode_stats,
                )
                path_summary["comparisons"][comparison_name] = {
                    **metadata,
                    "total_stats": stats_to_serializable(episode_stats),
                    "average_stats_per_observation_step": average_stats_per_step(
                        episode_stats, episode_comparison_observation_steps
                    ),
                    "metrics": summarize_metrics(episode_stats),
                }
            if path_summary["comparisons"]:
                episode_summary["paths"][path_name] = path_summary

        total_comparison_observation_steps += episode_comparison_observation_steps
        episode_comparison_summaries.append(episode_summary)

    episode_results.append(
        {
            "rollout_index": rollout_idx,
            "seed": seed,
            "success": bool(success),
        }
    )

env.close()
if dataset_file is not None:
    dataset_file.close()

if args.compare_difference:
    comparison_summary = {
        "compare_difference": True,
        "output_path": output_path,
        "episodes": episode_comparison_summaries,
        "overall": {
            "n_episodes": len(episode_comparison_summaries),
            "n_observation_steps": total_comparison_observation_steps,
            "paths": {},
        },
    }
    for path_name, path_metadata in comparison_metadata.items():
        path_summary = {
            "description": path_metadata["description"],
            "comparisons": {},
        }
        for comparison_name, metadata in path_metadata["comparisons"].items():
            overall_stats = overall_comparison_stats[path_name][comparison_name]
            if overall_stats is None:
                continue
            path_summary["comparisons"][comparison_name] = {
                **metadata,
                "total_stats": stats_to_serializable(overall_stats),
                "average_stats_per_observation_step": average_stats_per_step(
                    overall_stats, total_comparison_observation_steps
                ),
                "metrics": summarize_metrics(overall_stats),
            }
        if path_summary["comparisons"]:
            comparison_summary["overall"]["paths"][path_name] = path_summary

    comparison_path = os.path.join(output_path, "compare_difference.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"compare difference statistics saved to {comparison_path}")

num_successes = sum(result["success"] for result in episode_results)
evaluation_summary = {
    "task": args.task,
    "prompt": args.prompt,
    "exp_name": args.exp_name,
    "steer_scale": args.steer_scale,
    "bit_conditioned_checkpoint_dir": args.bit_conditioned_checkpoint_dir,
    "num_episodes": len(episode_results),
    "num_successes": num_successes,
    "success_rate": (
        num_successes / len(episode_results) if episode_results else 0.0
    ),
    "episodes": episode_results,
}
evaluation_summary_path = os.path.join(output_path, "evaluation_summary.json")
with open(evaluation_summary_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_summary, f, indent=2)
print(
    f"evaluation summary saved to {evaluation_summary_path}: "
    f"{num_successes}/{len(episode_results)} successes"
)

if total_inference_calls:
    avg_infer_ms = 1000.0 * total_inference_time_s / total_inference_calls
    print(
        f"overall infer_actions avg latency: {avg_infer_ms:.2f} ms "
        f"over {total_inference_calls} calls"
    )

# Close the simulation app after environment is closed
simulation_app.close()
