import sys
import os
from typing import Any
import subprocess

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPI_SRC_DIR = os.path.join(_REPO_DIR, "openpi", "src")
if _OPENPI_SRC_DIR not in sys.path:
    sys.path.insert(0, _OPENPI_SRC_DIR)
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
from sim_free_mpc import AccelActionMPC, AccelMPCConfig, SimFreeMPC, SimFreeMPCConfig
from sim_free_mpc.action_space import clamp_real_action_chunk
from sim_free_mpc.ddim import ddim_iteration_alphas
from sim_free_mpc.score_steering import combine_scores


DEFAULT_BASE_CHECKPOINT_DIR = "openpi/checkpoints/pytorch/pi05_droid_jointpos"
TASK_PROMPTS_PATH = os.path.join(_REPO_DIR, "task_prompts.json")

_SOUND_VIDEO_SCALE = None
_SOUND_VIDEO_MAX_DISTANCE_M = 0.05
_SOUND_AUDIO_CACHE = None
_SOUND_AUDIO_SAMPLE_RATE = 48_000
_SOUND_AUDIO_ATTENUATION_POWER = 2.0
_SOUND_AUDIO_REFERENCE_DISTANCE = 1.0
_SOUND_AUDIO_MIN_DISTANCE = 1e-3
_LAST_INFERENCE_RUNTIME = {}
_WEIGHT_SCALE_CENTER_OFFSET_DEBUG = (-0.0470425, 0.0, 0.0272255)
_WEIGHT_SCALE_TOP_OFFSET_Z_DEBUG = 0.0523800


def _score_update_mode_for_mpc_update(update_mode: str) -> str:
    if update_mode in ("mbd_score_action_prox", "mbd_score_action_warm"):
        return "mbd_score"
    return update_mode


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
        _model.ModelType.PROXY_SCORE,
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
    policies = {role: policy for role, policy in policies.items() if policy is not None}
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

    if "base" not in norm_info:
        return
    base_stats, base_use_quantiles = norm_info["base"]
    if base_stats is None:
        return

    for role in ("task", "ref"):
        if role not in norm_info:
            continue
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


def _assert_score_space_compatibility(base_policy, task_policy, ref_policy, args) -> None:
    score_mode = _score_steering_mode(args)
    if score_mode not in ("full", "task"):
        return

    if base_policy is None or task_policy is None:
        raise ValueError(f"{score_mode} score steering requires base and task policies.")
    score_policies = {"task": task_policy}
    if score_mode == "full":
        if ref_policy is None:
            raise ValueError("Full score steering requires a reference policy.")
        score_policies["ref"] = ref_policy

    for role, policy in score_policies.items():
        if not _is_score_proxy(policy._model):
            raise ValueError(
                f"{score_mode} score steering requires {role} to be a ProxyScore checkpoint; "
                f"got {policy._model.config.model_type.value!r}."
            )

    expected_timesteps = int(args.mpc_ddim_train_timesteps)
    for role, policy in score_policies.items():
        model = policy._model
        actual = int(getattr(model.config, "ddim_num_train_timesteps", -1))
        if actual != expected_timesteps:
            raise ValueError(
                f"{role} ProxyScore DDIM train timesteps ({actual}) do not match "
                f"--mpc_ddim_train_timesteps ({expected_timesteps})."
            )

    policies = {"base": base_policy, **score_policies}
    norm_info = {role: _policy_input_norm_stats(policy) for role, policy in policies.items()}
    base_stats, base_use_quantiles = norm_info["base"]
    if base_stats is None:
        raise ValueError("Base policy has no input norm_stats; score-space MPC cannot decode a shared action space.")

    action_dim = min(
        int(getattr(policy._model.config, "action_dim", 8))
        for policy in policies.values()
    )
    for role in score_policies:
        role_stats, role_use_quantiles = norm_info[role]
        if role_stats is None:
            raise ValueError(f"{role} ProxyScore policy has no input norm_stats.")
        if base_use_quantiles != role_use_quantiles:
            raise ValueError(
                "Score-space policy normalization mode mismatch: "
                f"base use_quantiles={base_use_quantiles}, {role} use_quantiles={role_use_quantiles}."
            )
        for key in ("state", "actions"):
            if key not in base_stats or key not in role_stats:
                raise ValueError(f"Missing {key!r} norm_stats in base or {role} policy.")
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
                    raise ValueError(
                        "Score-space normalized steering space mismatch: "
                        f"base and {role} {key} norm_stats differ in the first {dims} dims."
                    )


def _run_sequence_proxy_expert(
    model,
    prefix_embs,
    prefix_pad_masks,
    suffix_embs,
    suffix_pad_masks,
    adarms_cond,
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
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    position_ids = position_ids.to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=pad_masks,
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

    if model_type in (_model.ModelType.PROXY, _model.ModelType.PROXY_SCORE):
        images, img_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, _ = model.embed_prefix(images, img_masks)
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
        }

    if model_type == _model.ModelType.PROXY_SOUND:
        images, img_masks, sound, state = model._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, _ = model.embed_prefix(images, img_masks, sound)
        return {
            "kind": "sequence",
            "state": state,
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
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

    suffix_embs, suffix_pad_masks, _, adarms_cond = model.embed_suffix(
        prepared_proxy["state"],
        x_t_model,
        time_cond,
    )
    return _run_sequence_proxy_expert(
        model,
        prepared_proxy["prefix_embs"],
        prepared_proxy["prefix_pad_masks"],
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
    )


def _is_score_proxy(model) -> bool:
    return model.config.model_type == _model.ModelType.PROXY_SCORE


def _proxy_score_time_cond(args, iteration: int, device, dtype) -> torch.Tensor:
    ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=args.num_steps + 1,
        num_train_timesteps=args.mpc_ddim_train_timesteps,
    )
    step_ratio = int(args.mpc_ddim_train_timesteps) // int(args.num_steps + 1)
    timestep = int((int(args.num_steps + 1) - 1 - int(iteration)) * step_ratio)
    value = timestep / max(float(args.mpc_ddim_train_timesteps - 1), 1.0)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _score_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            lhs.detach().reshape(1, -1),
            rhs.detach().reshape(1, -1),
            dim=-1,
            eps=1e-8,
        )[0].cpu()
    )


def _relative_score_error(target: torch.Tensor, prediction: torch.Tensor) -> float:
    error_norm = torch.linalg.vector_norm((prediction - target).detach())
    target_norm = torch.linalg.vector_norm(target.detach()).clamp_min(1e-8)
    return float((error_norm / target_norm).cpu())


def _predict_proxy_score(prepared_proxy, model, x_t_path, time_cond):
    if not _is_score_proxy(model):
        raise ValueError(
            "Score-space PPS steering requires task/ref checkpoints with "
            f"model_type={_model.ModelType.PROXY_SCORE.value!r}; got "
            f"{model.config.model_type.value!r}."
        )
    action_dim = model.config.action_dim
    x_t_model = x_t_path[:, :, :action_dim]
    if prepared_proxy["kind"] != "sequence":
        raise ValueError("ProxyScorePytorch currently supports sequence image proxies only.")
    return model.predict_score_from_prefix(
        prepared_proxy["state"],
        prepared_proxy["prefix_embs"],
        prepared_proxy["prefix_pad_masks"],
        x_t_model,
        time_cond,
    )


def _sequence_proxy_flow_from_prefix(
    model,
    state,
    prefix_embs,
    prefix_pad_masks,
    x_t_path,
    time_cond,
    action_dim: int,
):
    x_t_model = x_t_path[:, :, :action_dim]
    suffix_embs, suffix_pad_masks, _, adarms_cond = model.embed_suffix(
        state,
        x_t_model,
        time_cond,
    )
    embs = torch.cat([prefix_embs, suffix_embs], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    position_ids = (torch.cumsum(pad_masks, dim=1) - 1).to(dtype=torch.long)

    hidden_states, _ = model.expert_model.forward(
        attention_mask=pad_masks,
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
    x_t,
    num_steps: int,
    proxy_action_dim: int,
    steer_step: torch.Tensor,
    steer_scale: torch.Tensor,
    share_proxy_dino: bool,
    only_steer: bool,
    no_steer: bool,
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

    task_prefix_embs, task_prefix_pad_masks, _ = task_model.embed_prefix(
        task_images, task_img_masks
    )
    if share_proxy_dino:
        ref_prefix_embs = task_prefix_embs
        ref_prefix_pad_masks = task_prefix_pad_masks
    else:
        ref_prefix_embs, ref_prefix_pad_masks, _ = ref_model.embed_prefix(
            ref_images, ref_img_masks
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
        )
        ref_v_t = _sequence_proxy_flow_from_prefix(
            ref_model,
            ref_state,
            ref_prefix_embs,
            ref_prefix_pad_masks,
            x_t,
            expanded_time,
            proxy_action_dim,
        )

        steer_mask = (denoise_time >= steer_step).to(dtype=base_v_t.dtype)
        if no_steer:
            v_t = base_v_t
        elif only_steer:
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


def _score_steering_mode(args) -> str | None:
    if getattr(args, "full_steer", False):
        return "full"
    if getattr(args, "task_steer", False):
        return "task"
    if getattr(args, "vlm_base", False):
        return "base"
    return None


def _uses_vlm_mpc_base(args) -> bool:
    return _score_steering_mode(args) is not None


def _standalone_policy_role(args) -> str | None:
    if getattr(args, "ref_only", False):
        return "ref"
    if getattr(args, "task_only", False):
        return "task"
    return None


def _required_policy_roles(args) -> set[str]:
    standalone_role = _standalone_policy_role(args)
    if standalone_role is not None:
        return {standalone_role}

    score_mode = _score_steering_mode(args)
    if score_mode == "base" or getattr(args, "no_steer", False):
        return {"base"}
    if score_mode == "task":
        return {"base", "task"}
    return {"base", "task", "ref"}


def _steering_mode_name(args) -> str:
    score_mode = _score_steering_mode(args)
    if score_mode == "base" or getattr(args, "no_steer", False):
        return "base_only"
    if score_mode == "full":
        return "base_plus_task_minus_ref"
    if score_mode == "task":
        return "base_plus_task"
    if getattr(args, "only_steer", False):
        return "only_steer"
    return "task_minus_ref"


def _uses_accel_action_mpc(args) -> bool:
    return _uses_vlm_mpc_base(args) and getattr(args, "mpc_optimize_space", "action") == "accel"


def _base_source_name(args) -> str:
    standalone_role = _standalone_policy_role(args)
    if standalone_role is not None:
        return f"{standalone_role}_only"
    score_mode = _score_steering_mode(args)
    if score_mode is None:
        return "pi_checkpoint"
    if _uses_accel_action_mpc(args):
        return "accel_action_mppi"
    if score_mode == "full":
        return "mbd_full_steer"
    if score_mode == "task":
        return "mbd_task_steer"
    return "mbd_base"


def _can_use_compiled_infer(base_policy, task_policy, ref_policy, args) -> bool:
    if _uses_vlm_mpc_base(args):
        return False
    if args.no_steer:
        return False
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
    global _LAST_INFERENCE_RUNTIME
    base_obs, base_inputs = _obs_to_input_checked(base_policy, raw_obs, "base")
    task_obs, _ = _obs_to_input_checked(task_policy, raw_obs, "task")
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
    ref_images, ref_img_masks, ref_state = ref_model._preprocess_observation(
        ref_obs, train=False
    )

    base_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"  # noqa: SLF001
    )
    share_proxy_dino = (
        getattr(task_model.config, "freeze_dino_encoder", False)
        and getattr(ref_model.config, "freeze_dino_encoder", False)
        and getattr(task_model.config, "dino_model_name", None)
        == getattr(ref_model.config, "dino_model_name", None)
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
        noise,
        args.num_steps,
        proxy_action_dim,
        torch.as_tensor(0.0, dtype=torch.float32, device=device),
        torch.as_tensor(args.steer_scale, dtype=torch.float32, device=device),
        share_proxy_dino,
        args.only_steer,
        args.no_steer,
    )

    actions = base_policy.output_to_actions(base_inputs, x_t)
    _LAST_INFERENCE_RUNTIME = {
        "base_source": "pi_checkpoint",
        "used_base_model_velocity": True,
        "steering_mode": _steering_mode_name(args),
        "checked_vlm_task_ref_shapes": False,
        "x_t_shape": tuple(x_t.shape),
        "v_vlm_shape": None,
        "score_shape": None,
        "v_task_shape": None,
        "v_ref_shape": None,
        "proxy_task_shape": None,
        "proxy_ref_shape": None,
        "mpc_last": None,
    }
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


def infer_actions_with_mpc(
    base_policy,
    task_policy,
    ref_policy,
    raw_obs,
    args,
    *,
    mpc_planner=None,
    mpc_context=None,
    warm_shift_steps=0,
):
    global _LAST_INFERENCE_RUNTIME
    standalone_role = _standalone_policy_role(args)
    if standalone_role is not None:
        standalone_policy = ref_policy if standalone_role == "ref" else task_policy
        if standalone_policy is None:
            raise ValueError(f"--{standalone_role}_only requires a {standalone_role} checkpoint.")
        _validate_policy_environment_inputs(standalone_policy, raw_obs, standalone_role)
        outputs = standalone_policy.infer(raw_obs)
        actions = np.asarray(outputs["actions"], dtype=np.float32)
        # Policy.infer() has already decoded and unnormalized the proxy output.
        # Clamp in executable joint space: the first target is relative to the
        # current robot state, and each later target is relative to the preceding
        # clamped target in the chunk.
        max_joint_delta = (
            args.mpc_joint_delta_clip if args.mpc_joint_delta_clip > 0.0 else None
        )
        actions = (
            clamp_real_action_chunk(
                torch.as_tensor(actions, dtype=torch.float32),
                current_joint_pos=raw_obs.get("observation/joint_position"),
                max_joint_delta=max_joint_delta,
            )
            .cpu()
            .numpy()
        )
        _LAST_INFERENCE_RUNTIME = {
            "base_source": f"{standalone_role}_only",
            "used_base_model_velocity": False,
            "steering_mode": f"{standalone_role}_only",
            "x_t_shape": None,
            "v_vlm_shape": None,
            "score_shape": None,
            "v_task_shape": None,
            "v_ref_shape": None,
            "proxy_task_shape": None,
            "proxy_ref_shape": None,
            "output_action_shape": tuple(actions.shape),
            "mpc_last": None,
            "mpc_trace": [],
        }
        return actions, {}
    if _uses_vlm_mpc_base(args) and mpc_planner is None:
        raise ValueError("VLM/MPC base mode requires a SimFreeMPC planner.")
    if not _uses_vlm_mpc_base(args):
        return infer_actions(base_policy, task_policy, ref_policy, raw_obs, args)
    if mpc_planner is None:
        return infer_actions(base_policy, task_policy, ref_policy, raw_obs, args)
    return _infer_actions_eager(
        base_policy,
        task_policy,
        ref_policy,
        raw_obs,
        args,
        mpc_planner=mpc_planner,
        mpc_context=mpc_context,
        warm_shift_steps=warm_shift_steps,
    )


def _infer_actions_eager(
    base_policy,
    task_policy,
    ref_policy,
    raw_obs,
    args,
    *,
    mpc_planner=None,
    mpc_context=None,
    warm_shift_steps=0,
):
    global _LAST_INFERENCE_RUNTIME
    base_obs, base_inputs = _obs_to_input_checked(base_policy, raw_obs, "base")

    bsize = base_obs.state.shape[0]
    device = base_obs.state.device
    need_compare = args.compare_difference
    use_vlm_mpc_base = _uses_vlm_mpc_base(args)
    score_steering_mode = _score_steering_mode(args)
    disable_steering = bool(getattr(args, "no_steer", False)) or score_steering_mode == "base"
    if score_steering_mode == "task":
        need_task = True
        need_ref = False
    elif score_steering_mode == "full":
        need_task = True
        need_ref = True
    else:
        need_task = (not disable_steering) or need_compare
        need_ref = need_task
    need_task_and_ref = need_task and need_ref
    if need_compare and use_vlm_mpc_base:
        raise ValueError(
            "--compare_difference compares against the pi checkpoint base path and is "
            "not compatible with --vlm_base."
        )
    if use_vlm_mpc_base and (mpc_planner is None or mpc_context is None):
        raise ValueError("VLM/MPC base mode requires mpc_planner and mpc_context.")

    base_model = base_policy._model
    task_model = task_policy._model if task_policy is not None else None
    ref_model = ref_policy._model if ref_policy is not None else None

    base_action_dim = base_model.config.action_dim
    proxy_action_dim = task_model.config.action_dim if need_task else None
    compare_action_dim = proxy_action_dim - 1 if need_compare else None

    actions_shape = (
        bsize,
        base_model.config.action_horizon,
        base_action_dim,
    )
    noise = base_model.sample_noise(actions_shape, device)

    if use_vlm_mpc_base:
        state = None
        base_prefix_pad_masks = None
        base_past_key_values = None
    else:
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

    if need_task:
        if task_policy is None or task_model is None:
            raise ValueError("The selected steering mode requires a task policy.")
        task_obs, _ = _obs_to_input_checked(task_policy, raw_obs, "task")
        prepared_task = _prepare_proxy_steering(task_model, task_obs)
        if need_ref:
            if ref_policy is None or ref_model is None:
                raise ValueError("The selected steering mode requires a reference policy.")
            ref_obs, _ = _obs_to_input_checked(ref_policy, raw_obs, "ref")
            prepared_ref = _prepare_proxy_steering(ref_model, ref_obs)
        else:
            prepared_ref = None
        if (not use_vlm_mpc_base) and _is_score_proxy(task_model):
            raise ValueError(
                "ProxyScore checkpoints require --full_steer or --task_steer. "
                "The default pi-checkpoint path operates in flow/velocity space."
            )
        if (not use_vlm_mpc_base) and need_ref and _is_score_proxy(ref_model):
            raise ValueError(
                "ProxyScore checkpoints require --full_steer or --task_steer. "
                "The default pi-checkpoint path operates in flow/velocity space."
            )
    else:
        prepared_task = None
        prepared_ref = None

    # if the DINO encoder is frozen and the model names are the same, use the same prefix embeddings for mimic
    if (
        need_task
        and need_ref
        and
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

    action_warm_started = False
    if use_vlm_mpc_base and args.mpc_update == "mbd_score_action_warm":
        x_t, action_warm_started = mpc_planner.warm_start_noise(
            noise,
            shift_steps=warm_shift_steps,
            current_state=base_inputs["state"],
        )
    else:
        x_t = noise
    teacher_path_x_t = noise.clone() if need_compare else None
    denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)
    runtime_stats = {
        "base_source": _base_source_name(args),
        "used_base_model_velocity": False,
        "steering_mode": _steering_mode_name(args),
        "checked_vlm_task_ref_shapes": False,
        "x_t_shape": tuple(x_t.shape),
        "v_vlm_shape": None,
        "score_shape": None,
        "v_task_shape": None,
        "v_ref_shape": None,
        "proxy_task_shape": None,
        "proxy_ref_shape": None,
        "mpc_last": None,
        "mpc_trace": [],
        "action_warm_started": bool(action_warm_started),
        "action_warm_shift_steps": int(warm_shift_steps) if action_warm_started else 0,
    }

    def record_mpc_stats(stats):
        if args.mpc_update == "mbd_score_action_warm":
            stats = dict(stats)
            stats.update(
                {
                    "action_warm_started": bool(action_warm_started),
                    "action_warm_shift_steps": int(warm_shift_steps) if action_warm_started else 0,
                }
            )
        runtime_stats["mpc_last"] = stats
        if args.mpc_debug:
            runtime_stats["mpc_trace"].append(_mpc_debug_stats(stats))

    if _uses_accel_action_mpc(args):
        if not disable_steering:
            raise ValueError("--mpc_optimize_space accel currently supports --vlm_base base-only mode.")
        if args.mpc_update not in ("legacy_score", "mbd_score"):
            raise ValueError(
                "Acceleration action-space MPC supports --mpc_update legacy_score or mbd_score. "
                "DDIM/action-prox variants are score-space only for this direct-action planner."
            )
        base_outputs = base_policy.infer(raw_obs)
        base_actions = np.asarray(base_outputs["actions"], dtype=np.float32)
        if base_actions.ndim != 2:
            raise ValueError(f"Expected base policy actions [H,D], got {tuple(base_actions.shape)}")
        if base_actions.shape[-1] < 8:
            current_gripper = raw_obs.get("observation/gripper_position")
            if current_gripper is None:
                raise ValueError(
                    "Acceleration action-space MPC needs a gripper trajectory, but base actions have "
                    f"{base_actions.shape[-1]} dims and no observation/gripper_position was found."
                )
            gripper_value = np.asarray(current_gripper, dtype=np.float32).reshape(-1)[0]
            gripper_traj = np.full((base_actions.shape[0], 1), gripper_value, dtype=np.float32)
        else:
            gripper_traj = base_actions[:, 7:8]
        gripper_tensor = torch.as_tensor(gripper_traj, device=device, dtype=torch.float32)
        if args.mpc_update == "mbd_score":
            planned_actions, geom_stats = mpc_planner.plan_mbd_score(
                context=mpc_context,
                gripper_traj=gripper_tensor,
                num_iterations=args.num_steps + 1,
                score_scale=args.gamma_base,
                device=device,
                dtype=torch.float32,
            )
        else:
            planned_actions, geom_stats = mpc_planner.plan(
                context=mpc_context,
                gripper_traj=gripper_tensor,
                device=device,
                dtype=torch.float32,
            )
        current_joint_pos = raw_obs.get("observation/joint_position")
        max_joint_delta = args.mpc_joint_delta_clip if args.mpc_joint_delta_clip > 0.0 else None
        actions = (
            clamp_real_action_chunk(
                planned_actions,
                current_joint_pos=current_joint_pos,
                max_joint_delta=max_joint_delta,
            )
            .detach()
            .cpu()
            .numpy()
        )
        runtime_stats["score_shape"] = tuple(planned_actions.shape)
        record_mpc_stats(geom_stats)
        if args.mpc_debug_stdout:
            print(
                f"vlm_mpc_{geom_stats['update_mode']} "
                f"cost_min={geom_stats['cost_min']:.4f} "
                f"cost_mean={geom_stats['cost_mean']:.4f} "
                f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                f"target_delta_norm={geom_stats['target_delta_norm']:.4f} "
                f"accel_norm={geom_stats['accel_norm']:.4f} "
                f"score_norm={geom_stats.get('score_norm', 0.0):.4f}"
                f"{_format_mpc_term_debug(geom_stats)}",
                flush=True,
            )
        _LAST_INFERENCE_RUNTIME = runtime_stats
        return actions, {}

    if need_compare:
        shared_compare_stats = {
            "ref_minus_base": None,
            "task_minus_ref": None,
        }
        teacher_compare_stats = {
            "ref_minus_base": None,
            "task_minus_ref": None,
        }

    mpc_denoise_iteration = 0
    mpc_denoise_iterations = args.num_steps + 1
    while denoise_time >= -dt / 2:
        expanded_time = denoise_time.expand(bsize)

        if use_vlm_mpc_base and disable_steering:
            if args.mpc_update == "legacy_score":
                x_t, geom_stats = mpc_planner.step_score_space(
                    x_t,
                    base_inputs,
                    mpc_context,
                    step_scale=args.gamma_base,
                )
            elif args.mpc_update == "mbd_score":
                x_t, geom_stats = mpc_planner.step_mbd_score(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    score_scale=args.gamma_base,
                )
            elif args.mpc_update == "mbd_score_action_prox":
                x_t, geom_stats = mpc_planner.step_mbd_score_action_prox(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    score_scale=args.gamma_base,
                )
            elif args.mpc_update == "mbd_score_action_warm":
                x_t, geom_stats = mpc_planner.step_mbd_score_action_warm(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    score_scale=args.gamma_base,
                )
            else:
                x_t, geom_stats = mpc_planner.step_ddim(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                    step_scale=args.gamma_base,
                )
            runtime_stats["score_shape"] = tuple(x_t.shape)
            record_mpc_stats(geom_stats)
            if args.mpc_debug_stdout:
                print(
                    f"vlm_mpc_{geom_stats['update_mode']} "
                    f"cost_min={geom_stats['cost_min']:.4f} "
                    f"cost_mean={geom_stats['cost_mean']:.4f} "
                    f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                    f"target_delta_norm={geom_stats['target_delta_norm']:.4f} "
                    f"score_norm={geom_stats['score_norm']:.4f}"
                    f"{_format_mpc_term_debug(geom_stats)}",
                    flush=True,
                )
            mpc_denoise_iteration += 1
            denoise_time += dt
            continue

        if use_vlm_mpc_base and score_steering_mode in ("full", "task"):
            if task_model is None or not _is_score_proxy(task_model):
                raise ValueError("Score steering requires a ProxyScore task checkpoint.")
            if score_steering_mode == "full" and (
                ref_model is None or not _is_score_proxy(ref_model)
            ):
                raise ValueError("Full score steering requires a ProxyScore reference checkpoint.")
            if args.mpc_update == "legacy_score":
                raise ValueError(
                    "Score steering does not support --mpc_update legacy_score."
                )

            if args.mpc_update == "mbd_score_action_prox":
                base_score, geom_stats = mpc_planner.estimate_mbd_score_action_prox(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                )
            elif args.mpc_update == "mbd_score_action_warm":
                base_score, geom_stats = mpc_planner.estimate_mbd_score_action_warm(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                )
            else:
                base_score, geom_stats = mpc_planner.estimate_mbd_score(
                    x_t,
                    base_inputs,
                    mpc_context,
                    iteration=mpc_denoise_iteration,
                    num_iterations=mpc_denoise_iterations,
                )
            score_time = _proxy_score_time_cond(
                args,
                mpc_denoise_iteration,
                device,
                x_t.dtype,
            ).expand(bsize)
            task_score = _predict_proxy_score(
                prepared_task,
                task_model,
                x_t,
                score_time,
            )
            if task_score.shape[:2] != x_t.shape[:2] or task_score.shape[-1] > x_t.shape[-1]:
                raise ValueError(
                    "Task score shape is incompatible with x_t: "
                    f"task={tuple(task_score.shape)}, x_t={tuple(x_t.shape)}."
                )

            task_full_score = torch.zeros_like(x_t)
            task_full_score[:, :, : task_score.shape[-1]] = task_score
            ref_score = None
            ref_full_score = None
            if score_steering_mode == "full":
                ref_score = _predict_proxy_score(
                    prepared_ref,
                    ref_model,
                    x_t,
                    score_time,
                )
                if ref_score.shape != task_score.shape:
                    raise ValueError(
                        "task/ref score shapes must match: "
                        f"task={tuple(task_score.shape)}, ref={tuple(ref_score.shape)}."
                    )
                ref_full_score = torch.zeros_like(x_t)
                ref_full_score[:, :, : ref_score.shape[-1]] = ref_score

            combined_score = combine_scores(
                base_score,
                task_full_score,
                mode=score_steering_mode,
                steer_scale=args.steer_scale,
                ref_score=ref_full_score,
                base_scale=args.gamma_base,
            )
            residual_score = (
                task_full_score - ref_full_score
                if ref_full_score is not None
                else task_full_score
            )
            proxy_dims = task_score.shape[-1]
            base_proxy_score = base_score[..., :proxy_dims]

            active_dims = int(geom_stats.get("active_dims", task_score.shape[-1]))
            x_t = mpc_planner.step_from_score(
                x_t,
                combined_score,
                iteration=mpc_denoise_iteration,
                num_iterations=mpc_denoise_iterations,
                update_mode=_score_update_mode_for_mpc_update(args.mpc_update),
                score_scale=1.0,
                active_dims=active_dims,
            )

            runtime_stats["score_shape"] = tuple(combined_score.shape)
            runtime_stats["proxy_task_shape"] = tuple(task_score.shape)
            runtime_stats["proxy_ref_shape"] = (
                tuple(ref_score.shape) if ref_score is not None else None
            )
            runtime_stats["v_task_shape"] = None
            runtime_stats["v_ref_shape"] = None
            runtime_stats["checked_vlm_task_ref_shapes"] = True
            geom_stats = dict(geom_stats)
            geom_stats.update(
                {
                    "update_mode": f"{args.mpc_update}_score_steer",
                    "score_steering_mode": score_steering_mode,
                    "score_base_norm": float(torch.linalg.vector_norm(base_score.detach()).cpu()),
                    "score_base_proxy_norm": float(
                        torch.linalg.vector_norm(base_proxy_score.detach()).cpu()
                    ),
                    "score_task_norm": float(torch.linalg.vector_norm(task_full_score.detach()).cpu()),
                    "score_residual_norm": float(torch.linalg.vector_norm(residual_score.detach()).cpu()),
                    "score_combined_norm": float(torch.linalg.vector_norm(combined_score.detach()).cpu()),
                    "proxy_score_time": float(score_time[0].detach().cpu()),
                }
            )
            if ref_full_score is not None:
                geom_stats["score_ref_norm"] = float(
                    torch.linalg.vector_norm(ref_full_score.detach()).cpu()
                )
                geom_stats["score_ref_base_cosine"] = _score_cosine(
                    base_proxy_score, ref_score
                )
                geom_stats["score_ref_base_relative_error"] = _relative_score_error(
                    base_proxy_score, ref_score
                )
                geom_stats["score_task_ref_cosine"] = _score_cosine(task_score, ref_score)
            if combined_score.shape[-1] > 7:
                score_components = {
                    "base": base_score,
                    "task": task_full_score,
                    "residual": residual_score,
                    "combined": combined_score,
                }
                if ref_full_score is not None:
                    score_components["ref"] = ref_full_score
                for component_name, component_score in score_components.items():
                    gripper_score = component_score[..., 7].detach()
                    geom_stats[f"score_{component_name}_gripper_mean"] = float(
                        gripper_score.mean().cpu()
                    )
                    geom_stats[f"score_{component_name}_gripper_norm"] = float(
                        torch.linalg.vector_norm(gripper_score).cpu()
                    )
            record_mpc_stats(geom_stats)
            if args.mpc_debug_stdout:
                print(
                    f"vlm_mpc_{geom_stats['update_mode']} "
                    f"cost_min={geom_stats['cost_min']:.4f} "
                    f"cost_mean={geom_stats['cost_mean']:.4f} "
                    f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                    f"base_score_norm={geom_stats['score_norm']:.4f} "
                    f"combined_score_norm={geom_stats['score_combined_norm']:.4f}"
                    f"{_format_mpc_term_debug(geom_stats)}",
                    flush=True,
                )
            mpc_denoise_iteration += 1
            denoise_time += dt
            continue

        if use_vlm_mpc_base:
            base_v_t, geom_stats = mpc_planner.step(
                x_t,
                base_inputs,
                mpc_context,
                dt=dt,
            )
            if base_v_t.shape != x_t.shape:
                raise ValueError(
                    "VLM/MPC base velocity shape must match x_t: "
                    f"got {tuple(base_v_t.shape)} vs {tuple(x_t.shape)}."
                )
            runtime_stats["v_vlm_shape"] = tuple(base_v_t.shape)
            record_mpc_stats(geom_stats)
            if args.mpc_debug_stdout:
                print(
                    "vlm_mpc_base "
                    f"cost_min={geom_stats['cost_min']:.4f} "
                    f"cost_mean={geom_stats['cost_mean']:.4f} "
                    f"cost_weighted={geom_stats['cost_weighted']:.4f} "
                    f"target_delta_norm={geom_stats['target_delta_norm']:.4f}"
                    f"{_format_mpc_term_debug(geom_stats)}",
                    flush=True,
                )
        else:
            base_v_t = base_model.denoise_step(
                state,
                base_prefix_pad_masks,
                base_past_key_values,
                x_t,
                expanded_time,
            )
            runtime_stats["used_base_model_velocity"] = True
        if need_compare:
            teacher_base_v_t = base_model.denoise_step(
                state,
                base_prefix_pad_masks,
                base_past_key_values,
                teacher_path_x_t,
                expanded_time,
            )

        if denoise_time >= 0.0:
            if need_task_and_ref:
                task_v_t = _predict_proxy_flow(
                    prepared_task, task_model, x_t, expanded_time
                )
                ref_v_t = _predict_proxy_flow(
                    prepared_ref, ref_model, x_t, expanded_time
                )
            else:
                task_v_t = None
                ref_v_t = None
            if need_task_and_ref:
                if task_v_t.shape != ref_v_t.shape:
                    raise ValueError(
                        "task/ref velocity shapes must match: "
                        f"task={tuple(task_v_t.shape)}, ref={tuple(ref_v_t.shape)}."
                    )
                if task_v_t.shape[:2] != x_t.shape[:2] or task_v_t.shape[-1] > x_t.shape[-1]:
                    raise ValueError(
                        "task/ref velocity shape is incompatible with x_t: "
                        f"task={tuple(task_v_t.shape)}, x_t={tuple(x_t.shape)}."
                    )
                runtime_stats["proxy_task_shape"] = tuple(task_v_t.shape)
                runtime_stats["proxy_ref_shape"] = tuple(ref_v_t.shape)
                runtime_stats["v_task_shape"] = tuple(task_v_t.shape)
                runtime_stats["v_ref_shape"] = tuple(ref_v_t.shape)
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

            if use_vlm_mpc_base:
                if disable_steering:
                    v_t = args.gamma_base * base_v_t
                    runtime_stats["checked_vlm_task_ref_shapes"] = False
                else:
                    task_full_v_t = torch.zeros_like(x_t)
                    ref_full_v_t = torch.zeros_like(x_t)
                    task_full_v_t[:, :, :proxy_action_dim] = task_v_t
                    ref_full_v_t[:, :, :proxy_action_dim] = ref_v_t
                    runtime_stats["v_task_shape"] = tuple(task_full_v_t.shape)
                    runtime_stats["v_ref_shape"] = tuple(ref_full_v_t.shape)
                    v_t = (
                        args.gamma_base * base_v_t
                        + args.steer_scale * (task_full_v_t - ref_full_v_t)
                    )
                    runtime_stats["checked_vlm_task_ref_shapes"] = True
                if args.only_steer and not disable_steering:
                    v_t = task_full_v_t
            else:
                if disable_steering:
                    v_t = base_v_t
                else:
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

    if use_vlm_mpc_base and args.mpc_update == "mbd_score_action_warm":
        mpc_planner.set_warm_action(x_t, state=base_inputs["state"])

    actions = base_policy.output_to_actions(base_inputs, x_t)
    if use_vlm_mpc_base:
        current_joint_pos = raw_obs.get("observation/joint_position")
        max_joint_delta = args.mpc_joint_delta_clip if args.mpc_joint_delta_clip > 0.0 else None
        action_tensor = torch.as_tensor(actions, device=device, dtype=torch.float32)
        actions = (
            clamp_real_action_chunk(
                action_tensor,
                current_joint_pos=current_joint_pos,
                max_joint_delta=max_joint_delta,
            )
            .detach()
            .cpu()
            .numpy()
        )
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
    _LAST_INFERENCE_RUNTIME = runtime_stats
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


def _task_name_for_mpc(task_name: str) -> str:
    task = task_name.lower()
    for name in ("pot", "weight", "tea", "capsule"):
        if name in task:
            return name
    return task_name


def _context_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach()
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        return value
    value = torch.as_tensor(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def _extract_subtasks(env_obs_dict):
    subtasks = {}
    raw = env_obs_dict.get("subtask_terms")
    if raw is None:
        return subtasks
    for key, value in raw.items():
        tensor = _context_tensor(value)
        if tensor is not None:
            subtasks[key] = tensor
    return subtasks


def _extract_scene_objects(env, names, env_origin=None):
    objects = {}
    for name in names:
        try:
            asset = env.scene[name]
        except Exception:
            continue
        data = getattr(asset, "data", None)
        if data is None:
            continue
        item = {}
        if hasattr(data, "root_pos_w"):
            pos = _context_tensor(data.root_pos_w)
            if pos is not None and env_origin is not None:
                pos = pos - env_origin.to(device=pos.device, dtype=pos.dtype)
            item["pos"] = pos
        if hasattr(data, "root_quat_w"):
            item["quat"] = _context_tensor(data.root_quat_w)
        if item:
            objects[name] = item
    return objects


def _extract_capsule_mpc_state(env, env_origin=None):
    """Extract the live lid link pose and joint position used by capsule_flow."""
    try:
        capsule = env.scene["capsule"]
    except Exception:
        return {}, None

    lid_joint_pos = None
    try:
        lid_joint_ids, _ = capsule.find_joints(["RevoluteJoint_capsule_coffee_maker_3_up"])
        if len(lid_joint_ids) == 1:
            lid_joint_pos = _context_tensor(capsule.data.joint_pos[:, lid_joint_ids[0]])
    except Exception:
        pass

    lid_objects = {}
    try:
        lid_body_ids, _ = capsule.find_bodies(["E_shell_8"])
        if len(lid_body_ids) == 1:
            lid_pos = _context_tensor(capsule.data.body_pos_w[:, lid_body_ids[0], :])
            lid_quat = _context_tensor(capsule.data.body_quat_w[:, lid_body_ids[0], :])
            if lid_pos is not None and env_origin is not None:
                lid_pos = lid_pos - env_origin.to(device=lid_pos.device, dtype=lid_pos.dtype)
            if lid_pos is not None and lid_quat is not None:
                lid_objects["capsule_lid"] = {"pos": lid_pos, "quat": lid_quat}
    except Exception:
        pass
    return lid_objects, lid_joint_pos


def build_mpc_context(env, env_obs_dict, args):
    policy_obs = env_obs_dict["policy"]
    env_origin = None
    if hasattr(env.scene, "env_origins"):
        env_origin = _context_tensor(env.scene.env_origins)
    robot_root_pos = None
    robot_root_quat = None
    try:
        robot = env.scene["robot"]
        robot_root_pos = _context_tensor(robot.data.root_pos_w)
        robot_root_quat = _context_tensor(robot.data.root_quat_w)
        if robot_root_pos is not None and env_origin is not None:
            robot_root_pos = robot_root_pos - env_origin.to(
                device=robot_root_pos.device,
                dtype=robot_root_pos.dtype,
            )
    except Exception:
        pass
    try:
        ee_frame = env.scene["ee_frame"]
        ee_frame.update(0.0, force_recompute=True)
        fk_source_pos = _context_tensor(ee_frame.data.source_pos_w)
        fk_source_quat = _context_tensor(ee_frame.data.source_quat_w)
        if fk_source_pos is not None:
            if env_origin is not None:
                fk_source_pos = fk_source_pos - env_origin.to(
                    device=fk_source_pos.device,
                    dtype=fk_source_pos.dtype,
                )
            robot_root_pos = fk_source_pos
        if fk_source_quat is not None:
            robot_root_quat = fk_source_quat
    except Exception:
        pass
    context = {
        "task": _task_name_for_mpc(args.task),
        "subtasks": _extract_subtasks(env_obs_dict),
        "joint_pos": _context_tensor(policy_obs.get("joint_pos")),
        "joint_vel": _context_tensor(policy_obs.get("joint_vel")),
        "eef_pos": _context_tensor(policy_obs.get("eef_pos")),
        "eef_quat": _context_tensor(policy_obs.get("eef_quat")),
        "gripper_pos": _context_tensor(policy_obs.get("gripper_pos")),
        "env_origin": env_origin,
        "robot_root_pos": robot_root_pos,
        "robot_root_quat": robot_root_quat,
    }
    context["objects"] = _extract_scene_objects(
        env,
        (
            "pot",
            "cover",
            "egg",
            "pear",
            "apple",
            "mango",
            "cabbage",
            "scale",
            "teapot",
            "teacup",
            "capsule",
            "can",
        ),
        env_origin=env_origin,
    )
    capsule_objects, capsule_lid_joint_pos = _extract_capsule_mpc_state(
        env,
        env_origin=env_origin,
    )
    context["objects"].update(capsule_objects)
    if capsule_lid_joint_pos is not None:
        context["capsule_lid_joint_pos"] = capsule_lid_joint_pos
    return context


def _format_mpc_term_debug(stats: dict[str, Any], *, limit: int = 8) -> str:
    terms = []
    suffix = "_weighted"
    for key, value in stats.items():
        if not (key.startswith("term_") and key.endswith(suffix)):
            continue
        name = key[len("term_") : -len(suffix)]
        try:
            terms.append((name, float(value)))
        except (TypeError, ValueError):
            continue
    if not terms:
        return ""
    terms.sort(key=lambda item: abs(item[1]), reverse=True)
    text = ",".join(f"{name}:{value:.4g}" for name, value in terms[:limit])
    return f" terms={text}"


def _jsonable_debug_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            scalar = value.reshape(-1)[0].item()
            if isinstance(scalar, (bool, np.bool_)):
                return bool(scalar)
            if isinstance(scalar, (int, np.integer)):
                return int(scalar)
            return float(scalar)
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _jsonable_debug_value(value.reshape(-1)[0])
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_debug_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_debug_value(item) for item in value]
    return value


def _debug_subtasks(env_obs_dict) -> dict[str, bool]:
    return {
        key: bool(_jsonable_debug_value(value))
        for key, value in _extract_subtasks(env_obs_dict).items()
    }


def _debug_phase_from_subtasks(task_name: str, subtasks: dict[str, bool]) -> str:
    task = task_name.lower()
    if "weight" in task:
        if subtasks.get("grasp_apple", False):
            return "place_apple"
        if subtasks.get("pear_on_scale", False):
            return "grasp_apple"
        if subtasks.get("grasp_pear", False):
            return "place_pear"
        return "grasp_pear"
    if "capsule" in task:
        if subtasks.get("grasp_pod", False):
            return "place_pod"
        if subtasks.get("open_coffee_lid", False):
            return "grasp_pod"
        return "open_lid"
    return "unknown"


def _mpc_debug_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    keys = (
        "update_mode",
        "cost_style",
        "cost_stage",
        "optimize_space",
        "cost_min",
        "cost_mean",
        "cost_weighted",
        "target_delta_norm",
        "accel_norm",
        "score_norm",
        "score_base_norm",
        "score_base_proxy_norm",
        "score_task_norm",
        "score_ref_norm",
        "score_residual_norm",
        "score_combined_norm",
        "score_ref_base_cosine",
        "score_ref_base_relative_error",
        "score_task_ref_cosine",
        "proxy_score_time",
        "score_base_gripper_mean",
        "score_base_gripper_norm",
        "score_task_gripper_mean",
        "score_task_gripper_norm",
        "score_ref_gripper_mean",
        "score_ref_gripper_norm",
        "score_residual_gripper_mean",
        "score_residual_gripper_norm",
        "score_combined_gripper_mean",
        "score_combined_gripper_norm",
        "gripper_mean",
        "proposal_center",
        "proposal_noise_scale",
        "action_warm_started",
        "action_warm_shift_steps",
    )
    payload = {key: _jsonable_debug_value(stats[key]) for key in keys if key in stats}
    best_terms = {}
    weighted_terms = {}
    best_debug = {}
    weighted_debug = {}
    for key, value in stats.items():
        if key.startswith("term_") and key.endswith("_best"):
            best_terms[key[len("term_") : -len("_best")]] = _jsonable_debug_value(value)
        elif key.startswith("term_") and key.endswith("_weighted"):
            weighted_terms[key[len("term_") : -len("_weighted")]] = _jsonable_debug_value(value)
        elif key.startswith("debug_") and key.endswith("_best"):
            best_debug[key[len("debug_") : -len("_best")]] = _jsonable_debug_value(value)
        elif key.startswith("debug_") and key.endswith("_weighted"):
            weighted_debug[key[len("debug_") : -len("_weighted")]] = _jsonable_debug_value(value)
    if weighted_terms:
        payload["terms_weighted"] = dict(
            sorted(weighted_terms.items(), key=lambda item: abs(float(item[1])), reverse=True)
        )
    if best_terms:
        payload["terms_best"] = dict(
            sorted(best_terms.items(), key=lambda item: abs(float(item[1])), reverse=True)
        )
    if weighted_debug:
        payload["debug_weighted"] = dict(sorted(weighted_debug.items()))
    if best_debug:
        payload["debug_best"] = dict(sorted(best_debug.items()))
    return payload


def _write_mpc_debug_log(handle, event: str, **payload) -> None:
    if handle is None:
        return
    record = {
        "time": time.time(),
        "event": event,
        **payload,
    }
    handle.write(json.dumps(_jsonable_debug_value(record), sort_keys=True) + "\n")
    handle.flush()


def _debug_action_gripper(action_step) -> float | None:
    action = np.asarray(action_step).reshape(-1)
    if action.shape[0] <= 7:
        return None
    return float(action[7])


def _collect_mpc_debug_frames(env, *, axis_length: float = 0.08):
    try:
        ee_frame = env.scene["ee_frame"]
        ee_frame.update(0.0, force_recompute=True)

        ee_pos = ee_frame.data.target_pos_w[:1]
        ee_quat = ee_frame.data.target_quat_w[:1]

        frames = []
        is_capsule_task = False
        try:
            coffee_maker = env.scene["capsule"]
            pod = env.scene["can"]
            is_capsule_task = True
            frames.append(("gripper", ee_pos[0, 0], ee_quat[0, 0]))

            pod_pos = pod.data.root_pos_w[:1]
            pod_quat = pod.data.root_quat_w[:1]
            frames.append(("capsule", pod_pos[0], pod_quat[0]))

            lid_body_ids, _ = coffee_maker.find_bodies(["E_shell_8"])
            if len(lid_body_ids) == 1:
                lid_body_id = lid_body_ids[0]
                lid_pos = coffee_maker.data.body_pos_w[:1, lid_body_id, :]
                lid_quat = coffee_maker.data.body_quat_w[:1, lid_body_id, :]
                frames.append(("lid", lid_pos[0], lid_quat[0]))
        except Exception:
            pass

        if not is_capsule_task:
            frames.append(("ee", ee_pos[0, 0], ee_quat[0, 0]))
            try:
                pear = env.scene["pear"]
                pear_pos = pear.data.root_pos_w[:1]
                pear_quat = pear.data.root_quat_w[:1]
                frames.append(("pear", pear_pos[0], pear_quat[0]))
            except Exception:
                pass

        try:
            scale = env.scene["scale"]
            scale_pos = scale.data.root_pos_w[:1]
            scale_quat = scale.data.root_quat_w[:1]
            scale_top_offset = torch.tensor(
                [
                    _WEIGHT_SCALE_CENTER_OFFSET_DEBUG[0],
                    _WEIGHT_SCALE_CENTER_OFFSET_DEBUG[1],
                    _WEIGHT_SCALE_CENTER_OFFSET_DEBUG[2] + _WEIGHT_SCALE_TOP_OFFSET_Z_DEBUG,
                ],
                device=scale_pos.device,
                dtype=scale_pos.dtype,
            ).view(1, 3)
            scale_top_pos = scale_pos + _quat_apply_wxyz(scale_quat, scale_top_offset)
            frames.append(("scale_top", scale_top_pos[0], scale_quat[0]))
        except Exception:
            pass
        if ee_pos.shape[1] >= 3:
            frames.extend(
                [
                    ("rf", ee_pos[0, 1], ee_quat[0, 1]),
                    ("lf", ee_pos[0, 2], ee_quat[0, 2]),
                ]
            )

        axes = {}
        local_axes = torch.eye(3, device=ee_pos.device, dtype=ee_pos.dtype)
        for name, pos, quat in frames:
            endpoints = [pos]
            for axis in local_axes:
                endpoints.append(pos + axis_length * _quat_apply_wxyz(quat, axis))
            axes[name] = torch.stack(endpoints, dim=0).detach()
        return axes
    except Exception as exc:
        print(f"mpc_debug_video_overlay collect failed: {exc}", flush=True)
        return {}


def _project_world_points_to_camera(points_w: torch.Tensor, camera) -> np.ndarray:
    data = camera.data
    cam_pos = data.pos_w[:1].to(device=points_w.device, dtype=points_w.dtype)
    cam_quat = data.quat_w_ros[:1].to(device=points_w.device, dtype=points_w.dtype)
    intr = data.intrinsic_matrices[0].to(device=points_w.device, dtype=points_w.dtype)
    points_cam = _quat_apply_inverse_wxyz(cam_quat, points_w - cam_pos).reshape(-1, 3)
    z = points_cam[:, 2]
    pixels = torch.full((points_cam.shape[0], 2), float("nan"), device=points_w.device, dtype=points_w.dtype)
    valid = z > 1e-4
    if valid.any():
        pixels[valid, 0] = intr[0, 0] * points_cam[valid, 0] / z[valid] + intr[0, 2]
        pixels[valid, 1] = intr[1, 1] * points_cam[valid, 1] / z[valid] + intr[1, 2]
    return pixels.detach().cpu().numpy()


def _draw_projected_debug_axes(image: np.ndarray, env, camera_name: str, axes: dict[str, torch.Tensor]) -> np.ndarray:
    if not axes:
        return image
    try:
        camera = env.scene[camera_name]
    except Exception:
        return image
    out = image.copy()
    height, width = out.shape[:2]
    axis_colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    label_colors = {
        "pear": (255, 255, 255),
        "scale_top": (128, 255, 128),
        "ee": (255, 255, 0),
        "gripper": (255, 255, 0),
        "lid": (0, 165, 255),
        "capsule": (255, 128, 255),
        "rf": (255, 0, 255),
        "lf": (0, 255, 255),
    }
    for name, points in axes.items():
        pixels = _project_world_points_to_camera(points, camera)
        if not np.isfinite(pixels[0]).all():
            continue
        origin = tuple(np.round(pixels[0]).astype(int))
        if not (0 <= origin[0] < width and 0 <= origin[1] < height):
            continue
        cv2.circle(out, origin, 3, label_colors.get(name, (255, 255, 255)), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            name,
            (origin[0] + 4, origin[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            label_colors.get(name, (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )
        for axis_idx, color in enumerate(axis_colors, start=1):
            if not np.isfinite(pixels[axis_idx]).all():
                continue
            end = tuple(np.round(pixels[axis_idx]).astype(int))
            cv2.line(out, origin, end, color, 2, lineType=cv2.LINE_AA)
    return out


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


def _quat_apply_inverse_wxyz(quat, vec):
    quat_inv = quat.clone()
    quat_inv[..., 1:] = -quat_inv[..., 1:]
    return _quat_apply_wxyz(quat_inv, vec)


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


def _build_rollout_frame(obs, use_thermal_overlay=False, debug_overlay=None):
    table_image = _to_uint8_image(obs["observation/exterior_image_1_left"])
    wrist_image = _to_uint8_image(obs["observation/wrist_image_left"])

    if use_thermal_overlay and _has_thermal_observation(obs):
        table_image = _overlay_thermal_on_rgb(
            table_image, obs["observation/thermal_exterior_image_1_left"]
        )
        wrist_image = _overlay_thermal_on_rgb(
            wrist_image, obs["observation/thermal_wrist_image_left"]
        )

    if debug_overlay is not None:
        env = debug_overlay.get("env")
        axes = debug_overlay.get("axes", {})
        table_image = _draw_projected_debug_axes(table_image, env, "table_cam", axes)
        wrist_image = _draw_projected_debug_axes(wrist_image, env, "wrist_cam", axes)

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


def _slugify(text: str) -> str:
    safe_chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("-")
    return "".join(safe_chars).strip("-") or "eval"


def _video_config_slug(args) -> str:
    parts = [
        _base_source_name(args),
        f"update-{args.mpc_update}" if _uses_vlm_mpc_base(args) else "update-flow",
        f"cost-{getattr(args, 'mpc_cost', 'na')}",
        f"space-{getattr(args, 'mpc_optimize_space', 'na')}",
        f"g{float(args.gamma_base):g}",
        f"n{float(args.mpc_noise):g}",
        f"t{float(args.mpc_temperature):g}",
        f"clip{float(args.mpc_joint_delta_clip):g}",
    ]
    return _slugify("_".join(parts))


def _experiment_output_name(args, *, run_id: str) -> str:
    return f"{run_id}_{_video_config_slug(args)}"


def _episode_video_name(seed, success, *, suffix=""):
    status = "success" if success else "fail"
    suffix = _slugify(str(suffix).strip("_")) if suffix else ""
    suffix_part = f"_{suffix}" if suffix else ""
    return f"{seed}_{status}{suffix_part}.mp4"


def _write_experiment_results(path: str, payload: dict[str, Any]) -> None:
    episodes = payload.get("episodes", [])
    successes = sum(bool(episode.get("success")) for episode in episodes)
    payload["summary"] = {
        "num_episodes": len(episodes),
        "num_successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable_debug_value(payload), handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _video_config_lines(args, *, seed: int) -> list[str]:
    return [
        f"seed={seed} task={args.task} prompt={args.prompt}",
        (
            f"base={_base_source_name(args)} update={args.mpc_update} "
            f"cost={getattr(args, 'mpc_cost', 'na')} "
            f"space={getattr(args, 'mpc_optimize_space', 'na')} "
            f"gamma={args.gamma_base:g} steps={args.num_steps} "
            f"rollout_steps={args.task_num_steps} spi={args.steps_per_inference}"
        ),
        (
            f"samples={args.mpc_num_samples} iters={args.mpc_iterations} "
            f"noise={args.mpc_noise:g} temp={args.mpc_temperature:g} "
            f"joint_clip={args.mpc_joint_delta_clip:g}"
        ),
    ]


def _add_video_header(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if frame.size == 0:
        return frame
    height = 26 * len(lines) + 12
    header = np.zeros((height, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
    header[:] = 12
    y = 24
    for line in lines:
        cv2.putText(
            header,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return np.concatenate((header, frame), axis=0)


def _first_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().flatten()[0].item())
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    return bool(value)


def _first_list(value):
    if torch.is_tensor(value):
        return value.detach().flatten().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _debug_scene_root_pos(env, asset_name: str):
    try:
        asset = env.scene[asset_name]
        return asset.data.root_pos_w[0].detach().cpu().tolist()
    except Exception:
        return None


def _capture_debug_object_xy_guard(env, asset_name: str, *, margin: float):
    try:
        asset = env.scene[asset_name]
        pos = asset.data.root_pos_w.detach()
        return {
            "asset_name": asset_name,
            "xy_min": pos[:, :2].clone() - float(margin),
            "xy_max": pos[:, :2].clone() + float(margin),
        }
    except Exception as exc:
        print(f"debug_object_xy_guard unavailable for {asset_name}: {exc}", flush=True)
        return None


def _subtask_flag_from_obs(env_obs_dict, key: str) -> bool:
    raw = env_obs_dict.get("subtask_terms", {})
    if key not in raw:
        return False
    return _first_bool(raw[key])


def _apply_debug_object_xy_guard(env, guard_state, *, release: bool) -> bool:
    if guard_state is None or release:
        return False
    asset_name = guard_state["asset_name"]
    try:
        asset = env.scene[asset_name]
        pos = asset.data.root_pos_w.detach().clone()
        quat = asset.data.root_quat_w.detach().clone()
        xy_min = guard_state["xy_min"].to(device=pos.device, dtype=pos.dtype)
        xy_max = guard_state["xy_max"].to(device=pos.device, dtype=pos.dtype)
        below = pos[:, :2] < xy_min
        above = pos[:, :2] > xy_max

        pose_changed = bool((below | above).any().item())
        if pose_changed:
            pos[:, :2] = torch.maximum(torch.minimum(pos[:, :2], xy_max), xy_min)
            asset.write_root_pose_to_sim(torch.cat((pos, quat), dim=-1))

        vel = asset.data.root_vel_w.detach().clone()
        outward_x = (below[:, 0] & (vel[:, 0] < 0.0)) | (above[:, 0] & (vel[:, 0] > 0.0))
        outward_y = (below[:, 1] & (vel[:, 1] < 0.0)) | (above[:, 1] & (vel[:, 1] > 0.0))
        if bool((outward_x | outward_y).any().item()):
            vel[outward_x, 0] = 0.0
            vel[outward_y, 1] = 0.0
            asset.write_root_velocity_to_sim(vel)
        return pose_changed
    except Exception as exc:
        print(f"debug_object_xy_guard failed for {asset_name}: {exc}", flush=True)
        return False


def _print_task_debug_done(
    *,
    env,
    step_idx: int,
    terminated,
    truncated,
    task_success: bool,
):
    print(
        "task_debug_done "
        f"step={step_idx} "
        f"terminated={_first_bool(terminated)} "
        f"truncated={_first_bool(truncated)} "
        f"task_success={task_success}",
        flush=True,
    )
    try:
        terms = env.termination_manager.get_active_iterable_terms(0)
        term_text = ", ".join(f"{name}={_first_list(values)[0]}" for name, values in terms)
        print(f"task_debug_done terms: {term_text}", flush=True)
    except Exception as exc:
        print(f"task_debug_done terms unavailable: {exc}", flush=True)

    scene_positions = {
        name: _debug_scene_root_pos(env, name)
        for name in ("apple", "pear", "pot", "egg", "cover")
    }
    scene_positions = {name: pos for name, pos in scene_positions.items() if pos is not None}
    if scene_positions:
        print(
            "task_debug_done scene_root_pos_after_possible_reset: "
            + json.dumps(scene_positions),
            flush=True,
        )


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
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task prompt. Defaults to the matching task_prompts.json entry.",
    )
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
        default=DEFAULT_BASE_CHECKPOINT_DIR,
        help="Base policy checkpoint directory. Defaults to the weight-task pi05 Droid joint-position checkpoint.",
    )
    parser.add_argument(
        "--task_checkpoint_dir",
        type=str,
        default=None,
        help="Task proxy checkpoint directory. Defaults to the matching task_prompts.json entry.",
    )
    parser.add_argument(
        "--ref_checkpoint_dir",
        type=str,
        default=None,
        help="Reference proxy checkpoint directory. Defaults to the matching task_prompts.json entry.",
    )
    standalone_group = parser.add_mutually_exclusive_group()
    standalone_group.add_argument(
        "--ref_only",
        "--ref-only",
        dest="ref_only",
        action="store_true",
        help="Evaluate the reference proxy directly from noise, without base/MPC/task steering.",
    )
    standalone_group.add_argument(
        "--task_only",
        "--task-only",
        dest="task_only",
        action="store_true",
        help="Evaluate the task proxy directly from noise, without base/MPC/ref steering.",
    )
    parser.add_argument(
        "--steer_scale",
        type=float,
        default=0.4,
        help="Lambda multiplying the task residual in full/task score steering.",
    )
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "--task_num_steps",
        type=int,
        default=225,
        help="Maximum rollout control steps. Default 225 is 15 seconds at 15 Hz.",
    )
    parser.add_argument(
        "--steps_per_inference",
        type=int,
        default=8,
        help=(
            "Number of environment control steps executed from each inferred action chunk. "
            "Smaller values replan more often. Default 8 preserves the previous behavior."
        ),
    )
    parser.add_argument("--only_steer", action="store_true")
    parser.add_argument(
        "--no_steer",
        action="store_true",
        help="Disable task/ref steering and use only the base path.",
    )
    parser.add_argument("--compare_difference", action="store_true")
    score_mode_group = parser.add_mutually_exclusive_group()
    score_mode_group.add_argument(
        "--vlm_base",
        "--vlm-base",
        dest="vlm_base",
        action="store_true",
        help="Evaluate only the FK/cost MBD base score, without task/ref steering.",
    )
    score_mode_group.add_argument(
        "--full_steer",
        "--full-steer",
        dest="full_steer",
        action="store_true",
        help="Run score-space MBD base + steer_scale * (task - ref).",
    )
    score_mode_group.add_argument(
        "--task_steer",
        "--task-steer",
        dest="task_steer",
        action="store_true",
        help="Run score-space MBD base + steer_scale * task, without loading ref.",
    )
    parser.add_argument(
        "--gamma_base",
        type=float,
        default=1.0,
        help="Scale multiplying the MBD base score before score composition.",
    )
    parser.add_argument("--mpc_num_samples", type=int, default=512)
    parser.add_argument("--mpc_iterations", type=int, default=8)
    parser.add_argument("--mpc_noise", type=float, default=0.35)
    parser.add_argument("--mpc_temperature", type=float, default=0.15)
    parser.add_argument(
        "--mpc_update",
        choices=(
            "ddim",
            "mbd_score",
            "mbd_score_action_prox",
            "mbd_score_action_warm",
            "legacy_score",
        ),
        default="mbd_score_action_prox",
        help=(
            "Reverse update used by MBD base/full/task score modes. Defaults to "
            "mbd_score_action_prox to match the current Weight ref distillation teacher."
        ),
    )
    parser.add_argument(
        "--mpc_cost",
        choices=("priority", "ref_style", "explore", "grasp_flow", "capsule_flow"),
        default="priority",
        help="Cost function used by sim-free MPC.",
    )
    parser.add_argument(
        "--mpc_optimize_space",
        choices=("action", "accel"),
        default="action",
        help=(
            "Parameterization used by sim-free MPC sampling. 'accel' runs pure "
            "MPPI over real joint accelerations and integrates directly to action chunks."
        ),
    )
    parser.add_argument(
        "--mpc_ddim_train_timesteps",
        type=int,
        default=100,
        help="Number of training timesteps used to discretize the DDIM cosine schedule.",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help=(
            "Optimize sim-free MPC at a lower knot rate and linearly "
            "interpolate back to the action horizon."
        ),
    )
    parser.add_argument(
        "--interpolate_low_frequency",
        type=float,
        default=5.0,
        help="Low-frequency knot rate used by --interpolate.",
    )
    parser.add_argument(
        "--interpolate_high_frequency",
        type=float,
        default=40.0,
        help=(
            "High-frequency rate used by --interpolate to compute the knot ratio. "
            "This does not change the Isaac env control rate."
        ),
    )
    parser.add_argument(
        "--mpc_joint_delta_clip",
        type=float,
        default=0.15,
        help=(
            "Clamp decoded MPC, --ref-only, or --task-only joint-position targets to this many "
            "radians per control step. "
            "Set to 0 to disable the per-step delta clamp. Joint limits are still enforced."
        ),
    )
    parser.add_argument("--mpc_debug", action="store_true")
    parser.add_argument(
        "--mpc_debug_stdout",
        action="store_true",
        help="Also print detailed MPC debug iterations to stdout. By default --mpc_debug writes them to a jsonl log.",
    )
    parser.add_argument("--task_debug", action="store_true")
    parser.add_argument(
        "--debug_hold_pear",
        action="store_true",
        help=(
            "Debug-only: before pear is detected as grasped, keep it inside a "
            "small horizontal XY guard window so it does not slide off the table edge."
        ),
    )
    parser.add_argument(
        "--debug_hold_pear_xy_margin",
        type=float,
        default=0.10,
        help="Half-width in meters of the debug pear XY guard window around the reset position.",
    )
    parser.add_argument(
        "--mpc_debug_axis_length",
        type=float,
        default=0.08,
        help="Axis length in meters for --mpc_debug_video_overlay.",
    )
    parser.add_argument(
        "--mpc_debug_video_overlay",
        action="store_true",
        help=(
            "Debug-only: draw projected task-object/gripper/finger axes directly "
            "on saved rollout videos. Capsule adds lid/gripper/capsule axes. "
            "Works in headless mode."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run one warmup inference, print base-source and shape diagnostics, then exit.",
    )
    return parser


def _task_prompt_entry(task_name: str) -> dict[str, str] | None:
    try:
        with open(TASK_PROMPTS_PATH, encoding="utf-8") as handle:
            entries = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load task prompt config {TASK_PROMPTS_PATH}: {exc}") from exc

    if not isinstance(entries, dict):
        raise ValueError(f"Expected an object at the top level of {TASK_PROMPTS_PATH}.")

    for entry in entries.values():
        if isinstance(entry, dict) and entry.get("task_id") == task_name:
            return entry

    task_lower = task_name.lower()
    key_matches = [
        entry
        for key, entry in entries.items()
        if isinstance(entry, dict) and (key.lower() == task_lower or key.lower() in task_lower)
    ]
    if len(key_matches) == 1:
        return key_matches[0]
    return None


def _repo_relative_path(value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(_REPO_DIR, value)


def _apply_task_prompt_defaults(args, parser: argparse.ArgumentParser) -> None:
    required_fields = ["prompt"]
    required_roles = _required_policy_roles(args)
    if "task" in required_roles:
        required_fields.append("task_checkpoint_dir")
    if "ref" in required_roles:
        required_fields.append("ref_checkpoint_dir")
    missing_fields = [
        field
        for field in required_fields
        if getattr(args, field) is None
    ]
    if not missing_fields:
        return

    try:
        entry = _task_prompt_entry(args.task)
    except ValueError as exc:
        parser.error(str(exc))
    if entry is None:
        parser.error(
            f"Task {args.task!r} has no matching entry in {TASK_PROMPTS_PATH}. "
            f"Provide {', '.join('--' + field for field in required_fields)} explicitly."
        )

    for field in missing_fields:
        entry_field = field
        if field.endswith("_checkpoint_dir") and (
            _score_steering_mode(args) in ("full", "task")
            or _standalone_policy_role(args) is not None
        ):
            entry_field = f"score_{field}"
        value = entry.get(entry_field)
        if not isinstance(value, str) or not value:
            parser.error(
                f"Task {args.task!r} is missing a valid {entry_field!r} in {TASK_PROMPTS_PATH}."
            )
        if field.endswith("_checkpoint_dir"):
            value = _repo_relative_path(value)
        setattr(args, field, value)


parser = parse_args()
AppLauncher.add_app_launcher_args(parser)
# Default the IsaacLab app to headless rendering with cameras enabled so these
# flags are not needed on the inference command line.
parser.set_defaults(enable_cameras=True, headless=True)

args = parser.parse_args()
_apply_task_prompt_defaults(args, parser)
standalone_role = _standalone_policy_role(args)
if standalone_role is not None:
    incompatible_flags = [
        flag
        for flag, enabled in (
            ("--vlm_base", args.vlm_base),
            ("--full_steer", args.full_steer),
            ("--task_steer", args.task_steer),
            ("--no_steer", args.no_steer),
            ("--only_steer", args.only_steer),
            ("--compare_difference", args.compare_difference),
        )
        if enabled
    ]
    if incompatible_flags:
        parser.error(
            f"--{standalone_role}_only directly evaluates one proxy and cannot be combined with "
            + ", ".join(incompatible_flags)
            + "."
        )
score_steering_mode = _score_steering_mode(args)
if score_steering_mode in ("full", "task"):
    incompatible_flags = [
        flag
        for flag, enabled in (
            ("--no_steer", args.no_steer),
            ("--only_steer", args.only_steer),
            ("--compare_difference", args.compare_difference),
        )
        if enabled
    ]
    if incompatible_flags:
        parser.error(
            f"--{score_steering_mode}_steer already defines the score composition and cannot "
            "be combined with " + ", ".join(incompatible_flags) + "."
        )
if score_steering_mode == "base" and (args.only_steer or args.compare_difference):
    parser.error("--vlm_base is base-only and cannot be combined with --only_steer or --compare_difference.")
if args.mpc_cost == "capsule_flow" and "capsule" not in args.task.lower():
    parser.error(
        "--mpc_cost capsule_flow requires a capsule task, for example "
        "--task Isaac-Capsule-Droid-Visuomotor-v0."
    )
if _uses_accel_action_mpc(args):
    if score_steering_mode != "base":
        raise ValueError("--mpc_optimize_space accel currently supports --vlm_base base-only mode.")
    if args.mpc_update not in ("legacy_score", "mbd_score"):
        raise ValueError(
            "Acceleration action-space MPC supports --mpc_update legacy_score or mbd_score. "
            "DDIM/action-prox variants are score-space only for this direct-action planner."
        )
if args.steps_per_inference <= 0:
    raise ValueError("--steps_per_inference must be positive.")

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

print(f"Environment name: {env_name}", flush=True)

# Configure environment
print("Parsing env cfg...", flush=True)
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
print("Creating eval env...", flush=True)
env = gym.make(env_name, cfg=env_cfg).unwrapped
print("Eval env created.", flush=True)

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
print("Resolving checkpoint configs...", flush=True)
base_policy = None
task_policy = None
ref_policy = None
required_policy_roles = _required_policy_roles(args)
if "base" in required_policy_roles:
    base_config = _config.get_config(_config_name_from_checkpoint_dir(base_checkpoint_dir))
    print("Loading base policy checkpoint...", flush=True)
    base_policy = policy_config.create_trained_policy(base_config, base_checkpoint_dir)
if "task" in required_policy_roles:
    task_config = _config.get_config(_config_name_from_checkpoint_dir(task_checkpoint_dir))
    print("Loading task policy checkpoint...", flush=True)
    task_policy = policy_config.create_trained_policy(
        task_config,
        task_checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps} if standalone_role == "task" else None,
    )
if "ref" in required_policy_roles:
    ref_config = _config.get_config(_config_name_from_checkpoint_dir(ref_checkpoint_dir))
    print("Loading ref policy checkpoint...", flush=True)
    ref_policy = policy_config.create_trained_policy(
        ref_config,
        ref_checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps} if standalone_role == "ref" else None,
    )

base_source = _base_source_name(args)
print(f"Base source: {base_source}", flush=True)
loaded_model_types = {
    role: policy._model.config.model_type.value
    for role, policy in (
        ("base", base_policy),
        ("task", task_policy),
        ("ref", ref_policy),
    )
    if policy is not None
}
print(
    "Loaded checkpoints: "
    + ", ".join(f"{role}={model_type}" for role, model_type in loaded_model_types.items()),
    flush=True,
)
if args.mpc_debug and base_policy is not None:
    base_policy._metadata = {
        **(getattr(base_policy, "_metadata", {}) or {}),
        "debug_torch_output_to_actions_norm_stats": True,
    }

CONTROL_FREQUENCY = 15

mpc_planner = None
if _uses_vlm_mpc_base(args):
    inferred_task = _task_name_for_mpc(args.task)
    if _uses_accel_action_mpc(args):
        mpc_planner = AccelActionMPC(
            AccelMPCConfig(
                task_name=inferred_task,
                num_samples=args.mpc_num_samples,
                iterations=args.mpc_iterations,
                noise=args.mpc_noise,
                temperature=args.mpc_temperature,
                cost_style=args.mpc_cost,
                control_frequency=CONTROL_FREQUENCY,
                ddim_num_train_timesteps=args.mpc_ddim_train_timesteps,
            )
        )
    else:
        mpc_planner = SimFreeMPC(
            base_policy,
            SimFreeMPCConfig(
                task_name=inferred_task,
                num_samples=args.mpc_num_samples,
                iterations=args.mpc_iterations,
                noise=args.mpc_noise,
                temperature=args.mpc_temperature,
                action_dims=8,
                joint_delta_clip=args.mpc_joint_delta_clip,
                ddim_num_train_timesteps=args.mpc_ddim_train_timesteps,
                interpolate=args.interpolate,
                control_frequency=args.interpolate_high_frequency,
                interpolate_frequency=args.interpolate_low_frequency,
                cost_style=args.mpc_cost,
                optimize_space=args.mpc_optimize_space,
            ),
        )
    print(
        "Sim-free MPC planner enabled: "
        f"base_source={base_source}, task={inferred_task}, samples={args.mpc_num_samples}, "
        f"iterations={args.mpc_iterations}, update={args.mpc_update}, cost={args.mpc_cost}, "
        f"optimize_space={args.mpc_optimize_space}, "
        f"ddim_train_timesteps={args.mpc_ddim_train_timesteps}, gamma_base={args.gamma_base}, "
        f"joint_delta_clip={args.mpc_joint_delta_clip}, interpolate={args.interpolate}, "
        f"interpolate_low_frequency={args.interpolate_low_frequency}, "
        f"interpolate_high_frequency={args.interpolate_high_frequency}",
        flush=True,
    )

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

if standalone_role is None:
    loaded_policies = {
        role: policy
        for role, policy in (
            ("base", base_policy),
            ("task", task_policy),
            ("ref", ref_policy),
        )
        if policy is not None
    }
    horizons = {
        role: int(policy._model.config.action_horizon)
        for role, policy in loaded_policies.items()
    }
    if len(set(horizons.values())) > 1:
        raise ValueError(f"Action horizon mismatch across loaded policies: {horizons}.")
    if task_policy is not None and ref_policy is not None:
        task_action_dim = int(task_policy._model.config.action_dim)
        ref_action_dim = int(ref_policy._model.config.action_dim)
        if task_action_dim != ref_action_dim:
            raise ValueError(
                "Action dimension mismatch between task/ref policies: "
                f"task={task_action_dim}, ref={ref_action_dim}."
            )

    proxy_action_dims = [
        int(policy._model.config.action_dim)
        for policy in (task_policy, ref_policy)
        if policy is not None
    ]
    _warn_if_norm_mismatch(
        base_policy,
        task_policy,
        ref_policy,
        action_dim=min(proxy_action_dims) if proxy_action_dims else 8,
    )
    _assert_score_space_compatibility(base_policy, task_policy, ref_policy, args)

steps_per_inference = int(args.steps_per_inference)
print(
    f"steps_per_inference={steps_per_inference} "
    f"({steps_per_inference / CONTROL_FREQUENCY:.3f}s between replans at {CONTROL_FREQUENCY}Hz)",
    flush=True,
)

print("Resetting env for warmup...", flush=True)
env_obs_dict, _ = env.reset()
print("Building warmup observation...", flush=True)
obs = get_pi_observation(env_obs_dict["policy"])
obs["prompt"] = args.prompt
print("Running warmup policy inference...", flush=True)
with torch.no_grad():
    warmup_actions, _ = infer_actions_with_mpc(
        base_policy,
        task_policy,
        ref_policy,
        copy.deepcopy(obs),
        args,
        mpc_planner=mpc_planner,
        mpc_context=(
            None
            if standalone_role is not None
            else build_mpc_context(env, env_obs_dict, args)
        ),
    )
print("Warmup policy inference finished.", flush=True)

if args.dry_run:
    print("Dry-run inference check passed", flush=True)
    print(
        "  checkpoints_loaded: "
        + ", ".join(
            f"{role}={model_type}" for role, model_type in loaded_model_types.items()
        ),
        flush=True,
    )
    print(f"  base_source: {_LAST_INFERENCE_RUNTIME.get('base_source')}", flush=True)
    print(
        "  used_base_model_velocity: "
        f"{_LAST_INFERENCE_RUNTIME.get('used_base_model_velocity')}",
        flush=True,
    )
    print(f"  steering_mode: {_LAST_INFERENCE_RUNTIME.get('steering_mode')}", flush=True)
    print(f"  x_t_shape: {_LAST_INFERENCE_RUNTIME.get('x_t_shape')}", flush=True)
    print(f"  v_vlm_shape: {_LAST_INFERENCE_RUNTIME.get('v_vlm_shape')}", flush=True)
    print(f"  score_shape: {_LAST_INFERENCE_RUNTIME.get('score_shape')}", flush=True)
    print(f"  v_task_shape: {_LAST_INFERENCE_RUNTIME.get('v_task_shape')}", flush=True)
    print(f"  v_ref_shape: {_LAST_INFERENCE_RUNTIME.get('v_ref_shape')}", flush=True)
    print(f"  proxy_task_shape: {_LAST_INFERENCE_RUNTIME.get('proxy_task_shape')}", flush=True)
    print(f"  proxy_ref_shape: {_LAST_INFERENCE_RUNTIME.get('proxy_ref_shape')}", flush=True)
    print(f"  output_action_shape: {np.asarray(warmup_actions).shape}", flush=True)
    if _LAST_INFERENCE_RUNTIME.get("mpc_last") is not None:
        print(f"  mpc_last: {_LAST_INFERENCE_RUNTIME['mpc_last']}", flush=True)
    env.close()
    if dataset_file is not None:
        dataset_file.close()
    simulation_app.close()
    sys.exit(0)

print("Ready!", flush=True)
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
total_comparison_observation_steps = 0
total_inference_time_s = 0.0
total_inference_calls = 0
video_run_id = time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
experiment_output_path = os.path.join(
    output_path,
    _experiment_output_name(args, run_id=video_run_id),
)
os.makedirs(experiment_output_path, exist_ok=False)
experiment_results_path = os.path.join(experiment_output_path, "results.json")
experiment_results = {
    "run_id": video_run_id,
    "config_slug": _video_config_slug(args),
    "command": sys.argv,
    "config": vars(args),
    "episodes": [],
}
_write_experiment_results(experiment_results_path, experiment_results)
print(
    f"Experiment output: {experiment_output_path}; max rollout duration "
    f"{args.task_num_steps / CONTROL_FREQUENCY:.1f}s",
    flush=True,
)
mpc_debug_log_file = None
mpc_debug_log_path = None
if args.mpc_debug:
    mpc_debug_log_path = os.path.join(experiment_output_path, "mpc_debug.jsonl")
    mpc_debug_log_file = open(mpc_debug_log_path, "w", encoding="utf-8")
    print(f"MPC debug log: {mpc_debug_log_path}", flush=True)
    _write_mpc_debug_log(
        mpc_debug_log_file,
        "run_start",
        run_id=video_run_id,
        task=args.task,
        prompt=args.prompt,
        mpc_cost=args.mpc_cost,
        mpc_update=args.mpc_update,
        mpc_optimize_space=args.mpc_optimize_space,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        steps_per_inference=steps_per_inference,
        task_num_steps=args.task_num_steps,
    )
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

    if mpc_planner is not None and args.mpc_update == "mbd_score_action_warm":
        mpc_planner.reset_action_warm()

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

    current_subtasks = _debug_subtasks(env_obs_dict)
    current_phase = _debug_phase_from_subtasks(args.task, current_subtasks)
    print(f"phase seed={seed} step=0 {current_phase} subtasks={current_subtasks}", flush=True)
    _write_mpc_debug_log(
        mpc_debug_log_file,
        "rollout_start",
        seed=seed,
        rollout_idx=rollout_idx,
        step=0,
        phase=current_phase,
        subtasks=current_subtasks,
    )

    pear_guard_state = (
        _capture_debug_object_xy_guard(
            env,
            "pear",
            margin=args.debug_hold_pear_xy_margin,
        )
        if args.debug_hold_pear
        else None
    )
    if pear_guard_state is not None:
        print(
            "debug_hold_pear enabled "
            f"xy_min={pear_guard_state['xy_min'].detach().cpu().tolist()} "
            f"xy_max={pear_guard_state['xy_max'].detach().cpu().tolist()}",
            flush=True,
        )

    excute_frames = []
    thermal_overlay_frames = []
    sound_audio_frames = []
    obs = get_pi_observation(env_obs_dict["policy"])
    obs["prompt"] = args.prompt
    video_header_lines = _video_config_lines(args, seed=seed)

    # ========== policy control loop ==============
    step_idx = 0
    success = False
    force_replan = False
    action_start_step = -steps_per_inference
    actions = None
    for step_idx in tqdm(range(args.task_num_steps), desc="Policy Control Loop"):
        try:
            if (
                actions is None
                or force_replan
                or step_idx - action_start_step >= steps_per_inference
            ):
                # print('predict_action')
                # run inference
                with torch.no_grad():
                    infer_start = time.perf_counter()
                    mpc_context = (
                        None
                        if standalone_role is not None
                        else build_mpc_context(env, env_obs_dict, args)
                    )
                    warm_shift_steps = (
                        0 if actions is None else max(step_idx - action_start_step, 0)
                    )
                    actions, compare_stats = infer_actions_with_mpc(
                        base_policy,
                        task_policy,
                        ref_policy,
                        copy.deepcopy(obs),
                        args,
                        mpc_planner=mpc_planner,
                        mpc_context=mpc_context,
                        warm_shift_steps=warm_shift_steps,
                    )
                    infer_elapsed = time.perf_counter() - infer_start
                    if args.mpc_debug:
                        _write_mpc_debug_log(
                            mpc_debug_log_file,
                            "inference",
                            seed=seed,
                            step=step_idx,
                            phase=current_phase,
                            subtasks=current_subtasks,
                            elapsed_s=infer_elapsed,
                            mpc=_mpc_debug_stats(_LAST_INFERENCE_RUNTIME.get("mpc_last")),
                            mpc_trace=_LAST_INFERENCE_RUNTIME.get("mpc_trace", []),
                        )
                    episode_inference_time_s += infer_elapsed
                    episode_inference_calls += 1
                    total_inference_time_s += infer_elapsed
                    total_inference_calls += 1
                    force_replan = False
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
                action_start_step = step_idx

            action_step = actions[step_idx - action_start_step]

            # perform step
            env_obs_dict, rewards, terminated, truncated, extras = env.step(
                torch.as_tensor(
                    action_step[None],
                    dtype=torch.float32,
                    device=env.device,
                )
            )
            if args.debug_hold_pear:
                pear_grasped = _subtask_flag_from_obs(env_obs_dict, "grasp_pear")
                pose_changed = _apply_debug_object_xy_guard(
                    env,
                    pear_guard_state,
                    release=pear_grasped,
                )
                if pose_changed:
                    env_obs_dict = env.observation_manager.compute(update_history=True)

            next_subtasks = _debug_subtasks(env_obs_dict)
            next_phase = _debug_phase_from_subtasks(args.task, next_subtasks)
            if next_phase != current_phase:
                print(
                    f"phase_transition seed={seed} step={step_idx + 1} "
                    f"{current_phase}->{next_phase} subtasks={next_subtasks}",
                    flush=True,
                )
                _write_mpc_debug_log(
                    mpc_debug_log_file,
                    "phase_transition",
                    seed=seed,
                    step=step_idx + 1,
                    from_phase=current_phase,
                    to_phase=next_phase,
                    subtasks=next_subtasks,
                )
                force_replan = True
            current_phase = next_phase
            current_subtasks = next_subtasks
            _write_mpc_debug_log(
                mpc_debug_log_file,
                "step",
                seed=seed,
                step=step_idx + 1,
                phase=current_phase,
                subtasks=current_subtasks,
                action_gripper=_debug_action_gripper(action_step),
            )

            obs = get_pi_observation(env_obs_dict["policy"])
            obs["prompt"] = args.prompt

            # Check for task success using success_term
            task_success = bool(success_term.func(env, **success_term.params)[0])

            # save visualization
            debug_overlay = None
            if args.mpc_debug_video_overlay:
                debug_overlay = {
                    "env": env,
                    "axes": _collect_mpc_debug_frames(
                        env,
                        axis_length=args.mpc_debug_axis_length,
                    ),
                }
            vis_image = _build_rollout_frame(
                obs,
                use_thermal_overlay=False,
                debug_overlay=debug_overlay,
            )
            excute_frames.append(_add_video_header(vis_image, video_header_lines))
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
                    _add_video_header(
                        _build_rollout_frame(
                            obs,
                            use_thermal_overlay=True,
                            debug_overlay=debug_overlay,
                        ),
                        video_header_lines,
                    )
                )

            step_idx += 1

            if terminated or truncated or task_success:
                if args.task_debug:
                    _print_task_debug_done(
                        env=env,
                        step_idx=step_idx,
                        terminated=terminated,
                        truncated=truncated,
                        task_success=task_success,
                    )
                print("terminated or truncated or task completed")
                success = task_success
                _write_mpc_debug_log(
                    mpc_debug_log_file,
                    "done",
                    seed=seed,
                    step=step_idx,
                    phase=current_phase,
                    subtasks=current_subtasks,
                    terminated=terminated,
                    truncated=truncated,
                    task_success=task_success,
                )
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

    video_path = os.path.join(experiment_output_path, video_name)
    audio_path = _write_stereo_sound_audio(
        os.path.join(experiment_output_path, f"{seed}_recording.wav"),
        sound_audio_frames,
    )
    video_write_path = (
        os.path.join(experiment_output_path, f"{seed}_recording.mp4")
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

    thermal_video_name = None
    if thermal_overlay_frames:
        thermal_video_name = _episode_video_name(
            seed,
            success,
            suffix="thermal_overlay",
        )
        thermal_video_path = os.path.join(experiment_output_path, thermal_video_name)
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
    _write_mpc_debug_log(
        mpc_debug_log_file,
        "rollout_end",
        seed=seed,
        rollout_idx=rollout_idx,
        steps=step_idx,
        phase=current_phase,
        subtasks=current_subtasks,
        success=success,
        video_path=video_path,
    )
    if episode_inference_calls:
        avg_infer_ms = 1000.0 * episode_inference_time_s / episode_inference_calls
        print(
            f"seed {seed} infer_actions avg latency: {avg_infer_ms:.2f} ms "
            f"over {episode_inference_calls} calls"
        )
    else:
        avg_infer_ms = None

    experiment_results["episodes"].append(
        {
            "rollout_index": rollout_idx,
            "seed": seed,
            "success": bool(success),
            "steps": step_idx,
            "video": video_name,
            "thermal_video": thermal_video_name,
            "inference_calls": episode_inference_calls,
            "average_inference_ms": avg_infer_ms,
        }
    )
    _write_experiment_results(experiment_results_path, experiment_results)

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

env.close()
if dataset_file is not None:
    dataset_file.close()

if args.compare_difference:
    comparison_summary = {
        "compare_difference": True,
        "output_path": experiment_output_path,
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

    comparison_path = os.path.join(experiment_output_path, "compare_difference.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"compare difference statistics saved to {comparison_path}")
    experiment_results["compare_difference"] = os.path.basename(comparison_path)

if total_inference_calls:
    avg_infer_ms = 1000.0 * total_inference_time_s / total_inference_calls
    print(
        f"overall infer_actions avg latency: {avg_infer_ms:.2f} ms "
        f"over {total_inference_calls} calls"
    )

experiment_results["total_inference_calls"] = total_inference_calls
experiment_results["total_inference_time_s"] = total_inference_time_s
_write_experiment_results(experiment_results_path, experiment_results)

_write_mpc_debug_log(
    mpc_debug_log_file,
    "run_end",
    total_inference_calls=total_inference_calls,
    total_inference_time_s=total_inference_time_s,
)
if mpc_debug_log_file is not None:
    mpc_debug_log_file.close()

# Close the simulation app after environment is closed
simulation_app.close()
