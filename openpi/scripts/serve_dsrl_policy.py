import copy
import dataclasses
import logging
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dsrl"))
sys.path.insert(0, str(ROOT / "dsrl" / "stable-baselines3"))
sys.path.insert(0, str(ROOT / "openpi" / "src"))
sys.path.insert(0, str(ROOT / "openpi" / "packages" / "openpi-client" / "src"))

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import torch
import tyro

from stable_baselines3 import DSRL
from stable_baselines3.common.save_util import load_from_zip_file

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

import train_openpi_dsrl_na_offline as dsrl_train_module


# Older checkpoints may reference either module name when unpickling custom objects.
sys.modules.setdefault("train_openpi_dsrl_na_offline", dsrl_train_module)
sys.modules.setdefault("train_openpi_dorl_na_offline", dsrl_train_module)


OPENPI_IMAGE_KEY = dsrl_train_module.OPENPI_IMAGE_KEY
OPENPI_WRIST_IMAGE_KEY = dsrl_train_module.OPENPI_WRIST_IMAGE_KEY
REQUIRED_OBSERVATION_KEYS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
)
DSRL_EXTERIOR_IMAGE_KEY = "dsrl/observation/exterior_image_1_left"
DSRL_WRIST_IMAGE_KEY = "dsrl/observation/wrist_image_left"


class OfflineDictEnv(gym.Env if gym is not None else object):
    metadata = {}

    def __init__(self, observation_space, action_space):
        if gym is None:
            raise ImportError(
                "Loading a DSRL checkpoint requires gymnasium or gym to be installed."
            )
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


class OpenPIBatchPolicyWrapper:
    """Small DSRL load-time wrapper around OpenPI batched inference."""

    def __init__(self, policy, noise_action_dim: int):
        self._policy = policy
        self._noise_action_dim = noise_action_dim

    def __call__(self, obs, initial_noise, return_numpy=True):
        if not isinstance(obs, dict):
            raise TypeError(f"Expected dict observations, got {type(obs)}")

        obs_np = {
            key: (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
            for key, value in obs.items()
        }
        noise_np = (
            initial_noise.detach().cpu().numpy()
            if isinstance(initial_noise, torch.Tensor)
            else np.asarray(initial_noise)
        )

        image_key = OPENPI_IMAGE_KEY if OPENPI_IMAGE_KEY in obs_np else "image"
        wrist_image_key = (
            OPENPI_WRIST_IMAGE_KEY
            if OPENPI_WRIST_IMAGE_KEY in obs_np
            else "wrist_image"
        )
        obs_list = []
        for i in range(obs_np["state"].shape[0]):
            obs_list.append(
                {
                    "observation/exterior_image_1_left": np.transpose(
                        obs_np[image_key][i], (1, 2, 0)
                    ),
                    "observation/wrist_image_left": np.transpose(
                        obs_np[wrist_image_key][i], (1, 2, 0)
                    ),
                    "observation/joint_position": obs_np["state"][i, :7].astype(
                        np.float32
                    ),
                    "observation/gripper_position": obs_np["state"][i, 7:8].astype(
                        np.float32
                    ),
                }
            )

        if noise_np.shape[-1] != self._noise_action_dim:
            raise ValueError(
                f"Expected latent noise dim {self._noise_action_dim}, "
                f"got {noise_np.shape[-1]}."
            )

        actions = self._policy.infer_batch(obs_list, noise=noise_np)["actions"].astype(
            np.float32
        )
        if return_numpy:
            return actions

        device = (
            initial_noise.device
            if isinstance(initial_noise, torch.Tensor)
            else torch.device("cpu")
        )
        return torch.as_tensor(actions, device=device, dtype=torch.float32)


def _image_hw_from_space(
    observation_space, key: str, default_hw: tuple[int, int]
) -> tuple[int, int]:
    if (
        observation_space is not None
        and hasattr(observation_space, "spaces")
        and key in observation_space.spaces
    ):
        shape = observation_space.spaces[key].shape
        if len(shape) == 3:
            return int(shape[1]), int(shape[2])
    return default_hw


def _resize_hwc_image(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] == target_hw:
        return image
    target_h, target_w = target_hw
    return np.asarray(
        Image.fromarray(image).resize((target_w, target_h), resample=Image.BICUBIC),
        dtype=np.uint8,
    )


def _validate_raw_observation(raw_obs: dict[str, Any]) -> None:
    missing_keys = [key for key in REQUIRED_OBSERVATION_KEYS if key not in raw_obs]
    if missing_keys:
        raise KeyError(
            "DSRL server request is missing required observation keys: "
            f"{missing_keys}. The DROID client should send keys "
            f"{list(REQUIRED_OBSERVATION_KEYS)}."
        )

    for key in (DSRL_EXTERIOR_IMAGE_KEY, "observation/exterior_image_1_left"):
        if key in raw_obs:
            exterior_image_key = key
            break
    else:
        exterior_image_key = "observation/exterior_image_1_left"

    for key in (DSRL_WRIST_IMAGE_KEY, "observation/wrist_image_left"):
        if key in raw_obs:
            wrist_image_key = key
            break
    else:
        wrist_image_key = "observation/wrist_image_left"

    for key in (exterior_image_key, wrist_image_key):
        image = np.asarray(raw_obs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"{key} must be an HWC RGB image with shape (H, W, 3), "
                f"got {image.shape}."
            )

    joint_position = np.asarray(raw_obs["observation/joint_position"])
    gripper_position = np.asarray(raw_obs["observation/gripper_position"])
    if joint_position.size < 7:
        raise ValueError(
            "observation/joint_position must contain at least 7 joints, "
            f"got shape {joint_position.shape}."
        )
    if gripper_position.size < 1:
        raise ValueError(
            "observation/gripper_position must contain one gripper value, "
            f"got shape {gripper_position.shape}."
        )


def _make_dsrl_observation(
    raw_obs: dict[str, Any], observation_space
) -> dict[str, np.ndarray]:
    _validate_raw_observation(raw_obs)

    image_hw = _image_hw_from_space(
        observation_space, "image", dsrl_train_module.ACTOR_CRITIC_IMAGE_SIZE
    )
    wrist_image_hw = _image_hw_from_space(observation_space, "wrist_image", image_hw)

    exterior_image = _resize_hwc_image(
        raw_obs.get(
            DSRL_EXTERIOR_IMAGE_KEY, raw_obs["observation/exterior_image_1_left"]
        ),
        image_hw,
    )
    wrist_image = _resize_hwc_image(
        raw_obs.get(DSRL_WRIST_IMAGE_KEY, raw_obs["observation/wrist_image_left"]),
        wrist_image_hw,
    )
    joint_position = np.asarray(raw_obs["observation/joint_position"], dtype=np.float32)
    gripper_position = np.asarray(
        raw_obs["observation/gripper_position"], dtype=np.float32
    )

    return {
        "image": np.transpose(exterior_image, (2, 0, 1)),
        "wrist_image": np.transpose(wrist_image, (2, 0, 1)),
        "state": np.concatenate(
            [joint_position[:7], gripper_position.reshape(-1)[:1]], axis=0
        ).astype(np.float32),
    }


def _format_openpi_noise(
    predicted_noise: np.ndarray, action_horizon: int, noise_dim: int
) -> np.ndarray:
    flat_noise = np.asarray(predicted_noise, dtype=np.float32).reshape(-1)
    if flat_noise.size == noise_dim:
        return np.repeat(flat_noise[None, :], action_horizon, axis=0)
    if flat_noise.size == action_horizon * noise_dim:
        return flat_noise.reshape(action_horizon, noise_dim)
    raise ValueError(
        "DSRL actor predicted incompatible noise size "
        f"{flat_noise.size}. Expected {noise_dim} for one-step noise or "
        f"{action_horizon * noise_dim} for full-horizon noise."
    )


class DSRLPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        *,
        base_policy,
        dsrl_model: DSRL,
        action_horizon: int,
        noise_dim: int,
        deterministic: bool,
        expose_noise: bool = False,
    ) -> None:
        self._base_policy = base_policy
        self._dsrl_model = dsrl_model
        self._action_horizon = action_horizon
        self._noise_dim = noise_dim
        self._deterministic = deterministic
        self._expose_noise = expose_noise

    def infer(self, obs: dict) -> dict:
        dsrl_obs = _make_dsrl_observation(obs, self._dsrl_model.observation_space)

        dsrl_start_time = time.monotonic()
        with torch.no_grad():
            predicted_noise, _ = self._dsrl_model.predict(
                dsrl_obs,
                deterministic=self._deterministic,
            )
        dsrl_infer_ms = (time.monotonic() - dsrl_start_time) * 1000

        openpi_noise = _format_openpi_noise(
            predicted_noise, self._action_horizon, self._noise_dim
        )
        results = self._base_policy.infer(copy.deepcopy(obs), noise=openpi_noise)
        results["dsrl_timing"] = {"infer_ms": dsrl_infer_ms}
        if self._expose_noise:
            results["dsrl_noise"] = openpi_noise
        results.setdefault(
            "visualize_vectors_step",
            {
                "num_steps": 0,
                "inputs": {"state": dsrl_obs["state"]},
                "vectors": [],
                "base_vectors": [],
                "steer_vectors": [],
                "mimic_vectors": [],
            },
        )
        return results


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    dsrl_checkpoint: str
    model_name: str = "pi05_droid_jointpos"
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    num_steps: int = 10
    device: str = "cuda:0"
    openpi_pytorch_device: str | None = None
    dsrl_deterministic: bool = True
    expose_noise: bool = False
    warmup: bool = True
    warmup_prompt: str | None = None


def _load_dsrl_model(
    *,
    dsrl_checkpoint: str,
    dsrl_device: str,
    base_policy,
    action_horizon: int,
    noise_dim: int,
) -> DSRL:
    diffusion_policy = OpenPIBatchPolicyWrapper(
        base_policy, noise_action_dim=noise_dim
    )
    custom_objects = {
        "diffusion_policy": diffusion_policy,
        "ActorCriticOnlyCombinedExtractor": (
            dsrl_train_module.ActorCriticOnlyCombinedExtractor
        ),
        "ACTOR_CRITIC_IMAGE_SIZE": dsrl_train_module.ACTOR_CRITIC_IMAGE_SIZE,
        "OPENPI_IMAGE_KEY": dsrl_train_module.OPENPI_IMAGE_KEY,
        "OPENPI_WRIST_IMAGE_KEY": dsrl_train_module.OPENPI_WRIST_IMAGE_KEY,
        "OPENPI_OBSERVATION_KEYS": dsrl_train_module.OPENPI_OBSERVATION_KEYS,
    }

    saved_data, _, _ = load_from_zip_file(
        dsrl_checkpoint,
        device=dsrl_device,
        custom_objects=custom_objects,
    )
    if saved_data is None:
        raise ValueError(f"No SB3 metadata found in {dsrl_checkpoint}")

    dsrl_env = OfflineDictEnv(
        saved_data["observation_space"], saved_data["action_space"]
    )
    dsrl_model = DSRL.load(
        dsrl_checkpoint,
        env=dsrl_env,
        device=dsrl_device,
        custom_objects=custom_objects,
        diffusion_policy=diffusion_policy,
        diffusion_act_dim=(action_horizon, noise_dim),
    )
    dsrl_model.diffusion_act_chunk = action_horizon
    dsrl_model.diffusion_act_dim = noise_dim
    dsrl_model.policy.set_training_mode(False)

    actor_output_dim = int(np.prod(dsrl_model.noise_action_space.shape))
    if actor_output_dim not in (noise_dim, action_horizon * noise_dim):
        raise ValueError(
            "Loaded DSRL actor has incompatible action dimension "
            f"{actor_output_dim}. Expected {noise_dim} or "
            f"{action_horizon * noise_dim}."
        )
    if actor_output_dim == noise_dim:
        logging.info(
            "Loaded one-step DSRL actor; noise will be repeated across horizon=%s.",
            action_horizon,
        )
    else:
        logging.info(
            "Loaded full-horizon DSRL actor; noise will be reshaped to (%s, %s).",
            action_horizon,
            noise_dim,
        )
    return dsrl_model


def create_policy(args: Args) -> tuple[DSRLPolicy, dict[str, Any]]:
    if not args.checkpoint_dir:
        raise ValueError("--checkpoint-dir is required.")
    if not args.dsrl_checkpoint:
        raise ValueError("--dsrl-checkpoint is required.")

    train_config = _config.get_config(args.model_name)
    base_policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.openpi_pytorch_device,
    )

    action_horizon = train_config.model.action_horizon
    noise_dim = train_config.model.action_dim
    dsrl_model = _load_dsrl_model(
        dsrl_checkpoint=args.dsrl_checkpoint,
        dsrl_device=args.device,
        base_policy=base_policy,
        action_horizon=action_horizon,
        noise_dim=noise_dim,
    )

    metadata = dict(base_policy.metadata)
    metadata.update(
        {
            "dsrl": True,
            "dsrl_checkpoint": args.dsrl_checkpoint,
            "dsrl_deterministic": args.dsrl_deterministic,
            "action_space": "joint_position",
            "action_horizon": action_horizon,
            "noise_dim": noise_dim,
        }
    )
    policy = DSRLPolicy(
        base_policy=base_policy,
        dsrl_model=dsrl_model,
        action_horizon=action_horizon,
        noise_dim=noise_dim,
        deterministic=args.dsrl_deterministic,
        expose_noise=args.expose_noise,
    )
    return policy, metadata


def _make_warmup_observation(default_prompt: str | None = None) -> dict[str, Any]:
    obs = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros((7,), dtype=np.float32),
        "observation/gripper_position": np.zeros((1,), dtype=np.float32),
    }
    if default_prompt is not None:
        obs["prompt"] = default_prompt
    return obs


def main(args: Args) -> None:
    policy, policy_metadata = create_policy(args)

    if args.warmup:
        warmup_prompt = args.warmup_prompt or args.default_prompt
        if warmup_prompt is None:
            logging.warning(
                "Skipping warmup because no prompt is available. Pass "
                "--warmup-prompt or --default-prompt to warm the OpenPI prompt path."
            )
        else:
            logging.info("Running one synthetic DSRL/OpenPI warmup inference...")
            warmup_start = time.monotonic()
            policy.infer(_make_warmup_observation(warmup_prompt))
            logging.info(
                "Warmup finished in %.1f ms.",
                (time.monotonic() - warmup_start) * 1000,
            )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "dsrl_policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating DSRL server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
