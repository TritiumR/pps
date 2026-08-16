#!/usr/bin/env python3
"""Collect exact live Base/MPC epsilon labels during ``eval_steering`` rollouts."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import shutil
from typing import Any

import numpy as np
import torch

from sim_free_mpc.ddim import ddim_iteration_alphas


CACHE_FORMAT_VERSION = 5
CACHE_LABEL_TYPE = "mpc_epsilon_action_prox_reverse_trajectory"
CACHE_STATE_SOURCE = "action_prox_reverse_trajectory_from_gaussian"
ACTION_PROX_NOISE_SCHEDULE = "mpc_noise_times_sqrt_one_minus_alpha_bar"
OBSERVATION_CACHE_FORMAT_VERSION = 2
OBSERVATION_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")


def norm_stats_fingerprint(norm_stats: dict[str, Any] | None) -> str | None:
    if norm_stats is None:
        return None
    digest = hashlib.sha256()
    for key in sorted(norm_stats):
        digest.update(key.encode("utf-8"))
        stat = norm_stats[key]
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stat, field, None)
            if value is None:
                digest.update(f"{field}:None".encode("utf-8"))
                continue
            array = np.asarray(value)
            digest.update(field.encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def file_sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_from_iteration(iteration: int, num_iterations: int, train_steps: int) -> float:
    # Validate against the same scheduler used by the teacher before exposing the time condition.
    ddim_iteration_alphas(
        iteration=iteration,
        num_iterations=num_iterations,
        num_train_timesteps=train_steps,
    )
    step_ratio = train_steps // num_iterations
    timestep = (num_iterations - 1 - iteration) * step_ratio
    return timestep / max(float(train_steps - 1), 1.0)


def _unbatch(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim and value.shape[0] == 1:
        value = value[0]
    return value


def _model_image_to_uint8(value: Any) -> np.ndarray:
    image = _unbatch(value)
    if image.ndim != 3:
        raise ValueError(f"Expected one model image, got {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (224, 224, 3):
        raise ValueError(f"Expected transformed image [224,224,3], got {image.shape}.")
    if image.dtype == np.uint8:
        return image.copy()
    image = np.asarray(image, dtype=np.float32)
    # Observation.from_dict maps uint8 to [-1, 1]. Be tolerant of a [0, 1] input too.
    if float(np.nanmin(image)) < -0.01:
        image = (image + 1.0) * 127.5
    else:
        image = image * 255.0
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


class RuntimeRefCacheCollector:
    """In-memory shard collector; each process writes its own NPZ and observation sidecar."""

    def __init__(
        self,
        *,
        output_path: str | pathlib.Path,
        trajectories_per_observation: int,
        max_observations: int,
        num_steps: int,
        ddim_num_train_timesteps: int,
        stored_action_dim: int,
        label_seed: int,
        metadata: dict[str, Any],
    ) -> None:
        if trajectories_per_observation <= 0 or max_observations <= 0:
            raise ValueError("K and max_observations must be positive.")
        self.output_path = pathlib.Path(output_path)
        self.observation_cache_path = pathlib.Path(f"{self.output_path}.observations")
        self.k = int(trajectories_per_observation)
        self.max_observations = int(max_observations)
        self.num_steps = int(num_steps)
        self.num_iterations = self.num_steps + 1
        self.train_steps = int(ddim_num_train_timesteps)
        self.stored_action_dim = int(stored_action_dim)
        self.label_seed = int(label_seed)
        self.base_metadata = dict(metadata)

        self._demo_names: list[str] = []
        self._step_indices: list[int] = []
        self._trajectory_ids: list[int] = []
        self._iterations: list[int] = []
        self._times: list[float] = []
        self._x_t: list[np.ndarray] = []
        self._epsilon: list[np.ndarray] = []
        self._cost_min: list[float] = []
        self._score_norm: list[float] = []
        self._images: list[np.ndarray] = []
        self._image_masks: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._tokenized_prompt: np.ndarray | None = None
        self._tokenized_prompt_mask: np.ndarray | None = None

    @property
    def num_observations(self) -> int:
        return len(self._images)

    @property
    def full(self) -> bool:
        return self.num_observations >= self.max_observations

    def _capture_observation(self, model_inputs: dict[str, Any]) -> None:
        images = np.stack(
            [_model_image_to_uint8(model_inputs["image"][key]) for key in OBSERVATION_IMAGE_KEYS]
        )
        masks = np.asarray(
            [bool(_unbatch(model_inputs["image_mask"][key])) for key in OBSERVATION_IMAGE_KEYS],
            dtype=np.bool_,
        )
        state = np.asarray(_unbatch(model_inputs["state"]), dtype=np.float32)
        if state.shape[-1] < self.stored_action_dim:
            raise ValueError(
                f"Observation state has {state.shape[-1]} dims; expected {self.stored_action_dim}."
            )
        state = state[: self.stored_action_dim].copy()
        prompt = np.asarray(_unbatch(model_inputs["tokenized_prompt"]), dtype=np.int64)
        prompt_mask = np.asarray(
            _unbatch(model_inputs["tokenized_prompt_mask"]), dtype=np.bool_
        )
        if self._tokenized_prompt is None:
            self._tokenized_prompt = prompt.copy()
            self._tokenized_prompt_mask = prompt_mask.copy()
        elif not (
            np.array_equal(prompt, self._tokenized_prompt)
            and np.array_equal(prompt_mask, self._tokenized_prompt_mask)
        ):
            raise ValueError("Runtime ref cache currently requires one shared prompt.")
        self._images.append(images)
        self._image_masks.append(masks)
        self._states.append(state)

    @torch.no_grad()
    def collect(
        self,
        *,
        planner: Any,
        base_model: Any,
        model_inputs: dict[str, Any],
        observation_inputs: dict[str, Any],
        context: dict[str, Any],
        seed: int,
        env_step: int,
    ) -> bool:
        """Collect K exact teacher chains for one live observation; return False when full."""
        if self.full:
            return False
        observation_idx = self.num_observations
        self._capture_observation(observation_inputs)
        model_config = base_model.config
        device = model_inputs["state"].device
        full_action_dim = int(model_config.action_dim)
        action_horizon = int(model_config.action_horizon)
        if full_action_dim < self.stored_action_dim:
            raise ValueError("Base model action dim is smaller than stored ref action dim.")
        cuda_devices = [device] if device.type == "cuda" else []

        # Extra teacher queries must not advance the RNG used by the executed Base rollout.
        with torch.random.fork_rng(devices=cuda_devices):
            # Hash the full observation identity instead of adding fields with fixed
            # multipliers. Separate rollout workers may offset label_seed by the same
            # multiplier used for environment seeds, so an additive formula can collide.
            seed_payload = (
                f"{self.label_seed}:{int(seed)}:{int(env_step)}:{observation_idx}"
            ).encode("utf-8")
            observation_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:8], "little"
            ) % (2**63 - 1)
            torch.manual_seed(observation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(observation_seed)
            for local_trajectory in range(self.k):
                trajectory_id = observation_idx * self.k + local_trajectory
                x_t = torch.randn(
                    (1, action_horizon, full_action_dim),
                    device=device,
                    dtype=torch.float32,
                )
                planner.begin_inference()
                for iteration in range(self.num_iterations):
                    score, diagnostics = planner.estimate_mbd_score_action_prox(
                        x_t,
                        model_inputs,
                        context,
                        iteration=iteration,
                        num_iterations=self.num_iterations,
                    )
                    alpha_bar, _ = ddim_iteration_alphas(
                        iteration=iteration,
                        num_iterations=self.num_iterations,
                        num_train_timesteps=self.train_steps,
                    )
                    epsilon = -math.sqrt(max(1.0 - float(alpha_bar), 1e-6)) * score
                    self._demo_names.append(f"seed_{int(seed):06d}")
                    self._step_indices.append(int(env_step))
                    self._trajectory_ids.append(trajectory_id)
                    self._iterations.append(iteration)
                    self._times.append(
                        _time_from_iteration(iteration, self.num_iterations, self.train_steps)
                    )
                    self._x_t.append(
                        x_t[0, :, : self.stored_action_dim]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    self._epsilon.append(
                        epsilon[0, :, : self.stored_action_dim]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    self._cost_min.append(float(diagnostics.get("cost_min", np.nan)))
                    self._score_norm.append(float(diagnostics.get("score_norm", np.nan)))
                    if iteration < self.num_steps:
                        x_t = planner.step_from_score(
                            x_t,
                            score,
                            iteration=iteration,
                            num_iterations=self.num_iterations,
                            update_mode="mbd_score",
                            active_dims=self.stored_action_dim,
                        )
        return True

    def save(self) -> None:
        if not self.num_observations:
            return
        labels = self.num_observations * self.k * self.num_iterations
        if len(self._x_t) != labels:
            raise RuntimeError(f"Incomplete runtime cache: got {len(self._x_t)}, expected {labels}.")
        metadata = {
            **self.base_metadata,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "label_type": CACHE_LABEL_TYPE,
            "state_source": CACHE_STATE_SOURCE,
            "initial_state_distribution": "standard_gaussian",
            "trajectory_update": "mbd_score",
            "num_observations": self.num_observations,
            "trajectories_per_observation": self.k,
            "num_trajectories": self.num_observations * self.k,
            "labels_per_trajectory": self.num_iterations,
            "num_steps": self.num_steps,
            "num_iterations": self.num_iterations,
            "ddim_num_train_timesteps": self.train_steps,
            "stored_action_dim": self.stored_action_dim,
            "label_seed": self.label_seed,
            "observation_seed_derivation": "sha256(label_seed,seed,env_step,observation_idx)",
            "observation_source": "eval_steering_live_model_inputs",
            "observation_cache_format_version": OBSERVATION_CACHE_FORMAT_VERSION,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(f"{self.output_path.name}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                demo_name=np.asarray(self._demo_names),
                step_index=np.asarray(self._step_indices, dtype=np.int64),
                trajectory_id=np.asarray(self._trajectory_ids, dtype=np.int64),
                iteration=np.asarray(self._iterations, dtype=np.int64),
                time=np.asarray(self._times, dtype=np.float32),
                x_t=np.asarray(self._x_t, dtype=np.float32),
                epsilon=np.asarray(self._epsilon, dtype=np.float32),
                cost_min=np.asarray(self._cost_min, dtype=np.float32),
                score_norm=np.asarray(self._score_norm, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        temporary.replace(self.output_path)
        self._save_observations(metadata)

    def _save_observations(self, cache_metadata: dict[str, Any]) -> None:
        target = self.observation_cache_path
        temporary = target.with_name(f"{target.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        np.save(temporary / "images.npy", np.asarray(self._images, dtype=np.uint8))
        np.save(temporary / "image_masks.npy", np.asarray(self._image_masks, dtype=np.bool_))
        np.save(temporary / "states.npy", np.asarray(self._states, dtype=np.float32))
        np.save(temporary / "tokenized_prompt.npy", self._tokenized_prompt)
        np.save(temporary / "tokenized_prompt_mask.npy", self._tokenized_prompt_mask)
        sidecar_metadata = {
            "format_version": OBSERVATION_CACHE_FORMAT_VERSION,
            "source": "eval_steering_live_model_inputs",
            "num_observations": self.num_observations,
            "prompt": cache_metadata["prompt"],
            "norm_stats_fingerprint": cache_metadata["norm_stats_fingerprint"],
            "use_quantile_norm": False,
            "image_keys": list(OBSERVATION_IMAGE_KEYS),
            "image_shape": [224, 224, 3],
        }
        (temporary / "metadata.json").write_text(
            json.dumps(sidecar_metadata, indent=2, sort_keys=True)
        )
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
