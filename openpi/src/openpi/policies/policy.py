from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            logging.info("Moving PyTorch policy model to %s...", pytorch_device)
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            logging.info("PyTorch policy model ready on %s.", pytorch_device)
        else:
            # JAX model setup
            logging.info("Jitting JAX policy sample_actions...")
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            logging.info("JAX policy sample_actions ready.")

    def obs_to_input(self, obs: dict) -> _model.Observation:
        """
        Convert the observation to a input observation for the model.
        Only for PyTorch models.
        TODO: bypass the conversion to numpy and directly convert to torch tensor.
        """
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Convert inputs to PyTorch tensors and move to correct device
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            inputs,
        )

        observation = _model.Observation.from_dict(inputs)

        return observation, inputs

    def output_to_actions(self, inputs: dict, x_t: torch.Tensor):
        outputs = {
            "state": inputs["state"],
            "actions": x_t,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)

        outputs = self._output_transform(outputs)

        return outputs["actions"]

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, start_time: float | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[
                    None, ...
                ],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )

            if (
                noise.ndim == 2
            ):  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if start_time is not None:
            sample_kwargs["start_time"] = start_time

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            ),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
            )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_batch(
        self, obs_list: list[dict[str, Any]], *, noise: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """
        Batch inference that processes multiple observations at once.

        Args:
            obs_list: List of observation dictionaries. All observations should have the same keys.
                     For prompts, all observations should have the same prompt value.
            noise: Optional batch of noise tensors with shape (batch_size, action_horizon, action_dim).
                   If None, noise will be sampled by the model.

        Returns:
            Dictionary with "actions" key containing batched actions with shape
            (batch_size, action_horizon, action_dim). The actions will have the output transform
            applied (e.g., first 8 dimensions extracted).
        """
        batch_size = len(obs_list)
        if batch_size == 0:
            raise ValueError("obs_list cannot be empty")

        # Apply input transform to each observation individually
        # Input transforms are designed for single samples and may not handle batches correctly
        # (e.g., _parse_image checks image.shape[0] == 3 which fails for batched data)
        transformed_inputs = []
        for obs in obs_list:
            obs_copy = jax.tree.map(lambda x: x, obs)
            transformed = self._input_transform(obs_copy)
            transformed_inputs.append(transformed)

        # Stack transformed inputs into batches
        # Handle special case for prompts (they should all be the same after tokenization)
        batched_inputs = {}
        for key in transformed_inputs[0].keys():
            if key == "prompt":
                # For string prompts, use the first one (they should all be the same)
                batched_inputs[key] = transformed_inputs[0][key]
            elif key in (
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "token_ar_mask",
                "token_loss_mask",
            ):
                # For tokenized prompts and masks, expand to batch dimension
                # They should all be the same since prompts are the same
                value = transformed_inputs[0][key]
                if value is not None:
                    value = np.asarray(value)
                    # Expand along batch dimension
                    if value.ndim == 1:
                        # (seq_len,) -> (batch_size, seq_len)
                        batched_inputs[key] = np.tile(
                            value[np.newaxis, ...], (batch_size, 1)
                        )
                    else:
                        # Already has batch dimension or is 2D, just tile along first axis
                        batched_inputs[key] = np.tile(
                            value[np.newaxis, ...],
                            (batch_size,) + (1,) * (value.ndim - 1),
                        )
                else:
                    batched_inputs[key] = None
            elif isinstance(transformed_inputs[0][key], dict):
                # For nested dicts (like "image" and "image_mask"), stack each value
                batched_inputs[key] = {}
                for sub_key in transformed_inputs[0][key].keys():
                    batched_inputs[key][sub_key] = np.stack(
                        [inp[key][sub_key] for inp in transformed_inputs], axis=0
                    )
            else:
                # Stack arrays along batch dimension
                batched_inputs[key] = np.stack(
                    [inp[key] for inp in transformed_inputs], axis=0
                )

        inputs = batched_inputs

        # Convert to tensors (don't add extra batch dimension since we already have it)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device),
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare noise
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )
            # Noise should already have batch dimension: (batch_size, action_horizon, action_dim)
            if noise.ndim == 2:
                # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(
            sample_rng_or_pytorch_device, observation, **sample_kwargs
        )
        model_time = time.monotonic() - start_time

        # Convert back to numpy
        if self._is_pytorch_model:
            actions = np.asarray(actions.detach().cpu())
            state = np.asarray(inputs["state"].detach().cpu())
        else:
            actions = np.asarray(actions)
            state = np.asarray(inputs["state"])

        # Apply output transform to each sample individually
        # The output transform may need state information (e.g., for AbsoluteActions),
        # so we need to apply it per sample rather than trying to batch it
        transformed_actions = []
        for i in range(batch_size):
            sample_output = {
                "actions": actions[i],
                "state": state[i],
            }
            transformed_sample = self._output_transform(sample_output)
            transformed_actions.append(transformed_sample["actions"])

        # Stack the transformed actions back into a batch
        actions = np.stack(transformed_actions, axis=0)

        return {
            "actions": actions,
            "policy_timing": {
                "infer_ms": model_time * 1000,
            },
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
