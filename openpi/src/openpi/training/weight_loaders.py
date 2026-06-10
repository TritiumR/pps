import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing adaptation weights. LoRA-only multimodal configs add
        # small prefix encoders that are initialized from the current model.
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*(lora|pointcloud_prefix_encoder|sound_prefix_encoder).*",
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class VLMEncoderWeightLoader(WeightLoader):
    """Loads only the VLM encoder weights from a checkpoint.

    This loader is designed for the VLM Action Expert model, which uses:
    - VLM encoder (PaliGemma vision + language) from a pretrained pi0/pi05 checkpoint
    - Action expert initialized from scratch (or kept from reference params)

    Only weights matching the VLM encoder patterns are loaded. All other weights
    (action expert, projection layers, etc.) are kept from the reference params.
    """

    checkpoint_path: str

    # Patterns for VLM encoder weights to load
    vlm_patterns: tuple[str, ...] = (
        ".*paligemma\\.vision_tower.*",
        ".*paligemma\\.language_model.*",
        ".*paligemma\\.multi_modal_projector.*",
    )

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(self.checkpoint_path), restore_type=np.ndarray
        )
        # Only load VLM encoder weights, keep all other weights from reference params
        return _merge_params_selective(
            loaded_params, params, load_patterns=self.vlm_patterns
        )


def _merge_params_selective(
    loaded_params: at.Params, params: at.Params, *, load_patterns: tuple[str, ...]
) -> at.Params:
    """Merges parameters, only loading weights that match the given patterns.

    Args:
        loaded_params: The parameters to merge from.
        params: The reference parameters (model's initialized weights).
        load_patterns: Regex patterns for weights to load from loaded_params.
                      All other weights are kept from params.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # Compile patterns
    patterns = [re.compile(p) for p in load_patterns]

    result = {}
    loaded_count = 0
    kept_count = 0

    for k, v in flat_ref.items():
        # Check if this key matches any of the load patterns
        should_load = any(p.fullmatch(k) for p in patterns)

        if should_load and k in flat_loaded:
            # Load from checkpoint
            loaded_v = flat_loaded[k]
            if loaded_v.shape == v.shape:
                result[k] = loaded_v.astype(v.dtype) if loaded_v.dtype != v.dtype else loaded_v
                loaded_count += 1
            else:
                # Shape mismatch - keep reference params
                logger.warning(
                    f"Shape mismatch for {k}: checkpoint {loaded_v.shape} vs model {v.shape}. "
                    "Keeping randomly initialized weights."
                )
                result[k] = v
                kept_count += 1
        else:
            # Keep from reference params (randomly initialized)
            result[k] = v
            kept_count += 1

    logger.info(
        f"VLMEncoderWeightLoader: Loaded {loaded_count} weights from checkpoint, "
        f"kept {kept_count} weights from reference params."
    )

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
