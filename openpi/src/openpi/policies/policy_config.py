import logging
import os
import pathlib
from typing import Any

from pathlib import Path
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms
from openpi.models.model import ModelType


def _output_norm_stats(
    norm_stats: dict[str, transforms.NormStats] | None,
) -> dict[str, transforms.NormStats] | None:
    if norm_stats is None:
        return None
    output_stats = {
        key: value
        for key, value in norm_stats.items()
        if key in ("state", "actions")
    }
    if not output_stats:
        return None
    return output_stats


def _load_checkpoint_norm_stats(
    checkpoint_dir: pathlib.Path,
    asset_id: str | None,
) -> dict[str, transforms.NormStats] | None:
    assets_dir = checkpoint_dir / "assets"
    if asset_id is not None:
        try:
            return _checkpoints.load_norm_stats(assets_dir, asset_id)
        except FileNotFoundError:
            logging.warning(
                "Norm stats were not found in checkpoint assets at %s; scanning checkpoint assets.",
                assets_dir / asset_id,
            )

    norm_stat_paths = list(assets_dir.glob("**/norm_stats.json"))
    if len(norm_stat_paths) == 1:
        norm_stats_dir = norm_stat_paths[0].parent
        logging.info("Loading norm stats from discovered checkpoint path %s", norm_stats_dir)
        return _checkpoints.load_norm_stats(norm_stats_dir=str(norm_stats_dir))
    if len(norm_stat_paths) > 1:
        logging.warning(
            "Found multiple norm_stats.json files under %s; falling back to config assets. Paths: %s",
            assets_dir,
            norm_stat_paths,
        )
    return None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    print(f"is_pytorch: {is_pytorch}")

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        if train_config.model.model_type in [
            ModelType.PI0,
            ModelType.PI05,
            ModelType.PI0_FAST,
        ]:
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(
            _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        norm_stats = _load_checkpoint_norm_stats(checkpoint_dir, data_config.asset_id)

        if norm_stats is None and train_config.data.norm_stats_dir is not None:
            project_root = Path(__file__).parent.parent.parent.parent
            norm_stats_dir = project_root / train_config.data.norm_stats_dir
            norm_stats = _checkpoints.load_norm_stats(
                norm_stats_dir=str(norm_stats_dir)
            )
        elif norm_stats is None:
            if data_config.asset_id is None:
                raise ValueError(
                    "Asset id or norm_stats_dir is required to load norm stats."
                )
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir / "assets", data_config.asset_id
            )

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    output_norm_stats = _output_norm_stats(norm_stats)

    logging.info("Creating policy wrapper...")
    policy = _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                output_norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
    policy._metadata = {
        **policy._metadata,
        "output_norm_stats": output_norm_stats,
        "use_quantile_norm": data_config.use_quantile_norm,
    }
    logging.info("Policy wrapper created.")
    return policy
