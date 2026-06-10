import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models_pytorch.residual_pytorch import ResidualPytorch


@dataclasses.dataclass(frozen=True)
class ResidualConfig(_model.BaseModelConfig):
    """Configuration for Residual Policy model.

    The Residual Policy learns to predict corrections to VLA model predictions.
    It takes VLA action as conditioning input and predicts the residual between
    VLA action and ground truth action.
    """

    dtype: str = "float32"
    action_expert_variant: _gemma.Variant = "gemma_100m"
    dino_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    freeze_dino_encoder: bool = False  # freeze DINO

    # Set the model specific defaults.
    action_dim: int = 8
    action_horizon: int = 10
    max_token_len: int = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 48)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.RESIDUAL

    @override
    def create(self, rng: at.KeyArrayLike) -> "ResidualPytorch":
        from openpi.models_pytorch.residual_pytorch import ResidualPytorch

        return ResidualPytorch(config=self)

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.max_token_len], bool
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )

        return observation_spec, action_spec

    def load_pytorch(self, train_config, weight_path: str) -> "ResidualPytorch":
        """Load a ResidualPytorch model with weights from a checkpoint.

        Args:
            train_config: The training configuration
            weight_path: Path to the model weights (safetensors file)

        Returns:
            ResidualPytorch model with loaded weights
        """
        import safetensors.torch
        from openpi.models_pytorch.residual_pytorch import ResidualPytorch

        model = ResidualPytorch(config=self)
        safetensors.torch.load_model(model, weight_path)
        return model
