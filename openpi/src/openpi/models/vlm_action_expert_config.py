"""VLM Action Expert Model Configuration.

This config is for a model that combines:
- VLM encoder from pi0 (PaliGemma) - frozen during training
- Action expert (small Gemma) - finetuned during training
- Projection layer to adapt VLM features to action expert dimension
"""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models_pytorch.vlm_action_expert_pytorch import VLMActionExpertPytorch


@dataclasses.dataclass(frozen=True)
class VLMActionExpertConfig(_model.BaseModelConfig):
    """Configuration for VLM Action Expert model.
    
    This model uses:
    - paligemma_variant: VLM encoder variant (frozen during training by default)
    - action_expert_variant: Action expert variant (finetuned during training)
    
    A projection layer is automatically added if the VLM and action expert
    have different hidden dimensions.
    """
    
    dtype: str = "bfloat16"
    # VLM encoder variant (will be frozen during training by default)
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Action expert variant (will be finetuned during training)
    action_expert_variant: _gemma.Variant = "gemma_100m"
    # Whether to freeze the VLM encoder (default True)
    freeze_vlm_encoder: bool = True

    # Set the model specific defaults.
    action_dim: int = 8  # DROID action dimension
    action_horizon: int = 10
    max_token_len: int = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 48)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.VLM_ACTION_EXPERT

    @override
    def create(self, rng: at.KeyArrayLike) -> "VLMActionExpertPytorch":
        from openpi.models_pytorch.vlm_action_expert_pytorch import VLMActionExpertPytorch

        return VLMActionExpertPytorch(config=self)

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
