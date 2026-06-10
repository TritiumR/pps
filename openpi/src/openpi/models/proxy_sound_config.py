import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models_pytorch.proxy_sound_pytorch import ProxySoundPytorch


@dataclasses.dataclass(frozen=True)
class ProxySoundConfig(_model.BaseModelConfig):
    dtype: str = "float32"
    action_expert_variant: _gemma.Variant = "gemma_100m"
    dino_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    freeze_dino_encoder: bool = False

    sound_channels: int = 2
    sound_mel_bins: int = 80
    sound_time_bins: int = 198
    sound_vmin: float = -18.420680743952367
    sound_vmax: float = 8.0349

    action_dim: int = 8
    action_horizon: int = 10
    max_token_len: int = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 48)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PROXY_SOUND

    @override
    def create(self, rng: at.KeyArrayLike) -> "ProxySoundPytorch":
        from openpi.models_pytorch.proxy_sound_pytorch import ProxySoundPytorch

        return ProxySoundPytorch(config=self)

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        sound_spec = jax.ShapeDtypeStruct(
            [batch_size, self.sound_channels, self.sound_mel_bins, self.sound_time_bins],
            jnp.float32,
        )

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
                sound=sound_spec,
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
