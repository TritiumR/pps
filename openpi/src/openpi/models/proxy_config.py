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
    from openpi.models_pytorch.proxy_pytorch import ProxyPytorch


DEFAULT_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
)

BIMANUAL_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

BIMANUAL_POINTCLOUD_KEYS = (
    "base_0_pointcloud",
    "left_wrist_0_pointcloud",
    "right_wrist_0_pointcloud",
)


@dataclasses.dataclass(frozen=True)
class ProxyConfig(_model.BaseModelConfig):
    dtype: str = "float32"
    action_expert_variant: _gemma.Variant = "gemma_100m"
    dino_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    freeze_dino_encoder: bool = False  # freeze DINO

    # Set the model specific defaults.
    action_dim: int = 8
    action_horizon: int = 10
    max_token_len: int = None  # type: ignore
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS
    use_dino_prefix: bool = True
    use_pointcloud_prefix: bool = False
    pointcloud_keys: tuple[str, ...] = ()
    pointcloud_num_points: int = 1024
    pointcloud_channels: int = 6
    pointcloud_prefix_tokens_per_camera: int = 128
    concerto_model_name: str = "concerto_small"
    concerto_repo_id: str = "Pointcept/Concerto"
    concerto_checkpoint_dir: str | None = None
    concerto_grid_size: float = 0.02
    concerto_enable_flash: bool = False
    freeze_concerto_encoder: bool = True
    attention_mode: str = "two_block_diffusion"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 48)
        if self.use_dino_prefix and not self.image_keys:
            raise ValueError("image_keys must contain at least one camera when use_dino_prefix=True.")
        if not self.use_dino_prefix and self.image_keys:
            raise ValueError("image_keys must be empty when use_dino_prefix=False.")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError(f"image_keys must be unique, got {self.image_keys}.")
        if self.use_pointcloud_prefix:
            if not self.pointcloud_keys:
                raise ValueError("pointcloud_keys must be set when use_pointcloud_prefix=True.")
            if len(set(self.pointcloud_keys)) != len(self.pointcloud_keys):
                raise ValueError(
                    f"pointcloud_keys must be unique, got {self.pointcloud_keys}."
                )
            if self.pointcloud_prefix_tokens_per_camera <= 0:
                raise ValueError("pointcloud_prefix_tokens_per_camera must be positive.")
        if self.attention_mode != "two_block_diffusion":
            raise ValueError(
                "Only attention_mode='two_block_diffusion' is supported; "
                f"got {self.attention_mode!r}."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PROXY

    @override
    def create(self, rng: at.KeyArrayLike) -> "ProxyPytorch":
        from openpi.models_pytorch.proxy_pytorch import ProxyPytorch

        return ProxyPytorch(config=self)

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
                images={key: image_spec for key in self.image_keys},
                image_masks={key: image_mask_spec for key in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                pointcloud=(
                    {
                        key: jax.ShapeDtypeStruct(
                            [batch_size, self.pointcloud_num_points, self.pointcloud_channels],
                            jnp.float32,
                        )
                        for key in self.pointcloud_keys
                    }
                    if self.use_pointcloud_prefix
                    else None
                ),
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

    # def get_freeze_filter(self) -> nnx.filterlib.Filter:
    #     """Returns the freeze filter based on the model config."""
    #     filters = []
    #     has_lora = False
    #     gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
    #     action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
    #     if "lora" in self.paligemma_variant:
    #         filters.append(
    #             gemma_params_filter,
    #         )
    #         if "lora" not in self.action_expert_variant:
    #             # If only freeze gemma params, exclude action expert params.
    #             filters.append(
    #                 nnx.Not(action_expert_params_filter),
    #             )
    #         has_lora = True
    #     elif "lora" in self.action_expert_variant:
    #         filters.append(
    #             action_expert_params_filter,
    #         )
    #         has_lora = True

    #     if has_lora:
    #         # If any lora is used, exclude all lora params.
    #         filters.append(
    #             nnx.Not(nnx_utils.PathRegex(".*lora.*")),
    #         )
    #     if not filters:
    #         return nnx.Nothing
    #     return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class BimanualProxyConfig(ProxyConfig):
    """Three RGB cameras plus Concerto features from their aligned point clouds."""

    image_keys: tuple[str, ...] = BIMANUAL_IMAGE_KEYS
    use_pointcloud_prefix: bool = True
    pointcloud_keys: tuple[str, ...] = BIMANUAL_POINTCLOUD_KEYS
