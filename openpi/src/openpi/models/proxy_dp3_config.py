import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models_pytorch.proxy_dp3_pytorch import ProxyDP3Pytorch


@dataclasses.dataclass(frozen=True)
class ProxyDP3Config(_model.BaseModelConfig):
    dtype: str = "float32"
    freeze_point_encoder: bool = False
    compile_sample_actions: bool = False

    action_dim: int = 8
    action_horizon: int = 10
    max_token_len: int = None  # type: ignore

    num_points: int = 1024
    use_pc_color: bool = True
    pointcloud_position_noise_std: float = 0.0
    pointcloud_dropout_ratio: float = 0.0
    pointcloud_random_resample: bool = False
    pointnet_type: str = "pointnet"
    encoder_output_dim: int = 64
    state_mlp_size: tuple[int, ...] = (64, 64)

    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 128
    # Flow matching uses continuous time in [0, 1], while the original DP3 U-Net
    # timestep embedding was tuned for diffusion steps around [0, 100].
    time_embedding_scale: float = 1.0
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    condition_type: str = "film"
    use_down_condition: bool = True
    use_mid_condition: bool = True
    use_up_condition: bool = True

    pointcloud_encoder_cfg: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "out_channels": 64,
            "use_layernorm": True,
            "final_norm": "layernorm",
            "use_projection": True,
        }
    )

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 48)
        point_cfg = dict(self.pointcloud_encoder_cfg)
        point_cfg["out_channels"] = self.encoder_output_dim
        object.__setattr__(self, "pointcloud_encoder_cfg", point_cfg)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PROXY_DP3

    @override
    def create(self, rng: at.KeyArrayLike) -> "ProxyDP3Pytorch":
        del rng
        from openpi.models_pytorch.proxy_dp3_pytorch import ProxyDP3Pytorch

        return ProxyDP3Pytorch(config=self)

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        pointcloud_spec = jax.ShapeDtypeStruct(
            [batch_size, self.num_points, 6], jnp.float32
        )

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                pointcloud=pointcloud_spec,
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
