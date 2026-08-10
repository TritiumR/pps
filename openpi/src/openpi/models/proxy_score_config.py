import dataclasses
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models_pytorch.proxy_score_pytorch import ProxyScorePytorch


@dataclasses.dataclass(frozen=True)
class ProxyScoreConfig(_model.BaseModelConfig):
    dtype: str = "float32"
    action_expert_variant: _gemma.Variant = "gemma_100m"
    dino_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    freeze_dino_encoder: bool = False
    ddim_num_train_timesteps: int = 100
    # "regress": plain chunked regression (no noising, no score); training only -- the
    # score-space serve path (predict_score_from_prefix) has no meaning for it.
    prediction_type: Literal["score", "epsilon", "x0", "regress"] = "score"
    compile_sample_actions: bool = False

    action_dim: int = 8
    action_horizon: int = 10
    # Goal conditioning. 0 (default) leaves the architecture byte-identical: no goal
    # projection is built and no goal token enters the suffix. >0 adds ONE context token,
    # embedded linearly from a goal vector of this width, placed in the state block -- it is
    # attended to by the action tokens, is never denoised, and carries no loss.
    goal_dim: int = 0
    # Joint action/goal denoising: p(a, g_hat | o). False (default) leaves the architecture
    # byte-identical. True adds ONE extra DENOISED row carrying the goal, with its own
    # encoder/decoder, so the goal is PREDICTED rather than given -- the opposite direction of
    # goal_dim, and the two are mutually exclusive (a model cannot be handed the goal it is
    # meant to infer).
    #
    # The row is dedicated and 3-D end to end: it does NOT ride in the [H, action_dim] chunk, so
    # there are no unused action dims on it to keep inert. It sits in the action tokens'
    # attention block, which is what lets the two influence each other -- and that requires
    # MG_PROXY_BIDIR_SUFFIX=1, since under causal attention the action rows could never see a
    # row placed after them.
    goal_row: bool = False
    goal_row_dim: int = 3
    # Scalar gain applied ONLY inside the goal row's diffusion space. The goal interface stays
    # exactly the shared workspace-box normalisation (no per-axis std division -- the z axis is
    # constant, so dividing by its std would divide by zero); this single constant puts the
    # row's target on the same numeric scale as the normalised action rows, so both denoise on
    # the same signal-to-noise schedule. Measured on the tray split the normalised goal has
    # per-axis std 0.128 / 0.155 / 0.0, so a gain of ~7 brings it to ~unit variance.
    goal_row_gain: float = 7.0
    max_token_len: int = None  # type: ignore

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 48)
        if self.ddim_num_train_timesteps <= 1:
            raise ValueError("ddim_num_train_timesteps must be greater than 1.")
        if self.prediction_type not in ("score", "epsilon", "x0", "regress"):
            raise ValueError(
                "prediction_type must be 'score', 'epsilon', 'x0' or 'regress'."
            )
        if self.goal_row and self.goal_dim:
            raise ValueError(
                "goal_row and goal_dim are mutually exclusive: goal_dim HANDS the model the "
                "goal as context, goal_row asks it to INFER the goal. Enabling both would let "
                "the goal row copy its own conditioning."
            )
        if self.goal_row and self.prediction_type != "x0":
            raise ValueError(
                "goal_row is only defined for prediction_type 'x0'; the goal row's target is "
                "the clean goal."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PROXY_SCORE

    @override
    def create(self, rng: at.KeyArrayLike) -> "ProxyScorePytorch":
        del rng
        from openpi.models_pytorch.proxy_score_pytorch import ProxyScorePytorch

        return ProxyScorePytorch(config=self)

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
