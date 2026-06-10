import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.expert_pytorch import DINOExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
)


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


# def make_att_2d_masks(pad_masks, att_masks):
#     """Copied from big_vision.

#     Tokens can attend to valid inputs tokens which have a cumulative mask_ar
#     smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
#     setup several types of attention, for example:

#       [[1 1 1 1 1 1]]: pure causal attention.

#       [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
#           themselves and the last 3 tokens have a causal attention. The first
#           entry could also be a 1 without changing behaviour.

#       [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
#           block can attend all previous blocks and all tokens on the same block.

#     Args:
#       input_mask: bool[B, N] true if its part of the input, false if padding.
#       mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
#         it and 0 where it shares the same attention mask as the previous token.
#     """
#     if att_masks.ndim != 2:
#         raise ValueError(att_masks.ndim)
#     if pad_masks.ndim != 2:
#         raise ValueError(pad_masks.ndim)

#     cumsum = torch.cumsum(att_masks, dim=1)
#     att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
#     pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
#     return att_2d_masks & pad_2d_masks


class ResidualPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.pi05 = config.pi05

        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.expert_model = DINOExpertModel(
            # dino_model_name="facebook/dinov3-vits16plus-pretrain-lvd1689m",
            dino_model_name=config.dino_model_name,
            action_expert_config=action_expert_config,
            # use_adarms=[False, True] if self.pi05 else [False, False],
            use_adarms=[False, False],
            precision=config.dtype,
            freeze_dino_encoder=getattr(config, "freeze_dino_encoder", False),
        )

        ACTION_DIM = 8  # for droid setup

        self.action_in_proj = nn.Linear(ACTION_DIM, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, ACTION_DIM)

        # VLA action projection (for residual policy - VLA action is part of prefix)
        self.vla_action_proj = nn.Linear(ACTION_DIM, action_expert_config.width)

        self.state_proj = nn.Linear(ACTION_DIM, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.width, action_expert_config.width
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        torch.set_float32_matmul_precision("high")
        if os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() not in ("1", "true", "yes"):
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    # def _prepare_attention_masks_4d(self, att_2d_masks):
    #     """Helper method to prepare 4D attention masks for transformer."""
    #     att_2d_masks_4d = att_2d_masks[:, None, :, :]
    #     return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, image_keys=IMAGE_KEYS, train=train
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with DINO."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            img_emb = self.expert_model.embed_image(img)

            # print("img_emb.shape", img_emb.shape)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_vla_action(self, vla_action):
        """Embed VLA action as part of prefix (fixed during denoising).

        Args:
            vla_action: VLA predicted action, shape (batch, action_horizon, action_dim)

        Returns:
            embs: VLA action embeddings, shape (batch, action_horizon, width)
            pad_masks: Padding masks, shape (batch, action_horizon)
            att_masks: Attention masks (0 = other tokens can attend)
        """
        if self.vla_action_proj.weight.dtype == torch.float32:
            vla_action = vla_action.to(torch.float32)

        vla_emb = self.vla_action_proj(vla_action)  # (batch, action_horizon, width)
        bsize, action_horizon = vla_emb.shape[:2]
        device = vla_emb.device

        pad_mask = torch.ones(bsize, action_horizon, dtype=torch.bool, device=device)
        # Attention mask: 0 means other tokens can attend to these (prefix-style)
        att_masks = torch.zeros(bsize, action_horizon, dtype=torch.bool, device=device)

        return vla_emb, pad_mask, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # if not self.pi05:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        # Embed state
        state_emb = self.state_proj(state)

        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # # Fuse timestep + action information using an MLP
        # def action_proj_func(noisy_actions):
        #     return self.action_in_proj(noisy_actions)

        if self.action_in_proj.weight.dtype == torch.float32:
            noisy_actions = noisy_actions.to(torch.float32)
        action_emb = self.action_in_proj(noisy_actions)

        # if not self.pi05:
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        # Apply MLP layers
        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)  # swish == silu
            return self.action_time_mlp_out(x)

        action_time_emb = mlp_func(action_time_emb)
        adarms_cond = None
        # else:
        #     # time MLP (for adaRMS)
        #     def time_mlp_func(time_emb):
        #         x = self.time_mlp_in(time_emb)
        #         x = F.silu(x)  # swish == silu
        #         x = self.time_mlp_out(x)
        #         return F.silu(x)

        #     time_emb = time_mlp_func(time_emb)
        #     action_time_emb = action_emb
        #     adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self, observation, actions, vla_action, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors).

        Args:
            observation: Observation from the environment
            actions: Ground truth actions (target for residual)
            vla_action: VLA predicted action (conditioning signal)
            noise: Optional noise tensor
            time: Optional time tensor

        Returns:
            Loss tensor with shape (batch_size, action_horizon, action_dim)
        """
        images, img_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Prefix: image features from DINO + VLA action (both fixed during denoising)
        img_embs, img_pad_masks, _ = self.embed_prefix(images, img_masks)
        vla_embs, vla_pad_masks, _ = self.embed_vla_action(vla_action)
        prefix_embs = torch.cat([img_embs, vla_embs], dim=1)
        prefix_pad_masks = torch.cat([img_pad_masks, vla_pad_masks], dim=1)

        # Suffix: state + (noisy_action, time) tokens
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        # Concatenate prefix and suffix sequences
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)

        # Use pad mask as attention mask (True == valid)
        attention_mask = pad_masks
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        position_ids = position_ids.to(dtype=torch.long)

        hidden_states, _ = self.expert_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        # Take only the last action_horizon tokens (suffix part)
        suffix_out = hidden_states[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self, device, observation, vla_action, noise=None, num_steps=10, start_time=1.0
    ) -> Tensor:
        """Do a full inference forward and compute the residual action.

        Args:
            device: Device to run on
            observation: Observation from the environment
            vla_action: VLA predicted action (conditioning signal)
                Shape: (batch_size, action_horizon, action_dim)
            noise: Optional noise tensor
            num_steps: Number of denoising steps
            start_time: Starting time for flow matching

        Returns:
            Predicted residual action with shape (batch_size, action_horizon, action_dim)
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, state = self._preprocess_observation(
            observation, train=False
        )

        # Prefix embeddings are static across denoising steps (images + VLA action)
        img_embs, img_pad_masks, _ = self.embed_prefix(images, img_masks)
        vla_embs, vla_pad_masks, _ = self.embed_vla_action(vla_action)
        prefix_embs = torch.cat([img_embs, vla_embs], dim=1)
        prefix_pad_masks = torch.cat([img_pad_masks, vla_pad_masks], dim=1)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(start_time, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Recompute suffix embeddings at current x_t and time
            suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
                state, x_t, expanded_time
            )

            embs = torch.cat([prefix_embs, suffix_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)

            attention_mask = pad_masks
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            position_ids = position_ids.to(dtype=torch.long)

            hidden_states, _ = self.expert_model.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=embs,
                use_cache=False,
                adarms_cond=adarms_cond,
            )

            suffix_out = hidden_states[:, -self.config.action_horizon :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        return x_t
