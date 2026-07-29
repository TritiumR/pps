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


def sample_bin(start, end, bsize, num_samples, device):
    bin_width = (end - start) / num_samples
    offsets = (
        torch.arange(num_samples, dtype=torch.float32, device=device) * bin_width
        + start
    )
    offsets = offsets.unsqueeze(0)
    noise = (
        torch.rand(bsize, num_samples, dtype=torch.float32, device=device) * bin_width
    )
    return offsets + noise


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_masks = pad_masks.to(torch.bool)
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def build_proxy_expert_masks(
    model,
    prefix_pad_masks,
    prefix_att_masks,
    suffix_pad_masks,
    suffix_att_masks,
    *,
    query_offset=0,
):
    """Build `(attention_mask, pad_masks)` for any proxy-style action expert.

    Used by the callers that inline the proxy forward pass (eval_steering.py,
    the steering policy servers). Models that do not implement
    `build_expert_masks` (e.g. the pointcloud/sound proxies) keep the legacy
    behavior of passing the 2D pad mask straight through.
    """
    build = getattr(model, "build_expert_masks", None)
    if build is not None and prefix_att_masks is not None and suffix_att_masks is not None:
        return build(
            prefix_pad_masks,
            prefix_att_masks,
            suffix_pad_masks,
            suffix_att_masks,
            query_offset=query_offset,
        )
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    return pad_masks, pad_masks


def build_proxy_prefix_mask(model, prefix_pad_masks, prefix_att_masks):
    """Mask for a prefix-only forward pass (KV cache construction)."""
    build = getattr(model, "build_attention_mask", None)
    if build is None or prefix_att_masks is None:
        return prefix_pad_masks
    return build(prefix_pad_masks, prefix_att_masks)


class ProxyPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.pi05 = config.pi05

        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.expert_model = DINOExpertModel(
            dino_model_name=config.dino_model_name,
            action_expert_config=action_expert_config,
            use_adarms=[False, False],
            precision=config.dtype,
            freeze_dino_encoder=getattr(config, "freeze_dino_encoder", False),
        )

        # pi0-style block attention over [images | state | actions] instead of the
        # token-level causal mask HF applies when only a 2D pad mask is passed.
        self.bidirectional_attention = getattr(config, "bidirectional_attention", False)

        ACTION_DIM = 8  # for droid setup

        self.action_in_proj = nn.Linear(ACTION_DIM, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, ACTION_DIM)

        # if self.pi05:
        #     self.time_mlp_in = nn.Linear(
        #         action_expert_config.width, action_expert_config.width
        #     )
        #     self.time_mlp_out = nn.Linear(
        #         action_expert_config.width, action_expert_config.width
        #     )
        # else:
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

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def build_attention_mask(self, pad_masks, att_masks, *, query_offset=0):
        """Build the mask handed to the Gemma expert.

        With `bidirectional_attention` disabled we pass the 2D pad mask, and
        HuggingFace builds a token-level causal mask from it (legacy behavior).
        With it enabled we build the pi0-style block mask ourselves and pass it
        as a 4D additive mask, which HF forwards as-is.

        `pad_masks` / `att_masks` always cover the *full* sequence (prefix +
        suffix). When only the tail of the sequence is being queried (KV-cached
        decoding, where the prefix is already in `past_key_values`), pass
        `query_offset` = number of cached tokens so the returned mask has the
        matching [B, 1, q_len, kv_len] shape.
        """
        if not self.bidirectional_attention:
            # HF wants the pad mask over the full kv length, which is what we
            # already have, so `query_offset` needs no handling here.
            return pad_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        if query_offset:
            att_2d_masks = att_2d_masks[:, query_offset:, :]
        return self._prepare_attention_masks_4d(att_2d_masks)

    def build_expert_masks(
        self,
        prefix_pad_masks,
        prefix_att_masks,
        suffix_pad_masks,
        suffix_att_masks,
        *,
        query_offset=0,
    ):
        """Concatenate prefix/suffix masks and build the expert attention mask.

        Returns `(attention_mask, pad_masks)`; `pad_masks` is the full-sequence
        2D pad mask, which callers also need for `position_ids`.
        """
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat(
            [prefix_att_masks.to(torch.int32), suffix_att_masks.to(torch.int32)], dim=1
        )
        attention_mask = self.build_attention_mask(
            pad_masks, att_masks, query_offset=query_offset
        )
        return attention_mask, pad_masks

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

    def sample_bin_times(self, bsize, num_steps, device):
        time = sample_bin(0.0, 1.0, bsize, num_steps, device)
        time = time.flip(dims=[1])
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
        self,
        observation,
        actions=None,
        noise=None,
        time=None,
        *,
        mode="train",
        noises=None,
        times=None,
        gradients=None,
        use_noise=True,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)."""
        if mode == "distill":
            return self.forward_distill(
                observation, noises, times, gradients, actions, use_noise=use_noise
            )
        if mode != "train":
            raise ValueError(f"Unsupported forward mode: {mode}")
        if actions is None:
            raise ValueError("actions must be provided for training mode.")

        images, img_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Prefix: image features from DINO
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )

        # print(prefix_embs.shape)

        # Suffix: state + (action, time) tokens
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )

        # print(suffix_embs.shape)

        # Concatenate prefix and suffix sequences
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        attention_mask, pad_masks = self.build_expert_masks(
            prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
        )
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

    def forward_distill(
        self,
        observation,
        noises,
        times,
        gradients,
        actions,
        use_noise=True,
    ) -> Tensor:
        """Train student model to mimic teacher model by supervising with teacher's gradients.

        Args:
            observation: Observation from the environment
            noises: Input noise for each flow-matching step of teacher model
                Shape: (batch_size, num_steps, action_horizon, action_dim)
            times: Time for each flow-matching step of teacher model
                Shape: (batch_size, num_steps)
            gradients: Gradients predicted by teacher model for each step
                Shape: (batch_size, num_steps, action_horizon, action_dim)
            actions: Actions predicted by teacher model
                Shape: (batch_size, action_horizon, action_dim)

        Returns:
            Loss tensor with shape (batch_size, num_steps, action_horizon, action_dim)
        """
        images, img_masks, state = self._preprocess_observation(observation, train=True)

        # Prefix: image features from DINO (static across all steps)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )

        initial_noise = noises[:, 0, :, : self.config.action_dim]

        # remove the extra action dimensions and remove first denoising step
        noises = noises[:, 1:, :, : self.config.action_dim]
        times = times[:, 1:]
        gradients = gradients[:, 1:, :, : self.config.action_dim]
        actions = actions[:, :, : self.config.action_dim]

        # print(f"initial_noise shape: {initial_noise.shape}")
        # print(f"noises shape: {noises.shape}")
        # print(f"times shape: {times.shape}")
        # print(f"gradients shape: {gradients.shape}")
        # print(f"actions shape: {actions.shape}")

        batch_size, num_steps = times.shape[:2]
        flat_times = times.reshape(batch_size * num_steps)
        flat_gradients = gradients.reshape(
            batch_size * num_steps, self.config.action_horizon, self.config.action_dim
        )

        if use_noise:
            flat_x_t = noises.reshape(
                batch_size * num_steps,
                self.config.action_horizon,
                self.config.action_dim,
            )
        else:
            time_expanded = times[:, :, None, None]
            x_t = time_expanded * initial_noise[:, None, :, :] + (
                1 - time_expanded
            ) * actions[:, None, :, :]
            flat_x_t = x_t.reshape(
                batch_size * num_steps,
                self.config.action_horizon,
                self.config.action_dim,
            )

        flat_state = state[:, None, :].expand(batch_size, num_steps, state.shape[-1])
        flat_state = flat_state.reshape(batch_size * num_steps, state.shape[-1])

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(flat_state, flat_x_t, flat_times)
        )

        prefix_embs = prefix_embs[:, None, :, :].expand(
            batch_size, num_steps, prefix_embs.shape[1], prefix_embs.shape[2]
        )
        prefix_embs = prefix_embs.reshape(
            batch_size * num_steps, prefix_embs.shape[2], prefix_embs.shape[3]
        )
        prefix_pad_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, num_steps, prefix_pad_masks.shape[1]
        )
        prefix_pad_masks = prefix_pad_masks.reshape(
            batch_size * num_steps, prefix_pad_masks.shape[2]
        )
        prefix_att_masks = prefix_att_masks[:, None, :].expand(
            batch_size, num_steps, prefix_att_masks.shape[1]
        )
        prefix_att_masks = prefix_att_masks.reshape(
            batch_size * num_steps, prefix_att_masks.shape[2]
        )

        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        attention_mask, pad_masks = self.build_expert_masks(
            prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
        )
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

        loss = F.mse_loss(v_t, flat_gradients, reduction="none")
        return loss.reshape(
            batch_size, num_steps, self.config.action_horizon, self.config.action_dim
        )

    @torch.no_grad()
    def forward_for_distill(
        self,
        observation,
        num_steps,
        teacher_flow_path_noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample proxy-teacher flow vectors at random denoising times for distillation.

        Args:
            observation: Observation batch.
            num_steps: Number of sampled flow-matching steps.
            teacher_flow_path_noise_std: Stddev of Gaussian noise injected into the
                teacher rollout state before each step. Set to 0 to disable.

        Returns:
            Tuple of (noises, times, gradients, actions) where:
            - noises: Input noise/state for each denoising step
              Shape: (batch_size, num_steps, action_horizon, action_dim)
            - times: Randomly sampled times for each step
              Shape: (batch_size, num_steps)
            - gradients: Proxy teacher flow predictions for each step
              Shape: (batch_size, num_steps, action_horizon, action_dim)
            - actions: Final denoised actions after all updates
              Shape: (batch_size, action_horizon, action_dim)
        """
        images, img_masks, state = self._preprocess_observation(observation, train=False)
        bsize = state.shape[0]
        device = state.device

        if num_steps <= 0:
            raise ValueError(f"num_steps must be greater than 0, got {num_steps}")
        if teacher_flow_path_noise_std < 0:
            raise ValueError(
                "teacher_flow_path_noise_std must be non-negative, got "
                f"{teacher_flow_path_noise_std}"
            )

        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        initial_noise = self.sample_noise(actions_shape, device)
        time_schedule = self.sample_bin_times(bsize, num_steps, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )

        x_t = initial_noise
        current_time = torch.tensor(1.0, dtype=torch.float32, device=device).expand(
            bsize
        )

        end_time = torch.tensor(0.0, dtype=torch.float32, device=device).expand(bsize)
        time_schedule = torch.cat([time_schedule, end_time.unsqueeze(1)], dim=1)
        time_schedule = time_schedule.transpose(0, 1)

        noises = []
        times = []
        gradients = []

        for target_time in time_schedule:
            if teacher_flow_path_noise_std > 0:
                x_t.add_(
                    torch.randn_like(x_t) * teacher_flow_path_noise_std
                )

            noises.append(x_t.clone())
            times.append(current_time.clone())

            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(state, x_t, current_time)
            )

            embs = torch.cat([prefix_embs, suffix_embs], dim=1)
            attention_mask, pad_masks = self.build_expert_masks(
                prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
            )
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
            gradients.append(v_t.clone())

            dt = (target_time - current_time)[:, None, None]
            x_t = x_t + dt * v_t
            current_time = target_time

        noises = torch.stack(noises, dim=1)
        times = torch.stack(times, dim=1)
        gradients = torch.stack(gradients, dim=1)

        return noises, times, gradients, x_t

    @torch.no_grad()
    def sample_actions(
        self, device, observation, noise=None, num_steps=10, start_time=1.0
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, state = self._preprocess_observation(
            observation, train=False
        )

        # Prefix embeddings are static across denoising steps
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(start_time, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Recompute suffix embeddings at current x_t and time
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(state, x_t, expanded_time)
            )

            embs = torch.cat([prefix_embs, suffix_embs], dim=1)
            attention_mask, pad_masks = self.build_expert_masks(
                prefix_pad_masks, prefix_att_masks, suffix_pad_masks, suffix_att_masks
            )
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
