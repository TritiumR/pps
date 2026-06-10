"""Residual-V (velocity-conditioned residual) policy, PyTorch implementation.

Differences from ``residual_pytorch.ResidualPytorch``:

  * Drops ``vla_action_proj`` / ``embed_vla_action``. The VLA action prefix is
    replaced by conditioning on the base VLA's velocity ``v_base_t``
    (recomputed per ODE step).
  * ``v_base_t`` is injected into the suffix action-time fusion MLP:
    ``[action_in_proj(x_t) || sincos(time) || v_base_proj(v_base_t)]`` goes
    through ``action_time_mlp_in`` (3W -> W) -> SiLU -> ``action_time_mlp_out``.
  * ``forward`` takes ``(x_t, time, v_base)`` explicitly (sampled outside the
    model) and returns the predicted correction velocity ``v_res``. The caller
    owns loss construction. This lets the training script feed the **same**
    ``(x_t, t)`` to both VLA and residual so ``v_base_t`` matches the residual's
    path point exactly.
  * ``sample_actions`` accepts a ``base_denoise_fn: (x_t, t) -> v_base_t``
    callable and a ``gamma`` scalar. The Euler update is
    ``x_t += dt * (v_base + gamma * v_res)``.
"""

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
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


class ResidualVPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.expert_model = DINOExpertModel(
            dino_model_name=config.dino_model_name,
            action_expert_config=action_expert_config,
            use_adarms=[False, False],
            precision=config.dtype,
            freeze_dino_encoder=getattr(config, "freeze_dino_encoder", False),
        )

        self.width = action_expert_config.width

        self.action_in_proj = nn.Linear(config.action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, config.action_dim)

        # Project the frozen base model's velocity into the action-expert width.
        # v_base_t has the same shape as noisy actions: (B, action_horizon, action_dim).
        self.v_base_proj = nn.Linear(config.action_dim, self.width)

        self.state_proj = nn.Linear(config.action_dim, self.width)

        # Widened fusion MLP: [action_emb || time_emb || v_base_emb] -> W
        self.action_time_mlp_in = nn.Linear(3 * self.width, self.width)
        self.action_time_mlp_out = nn.Linear(self.width, self.width)

        torch.set_float32_matmul_precision("high")
        if os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() not in (
            "1",
            "true",
            "yes",
        ):
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def _preprocess_observation(self, observation, *, train=True):
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
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks):
        """Embed images with DINO. Returns (embs, pad_masks, att_masks)."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.expert_model.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep, v_base):
        """Embed state + (action, time, v_base)-fused tokens for the Gemma expert."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.width,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        if self.action_in_proj.weight.dtype == torch.float32:
            noisy_actions = noisy_actions.to(torch.float32)
            v_base = v_base.to(torch.float32)

        action_emb = self.action_in_proj(noisy_actions)
        v_base_emb = self.v_base_proj(v_base)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        # 3-way fusion: [action || time || v_base] -> W
        fused = torch.cat([action_emb, time_emb, v_base_emb], dim=2)

        def mlp_func(x):
            x = self.action_time_mlp_in(x)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = mlp_func(fused)
        adarms_cond = None

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _expert_forward(self, prefix_embs, prefix_pad_masks, state, x_t, time, v_base):
        """Run the DINO-expert transformer with a given prefix + suffix inputs."""
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
            state, x_t, time, v_base
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
        return self.action_out_proj(suffix_out)

    def forward(self, observation, *, x_t, time, v_base) -> Tensor:
        """Predict the residual correction velocity ``v_res`` at the given path point.

        Args:
            observation: Observation batch (has images, image_masks, state, ...).
            x_t: Path state, shape (B, action_horizon, action_dim).
            time: Times in [0, 1], shape (B,).
            v_base: Base VLA velocity at (x_t, t, c), truncated to action_dim.
                Shape (B, action_horizon, action_dim). No gradient should flow
                into this tensor (caller should ``.detach()``).

        Returns:
            v_res: Predicted residual velocity, shape (B, action_horizon, action_dim).
        """
        images, img_masks, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks)
        return self._expert_forward(
            prefix_embs, prefix_pad_masks, state, x_t, time, v_base
        )

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        base_denoise_fn,
        noise=None,
        num_steps=10,
        gamma: float = 1.0,
        start_time: float = 1.0,
    ) -> Tensor:
        """Sample an action via Euler ODE with combined velocity ``v_base + gamma * v_res``.

        Args:
            device: Torch device.
            observation: Observation batch.
            base_denoise_fn: Callable ``(x_t, t) -> v_base_t`` returning the base VLA
                velocity at (x_t, t) truncated to ``action_dim``. Must already have
                any prefix caching pre-computed by the caller.
            noise: Optional initial noise, shape (B, action_horizon, action_dim).
            num_steps: Number of Euler steps.
            gamma: Mixing coefficient for the residual velocity.
            start_time: Starting time (1.0 by default).
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, state = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(start_time, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            v_base = base_denoise_fn(x_t, expanded_time)
            v_base = v_base.to(dtype=torch.float32).detach()

            v_res = self._expert_forward(
                prefix_embs, prefix_pad_masks, state, x_t, expanded_time, v_base
            )

            v_combined = v_base + gamma * v_res
            x_t = x_t + dt * v_combined
            time = time + dt

        return x_t
