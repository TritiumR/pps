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
    if device_type == "cpu":
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
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch_size].")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def squaredcos_cap_v2_alpha_bar(t: torch.Tensor | float) -> torch.Tensor | float:
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2


def ddim_alphas_cumprod(num_train_timesteps: int, *, device, dtype) -> torch.Tensor:
    if num_train_timesteps <= 1:
        raise ValueError("num_train_timesteps must be greater than 1.")

    alpha_cumprod = torch.ones((), device=device, dtype=dtype)
    alphas = []
    for idx in range(int(num_train_timesteps)):
        t1 = torch.as_tensor(idx / float(num_train_timesteps), device=device, dtype=dtype)
        t2 = torch.as_tensor((idx + 1) / float(num_train_timesteps), device=device, dtype=dtype)
        beta = 1.0 - squaredcos_cap_v2_alpha_bar(t2) / squaredcos_cap_v2_alpha_bar(t1)
        beta = torch.clamp(beta, max=0.999)
        alpha_cumprod = alpha_cumprod * (1.0 - beta)
        alphas.append(alpha_cumprod)
    return torch.stack(alphas)


def ddim_iteration_alphas(
    *,
    iteration: int,
    num_iterations: int,
    num_train_timesteps: int,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if iteration < 0 or iteration >= num_iterations:
        raise ValueError(f"iteration must be in [0, {num_iterations}), got {iteration}.")
    if num_iterations > num_train_timesteps:
        raise ValueError("num_iterations must be <= num_train_timesteps.")

    alphas = ddim_alphas_cumprod(num_train_timesteps, device=device, dtype=dtype)
    step_ratio = int(num_train_timesteps) // int(num_iterations)
    if step_ratio <= 0:
        raise ValueError("num_iterations must be <= num_train_timesteps.")

    timestep = int((int(num_iterations) - 1 - int(iteration)) * step_ratio)
    prev_timestep = timestep - step_ratio
    alpha_t = alphas[timestep]
    alpha_prev = (
        alphas[prev_timestep]
        if prev_timestep >= 0
        else torch.ones((), device=device, dtype=dtype)
    )
    time_cond = torch.as_tensor(
        timestep / max(float(num_train_timesteps - 1), 1.0),
        device=device,
        dtype=dtype,
    )
    return alpha_t, alpha_prev, time_cond


class ProxyScorePytorch(nn.Module):
    """Image/state-conditioned score proxy for DDIM/MBD score-space PPS steering."""

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

        action_dim = config.action_dim
        self.action_in_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, action_dim)
        self.state_proj = nn.Linear(action_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.width, action_expert_config.width
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        torch.set_float32_matmul_precision("high")
        if (
            getattr(config, "compile_sample_actions", False)
            and os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower()
            not in ("1", "true", "yes")
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
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def _sample_train_alpha(self, bsize: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        alphas = ddim_alphas_cumprod(
            self.config.ddim_num_train_timesteps,
            device=device,
            dtype=dtype,
        )
        idx = torch.randint(0, alphas.shape[0], (bsize,), device=device)
        alpha = alphas[idx]
        time = idx.to(dtype=dtype) / max(float(alphas.shape[0] - 1), 1.0)
        return alpha, time

    def _alpha_from_time(self, time: torch.Tensor, device, dtype) -> torch.Tensor:
        alphas = ddim_alphas_cumprod(
            self.config.ddim_num_train_timesteps,
            device=device,
            dtype=dtype,
        )
        idx = torch.round(
            torch.clamp(time.to(device=device, dtype=dtype), 0.0, 1.0)
            * max(float(alphas.shape[0] - 1), 1.0)
        ).to(dtype=torch.long)
        return alphas[idx]

    def embed_prefix(
        self, images, img_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)
        pad_masks.append(
            torch.ones(
                bsize,
                action_time_emb.shape[1],
                dtype=torch.bool,
                device=timestep.device,
            )
        )
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, None

    def _run_score_head(
        self,
        prefix_embs,
        prefix_pad_masks,
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
    ) -> torch.Tensor:
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        position_ids = position_ids.to(dtype=torch.long)
        hidden_states, _ = self.expert_model.forward(
            attention_mask=pad_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = hidden_states[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def predict_score_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
            state,
            x_t,
            time_cond,
        )
        return self._run_score_head(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            adarms_cond,
        )

    def forward(
        self,
        observation,
        actions=None,
        noise=None,
        time=None,
        score_target=None,
        *,
        mode="train",
        **_,
    ) -> Tensor:
        if mode != "train":
            raise ValueError(f"Unsupported forward mode for ProxyScorePytorch: {mode}")
        if actions is None:
            raise ValueError("actions must be provided for score training.")

        actions = actions[..., : self.config.action_dim]
        images, img_masks, state = self._preprocess_observation(observation, train=True)

        if score_target is not None:
            if time is None:
                raise ValueError("time must be provided when training from direct score targets.")
            x_t = actions
            score_target = score_target[..., : self.config.action_dim].to(
                device=actions.device,
                dtype=actions.dtype,
            )
            time = time.to(device=actions.device, dtype=actions.dtype)
        else:
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            else:
                noise = noise[:, :, : self.config.action_dim]
            if time is None:
                alpha, time = self._sample_train_alpha(
                    actions.shape[0],
                    actions.device,
                    actions.dtype,
                )
            else:
                alpha = self._alpha_from_time(time, actions.device, actions.dtype)

            alpha = alpha.to(device=actions.device, dtype=actions.dtype)
            time = time.to(device=actions.device, dtype=actions.dtype)
            beta = torch.clamp(1.0 - alpha, min=1e-6)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-6))
            sqrt_beta = torch.sqrt(beta)

            x_t = sqrt_alpha[:, None, None] * actions + sqrt_beta[:, None, None] * noise
            score_target = -noise / sqrt_beta[:, None, None]

        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks)
        grouped_labels = x_t.ndim == 4
        if grouped_labels:
            if score_target.ndim != 4 or time.ndim != 2:
                raise ValueError(
                    "Grouped score targets require x_t/score shape [B, L, H, D] "
                    "and time shape [B, L]."
                )
            batch_size, labels_per_observation = x_t.shape[:2]
            if prefix_embs.shape[0] != batch_size:
                raise ValueError("Observation batch does not match grouped score targets.")
            # The expensive visual prefix is computed once per observation. Repeating
            # the resulting embeddings preserves gradients from every score label
            # without rerunning DINO for each reverse-diffusion state.
            prefix_embs = prefix_embs.repeat_interleave(labels_per_observation, dim=0)
            prefix_pad_masks = prefix_pad_masks.repeat_interleave(
                labels_per_observation, dim=0
            )
            state = state.repeat_interleave(labels_per_observation, dim=0)
            x_t = x_t.flatten(0, 1)
            score_target = score_target.flatten(0, 1)
            time = time.flatten(0, 1)
        score_pred = self.predict_score_from_prefix(
            state,
            prefix_embs,
            prefix_pad_masks,
            x_t,
            time,
        )
        loss = F.mse_loss(score_pred, score_target, reduction="none")
        if grouped_labels:
            loss = loss.unflatten(0, (batch_size, labels_per_observation))
        return loss

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        start_time=1.0,
    ) -> Tensor:
        del start_time
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, state = self._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks)
        x_t = noise
        for iteration in range(int(num_steps)):
            alpha, alpha_prev, time_cond = ddim_iteration_alphas(
                iteration=iteration,
                num_iterations=int(num_steps),
                num_train_timesteps=self.config.ddim_num_train_timesteps,
                device=device,
                dtype=x_t.dtype,
            )
            expanded_time = time_cond.expand(bsize)
            score = self.predict_score_from_prefix(
                state,
                prefix_embs,
                prefix_pad_masks,
                x_t,
                expanded_time,
            )
            beta = torch.clamp(1.0 - alpha, min=1e-6)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-6))
            sqrt_beta = torch.sqrt(beta)
            x0_hat = (x_t + beta * score) / sqrt_alpha
            eps_hat = -sqrt_beta * score
            x_t = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0_hat
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps_hat
            )

        if torch.is_grad_enabled():
            logging.warning("ProxyScorePytorch.sample_actions is expected to run under no_grad.")
        return x_t
