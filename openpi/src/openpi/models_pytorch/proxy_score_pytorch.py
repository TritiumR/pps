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


def score_to_epsilon(score: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    sqrt_beta = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-6))
    while sqrt_beta.ndim < score.ndim:
        sqrt_beta = sqrt_beta.unsqueeze(-1)
    return -sqrt_beta * score


def make_att_2d_masks(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    """Build pi0-style block attention from padding and block-boundary masks."""
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be rank 2, got {pad_masks.ndim}.")
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be rank 2, got {att_masks.ndim}.")
    if pad_masks.shape != att_masks.shape:
        raise ValueError(
            f"pad_masks and att_masks must have the same shape, got "
            f"{tuple(pad_masks.shape)} and {tuple(att_masks.shape)}."
        )

    block_ids = torch.cumsum(att_masks.to(torch.int32), dim=1)
    block_mask = block_ids[:, None, :] <= block_ids[:, :, None]
    valid = pad_masks.to(torch.bool)
    return block_mask & valid[:, None, :] & valid[:, :, None]


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
            language_vocab_size=(
                config.language_vocab_size
                if getattr(config, "use_language_tokens", False)
                else None
            ),
        )
        self.bidirectional_attention = getattr(config, "bidirectional_attention", True)
        self.legacy_gemma_input_scale = getattr(
            config, "legacy_gemma_input_scale", False
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
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
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
        self, images, img_masks, lang_tokens=None, lang_masks=None
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

        if getattr(self.config, "use_language_tokens", False):
            if lang_tokens is None or lang_masks is None:
                raise ValueError(
                    "tokenized_prompt and tokenized_prompt_mask are required when "
                    "use_language_tokens=True."
                )
            lang_emb = self.expert_model.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            embs.append(lang_emb)
            pad_masks.append(lang_masks.to(torch.bool))
            att_masks += [0] * lang_emb.shape[1]

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

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def build_attention_mask(
        self,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> torch.Tensor:
        if not self.bidirectional_attention:
            return pad_masks
        return self._prepare_attention_masks_4d(
            make_att_2d_masks(pad_masks, att_masks)
        )

    def build_expert_masks(
        self,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat(
            [prefix_att_masks.to(torch.int32), suffix_att_masks.to(torch.int32)],
            dim=1,
        )
        return self.build_attention_mask(pad_masks, att_masks), pad_masks

    def _run_diffusion_head(
        self,
        prefix_embs,
        prefix_pad_masks,
        suffix_embs,
        suffix_pad_masks,
        adarms_cond,
        prefix_att_masks=None,
        suffix_att_masks=None,
    ) -> torch.Tensor:
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        if self.legacy_gemma_input_scale:
            # Stock Transformers Gemma scales all ``inputs_embeds`` before the
            # first decoder layer. Keep this checkpoint-compatibility behavior
            # local to ProxyScore; OpenPI's replacement intentionally omits the
            # global scaling for its continuous embeddings.
            normalizer = torch.tensor(
                embs.shape[-1] ** 0.5,
                dtype=embs.dtype,
                device=embs.device,
            )
            embs = embs * normalizer
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        attention_mask = pad_masks
        if prefix_att_masks is not None and suffix_att_masks is not None:
            attention_mask, pad_masks = self.build_expert_masks(
                prefix_pad_masks,
                prefix_att_masks,
                suffix_pad_masks,
                suffix_att_masks,
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
        return self.action_out_proj(suffix_out)

    def _predict_model_output_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
        prefix_att_masks=None,
    ) -> torch.Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            time_cond,
        )
        return self._run_diffusion_head(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            adarms_cond,
            prefix_att_masks,
            suffix_att_masks,
        )

    def predict_score_from_prefix(
        self,
        state,
        prefix_embs,
        prefix_pad_masks,
        x_t,
        time_cond,
        prefix_att_masks=None,
    ) -> torch.Tensor:
        output = self._predict_model_output_from_prefix(
            state,
            prefix_embs,
            prefix_pad_masks,
            x_t,
            time_cond,
            prefix_att_masks,
        )
        if self.config.prediction_type == "score":
            return output

        alpha = self._alpha_from_time(time_cond, x_t.device, x_t.dtype)
        sqrt_beta = torch.sqrt(torch.clamp(1.0 - alpha, min=1e-6))
        return -output / sqrt_beta[:, None, None]

    def forward(
        self,
        observation,
        actions=None,
        noise=None,
        time=None,
        score_target=None,
        epsilon_target=None,
        *,
        mode="train",
        **_,
    ) -> Tensor:
        if mode != "train":
            raise ValueError(f"Unsupported forward mode for ProxyScorePytorch: {mode}")
        if actions is None:
            raise ValueError("actions must be provided for score training.")
        if score_target is not None and epsilon_target is not None:
            raise ValueError("score_target and epsilon_target are mutually exclusive.")

        actions = actions[..., : self.config.action_dim]
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=True)
        )
        loss_weight = None
        direct_score_target = score_target is not None
        direct_epsilon_target = epsilon_target is not None

        if direct_score_target or direct_epsilon_target:
            if time is None:
                raise ValueError("time must be provided when training from direct targets.")
            if direct_epsilon_target and self.config.prediction_type != "epsilon":
                raise ValueError(
                    "epsilon_target requires ProxyScoreConfig.prediction_type='epsilon'."
                )
            x_t = actions
            direct_target = score_target if direct_score_target else epsilon_target
            target = direct_target[..., : self.config.action_dim].to(
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
            if self.config.prediction_type == "epsilon":
                target = noise
            else:
                target = -noise / sqrt_beta[:, None, None]
                loss_weight = beta[:, None, None]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        grouped_labels = x_t.ndim == 4
        if grouped_labels:
            if target.ndim != 4 or time.ndim != 2:
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
            prefix_att_masks = prefix_att_masks.repeat_interleave(
                labels_per_observation, dim=0
            )
            state = state.repeat_interleave(labels_per_observation, dim=0)
            x_t = x_t.flatten(0, 1)
            target = target.flatten(0, 1)
            time = time.flatten(0, 1)
        predict = (
            self.predict_score_from_prefix
            if direct_score_target or self.config.prediction_type == "score"
            else self._predict_model_output_from_prefix
        )
        pred = predict(
            state,
            prefix_embs,
            prefix_pad_masks,
            x_t,
            time,
            prefix_att_masks,
        )
        loss = F.mse_loss(pred, target, reduction="none")
        if loss_weight is not None:
            loss = loss * loss_weight
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

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
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
            output = self._predict_model_output_from_prefix(
                state,
                prefix_embs,
                prefix_pad_masks,
                x_t,
                expanded_time,
                prefix_att_masks,
            )
            beta = torch.clamp(1.0 - alpha, min=1e-6)
            sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-6))
            sqrt_beta = torch.sqrt(beta)
            if self.config.prediction_type == "epsilon":
                eps_hat = output
                x0_hat = (x_t - sqrt_beta * eps_hat) / sqrt_alpha
            else:
                score = output
                x0_hat = (x_t + beta * score) / sqrt_alpha
                eps_hat = -sqrt_beta * score
            x_t = (
                torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0_hat
                + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps_hat
            )

        if torch.is_grad_enabled():
            logging.warning("ProxyScorePytorch.sample_actions is expected to run under no_grad.")
        return x_t
