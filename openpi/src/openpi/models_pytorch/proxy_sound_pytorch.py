import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.proxy_pytorch import IMAGE_KEYS
from openpi.models_pytorch.proxy_pytorch import ProxyPytorch


class SoundSpectrogramEncoder(nn.Module):
    """Small CNN that turns two log-mel spectrograms into Gemma-width tokens."""

    def __init__(self, in_channels: int, hidden_size: int, mel_bins: int, time_bins: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        out_mel = math.ceil(mel_bins / 8)
        out_time = math.ceil(time_bins / 8)
        self.pos_embed = nn.Parameter(torch.zeros(1, out_mel * out_time, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, sound: torch.Tensor) -> torch.Tensor:
        x = self.conv(sound)
        x = x.flatten(2).transpose(1, 2)
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        return x


class ProxySoundPytorch(ProxyPytorch):
    """Proxy policy with RGB prefix tokens plus two-microphone spectrogram tokens."""

    def __init__(self, config):
        super().__init__(config)
        hidden_size = self.state_proj.out_features
        self.sound_encoder = SoundSpectrogramEncoder(
            in_channels=config.sound_channels,
            hidden_size=hidden_size,
            mel_bins=config.sound_mel_bins,
            time_bins=config.sound_time_bins,
        )

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_sound_pytorch(
            observation,
            image_keys=IMAGE_KEYS,
            train=train,
            sound_vmin=self.config.sound_vmin,
            sound_vmax=self.config.sound_vmax,
        )
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.sound,
            observation.state,
        )

    def embed_prefix(self, images, img_masks, sound):
        image_embs, image_pad_masks, image_att_masks = super().embed_prefix(images, img_masks)

        sound_embs = self.sound_encoder(sound)
        bsize, num_sound_tokens = sound_embs.shape[:2]
        sound_pad_masks = torch.ones(
            bsize, num_sound_tokens, dtype=torch.bool, device=sound_embs.device
        )
        sound_att_masks = torch.zeros(
            bsize, num_sound_tokens, dtype=image_att_masks.dtype, device=sound_embs.device
        )

        embs = torch.cat([image_embs, sound_embs], dim=1)
        pad_masks = torch.cat([image_pad_masks, sound_pad_masks], dim=1)
        att_masks = torch.cat([image_att_masks, sound_att_masks], dim=1)
        return embs, pad_masks, att_masks

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
    ):
        if mode == "distill":
            return self.forward_distill(
                observation, noises, times, gradients, actions, use_noise=use_noise
            )
        if mode != "train":
            raise ValueError(f"Unsupported forward mode: {mode}")
        if actions is None:
            raise ValueError("actions must be provided for training mode.")

        images, img_masks, sound, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, sound)
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(state, x_t, time)

        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        hidden_states, _ = self.expert_model.forward(
            attention_mask=pad_masks,
            position_ids=position_ids.to(dtype=torch.long),
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = hidden_states[:, -self.config.action_horizon :].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_distill(self, observation, noises, times, gradients, actions, use_noise=True):
        images, img_masks, sound, state = self._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, sound)

        initial_noise = noises[:, 0, :, : self.config.action_dim]
        noises = noises[:, 1:, :, : self.config.action_dim]
        times = times[:, 1:]
        gradients = gradients[:, 1:, :, : self.config.action_dim]
        actions = actions[:, :, : self.config.action_dim]

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
        suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
            flat_state, flat_x_t, flat_times
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

        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        hidden_states, _ = self.expert_model.forward(
            attention_mask=pad_masks,
            position_ids=position_ids.to(dtype=torch.long),
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )

        suffix_out = hidden_states[:, -self.config.action_horizon :].to(dtype=torch.float32)
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
    ):
        images, img_masks, sound, state = self._preprocess_observation(observation, train=False)
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
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, sound)

        x_t = initial_noise
        current_time = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)
        end_time = torch.tensor(0.0, dtype=torch.float32, device=device).expand(bsize)
        time_schedule = torch.cat([time_schedule, end_time.unsqueeze(1)], dim=1)
        time_schedule = time_schedule.transpose(0, 1)

        noises = []
        times = []
        gradients = []
        for target_time in time_schedule:
            if teacher_flow_path_noise_std > 0:
                x_t.add_(torch.randn_like(x_t) * teacher_flow_path_noise_std)

            noises.append(x_t.clone())
            times.append(current_time.clone())

            suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
                state, x_t, current_time
            )
            embs = torch.cat([prefix_embs, suffix_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            hidden_states, _ = self.expert_model.forward(
                attention_mask=pad_masks,
                position_ids=position_ids.to(dtype=torch.long),
                past_key_values=None,
                inputs_embeds=embs,
                use_cache=False,
                adarms_cond=adarms_cond,
            )

            suffix_out = hidden_states[:, -self.config.action_horizon :].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            gradients.append(v_t.clone())

            dt = (target_time - current_time)[:, None, None]
            x_t = x_t + dt * v_t
            current_time = target_time

        return torch.stack(noises, dim=1), torch.stack(times, dim=1), torch.stack(gradients, dim=1), x_t

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, start_time=1.0):
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, sound, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, sound)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(start_time, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            suffix_embs, suffix_pad_masks, _, adarms_cond = self.embed_suffix(
                state, x_t, expanded_time
            )
            embs = torch.cat([prefix_embs, suffix_embs], dim=1)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            hidden_states, _ = self.expert_model.forward(
                attention_mask=pad_masks,
                position_ids=position_ids.to(dtype=torch.long),
                past_key_values=None,
                inputs_embeds=embs,
                use_cache=False,
                adarms_cond=adarms_cond,
            )

            suffix_out = hidden_states[:, -self.config.action_horizon :].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
            time += dt

        return x_t
