import logging
import math
import os
from typing import Any

import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch import nn

import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: tuple[int, ...] | list[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> list[nn.Module]:
    if len(net_arch) > 0:
        modules: list[nn.Module] = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        **_: Any,
    ):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"Unsupported final_norm: {final_norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **_: Any,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"PointNetEncoderXYZ expects 3 channels, got {in_channels}")

        block_channel = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"Unsupported final_norm: {final_norm}")

        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class MultiStagePointNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        h_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 4,
        **_: Any,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"MultiStagePointNetEncoder expects 3 channels, got {in_channels}")

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        self.conv_in = nn.Conv1d(in_channels, h_dim, kernel_size=1)
        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * num_layers, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.act(self.conv_in(x))
        features = []
        for layer, global_layer in zip(self.layers, self.global_layers, strict=True):
            y = self.act(layer(y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(global_layer(y))
            features.append(y)
        x = torch.cat(features, dim=1)
        x = self.conv_out(x)
        return x.max(-1).values


class DP3Encoder(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        pointcloud_channels: int,
        encoder_output_dim: int,
        state_mlp_size: tuple[int, ...],
        pointcloud_encoder_cfg: dict[str, Any],
        use_pc_color: bool,
        pointnet_type: str,
    ):
        super().__init__()
        self.n_output_channels = encoder_output_dim
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        point_cfg = dict(pointcloud_encoder_cfg)
        if pointnet_type == "multi_stage_pointnet":
            point_cfg["in_channels"] = 3
            self.extractor = MultiStagePointNetEncoder(**point_cfg)
        elif pointnet_type == "pointnet" and use_pc_color:
            point_cfg["in_channels"] = min(pointcloud_channels, 6)
            self.extractor = PointNetEncoderXYZRGB(**point_cfg)
        elif pointnet_type == "pointnet":
            point_cfg["in_channels"] = 3
            self.extractor = PointNetEncoderXYZ(**point_cfg)
        else:
            raise NotImplementedError(f"Unsupported pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise ValueError("state_mlp_size cannot be empty")
        net_arch = list(state_mlp_size[:-1]) if len(state_mlp_size) > 1 else []
        state_output_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(
            *create_mlp(action_dim, state_output_dim, net_arch, nn.ReLU)
        )
        self.n_output_channels += state_output_dim

    def forward(self, pointcloud: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.pointnet_type == "multi_stage_pointnet" or not self.use_pc_color:
            pointcloud = pointcloud[..., :3]
        elif pointcloud.shape[-1] > 6:
            pointcloud = pointcloud[..., :6]

        point_feat = self.extractor(pointcloud)
        state_feat = self.state_mlp(state)
        return torch.cat([point_feat, state_feat], dim=-1)

    def output_shape(self) -> int:
        return self.n_output_channels


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossAttention(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, out_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key = self.key_proj(cond)
        value = self.value_proj(cond)
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        n_groups: int = 8,
        condition_type: str = "film",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.condition_type = condition_type
        self.out_channels = out_channels

        if condition_type == "film":
            self.cond_encoder: nn.Module = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels * 2),
                Rearrange("batch t -> batch t 1"),
            )
        elif condition_type == "add":
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif condition_type == "cross_attention_add":
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == "cross_attention_film":
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels * 2)
        elif condition_type == "mlp_film":
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, out_channels * 2),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            raise NotImplementedError(f"Unsupported condition_type: {condition_type}")

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        out = self.blocks[0](x)
        if cond is not None:
            if self.condition_type == "film":
                embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1)
                out = embed[:, 0] * out + embed[:, 1]
            elif self.condition_type == "add":
                out = out + self.cond_encoder(cond)
            elif self.condition_type == "cross_attention_add":
                embed = self.cond_encoder(x.permute(0, 2, 1), cond).permute(0, 2, 1)
                out = out + embed
            elif self.condition_type == "cross_attention_film":
                embed = self.cond_encoder(x.permute(0, 2, 1), cond).permute(0, 2, 1)
                embed = embed.reshape(cond.shape[0], 2, self.out_channels, -1)
                out = embed[:, 0] * out + embed[:, 1]
            elif self.condition_type == "mlp_film":
                embed = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, -1)
                out = embed[:, 0] * out + embed[:, 1]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        time_embedding_scale: float = 1.0,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition
        self.time_embedding_scale = float(time_embedding_scale)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:], strict=True))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        condition_type=condition_type,
                    ),
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        dim_out,
                        cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        condition_type=condition_type,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
            ]
        )

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(zip(all_dims[:-1], all_dims[1:], strict=True)):
            is_last = ind >= (len(all_dims) - 2)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        logging.getLogger("openpi").info(
            "DP3 UNet parameter count: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_horizon = sample.shape[1]
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0]).to(dtype=sample.dtype)

        timestep_embed = self.diffusion_step_encoder(timesteps * self.time_embedding_scale)
        global_feature = timestep_embed
        if global_cond is not None:
            if "cross_attention" in self.condition_type:
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
        elif "cross_attention" in self.condition_type:
            raise ValueError("cross_attention condition types require global_cond")

        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and h_local:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and h_local:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature) if self.use_mid_condition else mid_module(x)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and h_local:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and h_local:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        if x.shape[-1] != target_horizon:
            x = F.interpolate(x, size=target_horizon, mode="nearest")
        return einops.rearrange(x, "b t h -> b h t")


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha_t, beta_t).sample((bsize,))


def _gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    feature_dim = points.shape[-1]
    expanded_idx = idx.unsqueeze(-1).expand(*idx.shape, feature_dim)
    return torch.gather(points, 1, expanded_idx)


class ProxyDP3Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        pointcloud_channels = 6 if getattr(config, "use_pc_color", True) else 3
        pointcloud_encoder_cfg = dict(config.pointcloud_encoder_cfg)
        pointcloud_encoder_cfg.setdefault("out_channels", config.encoder_output_dim)

        self.obs_encoder = DP3Encoder(
            action_dim=config.action_dim,
            pointcloud_channels=pointcloud_channels,
            encoder_output_dim=config.encoder_output_dim,
            state_mlp_size=config.state_mlp_size,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=config.use_pc_color,
            pointnet_type=config.pointnet_type,
        )
        if getattr(config, "freeze_point_encoder", False):
            for param in self.obs_encoder.extractor.parameters():
                param.requires_grad = False

        obs_feature_dim = self.obs_encoder.output_shape()
        input_dim = config.action_dim
        global_cond_dim = obs_feature_dim
        if not config.obs_as_global_cond:
            input_dim = config.action_dim + obs_feature_dim
            global_cond_dim = None

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            time_embedding_scale=getattr(config, "time_embedding_scale", 1.0),
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            condition_type=config.condition_type,
            use_down_condition=config.use_down_condition,
            use_mid_condition=config.use_mid_condition,
            use_up_condition=config.use_up_condition,
        )

        torch.set_float32_matmul_precision("high")
        if (
            getattr(config, "compile_sample_actions", False)
            and os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "").lower() not in ("1", "true", "yes")
        ):
            self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        self.gradient_checkpointing_enabled = False
        self._logged_invalid_pointcloud_warning = False
        self._logged_point_resize_warning = False

    def _preprocess_observation(self, observation, *, train: bool = True):
        observation = _preprocessing.preprocess_observation_pointcloud_pytorch(
            observation, image_keys=IMAGE_KEYS, train=train
        )
        return observation.pointcloud, observation.pointcloud_masks, observation.state

    def _sanitize_pointcloud(self, pointcloud: torch.Tensor) -> torch.Tensor:
        xyz_valid = torch.isfinite(pointcloud[..., :3]).all(dim=-1)
        if xyz_valid.all():
            sanitized = pointcloud.clone()
            if sanitized.shape[-1] > 3:
                sanitized[..., 3:] = torch.nan_to_num(
                    sanitized[..., 3:], nan=0.0, posinf=255.0, neginf=0.0
                )
            return sanitized

        sanitized = pointcloud.clone()
        invalid_counts = (~xyz_valid).sum(dim=1)
        for batch_idx in range(pointcloud.shape[0]):
            valid_idx = torch.nonzero(xyz_valid[batch_idx], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                sanitized[batch_idx].zero_()
                continue
            invalid_idx = torch.nonzero(~xyz_valid[batch_idx], as_tuple=False).flatten()
            repeat = math.ceil(invalid_idx.numel() / valid_idx.numel())
            replacement = sanitized[batch_idx, valid_idx].repeat(repeat, 1)[: invalid_idx.numel()]
            sanitized[batch_idx, invalid_idx] = replacement

        if sanitized.shape[-1] > 3:
            sanitized[..., 3:] = torch.nan_to_num(
                sanitized[..., 3:], nan=0.0, posinf=255.0, neginf=0.0
            )

        if not self._logged_invalid_pointcloud_warning:
            logging.warning(
                "Detected non-finite XYZ values in pointcloud input. Replaced invalid points per sample before DP3 encoding. Invalid point counts: %s",
                invalid_counts.detach().cpu().tolist(),
            )
            self._logged_invalid_pointcloud_warning = True
        return sanitized

    def _dropout_pointcloud(self, pointcloud: torch.Tensor, *, train: bool) -> torch.Tensor:
        dropout_ratio = getattr(self.config, "pointcloud_dropout_ratio", 0.0)
        if not train or dropout_ratio <= 0.0:
            return pointcloud

        batch_size, num_points, _ = pointcloud.shape
        if num_points == 0:
            return pointcloud

        augmented = pointcloud.clone()
        dropout_ratio = min(max(float(dropout_ratio), 0.0), 1.0)
        drop_mask = torch.rand(
            batch_size,
            num_points,
            device=pointcloud.device,
        ) < dropout_ratio
        replacement_index = torch.randint(
            0,
            num_points,
            (batch_size, num_points),
            device=pointcloud.device,
        )
        replacement = _gather_points(pointcloud, replacement_index)
        return torch.where(drop_mask.unsqueeze(-1), replacement, augmented)

    def _jitter_pointcloud_positions(self, pointcloud: torch.Tensor, *, train: bool) -> torch.Tensor:
        noise_std = getattr(self.config, "pointcloud_position_noise_std", 0.0)
        if not train or noise_std <= 0.0:
            return pointcloud

        xyz_valid = torch.isfinite(pointcloud[..., :3]).all(dim=-1)
        if not xyz_valid.any():
            return pointcloud

        augmented = pointcloud.clone()
        xyz_noise = torch.randn_like(augmented[..., :3]) * noise_std
        noisy_xyz = augmented[..., :3] + xyz_noise
        augmented[..., :3] = torch.where(
            xyz_valid.unsqueeze(-1),
            noisy_xyz,
            augmented[..., :3],
        )
        return augmented

    def _augment_pointcloud(self, pointcloud: torch.Tensor, *, train: bool) -> torch.Tensor:
        if not train:
            return pointcloud
        augmented = self._dropout_pointcloud(pointcloud, train=train)
        augmented = self._jitter_pointcloud_positions(augmented, train=train)
        return augmented

    def _random_point_sample(self, xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
        num_points = xyz.shape[0]
        num_samples = min(num_samples, num_points)
        return torch.randperm(num_points, device=xyz.device)[:num_samples]

    def _resize_pointcloud(self, pointcloud: torch.Tensor, *, train: bool) -> torch.Tensor:
        target_points = self.config.num_points
        batch_size, num_points, _ = pointcloud.shape
        if train and getattr(self.config, "pointcloud_random_resample", False):
            index = torch.randint(
                0,
                num_points,
                (batch_size, target_points),
                device=pointcloud.device,
            )
            return _gather_points(pointcloud, index)

        if num_points == target_points:
            return pointcloud

        if num_points > target_points:
            if not train:
                index = torch.linspace(
                    0,
                    num_points - 1,
                    steps=target_points,
                    device=pointcloud.device,
                ).round().to(torch.long)
                if not self._logged_point_resize_warning:
                    logging.info(
                        "Resizing DP3 pointclouds from %s to %s points at inference time.",
                        num_points,
                        target_points,
                    )
                    self._logged_point_resize_warning = True
                return pointcloud.index_select(1, index)

            xyz = pointcloud[..., :3]
            indices = []
            for batch_idx in range(batch_size):
                indices.append(self._random_point_sample(xyz[batch_idx], target_points))
            index = torch.stack(indices, dim=0)
            return _gather_points(pointcloud, index)

        repeat = math.ceil(target_points / num_points)
        pointcloud = pointcloud.repeat(1, repeat, 1)
        return pointcloud[:, :target_points]

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

    def encode_observation(
        self,
        pointcloud: torch.Tensor,
        point_mask: torch.Tensor | None,
        state: torch.Tensor,
        *,
        train: bool,
    ) -> torch.Tensor:
        encoder_dtype = next(self.obs_encoder.parameters()).dtype
        pointcloud = pointcloud.to(dtype=encoder_dtype)
        state = state.to(dtype=encoder_dtype)
        pointcloud = self._sanitize_pointcloud(pointcloud)
        pointcloud = self._resize_pointcloud(pointcloud, train=train)
        pointcloud = self._augment_pointcloud(pointcloud, train=train)
        obs_features = self.obs_encoder(pointcloud, state)
        if point_mask is not None:
            obs_features = obs_features * point_mask[:, None].to(dtype=obs_features.dtype)
        return obs_features

    def _run_dp3(self, x_t: torch.Tensor, time: torch.Tensor, obs_features: torch.Tensor) -> torch.Tensor:
        if self.config.obs_as_global_cond:
            pred = self.model(sample=x_t, timestep=time, global_cond=obs_features)
        else:
            obs_tokens = obs_features[:, None, :].expand(-1, x_t.shape[1], -1)
            model_input = torch.cat([x_t, obs_tokens], dim=-1)
            pred = self.model(sample=model_input, timestep=time)
        return pred[..., : self.config.action_dim]

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
        if mode == "distill":
            return self.forward_distill(
                observation, noises, times, gradients, actions, use_noise=use_noise
            )
        if mode not in ("train", "val"):
            raise ValueError(f"Unsupported forward mode: {mode}")
        if actions is None:
            raise ValueError("actions must be provided for training mode.")

        augment_observation = mode == "train"
        pointcloud, point_mask, state = self._preprocess_observation(observation, train=augment_observation)
        obs_features = self.encode_observation(pointcloud, point_mask, state, train=augment_observation)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t = self._run_dp3(x_t, time, obs_features)
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
        pointcloud, point_mask, state = self._preprocess_observation(observation, train=True)
        obs_features = self.encode_observation(pointcloud, point_mask, state, train=True)

        initial_noise = noises[:, 0, :, : self.config.action_dim]
        noises = noises[:, 1:, :, : self.config.action_dim]
        times = times[:, 1:]
        gradients = gradients[:, 1:, :, : self.config.action_dim]
        actions = actions[:, :, : self.config.action_dim]

        losses = []
        for step_idx in range(times.shape[1]):
            noise_step = noises[:, step_idx]
            time_step = times[:, step_idx]
            gradient_step = gradients[:, step_idx]

            if use_noise:
                x_t = noise_step
            else:
                time_expanded = time_step[:, None, None]
                x_t = time_expanded * initial_noise + (1 - time_expanded) * actions

            v_t = self._run_dp3(x_t, time_step, obs_features)
            losses.append(F.mse_loss(v_t, gradient_step, reduction="none"))

        return torch.stack(losses, dim=1)

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        start_time=1.0,
    ) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        pointcloud, point_mask, state = self._preprocess_observation(observation, train=False)
        obs_features = self.encode_observation(pointcloud, point_mask, state, train=False)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(start_time, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._run_dp3(x_t, expanded_time, obs_features)
            x_t = x_t + dt * v_t
            time += dt

        return x_t
