"""Frozen Concerto point encoder with a fixed-size Perceiver bottleneck."""

from __future__ import annotations

import contextlib

import torch
from torch import nn


class PerceiverResampler(nn.Module):
    """Cross-attend learned queries to a variable-length point sequence."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int,
        num_heads: int = 8,
    ):
        super().__init__()
        if output_dim % num_heads:
            raise ValueError(
                f"output_dim={output_dim} must be divisible by num_heads={num_heads}."
            )
        self.queries = nn.Parameter(torch.empty(1, num_tokens, output_dim))
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.query_norm = nn.LayerNorm(output_dim)
        self.attention = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        key_value = self.input_proj(self.input_norm(features))
        query = self.query_norm(self.queries.expand(features.shape[0], -1, -1))
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        attended = query + attended
        return attended + self.mlp(self.output_norm(attended))


class ConcertoPointCloudPrefix(nn.Module):
    """Encode each camera with one shared Concerto-Small and emit fixed token blocks."""

    feature_dim = 512  # Concerto-Small's final encoder stage.

    def __init__(
        self,
        *,
        pointcloud_keys: tuple[str, ...],
        output_dim: int,
        tokens_per_camera: int = 128,
        model_name: str = "concerto_small",
        repo_id: str = "Pointcept/Concerto",
        checkpoint_dir: str | None = None,
        grid_size: float = 0.02,
        enable_flash: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        if model_name != "concerto_small":
            raise ValueError(
                "This integration is dimensioned for concerto_small; "
                f"got model_name={model_name!r}."
            )
        try:
            import concerto
        except ImportError as exc:
            raise ImportError(
                "Concerto point-cloud prefixes require the official Concerto package and "
                "its Pointcept dependencies (spconv, torch-scatter, timm, addict, and "
                "huggingface-hub). Install https://github.com/Pointcept/Concerto in the "
                "OpenPI training environment."
            ) from exc

        custom_config = {
            "enable_flash": enable_flash,
            "enc_patch_size": [1024] * 5,
        }
        self.encoder = concerto.load(
            model_name,
            repo_id=repo_id,
            download_root=checkpoint_dir,
            custom_config=custom_config,
        )
        self.pointcloud_keys = pointcloud_keys
        self.grid_size = grid_size
        self.freeze_encoder = freeze_encoder
        self.resampler = PerceiverResampler(
            input_dim=self.feature_dim,
            output_dim=output_dim,
            num_tokens=tokens_per_camera,
        )
        self.camera_embeddings = nn.Parameter(
            torch.empty(len(pointcloud_keys), 1, output_dim)
        )
        nn.init.normal_(self.camera_embeddings, std=0.02)

        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _make_concerto_input(self, cloud: torch.Tensor) -> dict[str, torch.Tensor]:
        if cloud.ndim != 3 or cloud.shape[-1] < 3:
            raise ValueError(
                f"Expected point cloud [B, N, C>=3], got {tuple(cloud.shape)}."
            )
        coord = cloud[..., :3].to(torch.float32)
        coord_min = coord.amin(dim=1, keepdim=True)
        coord_max = coord.amax(dim=1, keepdim=True)
        shift = torch.cat(
            ((coord_min[..., :2] + coord_max[..., :2]) / 2.0, coord_min[..., 2:3]),
            dim=-1,
        )
        centered_coord = coord - shift
        grid_coord = torch.div(
            coord - coord_min, self.grid_size, rounding_mode="trunc"
        ).int()

        if cloud.shape[-1] >= 6:
            color = cloud[..., 3:6].to(torch.float32)
            # Accept either byte-like RGB or normalized RGB.
            byte_scale = color.detach().abs().amax() > 2.0
            color = torch.where(byte_scale, color / 255.0, color).clamp(0.0, 1.0)
        else:
            color = torch.zeros_like(coord)
        normal = torch.zeros_like(coord)
        feat = torch.cat((centered_coord, color, normal), dim=-1)

        batch_size, num_points = coord.shape[:2]
        batch = torch.arange(batch_size, device=coord.device).repeat_interleave(
            num_points
        )
        return {
            "coord": centered_coord.reshape(-1, 3),
            "grid_coord": grid_coord.reshape(-1, 3),
            "feat": feat.reshape(-1, 9),
            "batch": batch,
        }

    @staticmethod
    def _pad_encoded_points(
        point, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = torch.bincount(point.batch, minlength=batch_size)
        max_points = int(counts.max().item())
        features = point.feat.new_zeros((batch_size, max_points, point.feat.shape[-1]))
        valid = torch.zeros(
            (batch_size, max_points), dtype=torch.bool, device=point.feat.device
        )
        for batch_index in range(batch_size):
            current = point.feat[point.batch == batch_index]
            features[batch_index, : current.shape[0]] = current
            valid[batch_index, : current.shape[0]] = True
        return features, valid

    def forward(self, pointclouds: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.pointcloud_keys) - set(pointclouds)
        if missing:
            raise ValueError(
                f"Point-cloud observation missing camera keys: {sorted(missing)}."
            )

        token_blocks = []
        context = torch.no_grad if self.freeze_encoder else contextlib.nullcontext
        for camera_index, key in enumerate(self.pointcloud_keys):
            cloud = pointclouds[key]
            with context():
                encoded = self.encoder(self._make_concerto_input(cloud))
            features, valid = self._pad_encoded_points(encoded, cloud.shape[0])
            tokens = self.resampler(features, valid)
            token_blocks.append(tokens + self.camera_embeddings[camera_index])
        return torch.cat(token_blocks, dim=1)
