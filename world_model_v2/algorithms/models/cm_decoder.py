"""Lightweight consistency-style decoder with multi-scale latent injection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DownBlock, ResidualBlock, UpBlock
from .embeddings import timestep_embedding


class CMDecoder(nn.Module):
    """Decode noisy inputs toward lower-noise targets conditioned on latent grids."""

    def __init__(
        self,
        image_channels: int,
        latent_channels: int,
        hidden_channels: int,
        latent_dim: int,
    ) -> None:
        """Create a lightweight consistency-model-inspired decoder."""

        super().__init__()
        self.latent_dim = latent_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.latents_to_input = nn.Conv2d(latent_channels, image_channels, kernel_size=1)
        self.down1 = DownBlock(image_channels, hidden_channels, latent_dim)
        self.down2 = DownBlock(hidden_channels, hidden_channels * 2, latent_dim)
        self.mid = ResidualBlock(hidden_channels * 2, latent_dim)
        self.latents_to_mid = nn.Conv2d(latent_channels, hidden_channels, kernel_size=1)
        self.latents_to_low = nn.Conv2d(latent_channels, hidden_channels * 2, kernel_size=1)
        self.up1 = UpBlock(hidden_channels * 2, hidden_channels * 2, hidden_channels, latent_dim)
        self.up2 = UpBlock(hidden_channels, hidden_channels, hidden_channels, latent_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        external_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the lower-noise target image from a noisy image input."""

        time_cond = torch.cat(
            [
                timestep_embedding(t, self.latent_dim),
                timestep_embedding(s, self.latent_dim),
            ],
            dim=1,
        )
        cond = self.time_mlp(time_cond)
        latent_full = F.interpolate(
            self.latents_to_input(external_cond),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = x + latent_full
        x, skip1 = self.down1(x, cond)
        latent_mid = F.interpolate(
            self.latents_to_mid(external_cond),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = x + latent_mid
        x, skip2 = self.down2(x, cond)
        latent_low = F.interpolate(
            self.latents_to_low(external_cond),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.mid(x + latent_low, cond)
        x = self.up1(x, skip2, cond)
        x = self.up2(x, skip1, cond)
        return torch.sigmoid(self.out(x))

