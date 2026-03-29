"""Shared convolutional blocks for the lightweight consistency decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Apply FiLM-style conditioning inside a small residual block."""

    def __init__(self, channels: int, latent_dim: int) -> None:
        """Create a conditioned residual block."""

        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(latent_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Refine a feature map with residual conditioning."""

        scale, shift = self.cond(cond).chunk(2, dim=1)
        scale = scale.view(-1, x.shape[1], 1, 1)
        shift = shift.view(-1, x.shape[1], 1, 1)
        residual = x
        x = self.norm1(x)
        x = x * (1.0 + scale) + shift
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Downsample while preserving a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, latent_dim: int) -> None:
        """Create a conditioned downsampling block."""

        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block = ResidualBlock(out_channels, latent_dim)
        self.down = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the downsampled feature map and its skip tensor."""

        x = self.proj(x)
        x = self.block(x, cond)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsample while merging a skip feature."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        latent_dim: int,
    ) -> None:
        """Create a conditioned upsampling block."""

        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.merge = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.block = ResidualBlock(out_channels, latent_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Upsample and fuse the matching skip connection."""

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        return self.block(x, cond)

