"""Tests for shared decoder blocks."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.models.blocks import DownBlock, ResidualBlock, UpBlock


def test_residual_block_preserves_spatial_shape() -> None:
    """Residual blocks should preserve input shape."""

    block = ResidualBlock(channels=32, latent_dim=64)
    x = torch.rand(2, 32, 16, 16)
    cond = torch.rand(2, 64)
    y = block(x, cond)
    assert y.shape == x.shape


def test_down_and_up_blocks_round_trip_shapes() -> None:
    """Down and up blocks should produce matching skip-compatible shapes."""

    down = DownBlock(in_channels=3, out_channels=32, latent_dim=64)
    up = UpBlock(in_channels=32, skip_channels=32, out_channels=16, latent_dim=64)
    x = torch.rand(2, 3, 32, 32)
    cond = torch.rand(2, 64)
    low, skip = down(x, cond)
    restored = up(low, skip, cond)
    assert low.shape == (2, 32, 16, 16)
    assert restored.shape == (2, 16, 32, 32)

