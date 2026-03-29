"""CNN encoder for Stage-1 latent world model training."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Encode RGB observations into a compact 2D latent grid."""

    def __init__(self, image_channels: int, latent_channels: int, hidden_channels: int) -> None:
        """Create a two-downsample CNN encoder."""

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a latent feature map."""

        return self.model(images)

