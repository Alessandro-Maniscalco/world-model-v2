"""Minimal deterministic world model with encoder, dynamics, and decoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """Apply a small residual convolutional refinement."""

    def __init__(self, channels: int) -> None:
        """Build a same-shape residual block."""

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a residual update of the input activation map."""

        return x + self.block(x)


class MinimalEncoder(nn.Module):
    """Encode an image into a `32x32` latent map."""

    def __init__(self, latent_channels: int, hidden_channels: int) -> None:
        """Create the small downsampling encoder."""

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into latent feature maps."""

        return self.model(images)


class MinimalDynamics(nn.Module):
    """Predict the next latent map from the current latent map."""

    def __init__(self, latent_channels: int, hidden_channels: int) -> None:
        """Create the small latent transition network."""

        super().__init__()
        self.input_proj = nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.residual_stack = nn.Sequential(
            nn.SiLU(),
            ResidualConvBlock(hidden_channels),
            nn.SiLU(),
            ResidualConvBlock(hidden_channels),
            nn.SiLU(),
        )
        self.output_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Predict the next latent map with a residual update."""

        hidden = self.input_proj(latents)
        delta = self.output_proj(self.residual_stack(hidden))
        return latents + delta


class MinimalDecoder(nn.Module):
    """Decode a `32x32` latent map back into a `128x128` image."""

    def __init__(self, latent_channels: int, hidden_channels: int) -> None:
        """Create the small upsampling decoder."""

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent feature maps into bounded RGB images."""

        return torch.sigmoid(self.model(latents))


class MinimalWorldModel(nn.Module):
    """Bundle encoder, dynamics, and decoder into one minimal model."""

    def __init__(self, latent_channels: int = 4, hidden_channels: int = 64) -> None:
        """Create the minimal model with shared defaults."""

        super().__init__()
        self.encoder = MinimalEncoder(latent_channels=latent_channels, hidden_channels=hidden_channels)
        self.dynamics = MinimalDynamics(latent_channels=latent_channels, hidden_channels=hidden_channels)
        self.decoder = MinimalDecoder(latent_channels=latent_channels, hidden_channels=hidden_channels)

    def configure_trainability(self, mode: str) -> None:
        """Freeze or unfreeze submodules for the requested training mode."""

        if mode not in {"joint", "ae_only", "dynamics_only"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self._set_module_trainable(self.encoder, mode in {"joint", "ae_only"})
        self._set_module_trainable(self.decoder, mode in {"joint", "ae_only"})
        self._set_module_trainable(self.dynamics, mode in {"joint", "dynamics_only"})

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        """Toggle gradient tracking for one submodule."""

        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into latent maps."""

        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent maps into RGB images."""

        return self.decoder(latents)

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of images through the autoencoder."""

        return self.decode(self.encode(images))

    def predict_next_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Predict the next latent map from the current latent map."""

        return self.dynamics(latents)

    def predict_next_frame(self, images: torch.Tensor) -> torch.Tensor:
        """Predict the next frame from the current frame."""

        current_latents = self.encode(images)
        next_latents = self.predict_next_latent(current_latents)
        return self.decode(next_latents)

    def rollout(self, seed_frame: torch.Tensor, steps: int) -> torch.Tensor:
        """Autoregressively predict future frames from a single seed frame."""

        if steps < 0:
            raise ValueError("steps must be non-negative.")
        predicted_frames = [seed_frame]
        current_latents = self.encode(seed_frame)
        for _ in range(steps):
            current_latents = self.predict_next_latent(current_latents)
            predicted_frames.append(self.decode(current_latents))
        return torch.stack(predicted_frames, dim=1)
