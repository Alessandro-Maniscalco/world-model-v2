"""Minimal world model with a fixed Wan VAE and latent dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from world_model_v2.minimal.wan_vae import (
    WanVAEConfig,
    WanVAEDecoder,
    WanVAEEncoder,
    kl_divergence_from_moments,
    sample_posterior as sample_posterior_latent,
)


@dataclass
class MinimalAutoencoderOutput:
    """Bundle reconstructions and posterior statistics for one AE pass."""

    reconstructed: torch.Tensor
    latent: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    kl_loss: torch.Tensor


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


class MinimalWorldModel(nn.Module):
    """Bundle the fixed Wan VAE with the minimal latent dynamics model."""

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        ae_backend: str = "wan",
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Create the minimal model around the Wan autoencoder path."""

        super().__init__()
        self._validate_autoencoder_backend(ae_backend)
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.ae_backend = "wan"
        self.resolution = resolution
        self.height = height
        self.width = width
        self.image_height = resolution if height is None else height
        self.image_width = resolution if width is None else width
        self.wan_cfg = WanVAEConfig(z_dim=latent_channels)
        self.encoder = WanVAEEncoder(self.wan_cfg)
        self.decoder = WanVAEDecoder(self.wan_cfg)
        self.backend_config = self.wan_cfg.to_dict()
        if self.image_height % self.spatial_downsample_factor != 0:
            raise ValueError(
                f"Image height {self.image_height} is incompatible with backend {self.ae_backend} "
                f"and downsample factor {self.spatial_downsample_factor}."
            )
        if self.image_width % self.spatial_downsample_factor != 0:
            raise ValueError(
                f"Image width {self.image_width} is incompatible with backend {self.ae_backend} "
                f"and downsample factor {self.spatial_downsample_factor}."
            )
        self.dynamics = MinimalDynamics(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
        )

    def _validate_autoencoder_backend(self, ae_backend: str) -> None:
        """Reject removed autoencoder backends with a clear migration hint."""

        if ae_backend != "wan":
            raise ValueError(
                f"Unsupported autoencoder backend: {ae_backend}. "
                "The minimal path now only supports the Wan VAE."
            )

    @property
    def spatial_downsample_factor(self) -> int:
        """Return the Wan VAE spatial compression factor."""

        return self.wan_cfg.spatial_downsample_factor()

    def autoencoder_config(self) -> dict[str, Any]:
        """Return the serializable autoencoder backend metadata."""

        return {"backend": self.ae_backend, "config": self.backend_config}

    def configure_trainability(self, mode: str) -> None:
        """Freeze or unfreeze submodules for the requested training mode."""

        if mode not in {"ae_only", "dynamics_only"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self._set_module_trainable(self.encoder, mode == "ae_only")
        self._set_module_trainable(self.decoder, mode == "ae_only")
        self._set_module_trainable(self.dynamics, mode == "dynamics_only")

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        """Toggle gradient tracking for one submodule."""

        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def encode_posterior(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior moments from the Wan encoder."""

        return self.encoder(images)

    def encode(self, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode images into latent maps using mean or sampled latents."""

        mu, log_var = self.encode_posterior(images)
        if deterministic:
            return mu
        return sample_posterior_latent(mu, log_var)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent maps into RGB images."""

        return self.decoder(latents)

    def autoencode(
        self,
        images: torch.Tensor,
        sample_posterior: bool,
    ) -> MinimalAutoencoderOutput:
        """Run one AE pass and return reconstructions plus posterior statistics."""

        mu, log_var = self.encode_posterior(images)
        latent = mu if not sample_posterior else sample_posterior_latent(mu, log_var)
        reconstructed = self.decode(latent)
        kl_loss = kl_divergence_from_moments(mu, log_var)
        return MinimalAutoencoderOutput(
            reconstructed=reconstructed,
            latent=latent,
            mu=mu,
            log_var=log_var,
            kl_loss=kl_loss,
        )

    def reconstruct(self, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Reconstruct a batch of images through the active autoencoder."""

        return self.decode(self.encode(images, deterministic=deterministic))

    def predict_next_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Predict the next latent map from the current latent map."""

        return self.dynamics(latents)

    def predict_next_frame(self, images: torch.Tensor) -> torch.Tensor:
        """Predict the next frame from the current frame."""

        current_latents = self.encode(images, deterministic=True)
        next_latents = self.predict_next_latent(current_latents)
        return self.decode(next_latents)

    def rollout(self, seed_frame: torch.Tensor, steps: int) -> torch.Tensor:
        """Autoregressively predict future frames from a single seed frame."""

        if steps < 0:
            raise ValueError("steps must be non-negative.")
        predicted_frames = [seed_frame]
        current_latents = self.encode(seed_frame, deterministic=True)
        for _ in range(steps):
            current_latents = self.predict_next_latent(current_latents)
            predicted_frames.append(self.decode(current_latents))
        return torch.stack(predicted_frames, dim=1)
