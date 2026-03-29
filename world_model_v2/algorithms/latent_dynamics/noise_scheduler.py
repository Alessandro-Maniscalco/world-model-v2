"""Minimal noise scheduling utilities for Stage-1 consistency training."""

from __future__ import annotations

import torch


class Stage1NoiseScheduler:
    """Add Gaussian noise and sample timestep pairs for Stage-1 training."""

    def __init__(
        self,
        timesteps: int = 32,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
    ) -> None:
        """Configure a simple monotonic noise schedule."""

        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigmas = torch.linspace(sigma_min, sigma_max, timesteps)

    def to(self, device: torch.device | str) -> "Stage1NoiseScheduler":
        """Move scheduler tensors onto a target device."""

        self.sigmas = self.sigmas.to(device)
        return self

    def sample_training_pair(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timestep pairs with `s < t`."""

        t = torch.randint(1, self.timesteps, (batch_size,), device=device)
        s = torch.floor(torch.rand(batch_size, device=device) * t.float()).long()
        return t, s

    def add_noise(self, clean_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add per-sample Gaussian noise at the selected timesteps."""

        sigma = self.sigmas[timesteps].view(-1, 1, 1, 1).to(clean_images.device)
        noise = torch.randn_like(clean_images)
        return clean_images + sigma * noise

    def get_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return a simple inverse-variance loss weight."""

        sigma = self.sigmas[timesteps].to(timesteps.device)
        return 1.0 / (sigma.square() + 1e-6)

    def make_sampling_schedule(
        self,
        num_steps: int,
        device: torch.device | str,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build a descending timestep schedule ending at zero noise."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        levels = torch.linspace(
            self.timesteps - 1,
            0,
            steps=num_steps + 1,
            device=device,
        )
        levels = torch.round(levels).long()
        schedule: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start, end in zip(levels[:-1], levels[1:], strict=False):
            schedule.append((start.view(1), end.view(1)))
        return schedule

