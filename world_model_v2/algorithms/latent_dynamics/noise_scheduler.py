"""Minimal noise scheduling utilities for image and latent denoising stages."""

from __future__ import annotations

import torch


class Stage1NoiseScheduler:
    """Add Gaussian noise and expose lightweight consistency-style update helpers."""

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
        batch_size: int | tuple[int, ...],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample timestep pairs with `s < t` for an arbitrary leading shape."""

        if isinstance(batch_size, int):
            shape = (batch_size,)
        else:
            shape = batch_size
        t = torch.randint(1, self.timesteps, shape, device=device)
        s = torch.floor(torch.rand(shape, device=device) * t.float()).long()
        return t, s

    def add_noise(self, clean_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add per-sample Gaussian noise at the selected timesteps."""

        return self.add_noise_with_shared_base(clean_images, timesteps, torch.randn_like(clean_images))

    def add_noise_with_shared_base(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Add per-sample Gaussian noise using a caller-provided shared noise tensor."""

        sigma = self.sigmas[timesteps].to(clean_images.device)
        while sigma.ndim < clean_images.ndim:
            sigma = sigma.unsqueeze(-1)
        return clean_images + sigma * noise

    def add_noise_to_t_s(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        stop_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise at two levels using the same Gaussian base sample."""

        noise = torch.randn_like(clean_images)
        return (
            self.add_noise_with_shared_base(clean_images, timesteps, noise),
            self.add_noise_with_shared_base(clean_images, stop_timesteps, noise),
        )

    def sample_stage2_noise_levels(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device | str,
        sampling_strategy: str,
        prev_frame_noise_scale: float,
        dyn_infer_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a full-window Stage-2 timestep layout similar to the upstream code."""

        if horizon < 2:
            raise ValueError("Stage 2 training requires horizon >= 2.")
        if prev_frame_noise_scale <= 0.0:
            raise ValueError("prev_frame_noise_scale must be positive.")

        max_prev_timestep = min(self.timesteps, max(2, int(self.timesteps * prev_frame_noise_scale)))
        prev_timesteps = torch.randint(
            1,
            max_prev_timestep,
            (batch_size, horizon - 1),
            device=device,
        )

        if sampling_strategy == "uniform":
            terminal_t = torch.randint(2, self.timesteps, (batch_size,), device=device)
            if dyn_infer_steps == 1:
                terminal_s = (
                    torch.floor(torch.rand(batch_size, device=device) * (terminal_t.float() - 1.0)).long()
                    + 1
                )
            else:
                intermediate = torch.linspace(
                    0,
                    self.timesteps - 1,
                    steps=dyn_infer_steps + 1,
                    device=device,
                ).round().long()[1:-1]
                if intermediate.numel() == 0:
                    raise ValueError("dyn_infer_steps must be at least 1.")
                terminal_s = intermediate[torch.randint(intermediate.numel(), (batch_size,), device=device)]
        elif sampling_strategy == "terminal_only":
            terminal_t = torch.full((batch_size,), self.timesteps - 1, device=device, dtype=torch.long)
            if dyn_infer_steps == 1:
                terminal_s = torch.zeros(batch_size, device=device, dtype=torch.long)
            else:
                intermediate = torch.linspace(
                    0,
                    self.timesteps - 1,
                    steps=dyn_infer_steps + 1,
                    device=device,
                ).round().long()[1:-1]
                if intermediate.numel() == 0:
                    raise ValueError("dyn_infer_steps must be at least 1.")
                terminal_s = intermediate[torch.randint(intermediate.numel(), (batch_size,), device=device)]
        else:
            raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

        return (
            torch.cat([prev_timesteps, terminal_t.unsqueeze(1)], dim=1),
            torch.cat([prev_timesteps, terminal_s.unsqueeze(1)], dim=1),
        )

    def get_weights(
        self,
        timesteps: torch.Tensor,
        weighting: str = "inverse_variance",
    ) -> torch.Tensor:
        """Return a loss-weight tensor using the requested weighting rule."""

        sigma = self.sigmas[timesteps].to(timesteps.device)
        if weighting == "inverse_variance":
            return 1.0 / (sigma.square() + 1e-6)
        if weighting == "uniform":
            return torch.ones_like(sigma)
        raise ValueError(f"Unsupported loss weighting: {weighting}")

    def ctm_ratio(
        self,
        timesteps: torch.Tensor,
        stop_timesteps: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the broadcastable interpolation ratio for a consistency update."""

        safe_timesteps = timesteps.clamp_min(1).to(dtype=dtype)
        ratio = stop_timesteps.to(dtype=dtype) / safe_timesteps
        return torch.where(timesteps == stop_timesteps, torch.ones_like(ratio), ratio)

    def ctm_calc_out(
        self,
        sample: torch.Tensor,
        predicted_clean: torch.Tensor,
        timesteps: torch.Tensor,
        stop_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Blend a model's clean prediction back toward the requested stop timestep."""

        ratio = self.ctm_ratio(
            timesteps,
            stop_timesteps,
            dtype=sample.dtype,
        )
        while ratio.ndim < sample.ndim:
            ratio = ratio.unsqueeze(-1)
        blended = sample * ratio + predicted_clean * (1.0 - ratio)
        mask = timesteps == stop_timesteps
        while mask.ndim < sample.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, sample, blended)

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
