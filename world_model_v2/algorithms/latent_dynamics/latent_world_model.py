"""Stage-1 latent world model with upstream-shaped responsibilities."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_v2.algorithms.latent_dynamics.noise_scheduler import Stage1NoiseScheduler
from world_model_v2.algorithms.models.cm_decoder import CMDecoder
from world_model_v2.algorithms.models.cnn_encoder import CNNEncoder
from world_model_v2.config import AlgorithmConfig


class LatentWorldModel(nn.Module):
    """Own Stage-1 encoding, decoding, training loss, and validation behavior."""

    def __init__(self, cfg: AlgorithmConfig, obs_keys: tuple[str, ...]) -> None:
        """Create the Stage-1 latent world model."""

        super().__init__()
        if cfg.training_stage != 1:
            raise NotImplementedError(
                "Only training_stage=1 is implemented in this refactor."
            )
        self.cfg = cfg
        self.obs_keys = tuple(obs_keys)
        self.image_channels = 3 * len(self.obs_keys)
        self.encoder = CNNEncoder(
            image_channels=self.image_channels,
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
        )
        self.decoder = CMDecoder(
            image_channels=self.image_channels,
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
            latent_dim=cfg.latent_dim,
        )
        self.noise_scheduler = Stage1NoiseScheduler(
            timesteps=cfg.timesteps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
        )

    def to(self, *args: object, **kwargs: object) -> "LatentWorldModel":
        """Move the model and its noise scheduler to the requested device."""

        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.noise_scheduler.to(device)
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and apply channel-wise latent normalization."""

        latents = self.encoder(images)
        norms = torch.linalg.vector_norm(latents, dim=1, keepdim=True).clamp_min(1e-6)
        return latents / norms

    def decode(
        self,
        noisy_images: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode noisy images toward their lower-noise targets."""

        return self.decoder(noisy_images, t, s, latents)

    def concatenate_observations(
        self,
        obs_batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        """Concatenate configured observations across the channel dimension."""

        views = [obs_batch[key] for key in self.obs_keys]
        if not views:
            raise ValueError("No observations were provided.")
        had_batch = views[0].ndim == 5
        if not had_batch:
            views = [view.unsqueeze(0) for view in views]
        combined = torch.cat(views, dim=2)
        return combined.float(), had_batch

    def training_step(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        """Compute the Stage-1 weighted reconstruction loss for one batch."""

        obs, _ = self.concatenate_observations(batch["obs"])  # type: ignore[arg-type]
        flat_images = obs.reshape(-1, obs.shape[2], obs.shape[3], obs.shape[4])
        latents = self.encode(flat_images)
        t, s = self.noise_scheduler.sample_training_pair(
            flat_images.shape[0],
            flat_images.device,
        )
        noisy_t = self.noise_scheduler.add_noise(flat_images, t)
        target_s = self.noise_scheduler.add_noise(flat_images, s)
        pred_s = self.decode(noisy_t, t, s, latents)
        weights = self.noise_scheduler.get_weights(t).view(-1, 1, 1, 1)
        recon_loss = ((pred_s - target_s) ** 2 * weights).mean()
        clean_pred = self.decode(
            flat_images,
            torch.zeros_like(t),
            torch.zeros_like(s),
            latents,
        )
        clean_loss = F.mse_loss(clean_pred, flat_images)
        loss = recon_loss + 0.1 * clean_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "clean_loss": clean_loss.detach(),
            "latents": latents.detach(),
        }

    @torch.no_grad()
    def reconstruct(
        self,
        obs_batch: Mapping[str, torch.Tensor],
        num_steps: int = 4,
        start_mode: str = "noisy-input",
    ) -> torch.Tensor:
        """Reconstruct a batch of observation sequences by iterative denoising."""

        obs, had_batch = self.concatenate_observations(obs_batch)
        batch_size, horizon, channels, height, width = obs.shape
        flat_images = obs.reshape(-1, channels, height, width)
        latents = self.encode(flat_images)
        if start_mode == "noise":
            current = torch.rand_like(flat_images)
        elif start_mode == "noisy-input":
            max_t = torch.full(
                (flat_images.shape[0],),
                self.noise_scheduler.timesteps - 1,
                device=flat_images.device,
                dtype=torch.long,
            )
            current = self.noise_scheduler.add_noise(flat_images, max_t)
        else:
            raise ValueError(f"Unsupported start_mode: {start_mode}")

        schedule = self.noise_scheduler.make_sampling_schedule(num_steps, flat_images.device)
        for start, end in schedule:
            t = start.expand(flat_images.shape[0])
            s = end.expand(flat_images.shape[0])
            current = self.decode(current, t, s, latents)

        reconstructed = current.reshape(batch_size, horizon, channels, height, width)
        if had_batch:
            return reconstructed.clamp(0.0, 1.0)
        return reconstructed.squeeze(0).clamp(0.0, 1.0)

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict[str, object],
        num_steps: int = 4,
        start_mode: str = "noisy-input",
    ) -> dict[str, object]:
        """Reconstruct a validation sample and return tensors plus summary stats."""

        obs, _ = self.concatenate_observations(batch["obs"])  # type: ignore[arg-type]
        reconstructed = self.reconstruct(
            batch["obs"],  # type: ignore[arg-type]
            num_steps=num_steps,
            start_mode=start_mode,
        )
        if reconstructed.ndim == 4:
            reconstructed = reconstructed.unsqueeze(0)
        latent_shape = list(self.encode(obs[:, :1].reshape(-1, obs.shape[2], obs.shape[3], obs.shape[4])).shape)
        stats = {
            "episode": int(torch.as_tensor(batch["episode_idx"]).reshape(-1)[0].item()),
            "input_frame_count": int(obs.shape[1]),
            "decoded_frame_count": int(reconstructed.shape[1]),
            "latent_shape": latent_shape,
            "action_shape": list(torch.as_tensor(batch["action"]).shape[-2:]),
            "start_mode": start_mode,
            "num_steps": num_steps,
        }
        return {
            "original": obs[0].detach().cpu(),
            "reconstructed": reconstructed[0].detach().cpu(),
            "stats": stats,
        }

