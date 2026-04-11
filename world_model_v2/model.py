"""World model with a fixed Wan VAE and configurable rectified-flow dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn

from world_model_v2.dynamics_transformer import (
    DYNAMICS_FRAME_LAYOUT,
    DynamicsTransformerConfig,
    RectifiedFlowDynamics,
)
from world_model_v2.wan_vae import (
    WanVAEConfig,
    WanVAEDecoder,
    WanVAEEncoder,
    kl_divergence_from_moments,
    sample_posterior as sample_posterior_latent,
)


@dataclass
class AutoencoderOutput:
    """Bundle reconstructions and posterior statistics for one AE pass."""

    reconstructed: torch.Tensor
    latent: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    kl_loss: torch.Tensor


class WorldModel(nn.Module):
    """Bundle the fixed Wan VAE with the rectified-flow dynamics model."""

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 64,
        ae_backend: str = "wan",
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        dynamics_infer_steps: int = 16,
        dynamics_train_timesteps: int = 1000,
        dynamics_rf_shift: float = 5.0,
        conditional_frame_timestep: float = -1.0,
        conditional_frame_sigma: float = 0.0,
        dynamics_video_condition_dropout: float = 0.0,
        dynamics_guidance_scale: float = 0.0,
        dynamics_context_frames: int = DYNAMICS_FRAME_LAYOUT.context_frames,
        dynamics_target_frames: int = DYNAMICS_FRAME_LAYOUT.target_frames,
        dynamics_conditioning_frame_choices: tuple[int, ...] | list[int] | None = None,
        dynamics_conditioning_frame_probabilities: tuple[float, ...] | list[float] | None = None,
        dynamics_validation_conditioning_frame_choices: tuple[int, ...] | list[int] | None = None,
        dynamics_open_rollout_context_frames: int | None = None,
        dynamics_open_rollout_stride_frames: int | None = None,
        dynamics_model_channels: int = 256,
        dynamics_num_blocks: int = 4,
        dynamics_num_heads: int = 4,
        dynamics_action_dim: int = 4,
        dynamics_action_conditioning_mode: str = "chunk_per_frame",
        dynamics_zero_init_action_embedder: bool = False,
        dynamics_use_adaln_lora: bool = True,
        dynamics_adaln_lora_dim: int = 64,
        dynamics_rope_t_extrapolation_ratio: float = 1.0,
        dynamics_use_learned_temporal_embedding: bool = False,
    ) -> None:
        """Create the world model around the Wan autoencoder path."""

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
        latent_height = self.image_height // self.spatial_downsample_factor
        latent_width = self.image_width // self.spatial_downsample_factor
        dynamics_patch_spatial = (
            2
            if latent_height >= 2
            and latent_width >= 2
            and latent_height % 2 == 0
            and latent_width % 2 == 0
            else 1
        )
        self.dynamics_backend = "rf_dit"
        self.dynamics = RectifiedFlowDynamics(
            DynamicsTransformerConfig(
                max_img_h=latent_height,
                max_img_w=latent_width,
                context_frames=dynamics_context_frames,
                target_frames=dynamics_target_frames,
                conditioning_frame_choices=dynamics_conditioning_frame_choices,
                conditioning_frame_probabilities=dynamics_conditioning_frame_probabilities,
                validation_conditioning_frame_choices=dynamics_validation_conditioning_frame_choices,
                open_rollout_context_frames=dynamics_open_rollout_context_frames,
                open_rollout_stride_frames=dynamics_open_rollout_stride_frames,
                in_channels=latent_channels,
                out_channels=latent_channels,
                patch_spatial=dynamics_patch_spatial,
                patch_temporal=1,
                model_channels=dynamics_model_channels,
                num_blocks=dynamics_num_blocks,
                num_heads=dynamics_num_heads,
                concat_padding_mask=False,
                pos_emb_cls="rope3d",
                pos_emb_learnable=False,
                pos_emb_interpolation="crop",
                use_adaln_lora=dynamics_use_adaln_lora,
                adaln_lora_dim=dynamics_adaln_lora_dim,
                atten_backend="torch",
                extra_per_block_abs_pos_emb=False,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=dynamics_rope_t_extrapolation_ratio,
                use_learned_temporal_embedding=dynamics_use_learned_temporal_embedding,
                action_dim=dynamics_action_dim,
                action_conditioning_mode=dynamics_action_conditioning_mode,
                zero_init_action_embedder=dynamics_zero_init_action_embedder,
                conditional_frame_timestep=conditional_frame_timestep,
                conditional_frame_sigma=conditional_frame_sigma,
                dynamics_infer_steps=dynamics_infer_steps,
                dynamics_train_timesteps=dynamics_train_timesteps,
                dynamics_rf_shift=dynamics_rf_shift,
                dynamics_video_condition_dropout=dynamics_video_condition_dropout,
                dynamics_guidance_scale=dynamics_guidance_scale,
            )
        )

    def _validate_autoencoder_backend(self, ae_backend: str) -> None:
        """Reject removed autoencoder backends with a clear migration hint."""

        if ae_backend != "wan":
            raise ValueError(
                f"Unsupported autoencoder backend: {ae_backend}. "
                "This codepath now only supports the Wan VAE."
            )

    @property
    def spatial_downsample_factor(self) -> int:
        """Return the Wan VAE spatial compression factor."""

        return self.wan_cfg.spatial_downsample_factor()

    def autoencoder_config(self) -> dict[str, Any]:
        """Return the serializable autoencoder backend metadata."""

        return {"backend": self.ae_backend, "config": self.backend_config}

    def dynamics_config(self) -> dict[str, Any]:
        """Return the serializable dynamics backend metadata."""

        return {"backend": self.dynamics_backend, "config": self.dynamics.to_dict()}

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
    ) -> AutoencoderOutput:
        """Run one AE pass and return reconstructions plus posterior statistics."""

        mu, log_var = self.encode_posterior(images)
        latent = mu if not sample_posterior else sample_posterior_latent(mu, log_var)
        reconstructed = self.decode(latent)
        kl_loss = kl_divergence_from_moments(mu, log_var)
        return AutoencoderOutput(
            reconstructed=reconstructed,
            latent=latent,
            mu=mu,
            log_var=log_var,
            kl_loss=kl_loss,
        )

    def reconstruct(self, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Reconstruct a batch of images through the active autoencoder."""

        return self.decode(self.encode(images, deterministic=deterministic))

    def encode_frame_sequence(
        self,
        images: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Encode an image sequence into a latent-video tensor."""

        if images.ndim != 5:
            raise ValueError(
                f"Expected image sequences with shape (B, T, C, H, W), received {tuple(images.shape)}."
            )
        batch_size, frames = images.shape[:2]
        flat_images = rearrange(images, "b t c h w -> (b t) c h w")
        flat_latents = self.encode(flat_images, deterministic=deterministic)
        return rearrange(
            flat_latents,
            "(b t) c h w -> b c t h w",
            b=batch_size,
            t=frames,
            c=flat_latents.shape[1],
            h=flat_latents.shape[2],
            w=flat_latents.shape[3],
        )

    def decode_frame_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a latent-video tensor into an image sequence."""

        if latents.ndim != 5:
            raise ValueError(
                f"Expected latent sequences with shape (B, C, T, H, W), received {tuple(latents.shape)}."
            )
        batch_size, _, frames, _, _ = latents.shape
        flat_latents = rearrange(latents, "b c t h w -> (b t) c h w")
        flat_images = self.decode(flat_latents)
        return rearrange(
            flat_images,
            "(b t) c h w -> b t c h w",
            b=batch_size,
            t=frames,
            c=flat_images.shape[1],
            h=flat_images.shape[2],
            w=flat_images.shape[3],
        )

    def encode_context_frames(
        self,
        images: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Encode a supported conditioning image context into a latent video tensor."""

        if images.ndim != 5:
            raise ValueError(
                f"Expected context images with shape (B, T, C, H, W), received {tuple(images.shape)}."
            )
        if images.shape[1] not in self.dynamics.cfg.conditioning_frame_choices:
            raise ValueError(
                f"Expected one of the supported context image frame counts "
                f"{self.dynamics.cfg.conditioning_frame_choices}, received {images.shape[1]}."
            )
        return self.encode_frame_sequence(images, deterministic=deterministic)

    def predict_next_latent(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor | None = None,
        infer_steps: int | None = None,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Predict the next latent chunk from one supported clean context length."""

        return self.dynamics.sample_next_latent(
            context_latent=latents,
            actions=actions,
            infer_steps=infer_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )

    def predict_next_frame(
        self,
        images: torch.Tensor,
        actions: torch.Tensor | None = None,
        infer_steps: int | None = None,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Predict the next frame chunk from a supported conditioning image context."""

        current_latents = self.encode_context_frames(images, deterministic=True)
        next_latents = self.predict_next_latent(
            current_latents,
            actions=actions,
            infer_steps=infer_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )
        return self.decode_frame_sequence(next_latents)

    def resolved_rollout_stride_frames(
        self,
        context_frames: int,
        stride_frames: int | None = None,
    ) -> int:
        """Resolve how many newly predicted frames a rollout chunk should append."""

        if context_frames < 1 or context_frames >= self.dynamics.cfg.max_frames:
            raise ValueError(
                "context_frames must stay within "
                f"[1, {self.dynamics.cfg.max_frames - 1}], received {context_frames}."
            )
        resolved_stride_frames = (
            self.dynamics.cfg.open_rollout_stride_frames
            if stride_frames is None
            else int(stride_frames)
        )
        if resolved_stride_frames is not None and resolved_stride_frames < 1:
            raise ValueError("stride_frames must be positive when provided.")
        chunk_target_capacity = self.dynamics.cfg.max_frames - context_frames
        return chunk_target_capacity if resolved_stride_frames is None else resolved_stride_frames

    def rollout(
        self,
        seed_frames: torch.Tensor,
        steps: int,
        actions: torch.Tensor | None = None,
        stride_frames: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively predict a requested number of future frames."""

        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if seed_frames.ndim != 5:
            raise ValueError(
                f"Expected seed frames with shape (B, T, C, H, W), received {tuple(seed_frames.shape)}."
            )
        seed_context_frames = int(seed_frames.shape[1])
        if seed_context_frames not in self.dynamics.cfg.conditioning_frame_choices:
            raise ValueError(
                "Expected one of the supported seed frame counts "
                f"{self.dynamics.cfg.conditioning_frame_choices}, received {seed_context_frames}."
            )
        if actions is not None:
            expected_action_steps = max(seed_context_frames - 1 + steps, 0)
            if actions.ndim != 3:
                raise ValueError(
                    "Expected rollout actions with shape "
                    f"(B, {expected_action_steps}, {self.dynamics.cfg.action_dim}), "
                    f"received {tuple(actions.shape)}."
                )
            if actions.shape[0] != seed_frames.shape[0]:
                raise ValueError(
                    f"Expected rollout action batch size {seed_frames.shape[0]}, received {actions.shape[0]}."
                )
            if actions.shape[1] != expected_action_steps:
                raise ValueError(
                    f"Expected {expected_action_steps} rollout action steps, received {actions.shape[1]}."
                )
            if actions.shape[2] != self.dynamics.cfg.action_dim:
                raise ValueError(
                    f"Expected rollout action dim {self.dynamics.cfg.action_dim}, received {actions.shape[2]}."
                )
        predicted_frames = [seed_frames[:, frame_index] for frame_index in range(seed_frames.shape[1])]
        full_rollout_latents = self.encode_context_frames(seed_frames, deterministic=True)
        generated_frames = 0
        generator = torch.Generator(device=seed_frames.device.type)
        generator.manual_seed(0)
        while generated_frames < steps:
            available_frames = int(full_rollout_latents.shape[2])
            rollout_context_limit = min(
                available_frames,
                int(self.dynamics.cfg.open_rollout_context_frames),
            )
            current_context_frames = max(
                conditioning_frames
                for conditioning_frames in self.dynamics.cfg.conditioning_frame_choices
                if conditioning_frames <= rollout_context_limit
            )
            current_latents = full_rollout_latents[:, :, -current_context_frames:]
            chunk_target_capacity = self.dynamics.cfg.max_frames - current_context_frames
            stride = WorldModel.resolved_rollout_stride_frames(
                self,
                current_context_frames,
                stride_frames=stride_frames,
            )
            chunk_target_frames = min(chunk_target_capacity, stride, steps - generated_frames)
            action_window = None
            if actions is not None:
                action_start = available_frames - current_context_frames
                action_stop = action_start + self.dynamics.cfg.num_action_per_chunk
                action_window = actions[:, action_start:min(action_stop, int(actions.shape[1]))]
                if action_window.shape[1] < self.dynamics.cfg.num_action_per_chunk:
                    pad_actions = torch.zeros(
                        action_window.shape[0],
                        self.dynamics.cfg.num_action_per_chunk - action_window.shape[1],
                        self.dynamics.cfg.action_dim,
                        device=actions.device,
                        dtype=actions.dtype,
                    )
                    action_window = torch.cat([action_window, pad_actions], dim=1)
            next_latents = self.predict_next_latent(
                current_latents,
                actions=action_window,
                generator=generator,
            )
            next_latents = next_latents[:, :, :chunk_target_frames]
            next_frames = self.decode_frame_sequence(next_latents)
            for frame_index in range(next_frames.shape[1]):
                predicted_frames.append(next_frames[:, frame_index])
            full_rollout_latents = torch.cat([full_rollout_latents, next_latents], dim=2)
            generated_frames += int(next_frames.shape[1])
        return torch.stack(predicted_frames, dim=1)
