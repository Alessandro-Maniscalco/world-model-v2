"""World model that pairs a Wan2.1-style temporal VAE with DreamDojo-style RF dynamics."""

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
    WanPosteriorEncoder,
    WanVAEConfig,
    WanVideoDecoder,
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


@dataclass(frozen=True)
class LatentNormalizationStats:
    """Store DreamDojo-style latent normalization statistics."""

    img_mean: tuple[float, ...]
    img_std: tuple[float, ...]
    video_mean: tuple[float, ...]
    video_std: tuple[float, ...]

    @classmethod
    def identity(cls, channels: int) -> "LatentNormalizationStats":
        """Return identity normalization for one latent-channel count."""

        zeros = tuple(0.0 for _ in range(channels))
        ones = tuple(1.0 for _ in range(channels))
        return cls(
            img_mean=zeros,
            img_std=ones,
            video_mean=zeros,
            video_std=ones,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        channels: int,
    ) -> "LatentNormalizationStats":
        """Build stats from checkpoint metadata or fall back to identity stats."""

        if payload is None:
            return cls.identity(channels)

        def _read_vector(key: str, default: float) -> tuple[float, ...]:
            """Read one per-channel vector from serialized metadata."""

            raw = payload.get(key)
            if raw is None:
                return tuple(default for _ in range(channels))
            if not isinstance(raw, (list, tuple)):
                raise TypeError(f"{key} must be a list of floats in latent normalization stats.")
            values = tuple(float(value) for value in raw)
            if len(values) != channels:
                raise ValueError(
                    f"{key} must contain {channels} values, received {len(values)}."
                )
            return values

        return cls(
            img_mean=_read_vector("img_mean", 0.0),
            img_std=_read_vector("img_std", 1.0),
            video_mean=_read_vector("video_mean", 0.0),
            video_std=_read_vector("video_std", 1.0),
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Return a JSON-serializable normalization payload."""

        return {
            "img_mean": list(self.img_mean),
            "img_std": list(self.img_std),
            "video_mean": list(self.video_mean),
            "video_std": list(self.video_std),
        }


class WorldModel(nn.Module):
    """Bundle the fixed Wan temporal tokenizer with the rectified-flow dynamics model."""

    def __init__(
        self,
        latent_channels: int = 32,
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
        wan_config: WanVAEConfig | None = None,
        latent_normalization_stats: LatentNormalizationStats | dict[str, Any] | None = None,
    ) -> None:
        """Create the world model around the Wan temporal autoencoder path."""

        super().__init__()
        self._validate_autoencoder_backend(ae_backend)
        self.hidden_channels = hidden_channels
        self.ae_backend = "wan"
        self.resolution = resolution
        self.height = height
        self.width = width
        self.image_height = resolution if height is None else height
        self.image_width = resolution if width is None else width
        self.wan_cfg = wan_config if wan_config is not None else WanVAEConfig(z_dim=latent_channels)
        self.latent_channels = self.wan_cfg.z_dim
        self.encoder = WanPosteriorEncoder(self.wan_cfg)
        self.decoder = WanVideoDecoder(self.wan_cfg)
        self.backend_config = self.wan_cfg.to_dict()
        self.normalization_stats = (
            latent_normalization_stats
            if isinstance(latent_normalization_stats, LatentNormalizationStats)
            else LatentNormalizationStats.from_dict(latent_normalization_stats, self.latent_channels)
        )
        self.register_buffer(
            "img_latent_mean",
            torch.tensor(self.normalization_stats.img_mean, dtype=torch.float32).view(1, self.latent_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_latent_std",
            torch.tensor(self.normalization_stats.img_std, dtype=torch.float32).view(1, self.latent_channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "video_latent_mean",
            torch.tensor(self.normalization_stats.video_mean, dtype=torch.float32).view(
                1, self.latent_channels, 1, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "video_latent_std",
            torch.tensor(self.normalization_stats.video_std, dtype=torch.float32).view(
                1, self.latent_channels, 1, 1, 1
            ),
            persistent=False,
        )
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
        # Keep the first DiT at full latent resolution because the Wan tokenizer
        # already performs the spatial compression.
        dynamics_patch_spatial = 1
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
                temporal_compression_ratio=self.temporal_downsample_factor,
                in_channels=self.latent_channels,
                out_channels=self.latent_channels,
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

    @property
    def temporal_downsample_factor(self) -> int:
        """Return the Wan VAE temporal compression factor."""

        return self.wan_cfg.temporal_downsample_factor()

    def pixel_frames_to_latent_frames(self, pixel_frames: int, *, exact: bool = False) -> int:
        """Return the latent count represented by one pixel-frame count."""

        if exact:
            return self.wan_cfg.exact_latent_frames_for_pixels(pixel_frames)
        return self.wan_cfg.pixel_frames_to_latent_frames(pixel_frames)

    def latent_frames_to_pixel_frames(self, latent_frames: int) -> int:
        """Return the pixel-frame count represented by one latent-frame count."""

        return self.wan_cfg.latent_frames_to_pixel_frames(latent_frames)

    def autoencoder_config(self) -> dict[str, Any]:
        """Return the serializable autoencoder backend metadata."""

        return {
            "backend": self.ae_backend,
            "config": self.backend_config,
            "normalization_stats": self.normalization_stats.to_dict(),
        }

    def set_latent_normalization_stats(self, stats: LatentNormalizationStats) -> None:
        """Update the active DreamDojo-style latent normalization statistics."""

        if len(stats.img_mean) != self.latent_channels:
            raise ValueError(
                f"Expected img_mean with {self.latent_channels} channels, received {len(stats.img_mean)}."
            )
        if len(stats.img_std) != self.latent_channels:
            raise ValueError(
                f"Expected img_std with {self.latent_channels} channels, received {len(stats.img_std)}."
            )
        if len(stats.video_mean) != self.latent_channels:
            raise ValueError(
                f"Expected video_mean with {self.latent_channels} channels, received {len(stats.video_mean)}."
            )
        if len(stats.video_std) != self.latent_channels:
            raise ValueError(
                f"Expected video_std with {self.latent_channels} channels, received {len(stats.video_std)}."
            )
        self.normalization_stats = stats
        self.img_latent_mean.copy_(
            torch.tensor(stats.img_mean, dtype=self.img_latent_mean.dtype, device=self.img_latent_mean.device).view(
                1,
                self.latent_channels,
                1,
                1,
            )
        )
        self.img_latent_std.copy_(
            torch.tensor(stats.img_std, dtype=self.img_latent_std.dtype, device=self.img_latent_std.device).view(
                1,
                self.latent_channels,
                1,
                1,
            )
        )
        self.video_latent_mean.copy_(
            torch.tensor(
                stats.video_mean,
                dtype=self.video_latent_mean.dtype,
                device=self.video_latent_mean.device,
            ).view(1, self.latent_channels, 1, 1, 1)
        )
        self.video_latent_std.copy_(
            torch.tensor(
                stats.video_std,
                dtype=self.video_latent_std.dtype,
                device=self.video_latent_std.device,
            ).view(1, self.latent_channels, 1, 1, 1)
        )

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

    def _normalize_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply image-latent normalization to one 4D tensor."""

        return (latents - self.img_latent_mean.to(dtype=latents.dtype)) / self.img_latent_std.to(
            dtype=latents.dtype
        )

    def _unnormalize_image_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert image-latent normalization for one 4D tensor."""

        return latents * self.img_latent_std.to(dtype=latents.dtype) + self.img_latent_mean.to(
            dtype=latents.dtype
        )

    def _normalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply video-latent normalization to one 5D tensor."""

        if latents.shape[2] == 1:
            return self._normalize_image_latents(latents.squeeze(2)).unsqueeze(2)
        return (latents - self.video_latent_mean.to(dtype=latents.dtype)) / self.video_latent_std.to(
            dtype=latents.dtype
        )

    def _unnormalize_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert video-latent normalization for one 5D tensor."""

        if latents.shape[2] == 1:
            return self._unnormalize_image_latents(latents.squeeze(2)).unsqueeze(2)
        return latents * self.video_latent_std.to(dtype=latents.dtype) + self.video_latent_mean.to(
            dtype=latents.dtype
        )

    def encode_posterior(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw posterior moments from the Wan encoder for 4D images."""

        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B, C, H, W), received {tuple(images.shape)}.")
        mu, log_var = self.encoder(images.unsqueeze(2))
        return mu.squeeze(2), log_var.squeeze(2)

    def encode(self, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode images into normalized latent maps."""

        mu, log_var = self.encode_posterior(images)
        raw_latent = mu if deterministic else sample_posterior_latent(mu, log_var)
        return self._normalize_image_latents(raw_latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized image latents into RGB images."""

        if latents.ndim != 4:
            raise ValueError(f"Expected latents with shape (B, C, H, W), received {tuple(latents.shape)}.")
        return self.decoder(self._unnormalize_image_latents(latents).unsqueeze(2)).squeeze(2)

    def autoencode(
        self,
        images: torch.Tensor,
        sample_posterior: bool,
    ) -> AutoencoderOutput:
        """Run one raw-image AE pass and return reconstructions plus posterior statistics."""

        mu, log_var = self.encode_posterior(images)
        raw_latent = mu if not sample_posterior else sample_posterior_latent(mu, log_var)
        reconstructed = self.decoder(raw_latent.unsqueeze(2)).squeeze(2)
        kl_loss = kl_divergence_from_moments(mu, log_var)
        return AutoencoderOutput(
            reconstructed=reconstructed,
            latent=raw_latent,
            mu=mu,
            log_var=log_var,
            kl_loss=kl_loss,
        )

    def autoencode_video(
        self,
        images: torch.Tensor,
        sample_posterior: bool,
    ) -> AutoencoderOutput:
        """Run one raw-video AE pass and return reconstructions plus posterior statistics."""

        if images.ndim != 5:
            raise ValueError(
                f"Expected video images with shape (B, T, C, H, W), received {tuple(images.shape)}."
            )
        video = rearrange(images, "b t c h w -> b c t h w")
        mu, log_var = self.encoder(video)
        raw_latent = mu if not sample_posterior else sample_posterior_latent(mu, log_var)
        reconstructed = rearrange(self.decoder(raw_latent), "b c t h w -> b t c h w")
        kl_loss = kl_divergence_from_moments(mu, log_var)
        return AutoencoderOutput(
            reconstructed=reconstructed,
            latent=raw_latent,
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
        """Encode an image sequence into a normalized latent-video tensor."""

        if images.ndim != 5:
            raise ValueError(
                f"Expected image sequences with shape (B, T, C, H, W), received {tuple(images.shape)}."
            )
        video = rearrange(images, "b t c h w -> b c t h w")
        mu, log_var = self.encoder(video)
        raw_latent = mu if deterministic else sample_posterior_latent(mu, log_var)
        return self._normalize_video_latents(raw_latent)

    def decode_frame_sequence(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a normalized latent-video tensor into an image sequence."""

        if latents.ndim != 5:
            raise ValueError(
                f"Expected latent sequences with shape (B, C, T, H, W), received {tuple(latents.shape)}."
            )
        decoded = self.decoder(self._unnormalize_video_latents(latents))
        return rearrange(decoded, "b c t h w -> b t c h w")

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
        latent_frames = self.pixel_frames_to_latent_frames(int(images.shape[1]), exact=True)
        if latent_frames not in self.dynamics.cfg.conditioning_frame_choices:
            raise ValueError(
                "Expected a context image length that maps to one of the supported conditioning "
                f"latent frame counts {self.dynamics.cfg.conditioning_frame_choices}, received "
                f"{images.shape[1]} pixel frames."
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

    def decode_target_latents(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor,
        *,
        context_pixel_frames: int,
        target_pixel_frames: int | None = None,
    ) -> torch.Tensor:
        """Decode target latents by decoding the full chunk and cropping off the context pixels."""

        full_latents = torch.cat([context_latents, target_latents], dim=2)
        full_frames = self.decode_frame_sequence(full_latents)
        resolved_target_pixel_frames = (
            self.temporal_downsample_factor * int(target_latents.shape[2])
            if target_pixel_frames is None
            else int(target_pixel_frames)
        )
        return full_frames[:, context_pixel_frames : context_pixel_frames + resolved_target_pixel_frames]

    def predict_next_frame(
        self,
        images: torch.Tensor,
        actions: torch.Tensor | None = None,
        infer_steps: int | None = None,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Predict the next pixel-frame chunk from a supported conditioning image context."""

        current_latents = self.encode_context_frames(images, deterministic=True)
        next_latents = self.predict_next_latent(
            current_latents,
            actions=actions,
            infer_steps=infer_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )
        return self.decode_target_latents(
            current_latents,
            next_latents,
            context_pixel_frames=int(images.shape[1]),
        )

    def resolved_rollout_stride_frames(
        self,
        context_frames: int,
        stride_frames: int | None = None,
    ) -> int:
        """Resolve how many new latent frames a rollout chunk should append."""

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
        """Autoregressively predict a requested number of future pixel frames."""

        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if seed_frames.ndim != 5:
            raise ValueError(
                f"Expected seed frames with shape (B, T, C, H, W), received {tuple(seed_frames.shape)}."
            )
        seed_context_latent_frames = self.pixel_frames_to_latent_frames(int(seed_frames.shape[1]), exact=True)
        if seed_context_latent_frames not in self.dynamics.cfg.conditioning_frame_choices:
            raise ValueError(
                "Expected one of the supported seed frame counts that map to "
                f"{self.dynamics.cfg.conditioning_frame_choices} latent frames, received "
                f"{seed_frames.shape[1]} pixel frames."
            )
        if actions is not None:
            expected_action_steps = max(int(seed_frames.shape[1]) - 1 + steps, 0)
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
        predicted = seed_frames.clone()
        generated_frames = 0
        generator = torch.Generator(device=seed_frames.device.type)
        generator.manual_seed(0)
        while generated_frames < steps:
            available_latent_frames = self.pixel_frames_to_latent_frames(int(predicted.shape[1]))
            rollout_context_limit = min(
                available_latent_frames,
                int(self.dynamics.cfg.open_rollout_context_frames),
            )
            current_context_latent_frames = max(
                conditioning_frames
                for conditioning_frames in self.dynamics.cfg.conditioning_frame_choices
                if conditioning_frames <= rollout_context_limit
            )
            current_context_pixel_frames = self.latent_frames_to_pixel_frames(current_context_latent_frames)
            current_context_images = predicted[:, -current_context_pixel_frames:]
            current_latents = self.encode_context_frames(current_context_images, deterministic=True)
            chunk_target_capacity = self.dynamics.cfg.max_frames - current_context_latent_frames
            stride_latent_frames = self.resolved_rollout_stride_frames(
                current_context_latent_frames,
                stride_frames=stride_frames,
            )
            chunk_target_latent_frames = min(chunk_target_capacity, stride_latent_frames)
            action_window = None
            if actions is not None:
                action_start = int(predicted.shape[1]) - current_context_pixel_frames
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
            next_latents = next_latents[:, :, :chunk_target_latent_frames]
            remaining_pixel_frames = steps - generated_frames
            max_decoded_pixel_frames = self.temporal_downsample_factor * int(next_latents.shape[2])
            next_frames = self.decode_target_latents(
                current_latents,
                next_latents,
                context_pixel_frames=current_context_pixel_frames,
                target_pixel_frames=min(max_decoded_pixel_frames, remaining_pixel_frames),
            )
            predicted = torch.cat([predicted, next_frames], dim=1)
            generated_frames += int(next_frames.shape[1])
        return predicted
