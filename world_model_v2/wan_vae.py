"""Wan2.1-style causal video VAE blocks used by the world-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE_T = 2
DEFAULT_WAN_DIM = 64
DEFAULT_WAN_Z_DIM = 64
DEFAULT_WAN_NUM_RES_BLOCKS = 1


@dataclass(frozen=True)
class WanVAEConfig:
    """Configure the local Wan2.1-style video autoencoder."""

    dim: int = DEFAULT_WAN_DIM
    z_dim: int = DEFAULT_WAN_Z_DIM
    dim_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = DEFAULT_WAN_NUM_RES_BLOCKS
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (True, True)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate the Wan VAE shape configuration."""

        if self.dim < 1:
            raise ValueError("dim must be positive.")
        if self.z_dim < 1:
            raise ValueError("z_dim must be positive.")
        if len(self.dim_mult) < 1:
            raise ValueError("dim_mult must contain at least one entry.")
        if len(self.temperal_downsample) != max(len(self.dim_mult) - 1, 0):
            raise ValueError(
                "temperal_downsample must have exactly len(dim_mult) - 1 entries."
            )

    def spatial_downsample_factor(self) -> int:
        """Return the total spatial compression factor."""

        return 2 ** max(len(self.dim_mult) - 1, 0)

    def temporal_downsample_factor(self) -> int:
        """Return the total temporal compression factor."""

        return 2 ** sum(int(flag) for flag in self.temperal_downsample)

    def pixel_frames_to_latent_frames(self, num_pixel_frames: int) -> int:
        """Return the Wan2.1 latent frame count for one pixel-frame count."""

        if num_pixel_frames < 1:
            raise ValueError("num_pixel_frames must be positive.")
        return 1 + (int(num_pixel_frames) - 1) // self.temporal_downsample_factor()

    def latent_frames_to_pixel_frames(self, num_latent_frames: int) -> int:
        """Return the Wan2.1 pixel-frame count for one latent-frame count."""

        if num_latent_frames < 1:
            raise ValueError("num_latent_frames must be positive.")
        return 1 + (int(num_latent_frames) - 1) * self.temporal_downsample_factor()

    def exact_latent_frames_for_pixels(self, num_pixel_frames: int) -> int:
        """Return the latent count when the pixel-frame count is exactly representable."""

        latent_frames = self.pixel_frames_to_latent_frames(num_pixel_frames)
        resolved_pixel_frames = self.latent_frames_to_pixel_frames(latent_frames)
        if resolved_pixel_frames != int(num_pixel_frames):
            raise ValueError(
                f"Expected a Wan-aligned pixel-frame count, received {num_pixel_frames}. "
                f"Supported counts follow 1 + 4 * k and nearest lower aligned count is "
                f"{resolved_pixel_frames - self.temporal_downsample_factor()}."
            )
        return latent_frames

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable config dictionary."""

        return {
            "dim": self.dim,
            "z_dim": self.z_dim,
            "dim_mult": list(self.dim_mult),
            "num_res_blocks": self.num_res_blocks,
            "attn_scales": list(self.attn_scales),
            "temperal_downsample": list(self.temperal_downsample),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WanVAEConfig":
        """Build a config from serialized checkpoint metadata."""

        return cls(
            dim=int(payload.get("dim", DEFAULT_WAN_DIM)),
            z_dim=int(payload.get("z_dim", DEFAULT_WAN_Z_DIM)),
            dim_mult=tuple(int(value) for value in payload.get("dim_mult", [1, 2, 4])),
            num_res_blocks=int(payload.get("num_res_blocks", DEFAULT_WAN_NUM_RES_BLOCKS)),
            attn_scales=tuple(float(value) for value in payload.get("attn_scales", [])),
            temperal_downsample=tuple(
                bool(value) for value in payload.get("temperal_downsample", [True, True])
            ),
            dropout=float(payload.get("dropout", 0.0)),
        )


class CausalConv3d(nn.Conv3d):
    """Apply causal temporal padding before a 3D convolution."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Create the causal convolution and cache its padding layout."""

        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        """Convolve a video tensor with optional cached past activations."""

        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(device=x.device, dtype=x.dtype)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= int(cache_x.shape[2])
        return super().forward(F.pad(x, padding))


class RMSNorm(nn.Module):
    """Apply Wan-style RMS normalization."""

    def __init__(
        self,
        dim: int,
        *,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        """Create the broadcasted normalization parameters."""

        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized activations with learned gain and bias."""

        normalized = F.normalize(x, dim=(1 if self.channel_first else -1))
        output = normalized * self.scale * self.gamma
        if self.bias is not None:
            output = output + self.bias
        return output


class Upsample(nn.Upsample):
    """Nearest-neighbor upsampling that preserves the input activation dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one nearest-exact upsampled tensor without promoting precision."""

        return super().forward(x)


class Resample(nn.Module):
    """Apply Wan2.1 spatial or spatio-temporal resampling."""

    def __init__(self, dim: int, mode: str) -> None:
        """Create the requested resampling path."""

        super().__init__()
        if mode not in {"none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"}:
            raise ValueError(f"Unsupported resample mode: {mode}")
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
            self.time_conv = None
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=(2, 2)),
            )
            self.time_conv = None
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(
                dim,
                dim,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
            )
        else:
            self.resample = nn.Identity()
            self.time_conv = None

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Return the resampled tensor with optional feature-cache updates."""

        resolved_feat_idx = [0] if feat_idx is None else feat_idx
        batch, channels, frames, _, _ = x.shape
        if self.mode == "upsample3d":
            if feat_cache is not None and self.time_conv is not None:
                cache_index = resolved_feat_idx[0]
                if feat_cache[cache_index] is None:
                    feat_cache[cache_index] = "Rep"
                    resolved_feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[cache_index] not in {None, "Rep"}:
                        cache_x = torch.cat(
                            [
                                feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if cache_x.shape[2] < 2 and feat_cache[cache_index] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    if feat_cache[cache_index] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[cache_index])
                    feat_cache[cache_index] = cache_x
                    resolved_feat_idx[0] += 1
                    x = x.reshape(batch, 2, channels, frames, x.shape[-2], x.shape[-1])
                    x = torch.stack((x[:, 0], x[:, 1]), dim=3)
                    x = x.reshape(batch, channels, frames * 2, x.shape[-2], x.shape[-1])
        frames = int(x.shape[2])
        flat = rearrange(x, "b c t h w -> (b t) c h w")
        flat = self.resample(flat)
        x = rearrange(flat, "(b t) c h w -> b c t h w", b=batch, t=frames)
        if self.mode == "downsample3d" and feat_cache is not None and self.time_conv is not None:
            cache_index = resolved_feat_idx[0]
            if feat_cache[cache_index] is None:
                feat_cache[cache_index] = x.clone()
                resolved_feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[cache_index][:, :, -1:, :, :], x], dim=2))
                feat_cache[cache_index] = cache_x
                resolved_feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    """Apply a causal Wan2.1 residual update."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        """Create the residual and shortcut paths for one block."""

        super().__init__()
        self.shortcut = CausalConv3d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, kernel_size=3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Return the shortcut plus residual update."""

        resolved_feat_idx = [0] if feat_idx is None else feat_idx
        shortcut = self.shortcut(x)
        hidden = x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                cache_index = resolved_feat_idx[0]
                cache_x = hidden[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[cache_index] is not None:
                    cache_x = torch.cat(
                        [feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device), cache_x],
                        dim=2,
                    )
                hidden = layer(hidden, feat_cache[cache_index])
                feat_cache[cache_index] = cache_x
                resolved_feat_idx[0] += 1
            else:
                hidden = layer(hidden)
        return hidden + shortcut


class AttentionBlock(nn.Module):
    """Apply single-head spatial attention independently per frame."""

    def __init__(self, dim: int) -> None:
        """Create the spatial attention projections."""

        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spatial-attention residual update."""

        identity = x
        batch, channels, frames, height, width = x.shape
        flat = rearrange(x, "b c t h w -> (b t) c h w")
        flat = self.norm(flat)
        q, k, v = (
            self.to_qkv(flat)
            .reshape(batch * frames, 1, channels * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.squeeze(1).permute(0, 2, 1).reshape(batch * frames, channels, height, width)
        projected = self.proj(attended)
        return rearrange(projected, "(b t) c h w -> b c t h w", b=batch, t=frames) + identity


class WanEncoder3d(nn.Module):
    """Encode RGB videos into pre-moment latent features."""

    def __init__(self, cfg: WanVAEConfig, out_channels: int) -> None:
        """Create the causal Wan2.1 encoder backbone."""

        super().__init__()
        dims = [cfg.dim * scale for scale in (1, *cfg.dim_mult)]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], kernel_size=3, padding=1)
        downsample_layers: list[nn.Module] = []
        current_dim = dims[0]
        for index, out_dim in enumerate(dims[1:]):
            for _ in range(cfg.num_res_blocks):
                downsample_layers.append(ResidualBlock(current_dim, out_dim, cfg.dropout))
                if scale in cfg.attn_scales:
                    downsample_layers.append(AttentionBlock(out_dim))
                current_dim = out_dim
            if index != len(cfg.dim_mult) - 1:
                mode = "downsample3d" if cfg.temperal_downsample[index] else "downsample2d"
                downsample_layers.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.ModuleList(downsample_layers)
        self.middle = nn.ModuleList(
            [
                ResidualBlock(current_dim, current_dim, cfg.dropout),
                AttentionBlock(current_dim),
                ResidualBlock(current_dim, current_dim, cfg.dropout),
            ]
        )
        self.head = nn.ModuleList(
            [
                RMSNorm(current_dim, images=False),
                nn.SiLU(),
                CausalConv3d(current_dim, out_channels, kernel_size=3, padding=1),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Return encoded video features for one RGB clip."""

        resolved_feat_idx = [0] if feat_idx is None else feat_idx
        if feat_cache is not None:
            cache_index = resolved_feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[cache_index] is not None:
                cache_x = torch.cat(
                    [feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[cache_index])
            feat_cache[cache_index] = cache_x
            resolved_feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, resolved_feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, resolved_feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                cache_index = resolved_feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[cache_index] is not None:
                    cache_x = torch.cat(
                        [feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[cache_index])
                feat_cache[cache_index] = cache_x
                resolved_feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class WanDecoder3d(nn.Module):
    """Decode latent videos back into RGB frames."""

    def __init__(self, cfg: WanVAEConfig, in_channels: int) -> None:
        """Create the causal Wan2.1 decoder backbone."""

        super().__init__()
        dims = [cfg.dim * scale for scale in (cfg.dim_mult[-1], *reversed(cfg.dim_mult))]
        scale = 1.0 / (2 ** max(len(cfg.dim_mult) - 2, 0))
        temperal_upsample = tuple(reversed(cfg.temperal_downsample))
        self.conv1 = CausalConv3d(in_channels, dims[0], kernel_size=3, padding=1)
        self.middle = nn.ModuleList(
            [
                ResidualBlock(dims[0], dims[0], cfg.dropout),
                AttentionBlock(dims[0]),
                ResidualBlock(dims[0], dims[0], cfg.dropout),
            ]
        )
        upsample_layers: list[nn.Module] = []
        current_dim = dims[0]
        for index, out_dim in enumerate(dims[1:]):
            if index in {1, 2, 3}:
                current_dim = current_dim // 2
            for _ in range(cfg.num_res_blocks + 1):
                upsample_layers.append(ResidualBlock(current_dim, out_dim, cfg.dropout))
                if scale in cfg.attn_scales:
                    upsample_layers.append(AttentionBlock(out_dim))
                current_dim = out_dim
            if index != len(cfg.dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[index] else "upsample2d"
                upsample_layers.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.ModuleList(upsample_layers)
        self.head = nn.ModuleList(
            [
                RMSNorm(current_dim, images=False),
                nn.SiLU(),
                CausalConv3d(current_dim, 3, kernel_size=3, padding=1),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Return decoded RGB videos in the repo's `[0, 1]` convention."""

        resolved_feat_idx = [0] if feat_idx is None else feat_idx
        if feat_cache is not None:
            cache_index = resolved_feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[cache_index] is not None:
                cache_x = torch.cat(
                    [feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[cache_index])
            feat_cache[cache_index] = cache_x
            resolved_feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, resolved_feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, resolved_feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                cache_index = resolved_feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[cache_index] is not None:
                    cache_x = torch.cat(
                        [feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[cache_index])
                feat_cache[cache_index] = cache_x
                resolved_feat_idx[0] += 1
            else:
                x = layer(x)
        return torch.sigmoid(x)


class WanPosteriorEncoder(nn.Module):
    """Encode video clips into posterior moments."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the Wan2.1-style posterior encoder."""

        super().__init__()
        self.cfg = cfg
        self.temporal_window = cfg.temporal_downsample_factor()
        self.backbone = WanEncoder3d(cfg, out_channels=cfg.z_dim * 2)
        self.moments_conv = CausalConv3d(cfg.z_dim * 2, cfg.z_dim * 2, kernel_size=1)
        self._enc_feat_map: list[torch.Tensor | str | None] = []

    def clear_cache(self) -> None:
        """Reset the cached causal feature maps used by chunked video encoding."""

        self._enc_feat_map = [None] * sum(
            1 for module in self.backbone.modules() if isinstance(module, CausalConv3d)
        )

    def _encode_first_chunk(self, video: torch.Tensor) -> torch.Tensor:
        """Encode the first single-frame chunk that seeds the causal cache."""

        return self.backbone(video[:, :, :1], feat_cache=self._enc_feat_map, feat_idx=[0])

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log-variance tensors for one video batch."""

        self.clear_cache()
        total_frames = int(video.shape[2])
        if total_frames < 1:
            raise ValueError("video must contain at least one frame.")
        iter_count = 1 + (total_frames - 1) // self.temporal_window
        encoded = self._encode_first_chunk(video)
        for index in range(1, iter_count):
            start = 1 + self.temporal_window * (index - 1)
            stop = 1 + self.temporal_window * index
            encoded = torch.cat(
                [
                    encoded,
                    self.backbone(
                        video[:, :, start:stop],
                        feat_cache=self._enc_feat_map,
                        feat_idx=[0],
                    ),
                ],
                dim=2,
            )
        if (total_frames - 1) % self.temporal_window:
            start = 1 + self.temporal_window * (iter_count - 1)
            encoded = torch.cat(
                [
                    encoded,
                    self.backbone(
                        video[:, :, start:],
                        feat_cache=self._enc_feat_map,
                        feat_idx=[0],
                    ),
                ],
                dim=2,
            )
        moments = self.moments_conv(encoded)
        self.clear_cache()
        return moments.chunk(2, dim=1)


class WanVideoDecoder(nn.Module):
    """Decode raw latent videos into RGB videos."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the Wan2.1-style video decoder."""

        super().__init__()
        self.cfg = cfg
        self.temporal_window = cfg.temporal_downsample_factor()
        self.pre_decode_conv = CausalConv3d(cfg.z_dim, cfg.z_dim, kernel_size=1)
        self.backbone = WanDecoder3d(cfg, in_channels=cfg.z_dim)
        self._feat_map: list[torch.Tensor | str | None] = []

    def clear_cache(self) -> None:
        """Reset the cached causal feature maps used by chunked video decoding."""

        self._feat_map = [None] * sum(
            1 for module in self.backbone.modules() if isinstance(module, CausalConv3d)
        )

    def _decode_first_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the first latent frame that seeds the causal cache."""

        return self.backbone(latents[:, :, :1], feat_cache=self._feat_map, feat_idx=[0])

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return decoded RGB videos for one latent batch."""

        self.clear_cache()
        projected = self.pre_decode_conv(latents)
        decoded = self._decode_first_chunk(projected)
        for index in range(1, int(projected.shape[2])):
            decoded = torch.cat(
                [
                    decoded,
                    self.backbone(
                        projected[:, :, index : index + 1],
                        feat_cache=self._feat_map,
                        feat_idx=[0],
                    ),
                ],
                dim=2,
            )
        self.clear_cache()
        return decoded


class WanVAEEncoder(nn.Module):
    """Wrap the video posterior encoder for 4D image batches."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the image-facing encoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.video_encoder = WanPosteriorEncoder(cfg)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior moments for 4D image batches."""

        mu, log_var = self.video_encoder(images.unsqueeze(2))
        return mu.squeeze(2), log_var.squeeze(2)


class WanVAEDecoder(nn.Module):
    """Wrap the video decoder for 4D latent image batches."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the image-facing decoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.video_decoder = WanVideoDecoder(cfg)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return decoded images for 4D latent batches."""

        return self.video_decoder(latents.unsqueeze(2)).squeeze(2)


def kl_divergence_from_moments(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Return the batch-averaged KL divergence to a unit Gaussian prior."""

    return 0.5 * torch.mean(torch.exp(log_var) + mu.square() - 1.0 - log_var)


def sample_posterior(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample a diagonal-Gaussian posterior with the reparameterization trick."""

    std = torch.exp(0.5 * log_var)
    return mu + std * torch.randn_like(std)
