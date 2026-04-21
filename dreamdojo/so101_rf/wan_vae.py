"""DreamDojo-compatible Wan 2.2 causal video VAE blocks for the world-model pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE_T = 2
DEFAULT_WAN_DIM = 160
DEFAULT_WAN_DEC_DIM = 256
DEFAULT_WAN_Z_DIM = 48
DEFAULT_WAN_NUM_RES_BLOCKS = 2

WAN_2PT2_MEAN = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
WAN_2PT2_STD = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


@dataclass(frozen=True)
class WanVAEConfig:
    """Configure the DreamDojo-compatible Wan 2.2 video autoencoder."""

    dim: int = DEFAULT_WAN_DIM
    dec_dim: int = DEFAULT_WAN_DEC_DIM
    z_dim: int = DEFAULT_WAN_Z_DIM
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = DEFAULT_WAN_NUM_RES_BLOCKS
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    patch_size: int = 2
    temporal_window: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate the Wan 2.2 architecture and chunking parameters."""

        if self.dim < 1:
            raise ValueError("dim must be positive.")
        if self.dec_dim < 1:
            raise ValueError("dec_dim must be positive.")
        if self.z_dim < 1:
            raise ValueError("z_dim must be positive.")
        if len(self.dim_mult) < 1:
            raise ValueError("dim_mult must contain at least one entry.")
        if len(self.temperal_downsample) != max(len(self.dim_mult) - 1, 0):
            raise ValueError(
                "temperal_downsample must have exactly len(dim_mult) - 1 entries."
            )
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive.")
        if self.temporal_window < 1:
            raise ValueError("temporal_window must be positive.")
        if self.temporal_window != self.temporal_downsample_factor():
            raise ValueError(
                "temporal_window must equal the temporal compression factor derived from "
                "temperal_downsample."
            )

    def spatial_downsample_factor(self) -> int:
        """Return the total Wan 2.2 spatial compression factor."""

        return int(self.patch_size) * (2 ** max(len(self.dim_mult) - 1, 0))

    def temporal_downsample_factor(self) -> int:
        """Return the total Wan 2.2 temporal compression factor."""

        return 2 ** sum(int(flag) for flag in self.temperal_downsample)

    def pixel_frames_to_latent_frames(self, num_pixel_frames: int) -> int:
        """Return the Wan 2.2 latent frame count for one pixel-frame count."""

        if num_pixel_frames < 1:
            raise ValueError("num_pixel_frames must be positive.")
        return 1 + (int(num_pixel_frames) - 1) // self.temporal_downsample_factor()

    def latent_frames_to_pixel_frames(self, num_latent_frames: int) -> int:
        """Return the Wan 2.2 pixel-frame count for one latent-frame count."""

        if num_latent_frames < 1:
            raise ValueError("num_latent_frames must be positive.")
        return 1 + (int(num_latent_frames) - 1) * self.temporal_downsample_factor()

    def exact_latent_frames_for_pixels(self, num_pixel_frames: int) -> int:
        """Return the latent count when one pixel-frame count is exactly representable."""

        latent_frames = self.pixel_frames_to_latent_frames(num_pixel_frames)
        resolved_pixel_frames = self.latent_frames_to_pixel_frames(latent_frames)
        if resolved_pixel_frames != int(num_pixel_frames):
            raise ValueError(
                f"Expected a Wan 2.2-aligned pixel-frame count, received {num_pixel_frames}. "
                "Supported counts follow 1 + 4 * k."
            )
        return latent_frames

    def is_dreamdojo_exact(self) -> bool:
        """Return whether this config matches DreamDojo's public Wan 2.2 architecture."""

        return self == WanVAEConfig()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable config dictionary."""

        return {
            "dim": self.dim,
            "dec_dim": self.dec_dim,
            "z_dim": self.z_dim,
            "dim_mult": list(self.dim_mult),
            "num_res_blocks": self.num_res_blocks,
            "attn_scales": list(self.attn_scales),
            "temperal_downsample": list(self.temperal_downsample),
            "patch_size": self.patch_size,
            "temporal_window": self.temporal_window,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WanVAEConfig":
        """Build one config from serialized checkpoint metadata."""

        return cls(
            dim=int(payload.get("dim", DEFAULT_WAN_DIM)),
            dec_dim=int(payload.get("dec_dim", DEFAULT_WAN_DEC_DIM)),
            z_dim=int(payload.get("z_dim", DEFAULT_WAN_Z_DIM)),
            dim_mult=tuple(int(value) for value in payload.get("dim_mult", [1, 2, 4, 4])),
            num_res_blocks=int(payload.get("num_res_blocks", DEFAULT_WAN_NUM_RES_BLOCKS)),
            attn_scales=tuple(float(value) for value in payload.get("attn_scales", [])),
            temperal_downsample=tuple(
                bool(value) for value in payload.get("temperal_downsample", [False, True, True])
            ),
            patch_size=int(payload.get("patch_size", 2)),
            temporal_window=int(payload.get("temporal_window", 4)),
            dropout=float(payload.get("dropout", 0.0)),
        )


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Apply Wan 2.2 spatial patchification to one image or video tensor."""

    if patch_size == 1:
        return x
    if x.ndim == 4:
        return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    if x.ndim == 5:
        return rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    raise ValueError(f"Invalid input shape: {tuple(x.shape)}")


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Undo Wan 2.2 spatial patchification for one image or video tensor."""

    if patch_size == 1:
        return x
    if x.ndim == 4:
        return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    if x.ndim == 5:
        return rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    raise ValueError(f"Invalid input shape: {tuple(x.shape)}")


class CausalConv3d(nn.Conv3d):
    """Apply causal temporal padding before one 3D convolution."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Create the causal convolution and cache its explicit pad layout."""

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
        """Convolve one video tensor with optional cached past activations."""

        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(device=x.device, dtype=x.dtype)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= int(cache_x.shape[2])
        return super().forward(F.pad(x, padding))


class RMSNorm(nn.Module):
    """Apply Wan-style RMS normalization with broadcasted gain and bias."""

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

        output = F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma
        if self.bias is not None:
            output = output + self.bias
        return output


RMS_norm = RMSNorm


class Upsample(nn.Upsample):
    """Nearest-neighbor upsampling that preserves bf16 compatibility."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample in fp32 and cast back to the original dtype."""

        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """Apply Wan 2.2 spatial or spatio-temporal resampling."""

    def __init__(self, dim: int, mode: str) -> None:
        """Create one Wan 2.2 resampling path."""

        super().__init__()
        if mode not in {"none", "upsample2d", "upsample3d", "downsample2d", "downsample3d"}:
            raise ValueError(f"Unsupported resample mode: {mode}")
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
            self.time_conv: CausalConv3d | None = None
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
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
        """Return the resampled tensor and update causal caches when provided."""

        resolved_feat_idx = [0] if feat_idx is None else feat_idx
        batch, channels, frames, _, _ = x.shape
        if self.mode == "upsample3d" and feat_cache is not None and self.time_conv is not None:
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
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x, device=cache_x.device), cache_x],
                        dim=2,
                    )
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


class AvgDown3D(nn.Module):
    """Average grouped channels after one spatio-temporal downsampling reshape."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        """Create the reshape-and-average shortcut used by Wan 2.2 down blocks."""

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        if in_channels * self.factor % out_channels != 0:
            raise ValueError("in_channels * factor must be divisible by out_channels.")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the averaged downsampled shortcut tensor."""

        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        batch, channels, frames, height, width = x.shape
        x = x.view(
            batch,
            channels,
            frames // self.factor_t,
            self.factor_t,
            height // self.factor_s,
            self.factor_s,
            width // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            batch,
            channels * self.factor,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        x = x.view(
            batch,
            self.out_channels,
            self.group_size,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    """Duplicate grouped channels into one spatio-temporal upsampling reshape."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ) -> None:
        """Create the reshape-and-duplicate shortcut used by Wan 2.2 up blocks."""

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        if out_channels * self.factor % in_channels != 0:
            raise ValueError("out_channels * factor must be divisible by in_channels.")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        """Return the duplicated upsampled shortcut tensor."""

        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class ResidualBlock(nn.Module):
    """Apply a causal Wan 2.2 residual update."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        """Create the residual and shortcut paths for one block."""

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, kernel_size=3, padding=1),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

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
                        [
                            feat_cache[cache_index][:, :, -1:, :, :].to(cache_x.device),
                            cache_x,
                        ],
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
        """Create the Wan 2.2 spatial attention projections."""

        super().__init__()
        self.dim = dim
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


class Down_ResidualBlock(nn.Module):
    """Apply the Wan 2.2 residual downsampling block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        """Create the residual path and grouped-average shortcut."""

        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        downsamples: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(mult):
            downsamples.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Return one residual downsampling update."""

        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    """Apply the Wan 2.2 residual upsampling block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        """Create the residual path and grouped-duplication shortcut."""

        super().__init__()
        self.avg_shortcut = (
            DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
            if up_flag
            else None
        )
        upsamples: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(mult):
            upsamples.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*upsamples)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        """Return one residual upsampling update."""

        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is None:
            return x_main
        return x_main + self.avg_shortcut(x, first_chunk=first_chunk)


class Encoder3d(nn.Module):
    """Encode patchified RGB videos into pre-moment latent features."""

    def __init__(self, cfg: WanVAEConfig, out_channels: int) -> None:
        """Create the DreamDojo-compatible Wan 2.2 encoder backbone."""

        super().__init__()
        patch_channels = 3 * (cfg.patch_size**2)
        dims = [cfg.dim * scale for scale in (1, *cfg.dim_mult)]
        self.conv1 = CausalConv3d(patch_channels, dims[0], kernel_size=3, padding=1)
        downsample_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = cfg.temperal_downsample[index] if index < len(cfg.temperal_downsample) else False
            downsample_layers.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=cfg.dropout,
                    mult=cfg.num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=index != len(cfg.dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsample_layers)
        self.middle = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1], cfg.dropout),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], cfg.dropout),
        )
        self.head = nn.Sequential(
            RMSNorm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Return encoded video features for one patchified RGB clip."""

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
            x = layer(x, feat_cache, resolved_feat_idx) if feat_cache is not None else layer(x)
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


class Decoder3d(nn.Module):
    """Decode latent videos back into patchified RGB frames."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the DreamDojo-compatible Wan 2.2 decoder backbone."""

        super().__init__()
        patch_channels = 3 * (cfg.patch_size**2)
        dims = [cfg.dec_dim * scale for scale in (cfg.dim_mult[-1], *reversed(cfg.dim_mult))]
        temperal_upsample = tuple(reversed(cfg.temperal_downsample))
        self.conv1 = CausalConv3d(cfg.z_dim, dims[0], kernel_size=3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], cfg.dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], cfg.dropout),
        )
        upsample_layers: list[nn.Module] = []
        current_dim = dims[0]
        for index, out_dim in enumerate(dims[1:]):
            t_up_flag = temperal_upsample[index] if index < len(temperal_upsample) else False
            upsample_layers.append(
                Up_ResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    dropout=cfg.dropout,
                    mult=cfg.num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=index != len(cfg.dim_mult) - 1,
                )
            )
            current_dim = out_dim
        self.upsamples = nn.Sequential(*upsample_layers)
        self.head = nn.Sequential(
            RMSNorm(current_dim, images=False),
            nn.SiLU(),
            CausalConv3d(current_dim, patch_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        """Return decoded patchified RGB videos with DreamDojo's raw decoder output."""

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
            if isinstance(layer, Up_ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, resolved_feat_idx, first_chunk=first_chunk)
            elif isinstance(layer, Up_ResidualBlock):
                x = layer(x, first_chunk=first_chunk)
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


def count_conv3d(model: nn.Module) -> int:
    """Count the causal convolutions in one module tree."""

    return sum(1 for module in model.modules() if isinstance(module, CausalConv3d))


class WanPosteriorEncoder(nn.Module):
    """Encode video clips into Wan 2.2 posterior moments."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the Wan 2.2 posterior encoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.temporal_window = cfg.temporal_window
        self.backbone = Encoder3d(cfg, out_channels=cfg.z_dim * 2)
        self.moments_conv = CausalConv3d(cfg.z_dim * 2, cfg.z_dim * 2, kernel_size=1)
        self._enc_feat_map: list[torch.Tensor | str | None] = []

    def clear_cache(self) -> None:
        """Reset the cached causal feature maps used by chunked video encoding."""

        self._enc_feat_map = [None] * count_conv3d(self.backbone)

    def _encode_first_chunk(self, video: torch.Tensor) -> torch.Tensor:
        """Encode the first single-frame chunk that seeds the causal cache."""

        return self.backbone(video[:, :, :1], feat_cache=self._enc_feat_map, feat_idx=[0])

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log-variance tensors for one video batch."""

        self.clear_cache()
        patchified = patchify(video, patch_size=self.cfg.patch_size)
        total_frames = int(patchified.shape[2])
        if total_frames < 1:
            raise ValueError("video must contain at least one frame.")
        iter_count = 1 + (total_frames - 1) // self.temporal_window
        encoded = self._encode_first_chunk(patchified)
        for index in range(1, iter_count):
            start = 1 + self.temporal_window * (index - 1)
            stop = 1 + self.temporal_window * index
            encoded = torch.cat(
                [
                    encoded,
                    self.backbone(
                        patchified[:, :, start:stop],
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
                        patchified[:, :, start:],
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
    """Decode raw Wan 2.2 latent videos into RGB videos."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the Wan 2.2 video decoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.temporal_window = cfg.temporal_window
        self.pre_decode_conv = CausalConv3d(cfg.z_dim, cfg.z_dim, kernel_size=1)
        self.backbone = Decoder3d(cfg)
        self._feat_map: list[torch.Tensor | str | None] = []

    def clear_cache(self) -> None:
        """Reset the cached causal feature maps used by chunked video decoding."""

        self._feat_map = [None] * count_conv3d(self.backbone)

    def _decode_first_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the first latent frame that seeds the causal cache."""

        return self.backbone(latents[:, :, :1], feat_cache=self._feat_map, feat_idx=[0], first_chunk=True)

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
        return unpatchify(decoded, patch_size=self.cfg.patch_size)


class WanVAEEncoder(nn.Module):
    """Wrap the video posterior encoder for 4D image batches."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the image-facing Wan 2.2 encoder wrapper."""

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
        """Create the image-facing Wan 2.2 decoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.video_decoder = WanVideoDecoder(cfg)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return decoded images for 4D latent batches."""

        return self.video_decoder(latents.unsqueeze(2)).squeeze(2)


def dreamdojo_wan_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    """Extract one raw DreamDojo Wan 2.2 state dict from a loaded checkpoint payload."""

    if isinstance(payload, Mapping):
        if payload and all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
            return dict(payload)
        nested_state_dict = payload.get("state_dict")
        if isinstance(nested_state_dict, Mapping) and nested_state_dict:
            if all(
                isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in nested_state_dict.items()
            ):
                return dict(nested_state_dict)
    return None


def remap_dreamdojo_wan_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap raw DreamDojo Wan 2.2 weights into this repo's encoder and decoder wrappers."""

    prefix_map = {
        "encoder.": "encoder.backbone.",
        "conv1.": "encoder.moments_conv.",
        "conv2.": "decoder.pre_decode_conv.",
        "decoder.": "decoder.backbone.",
    }
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for source_prefix, target_prefix in prefix_map.items():
            if key.startswith(source_prefix):
                remapped[f"{target_prefix}{key.removeprefix(source_prefix)}"] = value
                break
    return remapped


def kl_divergence_from_moments(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Return the batch-averaged KL divergence to a unit Gaussian prior."""

    return 0.5 * torch.mean(torch.exp(log_var) + mu.square() - 1.0 - log_var)


def sample_posterior(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample one diagonal-Gaussian posterior with the reparameterization trick."""

    std = torch.exp(0.5 * log_var)
    return mu + std * torch.randn_like(std)
