"""Wan-style VAE blocks adapted for the root world-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WanVAEConfig:
    """Configure the Wan-style VAE backend."""

    dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 1
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, False, False)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate the shape of the Wan-style configuration."""

        if len(self.dim_mult) == 0:
            raise ValueError("dim_mult must contain at least one entry.")
        if len(self.temperal_downsample) != max(len(self.dim_mult) - 1, 0):
            raise ValueError(
                "temperal_downsample must have exactly len(dim_mult) - 1 entries."
            )

    def spatial_downsample_factor(self) -> int:
        """Return the total spatial compression factor of the encoder."""

        return 2 ** max(len(self.dim_mult) - 1, 0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the config."""

        return {
            "dim": self.dim,
            "z_dim": self.z_dim,
            "dim_mult": list(self.dim_mult),
            "num_res_blocks": self.num_res_blocks,
            "attn_scales": list(self.attn_scales),
            "temperal_downsample": list(self.temperal_downsample),
            "dropout": self.dropout,
        }


class CausalConv3d(nn.Conv3d):
    """Apply causal temporal padding before a 3D convolution."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Create a causal 3D convolution with explicit padding control."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the causally padded 3D convolution result."""

        return super().forward(F.pad(x, self._padding))


class RMSNorm(nn.Module):
    """Normalize activations with an RMS-style channel normalization."""

    def __init__(self, dim: int, images: bool = True, bias: bool = False) -> None:
        """Create a channel-first RMS normalization layer."""

        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        shape = (dim, *broadcastable_dims)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize along channels and apply learnable scale and bias."""

        output = F.normalize(x, dim=1) * self.scale * self.gamma
        if self.bias is not None:
            output = output + self.bias
        return output


class AttentionBlock(nn.Module):
    """Apply per-frame spatial self-attention over a 5D video tensor."""

    def __init__(self, dim: int) -> None:
        """Create a single-head attention block with zero-initialized output."""

        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a video tensor with spatial attention at each time step."""

        identity = x
        batch, channels, steps, height, width = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(batch * steps, channels, height, width)
        flat = self.norm(flat)
        qkv = self.to_qkv(flat)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(batch * steps, channels, height * width).transpose(1, 2).unsqueeze(1)
        k = k.reshape(batch * steps, channels, height * width).transpose(1, 2).unsqueeze(1)
        v = v.reshape(batch * steps, channels, height * width).transpose(1, 2).unsqueeze(1)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.squeeze(1).transpose(1, 2).reshape(batch * steps, channels, height, width)
        projected = self.proj(attended)
        projected = projected.reshape(batch, steps, channels, height, width).permute(0, 2, 1, 3, 4)
        return projected + identity


class Resample(nn.Module):
    """Upsample or downsample a 5D tensor in spatial and optional temporal axes."""

    def __init__(self, dim: int, mode: str, out_dim: int | None = None) -> None:
        """Create one spatial or spatio-temporal resampling block."""

        super().__init__()
        self.mode = mode
        self.out_dim = dim if out_dim is None else out_dim
        if mode == "upsample2d":
            self.proj2d = nn.Conv2d(dim, self.out_dim, kernel_size=3, padding=1)
        elif mode == "downsample2d":
            self.proj2d = nn.Conv2d(dim, self.out_dim, kernel_size=3, stride=2)
        elif mode == "upsample3d":
            self.proj3d = CausalConv3d(dim, self.out_dim, kernel_size=3, padding=1)
        elif mode == "downsample3d":
            self.proj3d = CausalConv3d(dim, self.out_dim, kernel_size=3, stride=2, padding=1)
        elif mode != "none":
            raise ValueError(f"Unsupported resample mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the requested resampled tensor."""

        batch, channels, steps, height, width = x.shape
        if self.mode == "none":
            return x
        if self.mode == "upsample2d":
            flat = x.permute(0, 2, 1, 3, 4).reshape(batch * steps, channels, height, width)
            flat = F.interpolate(flat, scale_factor=2.0, mode="nearest-exact")
            flat = self.proj2d(flat)
            return flat.reshape(batch, steps, self.out_dim, flat.shape[-2], flat.shape[-1]).permute(
                0, 2, 1, 3, 4
            )
        if self.mode == "downsample2d":
            flat = x.permute(0, 2, 1, 3, 4).reshape(batch * steps, channels, height, width)
            flat = F.pad(flat, (0, 1, 0, 1))
            flat = self.proj2d(flat)
            return flat.reshape(batch, steps, self.out_dim, flat.shape[-2], flat.shape[-1]).permute(
                0, 2, 1, 3, 4
            )
        if self.mode == "upsample3d":
            upsampled = F.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
            return self.proj3d(upsampled)
        return self.proj3d(x)


class ResidualBlock(nn.Module):
    """Apply a Wan-style residual refinement to a 5D activation map."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        """Create the residual and shortcut paths for one block."""

        super().__init__()
        self.norm1 = RMSNorm(in_dim, images=False)
        self.act1 = nn.SiLU()
        self.conv1 = CausalConv3d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = RMSNorm(out_dim, images=False)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_dim, out_dim, kernel_size=3, padding=1)
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual update added to the shortcut path."""

        hidden = self.conv1(self.act1(self.norm1(x)))
        hidden = self.conv2(self.dropout(self.act2(self.norm2(hidden))))
        return hidden + self.shortcut(x)


class WanEncoder3d(nn.Module):
    """Encode a single-frame video tensor into VAE moments."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the Wan-style encoder backbone."""

        super().__init__()
        dims = [cfg.dim * scale for scale in (1, *cfg.dim_mult)]
        scale = 1.0
        self.conv_in = CausalConv3d(3, dims[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = nn.ModuleList()
            current_dim = in_dim
            for _ in range(cfg.num_res_blocks):
                block.append(ResidualBlock(current_dim, out_dim, cfg.dropout))
                if scale in cfg.attn_scales:
                    block.append(AttentionBlock(out_dim))
                current_dim = out_dim
            if index != len(cfg.dim_mult) - 1:
                mode = "downsample3d" if cfg.temperal_downsample[index] else "downsample2d"
                block.append(Resample(out_dim, mode=mode, out_dim=out_dim))
                scale /= 2.0
            self.down_blocks.append(block)
        final_dim = dims[-1]
        self.middle = nn.ModuleList(
            [
                ResidualBlock(final_dim, final_dim, cfg.dropout),
                AttentionBlock(final_dim),
                ResidualBlock(final_dim, final_dim, cfg.dropout),
            ]
        )
        self.norm_out = RMSNorm(final_dim, images=False)
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(final_dim, cfg.z_dim * 2, kernel_size=3, padding=1)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log variance tensors for the input video."""

        hidden = self.conv_in(video)
        for block in self.down_blocks:
            for layer in block:
                hidden = layer(hidden)
        for layer in self.middle:
            hidden = layer(hidden)
        moments = self.conv_out(self.act_out(self.norm_out(hidden)))
        return moments.chunk(2, dim=1)


class WanDecoder3d(nn.Module):
    """Decode a latent video tensor back into RGB frames."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the Wan-style decoder backbone."""

        super().__init__()
        reversed_mult = tuple(reversed(cfg.dim_mult))
        dims = [cfg.dim * scale for scale in (reversed_mult[0], *reversed_mult)]
        scale = 1.0 / (2 ** max(len(cfg.dim_mult) - 2, 0))
        temperal_upsample = tuple(reversed(cfg.temperal_downsample))
        self.conv_in = CausalConv3d(cfg.z_dim, dims[0], kernel_size=3, padding=1)
        self.middle = nn.ModuleList(
            [
                ResidualBlock(dims[0], dims[0], cfg.dropout),
                AttentionBlock(dims[0]),
                ResidualBlock(dims[0], dims[0], cfg.dropout),
            ]
        )
        self.up_blocks = nn.ModuleList()
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = nn.ModuleList()
            current_dim = in_dim
            for _ in range(cfg.num_res_blocks + 1):
                block.append(ResidualBlock(current_dim, out_dim, cfg.dropout))
                if scale in cfg.attn_scales:
                    block.append(AttentionBlock(out_dim))
                current_dim = out_dim
            if index != len(cfg.dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[index] else "upsample2d"
                block.append(Resample(out_dim, mode=mode, out_dim=out_dim))
                scale *= 2.0
            self.up_blocks.append(block)
        self.norm_out = RMSNorm(dims[-1], images=False)
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(dims[-1], 3, kernel_size=3, padding=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return decoded RGB video frames in the `[0, 1]` range."""

        hidden = self.conv_in(latents)
        for layer in self.middle:
            hidden = layer(hidden)
        for block in self.up_blocks:
            for layer in block:
                hidden = layer(hidden)
        return torch.sigmoid(self.conv_out(self.act_out(self.norm_out(hidden))))


class WanVAEEncoder(nn.Module):
    """Wrap the Wan-style encoder for 4D image batches."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the image-to-latent encoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.backbone = WanEncoder3d(cfg)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior moments for an image batch."""

        mu, log_var = self.backbone(images.unsqueeze(2))
        return mu.squeeze(2), log_var.squeeze(2)


class WanVAEDecoder(nn.Module):
    """Wrap the Wan-style decoder for 4D latent image batches."""

    def __init__(self, cfg: WanVAEConfig) -> None:
        """Create the latent-to-image decoder wrapper."""

        super().__init__()
        self.cfg = cfg
        self.backbone = WanDecoder3d(cfg)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return decoded images for a batch of latent tensors."""

        return self.backbone(latents.unsqueeze(2)).squeeze(2)


def kl_divergence_from_moments(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Return the batch-averaged KL divergence to a unit Gaussian prior."""

    return 0.5 * torch.mean(torch.exp(log_var) + mu.square() - 1.0 - log_var)


def sample_posterior(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Draw a reparameterized sample from a diagonal Gaussian posterior."""

    std = torch.exp(0.5 * log_var)
    return mu + std * torch.randn_like(std)
