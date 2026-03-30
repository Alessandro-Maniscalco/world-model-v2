"""Lightweight action-conditioned latent dynamics network for Stage 2 and 3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_v2.algorithms.models.embeddings import timestep_embedding


def _group_norm_groups(channels: int, max_groups: int = 8) -> int:
    """Return a valid GroupNorm group count for the requested channel size."""

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalAttentionBlock(nn.Module):
    """Apply causal temporal self-attention at each spatial location."""

    def __init__(self, channels: int, heads: int) -> None:
        """Create a lightweight temporal attention block."""

        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix information across time while preserving causal order."""

        batch_size, channels, steps, height, width = x.shape
        sequence = x.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, steps, channels)
        normalized = self.norm(sequence)
        query, key, value = self.to_qkv(normalized).chunk(3, dim=-1)
        query = query.view(-1, steps, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(-1, steps, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(-1, steps, self.heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(-1, steps, channels)
        mixed = self.to_out(attended)
        output = sequence + mixed
        return output.view(batch_size, height, width, steps, channels).permute(0, 4, 3, 1, 2)


class FiLMResidual3DBlock(nn.Module):
    """Apply FiLM-conditioned residual refinement with spatial Conv3d kernels."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        """Create a residual block with per-frame FiLM conditioning."""

        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.cond1 = nn.Linear(cond_dim, in_channels * 2)
        self.cond2 = nn.Linear(cond_dim, out_channels * 2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Refine a latent video tensor with timestep and action conditioning."""

        batch_size, _, steps, height, width = x.shape
        scale1, shift1 = self.cond1(cond).chunk(2, dim=-1)
        scale1 = scale1.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        shift1 = shift1.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        hidden = self._apply_group_norm(self.norm1, x, batch_size, steps, height, width)
        hidden = hidden * (1.0 + scale1) + shift1
        hidden = F.silu(hidden)
        hidden = self.conv1(hidden)

        scale2, shift2 = self.cond2(cond).chunk(2, dim=-1)
        scale2 = scale2.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        shift2 = shift2.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        hidden = self._apply_group_norm(self.norm2, hidden, batch_size, steps, height, width)
        hidden = hidden * (1.0 + scale2) + shift2
        hidden = F.silu(hidden)
        hidden = self.conv2(hidden)
        return hidden + self.skip(x)

    def _apply_group_norm(
        self,
        norm: nn.GroupNorm,
        x: torch.Tensor,
        batch_size: int,
        steps: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Apply GroupNorm independently at each frame to preserve causality."""

        normalized = x.permute(0, 2, 1, 3, 4).reshape(batch_size * steps, x.shape[1], height, width)
        normalized = norm(normalized)
        return normalized.reshape(batch_size, steps, x.shape[1], height, width).permute(0, 2, 1, 3, 4)


class CMLatentDynamics(nn.Module):
    """Predict denoised latent sequences conditioned on actions and past latents."""

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        cond_dim: int,
        action_dim: int,
        action_emb_dim: int,
        attention_heads: int,
    ) -> None:
        """Create the lightweight causal latent-dynamics model."""

        super().__init__()
        if hidden_channels % attention_heads != 0:
            raise ValueError("hidden_channels must be divisible by attention_heads")
        self.cond_dim = cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.SiLU(),
            nn.Linear(action_emb_dim, cond_dim),
        )
        self.init = nn.Conv3d(
            latent_channels,
            hidden_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.init_attn = TemporalAttentionBlock(hidden_channels, attention_heads)
        self.down_block = FiLMResidual3DBlock(hidden_channels, hidden_channels, cond_dim)
        self.down_attn = TemporalAttentionBlock(hidden_channels, attention_heads)
        self.downsample = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.mid_block = FiLMResidual3DBlock(hidden_channels, hidden_channels, cond_dim)
        self.mid_attn = TemporalAttentionBlock(hidden_channels, attention_heads)
        self.upsample = nn.ConvTranspose3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.up_block = FiLMResidual3DBlock(hidden_channels * 2, hidden_channels, cond_dim)
        self.up_attn = TemporalAttentionBlock(hidden_channels, attention_heads)
        self.out = nn.Conv3d(hidden_channels, latent_channels, kernel_size=1)

    def build_condition(
        self,
        timesteps: torch.Tensor,
        stop_timesteps: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-frame FiLM conditioning from timestep pairs and actions."""

        time_cond = torch.cat(
            [
                timestep_embedding(timesteps, self.cond_dim),
                timestep_embedding(stop_timesteps, self.cond_dim),
            ],
            dim=-1,
        )
        return self.time_mlp(time_cond) + self.action_mlp(actions)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        stop_timesteps: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a lower-noise latent sequence for the requested timestep pairs."""

        cond = self.build_condition(timesteps, stop_timesteps, actions)
        x = latents.permute(0, 2, 1, 3, 4)
        base = self.init_attn(self.init(x))
        skip = self.down_attn(self.down_block(base, cond))
        hidden = self.downsample(skip)
        hidden = self.mid_attn(self.mid_block(hidden, cond))
        hidden = self.upsample(hidden)
        hidden = torch.cat([hidden, skip], dim=1)
        hidden = self.up_attn(self.up_block(hidden, cond))
        output = self.out(hidden + base)
        return output.permute(0, 2, 1, 3, 4)
