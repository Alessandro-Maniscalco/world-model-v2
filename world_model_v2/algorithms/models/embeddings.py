"""Embedding helpers used by latent-dynamics model components."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Build sinusoidal embeddings for consistency-model timestep pairs."""

    original_shape = timesteps.shape
    flat_timesteps = timesteps.reshape(-1)
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
    args = flat_timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding.view(*original_shape, dim)
