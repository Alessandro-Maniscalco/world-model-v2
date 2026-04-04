"""Tests for the minimal world-model architecture."""

from __future__ import annotations

import torch

from world_model_v2.minimal.model import MinimalWorldModel


def test_minimal_world_model_preserves_expected_shapes() -> None:
    """The encoder, decoder, and dynamics should use the requested tensor shapes."""

    model = MinimalWorldModel()
    images = torch.rand(2, 3, 128, 128)
    latents = model.encode(images)
    reconstructed = model.decode(latents)
    next_latents = model.predict_next_latent(latents)
    assert latents.shape == (2, 4, 32, 32)
    assert reconstructed.shape == (2, 3, 128, 128)
    assert next_latents.shape == latents.shape


def test_minimal_world_model_rollout_includes_seed_frame() -> None:
    """Autoregressive rollout should return the seed plus the requested predictions."""

    model = MinimalWorldModel()
    seed = torch.rand(1, 3, 128, 128)
    rollout = model.rollout(seed, steps=5)
    assert rollout.shape == (1, 6, 3, 128, 128)
