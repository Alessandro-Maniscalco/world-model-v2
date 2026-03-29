"""Tests for the consistency-style decoder."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.models.cm_decoder import CMDecoder


def test_cm_decoder_outputs_image_shaped_predictions() -> None:
    """The decoder should emit image-shaped outputs in `[0, 1]`."""

    decoder = CMDecoder(image_channels=3, latent_channels=4, hidden_channels=32, latent_dim=64)
    noisy = torch.rand(2, 3, 128, 128)
    latents = torch.rand(2, 4, 32, 32)
    prediction = decoder(noisy, torch.tensor([5, 4]), torch.tensor([1, 0]), latents)
    assert prediction.shape == noisy.shape
    assert torch.all(prediction >= 0.0)
    assert torch.all(prediction <= 1.0)

