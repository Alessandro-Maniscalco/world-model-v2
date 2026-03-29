"""Tests for the CNN encoder."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.models.cnn_encoder import CNNEncoder


def test_cnn_encoder_reduces_resolution_twice() -> None:
    """The encoder should map 128x128 images to a 32x32 latent grid."""

    encoder = CNNEncoder(image_channels=3, latent_channels=4, hidden_channels=32)
    images = torch.rand(2, 3, 128, 128)
    latents = encoder(images)
    assert latents.shape == (2, 4, 32, 32)

