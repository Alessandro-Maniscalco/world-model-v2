"""Tests for timestep embeddings."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.models.embeddings import timestep_embedding


def test_timestep_embedding_returns_expected_shape() -> None:
    """Timestep embeddings should have a stable dense shape."""

    embedding = timestep_embedding(torch.tensor([1, 3, 5]), dim=64)
    assert embedding.shape == (3, 64)
    assert torch.isfinite(embedding).all()

