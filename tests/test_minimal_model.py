"""Tests for the minimal world-model architecture."""

from __future__ import annotations

import pytest

from world_model_v2.minimal.model import MinimalWorldModel


def test_minimal_world_model_rejects_removed_conv_backend() -> None:
    """The minimal model should reject the removed conv fallback backend."""

    with pytest.raises(ValueError, match="only supports the Wan VAE"):
        MinimalWorldModel(ae_backend="conv", resolution=128)
