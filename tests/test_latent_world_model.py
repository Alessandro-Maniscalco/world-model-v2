"""Tests for the upstream-shaped latent world model."""

from __future__ import annotations

import pytest
import torch

from world_model_v2.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from world_model_v2.config import AlgorithmConfig


def make_batch(batch_size: int = 2, horizon: int = 1) -> dict[str, object]:
    """Create a small synthetic batch with upstream-shaped observation keys."""

    return {
        "obs": {"camera_1_color": torch.rand(batch_size, horizon, 3, 128, 128)},
        "action": torch.rand(batch_size, horizon, 4),
        "episode_idx": torch.zeros(batch_size, dtype=torch.long),
    }


def test_training_step_returns_loss_dict() -> None:
    """Stage-1 training should return the expected loss payload."""

    model = LatentWorldModel(
        AlgorithmConfig(training_stage=1, latent_channels=4, latent_dim=64, hidden_channels=32, timesteps=8),
        obs_keys=("camera_1_color",),
    )
    outputs = model.training_step(make_batch())
    assert set(outputs) == {"loss", "recon_loss", "clean_loss", "latents"}
    assert outputs["loss"].ndim == 0
    assert outputs["latents"].shape == (2, 4, 32, 32)


def test_reconstruct_preserves_sequence_shape() -> None:
    """Reconstruction should preserve batch and horizon axes."""

    model = LatentWorldModel(
        AlgorithmConfig(training_stage=1, latent_channels=4, latent_dim=64, hidden_channels=32, timesteps=8, infer_steps=2),
        obs_keys=("camera_1_color",),
    )
    reconstructed = model.reconstruct(make_batch()["obs"], num_steps=2)
    assert reconstructed.shape == (2, 1, 3, 128, 128)


def test_training_stage_other_than_one_is_not_implemented() -> None:
    """Only Stage 1 should be accepted in this refactor."""

    with pytest.raises(NotImplementedError):
        LatentWorldModel(AlgorithmConfig(training_stage=2), obs_keys=("camera_1_color",))

