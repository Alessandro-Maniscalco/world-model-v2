"""Runtime-heavy Wan model tests kept out of the fast minimal suite."""

from __future__ import annotations

import torch

from world_model_v2.minimal.model import MinimalWorldModel


def test_wan_world_model_preserves_expected_shapes() -> None:
    """The Wan model should keep the expected image and latent tensor shapes."""

    model = MinimalWorldModel(resolution=128)
    images = torch.rand(2, 3, 128, 128)
    latents = model.encode(images, deterministic=True)
    reconstructed = model.decode(latents)
    next_latents = model.predict_next_latent(latents)
    assert latents.shape == (2, 16, 16, 16)
    assert reconstructed.shape == (2, 3, 128, 128)
    assert next_latents.shape == latents.shape


def test_wan_world_model_derives_latent_size_from_resolution() -> None:
    """The Wan model should derive latent height and width from the input resolution."""

    model = MinimalWorldModel(resolution=64)
    images = torch.rand(2, 3, 64, 64)
    latents = model.encode(images, deterministic=True)
    assert latents.shape == (2, 16, 8, 8)


def test_wan_world_model_supports_rectangular_inputs() -> None:
    """The Wan model should derive latent height and width for rectangular inputs."""

    model = MinimalWorldModel(resolution=128, height=240, width=320)
    images = torch.rand(2, 3, 240, 320)
    latents = model.encode(images, deterministic=True)
    reconstructed = model.decode(latents)
    assert latents.shape == (2, 16, 30, 40)
    assert reconstructed.shape == (2, 3, 240, 320)


def test_wan_world_model_autoencode_reports_kl_statistics() -> None:
    """The Wan backend should expose posterior statistics for KL-regularized training."""

    model = MinimalWorldModel(resolution=128)
    images = torch.rand(2, 3, 128, 128)
    output = model.autoencode(images, sample_posterior=True)
    assert output.reconstructed.shape == (2, 3, 128, 128)
    assert output.mu.shape == (2, 16, 16, 16)
    assert output.log_var.shape == (2, 16, 16, 16)
    assert output.latent.shape == output.mu.shape
    assert output.kl_loss.ndim == 0
    assert float(output.kl_loss) >= 0.0


def test_wan_world_model_rollout_includes_seed_frame() -> None:
    """Autoregressive rollout should return the seed plus the requested predictions."""

    model = MinimalWorldModel(resolution=128)
    seed = torch.rand(1, 3, 128, 128)
    rollout = model.rollout(seed, steps=5)
    assert rollout.shape == (1, 6, 3, 128, 128)
