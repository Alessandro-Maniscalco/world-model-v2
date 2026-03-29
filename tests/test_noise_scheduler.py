"""Tests for the Stage-1 latent-dynamics noise scheduler."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.latent_dynamics.noise_scheduler import Stage1NoiseScheduler


def test_sample_training_pair_returns_ordered_timesteps() -> None:
    """Sampled timestep pairs should satisfy s < t."""

    scheduler = Stage1NoiseScheduler(timesteps=12)
    t, s = scheduler.sample_training_pair(batch_size=32, device="cpu")
    assert torch.all(t >= 1)
    assert torch.all(s >= 0)
    assert torch.all(s < t)


def test_add_noise_and_weights_preserve_expected_shapes() -> None:
    """Noise addition and weight lookup should broadcast over image batches."""

    scheduler = Stage1NoiseScheduler(timesteps=10)
    images = torch.zeros(4, 3, 8, 8)
    timesteps = torch.tensor([1, 2, 3, 4])
    noisy = scheduler.add_noise(images, timesteps)
    weights = scheduler.get_weights(timesteps)
    assert noisy.shape == images.shape
    assert weights.shape == (4,)
    assert torch.all(weights > 0)


def test_make_sampling_schedule_reaches_zero() -> None:
    """Sampling schedules should descend to timestep zero."""

    scheduler = Stage1NoiseScheduler(timesteps=9)
    schedule = scheduler.make_sampling_schedule(num_steps=4, device="cpu")
    assert len(schedule) == 4
    assert int(schedule[0][0].item()) == 8
    assert int(schedule[-1][1].item()) == 0
