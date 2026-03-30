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


def test_scheduler_supports_sequence_batch_shapes() -> None:
    """Noise sampling and injection should work on `(B, T, ...)` latent tensors."""

    scheduler = Stage1NoiseScheduler(timesteps=10)
    timesteps, stop_timesteps = scheduler.sample_training_pair((2, 3), device="cpu")
    latents = torch.zeros(2, 3, 4, 8, 8)
    noisy = scheduler.add_noise(latents, timesteps)
    weights = scheduler.get_weights(stop_timesteps)
    assert timesteps.shape == (2, 3)
    assert stop_timesteps.shape == (2, 3)
    assert noisy.shape == latents.shape
    assert weights.shape == (2, 3)


def test_add_noise_to_t_s_reuses_one_shared_noise_sample() -> None:
    """Stage-2 noise pairs should differ only by the requested sigma scaling."""

    scheduler = Stage1NoiseScheduler(timesteps=10, sigma_min=0.1, sigma_max=1.0)
    latents = torch.zeros(2, 3, 4, 8, 8)
    t = torch.full((2, 3), 8, dtype=torch.long)
    s = torch.full((2, 3), 2, dtype=torch.long)
    noisy_t, noisy_s = scheduler.add_noise_to_t_s(latents, t, s)
    sigma_t = scheduler.sigmas[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    sigma_s = scheduler.sigmas[s].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    recovered_t = noisy_t / sigma_t
    recovered_s = noisy_s / sigma_s
    assert torch.allclose(recovered_t, recovered_s, atol=1e-6, rtol=1e-6)


def test_sample_stage2_noise_levels_matches_terminal_only_layout() -> None:
    """Stage-2 terminal-only sampling should keep context frames at matched low noise."""

    scheduler = Stage1NoiseScheduler(timesteps=20)
    t, s = scheduler.sample_stage2_noise_levels(
        batch_size=3,
        horizon=5,
        device="cpu",
        sampling_strategy="terminal_only",
        prev_frame_noise_scale=0.1,
        dyn_infer_steps=1,
    )
    assert t.shape == (3, 5)
    assert s.shape == (3, 5)
    assert torch.equal(t[:, :-1], s[:, :-1])
    assert torch.all(t[:, :-1] >= 1)
    assert torch.all(t[:, :-1] < max(2, int(20 * 0.1)))
    assert torch.all(t[:, -1] == 19)
    assert torch.all(s[:, -1] == 0)


def test_uniform_loss_weighting_returns_ones() -> None:
    """Uniform weighting should produce a ones tensor for Stage-2 loss matching."""

    scheduler = Stage1NoiseScheduler(timesteps=10)
    weights = scheduler.get_weights(torch.tensor([1, 4, 7]), weighting="uniform")
    assert torch.equal(weights, torch.ones(3))


def test_make_sampling_schedule_reaches_zero() -> None:
    """Sampling schedules should descend to timestep zero."""

    scheduler = Stage1NoiseScheduler(timesteps=9)
    schedule = scheduler.make_sampling_schedule(num_steps=4, device="cpu")
    assert len(schedule) == 4
    assert int(schedule[0][0].item()) == 8
    assert int(schedule[-1][1].item()) == 0
