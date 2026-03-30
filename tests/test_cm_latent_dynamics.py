"""Tests for the Stage-2 latent dynamics model."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.models.cm_latent_dynamics import CMLatentDynamics


def build_model() -> CMLatentDynamics:
    """Create a small latent dynamics model for unit tests."""

    return CMLatentDynamics(
        latent_channels=4,
        hidden_channels=32,
        cond_dim=64,
        action_dim=4,
        action_emb_dim=64,
        attention_heads=4,
    )


def test_cm_latent_dynamics_preserves_sequence_shape() -> None:
    """The dynamics model should return one latent prediction per input frame."""

    model = build_model()
    latents = torch.randn(2, 4, 4, 32, 32)
    t = torch.randint(0, 8, (2, 4))
    s = torch.zeros_like(t)
    actions = torch.randn(2, 4, 4)
    outputs = model(latents, t, s, actions)
    assert outputs.shape == latents.shape
    assert torch.isfinite(outputs).all()


def test_cm_latent_dynamics_is_causal_across_time() -> None:
    """Future latent inputs should not change earlier outputs."""

    model = build_model()
    latents = torch.randn(1, 4, 4, 32, 32)
    latents_changed = latents.clone()
    latents_changed[:, -1] += 10.0
    t = torch.randint(0, 8, (1, 4))
    s = torch.zeros_like(t)
    actions = torch.randn(1, 4, 4)
    base_outputs = model(latents, t, s, actions)
    changed_outputs = model(latents_changed, t, s, actions)
    assert torch.allclose(base_outputs[:, :-1], changed_outputs[:, :-1], atol=1e-5, rtol=1e-4)
