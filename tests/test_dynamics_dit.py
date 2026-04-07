"""Tests for the minimal rectified-flow DiT dynamics module."""

from __future__ import annotations

import pytest
import torch

from world_model_v2.minimal.dynamics_dit import (
    MinimalRFDiTConfig,
    MinimalRectifiedFlowDynamics,
)


def build_dynamics() -> MinimalRectifiedFlowDynamics:
    """Create a small RF DiT dynamics module for unit tests."""

    return MinimalRectifiedFlowDynamics(
        MinimalRFDiTConfig(
            max_img_h=8,
            max_img_w=8,
            max_frames=5,
            in_channels=4,
            out_channels=4,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=64,
            num_blocks=2,
            num_heads=4,
            action_dim=4,
            dynamics_infer_steps=4,
            dynamics_train_timesteps=32,
            dynamics_rf_shift=5.0,
        )
    )


def test_rf_dit_forward_preserves_video_shape() -> None:
    """The tiny DiT should return one velocity tensor per latent input."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(2, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((2, 5), 10.0)
    condition_mask = dynamics.make_condition_mask(conditioning_latent_video)
    actions = dynamics.make_zero_actions(batch_size=2, device=noisy_latent_video.device, dtype=noisy_latent_video.dtype)
    output = dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=actions,
        conditioning_latent_video=conditioning_latent_video,
    )
    assert output.shape == noisy_latent_video.shape
    assert torch.isfinite(output).all()


def test_prepare_training_inputs_uses_full_clip_rf_interpolation() -> None:
    """RF training inputs should noise the full clip and share one timestep per item."""

    dynamics = build_dynamics()
    clean_latent_video = torch.randn(2, 4, 5, 8, 8)
    prepared = dynamics.prepare_training_inputs(clean_latent_video)
    assert prepared.noisy_latent_video.shape == clean_latent_video.shape
    assert prepared.conditioning_latent_video.shape == clean_latent_video.shape
    assert prepared.target_velocity.shape == clean_latent_video.shape
    assert prepared.timesteps.shape == (2, 5)
    assert prepared.actions.shape == (2, 4, 4)
    sigma = prepared.target_sigmas.view(-1, 1, 1, 1, 1)
    assert torch.allclose(prepared.conditioning_latent_video, clean_latent_video)
    assert torch.allclose(
        prepared.noisy_latent_video,
        clean_latent_video + sigma * prepared.target_velocity,
        atol=1e-6,
    )
    assert not torch.allclose(
        prepared.target_velocity[:, :, : dynamics.cfg.context_frames],
        torch.zeros_like(prepared.target_velocity[:, :, : dynamics.cfg.context_frames]),
    )
    assert torch.all(prepared.condition_mask[:, :, 0] == 1.0)
    assert torch.all(prepared.condition_mask[:, :, 1] == 1.0)
    assert torch.all(prepared.condition_mask[:, :, 2] == 1.0)
    assert torch.all(prepared.condition_mask[:, :, 3] == 0.0)
    assert torch.all(prepared.condition_mask[:, :, 4] == 0.0)
    assert torch.allclose(prepared.timesteps, prepared.timesteps[:, :1].expand_as(prepared.timesteps))
    assert torch.all(prepared.target_sigmas > 0.0)


def test_make_zero_actions_matches_transition_chunk_shape() -> None:
    """Zero-action fallback should follow DreamDojo's four-action chunk semantics."""

    dynamics = build_dynamics()
    actions = dynamics.make_zero_actions(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)
    assert actions.shape == (3, 4, 4)
    assert torch.count_nonzero(actions) == 0


def test_forward_rejects_wrong_action_horizon() -> None:
    """The DiT should fail clearly when the action chunk length is wrong."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 10.0)
    condition_mask = dynamics.make_condition_mask(conditioning_latent_video)
    with pytest.raises(ValueError, match="action steps per chunk"):
        dynamics(
            noisy_latent_video=noisy_latent_video,
            timesteps=timesteps,
            condition_mask=condition_mask,
            actions=torch.zeros(1, 5, 4),
            conditioning_latent_video=conditioning_latent_video,
        )


def test_nonzero_actions_change_timestep_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Active action conditioning should change the timestep embedding passed into the blocks."""

    dynamics = build_dynamics()
    clean_latent_video = torch.randn(1, 4, 5, 8, 8)
    timesteps = torch.full((1, 5), 9.0)
    condition_mask = dynamics.make_condition_mask(clean_latent_video)
    captured: list[torch.Tensor] = []
    original_forward = dynamics.net.blocks[0].forward

    def capture_forward(
        self: object,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the first block's timestep embedding before delegating."""

        captured.append(timestep_embedding.detach().clone())
        return original_forward(x, timestep_embedding, cos, sin)

    monkeypatch.setattr(
        dynamics.net.blocks[0],
        "forward",
        capture_forward.__get__(dynamics.net.blocks[0], type(dynamics.net.blocks[0])),
    )
    zero_actions = dynamics.make_zero_actions(batch_size=1, device=clean_latent_video.device, dtype=clean_latent_video.dtype)
    nonzero_actions = torch.arange(16, dtype=clean_latent_video.dtype).view(1, 4, 4)
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=zero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=nonzero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    assert len(captured) == 2
    assert not torch.allclose(captured[0], captured[1])


def test_forward_repins_conditioned_frames_and_overwrites_conditioned_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper should repin context frames before the net and overwrite their velocity."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 12.0)
    condition_mask = dynamics.make_condition_mask(conditioning_latent_video)
    actions = dynamics.make_zero_actions(batch_size=1, device=noisy_latent_video.device, dtype=noisy_latent_video.dtype)
    target_velocity = torch.randn_like(conditioning_latent_video)
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the repinned model input and return a constant target-frame velocity."""

        del timesteps_B_T, condition_video_input_mask_B_C_T_H_W, action
        captured["x"] = x_B_C_T_H_W.detach().clone()
        return torch.full_like(x_B_C_T_H_W, 7.0)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    output = dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=actions,
        conditioning_latent_video=conditioning_latent_video,
        target_velocity=target_velocity,
    )
    assert torch.allclose(
        captured["x"][:, :, : dynamics.cfg.context_frames],
        conditioning_latent_video[:, :, : dynamics.cfg.context_frames],
    )
    assert torch.allclose(
        output[:, :, : dynamics.cfg.context_frames],
        target_velocity[:, :, : dynamics.cfg.context_frames],
    )
    assert torch.allclose(
        output[:, :, dynamics.cfg.context_frames :],
        torch.full_like(output[:, :, dynamics.cfg.context_frames :], 7.0),
    )


def test_sample_next_latent_keeps_context_pinned_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RF sampling should repin the context frames before every DiT forward."""

    dynamics = build_dynamics()
    context = torch.randn(2, 4, 3, 8, 8)
    captured_inputs: list[torch.Tensor] = []

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture every full latent clip passed into the DiT during sampling."""

        del timesteps_B_T, condition_video_input_mask_B_C_T_H_W, action
        captured_inputs.append(x_B_C_T_H_W.detach().clone())
        return torch.zeros_like(x_B_C_T_H_W)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    generator = torch.Generator(device=context.device.type)
    generator.manual_seed(7)
    next_latent = dynamics.sample_next_latent(context, generator=generator)
    assert next_latent.shape == (2, 4, 2, 8, 8)
    assert torch.isfinite(next_latent).all()
    assert len(captured_inputs) == dynamics.cfg.dynamics_infer_steps
    for latent_video in captured_inputs:
        assert torch.allclose(latent_video[:, :, : dynamics.cfg.context_frames], context)
