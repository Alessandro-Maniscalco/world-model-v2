"""Tests for the root world-model architecture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from world_model_v2.model import WorldModel


def test_world_model_rejects_removed_conv_backend() -> None:
    """The world model should reject the removed conv fallback backend."""

    with pytest.raises(ValueError, match="only supports the Wan VAE"):
        WorldModel(ae_backend="conv", resolution=128)


def test_world_model_preserves_requested_dynamics_architecture() -> None:
    """The world model should build the RF DiT with the requested dynamics settings."""

    model = WorldModel(
        ae_backend="wan",
        resolution=128,
        dynamics_model_channels=384,
        dynamics_num_blocks=6,
        dynamics_num_heads=8,
        dynamics_action_conditioning_mode="global_chunk",
        dynamics_zero_init_action_embedder=True,
        dynamics_use_adaln_lora=True,
        dynamics_adaln_lora_dim=96,
        dynamics_rope_t_extrapolation_ratio=1.5,
    )

    assert model.dynamics.cfg.model_channels == 384
    assert model.dynamics.cfg.num_blocks == 6
    assert model.dynamics.cfg.num_heads == 8
    assert model.dynamics.cfg.action_conditioning_mode == "global_chunk"
    assert model.dynamics.cfg.zero_init_action_embedder is True
    assert model.dynamics.cfg.use_adaln_lora is True
    assert model.dynamics.cfg.adaln_lora_dim == 96
    assert model.dynamics.cfg.rope_t_extrapolation_ratio == 1.5


def test_world_model_preserves_requested_dynamics_layout_controls() -> None:
    """The world model should retain custom frame-layout and validation settings."""

    model = WorldModel(
        ae_backend="wan",
        resolution=128,
        dynamics_context_frames=1,
        dynamics_target_frames=3,
        dynamics_conditioning_frame_choices=(1,),
        dynamics_conditioning_frame_probabilities=(1.0,),
        dynamics_validation_conditioning_frame_choices=(1,),
        dynamics_open_rollout_context_frames=1,
    )

    assert model.dynamics.cfg.context_frames == 1
    assert model.dynamics.cfg.target_frames == 3
    assert model.dynamics.cfg.max_frames == 4
    assert model.dynamics.cfg.conditioning_frame_choices == (1,)
    assert model.dynamics.cfg.conditioning_frame_probabilities == (1.0,)
    assert model.dynamics.cfg.validation_conditioning_frame_choices == (1,)
    assert model.dynamics.cfg.open_rollout_context_frames == 1


def test_world_model_rollout_supports_multi_target_layouts() -> None:
    """Rollout should pad late action windows and stop after the requested future frames."""

    captured_action_windows: list[torch.Tensor] = []
    cfg = SimpleNamespace(
        conditioning_frame_choices=(1,),
        context_frames=1,
        target_frames=3,
        max_frames=4,
        num_action_per_chunk=3,
        action_dim=4,
    )

    class DummyWorldModel:
        """Minimal rollout harness that exercises the shared rollout logic."""

        dynamics = SimpleNamespace(cfg=cfg)

        def encode_context_frames(self, images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
            """Encode images into a fake latent tensor with matching frame order."""

            del deterministic
            return images.permute(0, 2, 1, 3, 4)

        def predict_next_latent(
            self,
            latents: torch.Tensor,
            actions: torch.Tensor | None = None,
            infer_steps: int | None = None,
            generator: torch.Generator | None = None,
            guidance_scale: float | None = None,
        ) -> torch.Tensor:
            """Return one fixed three-frame latent chunk while recording the action window."""

            del infer_steps, generator, guidance_scale
            if actions is not None:
                captured_action_windows.append(actions.detach().clone())
            batch_size, channels, _, height, width = latents.shape
            target = torch.zeros(batch_size, channels, 3, height, width)
            target[:, :, 0] = 1.0
            target[:, :, 1] = 2.0
            target[:, :, 2] = 3.0
            return target

        def decode_frame_sequence(self, latents: torch.Tensor) -> torch.Tensor:
            """Decode the fake latent tensor back into image-frame ordering."""

            return latents.permute(0, 2, 1, 3, 4)

    seed_frames = torch.zeros(1, 1, 3, 2, 2)
    actions = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    rollout = WorldModel.rollout(DummyWorldModel(), seed_frames, steps=4, actions=actions)

    assert rollout.shape == (1, 5, 3, 2, 2)
    assert len(captured_action_windows) == 2
    assert torch.equal(captured_action_windows[0], actions[:, :3])
    assert torch.equal(captured_action_windows[1][:, :1], actions[:, 3:4])
    assert torch.count_nonzero(captured_action_windows[1][:, 1:]) == 0
