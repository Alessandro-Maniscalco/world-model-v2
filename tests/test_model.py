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
        dynamics_action_conditioning_mode="chunk_per_frame",
        dynamics_zero_init_action_embedder=True,
        dynamics_use_adaln_lora=True,
        dynamics_adaln_lora_dim=96,
        dynamics_rope_t_extrapolation_ratio=1.5,
    )

    assert model.dynamics.cfg.model_channels == 384
    assert model.dynamics.cfg.num_blocks == 6
    assert model.dynamics.cfg.num_heads == 8
    assert model.dynamics.cfg.action_conditioning_mode == "chunk_per_frame"
    assert model.dynamics.cfg.zero_init_action_embedder is True
    assert model.dynamics.cfg.use_adaln_lora is True
    assert model.dynamics.cfg.adaln_lora_dim == 96
    assert model.dynamics.cfg.rope_t_extrapolation_ratio == 1.5


def test_world_model_rejects_unsupported_dynamics_variants() -> None:
    """The world model should fail fast on removed non-DreamDojo dynamics options."""

    with pytest.raises(ValueError, match="chunk_per_frame"):
        WorldModel(ae_backend="wan", resolution=128, dynamics_action_conditioning_mode="global_chunk")
    with pytest.raises(ValueError, match="use_learned_temporal_embedding"):
        WorldModel(ae_backend="wan", resolution=128, dynamics_use_learned_temporal_embedding=True)
    with pytest.raises(ValueError, match="use_adaln_lora=True"):
        WorldModel(ae_backend="wan", resolution=128, dynamics_use_adaln_lora=False)


def test_world_model_preserves_requested_dynamics_layout_controls() -> None:
    """The world model should retain custom frame-layout and validation settings."""

    model = WorldModel(
        ae_backend="wan",
        resolution=128,
        dynamics_context_frames=1,
        dynamics_target_frames=3,
        conditional_frame_sigma=1e-4,
        dynamics_conditioning_frame_choices=(1,),
        dynamics_conditioning_frame_probabilities=(1.0,),
        dynamics_validation_conditioning_frame_choices=(1,),
        dynamics_open_rollout_context_frames=1,
        dynamics_open_rollout_stride_frames=1,
    )

    assert model.dynamics.cfg.context_frames == 1
    assert model.dynamics.cfg.target_frames == 3
    assert model.dynamics.cfg.max_frames == 4
    assert model.dynamics.cfg.conditional_frame_sigma == pytest.approx(1e-4)
    assert model.dynamics.cfg.conditioning_frame_choices == (1,)
    assert model.dynamics.cfg.conditioning_frame_probabilities == (1.0,)
    assert model.dynamics.cfg.validation_conditioning_frame_choices == (1,)
    assert model.dynamics.cfg.open_rollout_context_frames == 1
    assert model.dynamics.cfg.open_rollout_stride_frames == 1


def test_world_model_rollout_supports_multi_target_layouts() -> None:
    """Rollout should pad late action windows and stop after the requested future frames."""

    captured_action_windows: list[torch.Tensor] = []
    cfg = SimpleNamespace(
        conditioning_frame_choices=(1,),
        context_frames=1,
        target_frames=3,
        max_frames=4,
        num_action_per_chunk=12,
        action_dim=4,
        temporal_compression_ratio=4,
        open_rollout_context_frames=1,
        open_rollout_stride_frames=None,
    )

    class DummyWorldModel:
        """Minimal rollout harness that exercises the shared rollout logic."""

        dynamics = SimpleNamespace(cfg=cfg)
        temporal_downsample_factor = 4

        def pixel_frames_to_latent_frames(self, pixel_frames: int, *, exact: bool = False) -> int:
            """Map Wan pixel-frame counts into latent-frame counts for the rollout harness."""

            latent_frames = 1 + (pixel_frames - 1) // self.temporal_downsample_factor
            if exact and (1 + (latent_frames - 1) * self.temporal_downsample_factor) != pixel_frames:
                raise ValueError("pixel_frames must align to the harness temporal ratio.")
            return latent_frames

        def latent_frames_to_pixel_frames(self, latent_frames: int) -> int:
            """Map harness latent-frame counts back into Wan pixel-frame counts."""

            return 1 + (latent_frames - 1) * self.temporal_downsample_factor

        def resolved_rollout_stride_frames(
            self,
            context_frames: int,
            stride_frames: int | None = None,
        ) -> int:
            """Delegate stride resolution to the shared world-model helper."""

            return WorldModel.resolved_rollout_stride_frames(self, context_frames, stride_frames)

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

        def decode_target_latents(
            self,
            context_latents: torch.Tensor,
            target_latents: torch.Tensor,
            *,
            context_pixel_frames: int,
            target_pixel_frames: int | None = None,
        ) -> torch.Tensor:
            """Expand each predicted latent into four pixel frames for rollout assertions."""

            del context_latents, context_pixel_frames
            expanded = torch.cat(
                [
                    target_latents[:, :, index : index + 1]
                    .permute(0, 2, 1, 3, 4)
                    .repeat(1, self.temporal_downsample_factor, 1, 1, 1)
                    for index in range(target_latents.shape[2])
                ],
                dim=1,
            )
            return expanded[:, :target_pixel_frames]

    seed_frames = torch.zeros(1, 1, 3, 2, 2)
    actions = torch.arange(64, dtype=torch.float32).view(1, 16, 4)

    rollout = WorldModel.rollout(DummyWorldModel(), seed_frames, steps=16, actions=actions)

    assert rollout.shape == (1, 17, 3, 2, 2)
    assert len(captured_action_windows) == 2
    assert torch.equal(captured_action_windows[0], actions[:, :12])
    assert torch.equal(captured_action_windows[1][:, :4], actions[:, 12:16])
    assert torch.count_nonzero(captured_action_windows[1][:, 4:]) == 0


def test_world_model_resolves_default_rollout_stride_from_layout() -> None:
    """Default rollout stride should expand to the chunk capacity for the current context."""

    implicit_stride_model = SimpleNamespace(
        dynamics=SimpleNamespace(
            cfg=SimpleNamespace(
                max_frames=4,
                open_rollout_stride_frames=None,
            )
        )
    )
    explicit_stride_model = SimpleNamespace(
        dynamics=SimpleNamespace(
            cfg=SimpleNamespace(
                max_frames=4,
                open_rollout_stride_frames=1,
            )
        )
    )

    assert WorldModel.resolved_rollout_stride_frames(implicit_stride_model, 1) == 3
    assert WorldModel.resolved_rollout_stride_frames(implicit_stride_model, 2) == 2
    assert WorldModel.resolved_rollout_stride_frames(explicit_stride_model, 1) == 1


def test_world_model_rollout_can_use_overlap_stride() -> None:
    """Rollout should support chunk overlap by appending fewer frames than the chunk predicts."""

    captured_action_windows: list[torch.Tensor] = []
    cfg = SimpleNamespace(
        conditioning_frame_choices=(1,),
        context_frames=1,
        target_frames=2,
        max_frames=3,
        num_action_per_chunk=8,
        action_dim=4,
        temporal_compression_ratio=4,
        open_rollout_context_frames=1,
        open_rollout_stride_frames=1,
    )

    class DummyWorldModel:
        """Minimal rollout harness that exercises overlap-stride logic."""

        dynamics = SimpleNamespace(cfg=cfg)
        temporal_downsample_factor = 4

        def pixel_frames_to_latent_frames(self, pixel_frames: int, *, exact: bool = False) -> int:
            """Map Wan pixel-frame counts into latent-frame counts for the rollout harness."""

            latent_frames = 1 + (pixel_frames - 1) // self.temporal_downsample_factor
            if exact and (1 + (latent_frames - 1) * self.temporal_downsample_factor) != pixel_frames:
                raise ValueError("pixel_frames must align to the harness temporal ratio.")
            return latent_frames

        def latent_frames_to_pixel_frames(self, latent_frames: int) -> int:
            """Map harness latent-frame counts back into Wan pixel-frame counts."""

            return 1 + (latent_frames - 1) * self.temporal_downsample_factor

        def resolved_rollout_stride_frames(
            self,
            context_frames: int,
            stride_frames: int | None = None,
        ) -> int:
            """Delegate stride resolution to the shared world-model helper."""

            return WorldModel.resolved_rollout_stride_frames(self, context_frames, stride_frames)

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
            """Return one fixed two-frame chunk while recording each shifted action window."""

            del infer_steps, generator, guidance_scale
            if actions is not None:
                captured_action_windows.append(actions.detach().clone())
            batch_size, channels, _, height, width = latents.shape
            target = torch.zeros(batch_size, channels, 2, height, width)
            target[:, :, 0] = 1.0
            target[:, :, 1] = 2.0
            return target

        def decode_frame_sequence(self, latents: torch.Tensor) -> torch.Tensor:
            """Decode the fake latent tensor back into image-frame ordering."""

            return latents.permute(0, 2, 1, 3, 4)

        def decode_target_latents(
            self,
            context_latents: torch.Tensor,
            target_latents: torch.Tensor,
            *,
            context_pixel_frames: int,
            target_pixel_frames: int | None = None,
        ) -> torch.Tensor:
            """Expand each predicted latent into four pixel frames for rollout assertions."""

            del context_latents, context_pixel_frames
            expanded = torch.cat(
                [
                    target_latents[:, :, index : index + 1]
                    .permute(0, 2, 1, 3, 4)
                    .repeat(1, self.temporal_downsample_factor, 1, 1, 1)
                    for index in range(target_latents.shape[2])
                ],
                dim=1,
            )
            return expanded[:, :target_pixel_frames]

    seed_frames = torch.zeros(1, 1, 3, 2, 2)
    actions = torch.arange(36, dtype=torch.float32).view(1, 9, 4)

    rollout = WorldModel.rollout(
        DummyWorldModel(),
        seed_frames,
        steps=9,
        actions=actions,
        stride_frames=1,
    )

    assert rollout.shape == (1, 10, 3, 2, 2)
    assert len(captured_action_windows) == 3
    assert torch.equal(captured_action_windows[0], actions[:, 0:8])
    assert torch.equal(captured_action_windows[1][:, :5], actions[:, 4:9])
    assert torch.count_nonzero(captured_action_windows[1][:, 5:]) == 0
    assert torch.equal(captured_action_windows[2][:, :1], actions[:, 8:9])
    assert torch.count_nonzero(captured_action_windows[2][:, 1:]) == 0


def test_world_model_rollout_respects_fixed_open_rollout_context() -> None:
    """Rollout should not silently expand beyond the configured open-rollout context length."""

    captured_action_windows: list[torch.Tensor] = []
    captured_context_lengths: list[int] = []
    cfg = SimpleNamespace(
        conditioning_frame_choices=(1, 3),
        context_frames=1,
        target_frames=3,
        max_frames=4,
        num_action_per_chunk=12,
        action_dim=4,
        temporal_compression_ratio=4,
        open_rollout_context_frames=1,
        open_rollout_stride_frames=None,
    )

    class DummyWorldModel:
        """Minimal rollout harness that records the actual context length per chunk."""

        dynamics = SimpleNamespace(cfg=cfg)
        temporal_downsample_factor = 4

        def pixel_frames_to_latent_frames(self, pixel_frames: int, *, exact: bool = False) -> int:
            """Map Wan pixel-frame counts into latent-frame counts for the rollout harness."""

            latent_frames = 1 + (pixel_frames - 1) // self.temporal_downsample_factor
            if exact and (1 + (latent_frames - 1) * self.temporal_downsample_factor) != pixel_frames:
                raise ValueError("pixel_frames must align to the harness temporal ratio.")
            return latent_frames

        def latent_frames_to_pixel_frames(self, latent_frames: int) -> int:
            """Map harness latent-frame counts back into Wan pixel-frame counts."""

            return 1 + (latent_frames - 1) * self.temporal_downsample_factor

        def resolved_rollout_stride_frames(
            self,
            context_frames: int,
            stride_frames: int | None = None,
        ) -> int:
            """Delegate stride resolution to the shared world-model helper."""

            return WorldModel.resolved_rollout_stride_frames(self, context_frames, stride_frames)

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
            """Return one fixed three-frame chunk while recording context and action windows."""

            del infer_steps, generator, guidance_scale
            captured_context_lengths.append(int(latents.shape[2]))
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

        def decode_target_latents(
            self,
            context_latents: torch.Tensor,
            target_latents: torch.Tensor,
            *,
            context_pixel_frames: int,
            target_pixel_frames: int | None = None,
        ) -> torch.Tensor:
            """Expand each predicted latent into four pixel frames for rollout assertions."""

            del context_latents, context_pixel_frames
            expanded = torch.cat(
                [
                    target_latents[:, :, index : index + 1]
                    .permute(0, 2, 1, 3, 4)
                    .repeat(1, self.temporal_downsample_factor, 1, 1, 1)
                    for index in range(target_latents.shape[2])
                ],
                dim=1,
            )
            return expanded[:, :target_pixel_frames]

    seed_frames = torch.zeros(1, 1, 3, 2, 2)
    actions = torch.arange(64, dtype=torch.float32).view(1, 16, 4)

    rollout = WorldModel.rollout(DummyWorldModel(), seed_frames, steps=16, actions=actions)

    assert rollout.shape == (1, 17, 3, 2, 2)
    assert captured_context_lengths == [1, 1]
    assert len(captured_action_windows) == 2
    assert torch.equal(captured_action_windows[0], actions[:, :12])
    assert torch.equal(captured_action_windows[1][:, :4], actions[:, 12:16])
    assert torch.count_nonzero(captured_action_windows[1][:, 4:]) == 0
