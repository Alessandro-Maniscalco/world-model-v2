"""Runtime-heavy Wan model tests kept out of the fast root suite."""

from __future__ import annotations

import pytest
import torch

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from world_model_v2.model import WorldModel
from world_model_v2.wan_vae import WanVAEConfig


pytestmark = pytest.mark.slow

TEST_RUNTIME_WAN_CONFIG = WanVAEConfig(dim=16, dec_dim=16, z_dim=48, num_res_blocks=1)


def _build_runtime_world_model(
    *,
    resolution: int = 32,
    height: int | None = None,
    width: int | None = None,
    **overrides: object,
) -> WorldModel:
    """Create a much smaller runtime test model while preserving Wan semantics."""

    return WorldModel(
        resolution=resolution,
        height=height,
        width=width,
        latent_channels=TEST_RUNTIME_WAN_CONFIG.z_dim,
        wan_config=TEST_RUNTIME_WAN_CONFIG,
        dynamics_model_channels=32,
        dynamics_num_blocks=1,
        dynamics_num_heads=1,
        dynamics_infer_steps=1,
        **overrides,
    )


def test_wan_world_model_preserves_expected_shapes() -> None:
    """The Wan model should keep the expected image and latent tensor shapes."""

    model = _build_runtime_world_model()
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor
    images = torch.rand(2, 3, model.image_height, model.image_width)
    context_images = torch.rand(
        2,
        DYNAMICS_FRAME_LAYOUT.context_pixel_frames,
        3,
        model.image_height,
        model.image_width,
    )
    latents = model.encode(images, deterministic=True)
    context_latents = model.encode_context_frames(context_images, deterministic=True)
    reconstructed = model.decode(latents)
    next_latents = model.predict_next_latent(context_latents)
    assert latents.shape == (2, model.latent_channels, latent_height, latent_width)
    assert context_latents.shape == (
        2,
        model.latent_channels,
        DYNAMICS_FRAME_LAYOUT.context_frames,
        latent_height,
        latent_width,
    )
    assert reconstructed.shape == (2, 3, model.image_height, model.image_width)
    assert next_latents.shape == (
        2,
        model.latent_channels,
        DYNAMICS_FRAME_LAYOUT.target_frames,
        latent_height,
        latent_width,
    )


def test_wan_world_model_accepts_explicit_action_chunks() -> None:
    """The Wan model should accept DreamDojo-style twelve-step action chunks."""

    model = _build_runtime_world_model()
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor
    context_images = torch.rand(
        1,
        DYNAMICS_FRAME_LAYOUT.context_pixel_frames,
        3,
        model.image_height,
        model.image_width,
    )
    context_latents = model.encode_context_frames(context_images, deterministic=True)
    actions = torch.zeros(1, DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    next_latents = model.predict_next_latent(context_latents, actions=actions)
    assert next_latents.shape == (
        1,
        model.latent_channels,
        DYNAMICS_FRAME_LAYOUT.target_frames,
        latent_height,
        latent_width,
    )


def test_wan_world_model_supports_five_frame_two_latent_context_prediction() -> None:
    """The Wan model should support a five-frame context that maps to two latent frames."""

    model = _build_runtime_world_model(
        dynamics_conditioning_frame_choices=(1, 2),
        dynamics_conditioning_frame_probabilities=(0.5, 0.5),
        dynamics_validation_conditioning_frame_choices=(1, 2),
        dynamics_open_rollout_context_frames=1,
    )
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor
    context_images = torch.rand(
        1,
        model.latent_frames_to_pixel_frames(2),
        3,
        model.image_height,
        model.image_width,
    )
    context_latents = model.encode_context_frames(context_images, deterministic=True)
    actions = torch.zeros(1, model.dynamics.cfg.num_action_per_chunk, 4)
    next_latents = model.predict_next_latent(context_latents, actions=actions)
    assert next_latents.shape == (
        1,
        model.latent_channels,
        model.dynamics.cfg.max_frames - 2,
        latent_height,
        latent_width,
    )


def test_wan_world_model_derives_latent_size_from_resolution() -> None:
    """The Wan model should derive latent height and width from the input resolution."""

    model = _build_runtime_world_model(resolution=16)
    images = torch.rand(2, 3, model.image_height, model.image_width)
    latents = model.encode(images, deterministic=True)
    assert latents.shape == (
        2,
        model.latent_channels,
        model.image_height // model.spatial_downsample_factor,
        model.image_width // model.spatial_downsample_factor,
    )


def test_wan_world_model_supports_rectangular_inputs() -> None:
    """The Wan model should derive latent height and width for rectangular inputs."""

    model = _build_runtime_world_model(resolution=32, height=32, width=48)
    images = torch.rand(2, 3, model.image_height, model.image_width)
    latents = model.encode(images, deterministic=True)
    reconstructed = model.decode(latents)
    assert latents.shape == (
        2,
        model.latent_channels,
        model.image_height // model.spatial_downsample_factor,
        model.image_width // model.spatial_downsample_factor,
    )
    assert reconstructed.shape == (2, 3, model.image_height, model.image_width)


def test_wan_world_model_autoencode_reports_kl_statistics() -> None:
    """The Wan backend should expose posterior statistics for KL-regularized training."""

    model = _build_runtime_world_model()
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor
    images = torch.rand(2, 3, model.image_height, model.image_width)
    output = model.autoencode(images, sample_posterior=True)
    assert output.reconstructed.shape == (2, 3, model.image_height, model.image_width)
    assert output.mu.shape == (2, model.latent_channels, latent_height, latent_width)
    assert output.log_var.shape == (2, model.latent_channels, latent_height, latent_width)
    assert output.latent.shape == output.mu.shape
    assert output.kl_loss.ndim == 0
    assert float(output.kl_loss) >= 0.0


def test_wan_world_model_rollout_includes_seed_frame() -> None:
    """Autoregressive rollout should return the seed context plus the requested steps."""

    model = _build_runtime_world_model()
    seed = torch.rand(
        1,
        DYNAMICS_FRAME_LAYOUT.context_pixel_frames,
        3,
        model.image_height,
        model.image_width,
    )
    rollout = model.rollout(seed, steps=2)
    assert rollout.shape == (
        1,
        DYNAMICS_FRAME_LAYOUT.context_pixel_frames + 2,
        3,
        model.image_height,
        model.image_width,
    )


def test_wan_world_model_rollout_accepts_overlap_stride() -> None:
    """Runtime rollout should allow appending one frame at a time from a wider chunk head."""

    model = _build_runtime_world_model(
        dynamics_context_frames=1,
        dynamics_target_frames=2,
        dynamics_conditioning_frame_choices=(1,),
        dynamics_conditioning_frame_probabilities=(1.0,),
        dynamics_validation_conditioning_frame_choices=(1,),
        dynamics_open_rollout_context_frames=1,
        dynamics_open_rollout_stride_frames=1,
    )
    seed = torch.rand(1, 1, 3, model.image_height, model.image_width)
    rollout = model.rollout(seed, steps=2, stride_frames=1)
    assert rollout.shape == (1, 3, 3, model.image_height, model.image_width)
