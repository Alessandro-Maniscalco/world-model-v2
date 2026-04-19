"""Tests for the Wan2.1-style temporal VAE helpers and frame mappings."""

from __future__ import annotations

import torch

from world_model_v2.model import LatentNormalizationStats, WorldModel
from world_model_v2.wan_vae import (
    DEFAULT_WAN_DIM,
    DEFAULT_WAN_NUM_RES_BLOCKS,
    DEFAULT_WAN_Z_DIM,
    Upsample,
    WanVAEConfig,
)

TEST_WAN_CONFIG = WanVAEConfig(dim=16, z_dim=8, num_res_blocks=1)


def _build_small_world_model(
    *,
    resolution: int = 16,
    latent_normalization_stats: LatentNormalizationStats | None = None,
) -> WorldModel:
    """Create a smaller Wan world model for semantic encode/decode tests."""

    return WorldModel(
        resolution=resolution,
        latent_channels=TEST_WAN_CONFIG.z_dim,
        wan_config=TEST_WAN_CONFIG,
        latent_normalization_stats=latent_normalization_stats,
        dynamics_model_channels=32,
        dynamics_num_blocks=1,
        dynamics_num_heads=1,
        dynamics_infer_steps=1,
    )


def test_wan_vae_config_uses_temporal_two_x_mapping() -> None:
    """The default Wan config should expose the expected 2x temporal helper formulas."""

    cfg = WanVAEConfig()

    assert cfg.dim == DEFAULT_WAN_DIM
    assert cfg.z_dim == DEFAULT_WAN_Z_DIM
    assert cfg.num_res_blocks == DEFAULT_WAN_NUM_RES_BLOCKS
    assert cfg.temporal_downsample_factor() == 2
    assert cfg.pixel_frames_to_latent_frames(1) == 1
    assert cfg.pixel_frames_to_latent_frames(3) == 2
    assert cfg.pixel_frames_to_latent_frames(7) == 4
    assert cfg.latent_frames_to_pixel_frames(1) == 1
    assert cfg.latent_frames_to_pixel_frames(2) == 3
    assert cfg.latent_frames_to_pixel_frames(4) == 7


def test_wan_video_tokenizer_preserves_expected_temporal_shapes() -> None:
    """Encoding and decoding should follow the Wan 1/3/7 to 1/2/4 frame mapping."""

    model = _build_small_world_model()
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor

    for pixel_frames, latent_frames in ((1, 1), (3, 2), (7, 4)):
        images = torch.rand(1, pixel_frames, 3, model.image_height, model.image_width)
        latents = model.encode_frame_sequence(images, deterministic=True)
        reconstructed = model.decode_frame_sequence(latents)

        assert latents.shape == (1, model.latent_channels, latent_frames, latent_height, latent_width)
        assert reconstructed.shape == (1, pixel_frames, 3, model.image_height, model.image_width)


def test_wan_image_wrappers_still_support_single_frames() -> None:
    """The image-facing encode/decode wrappers should still work for one RGB frame batch."""

    model = _build_small_world_model()
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor
    images = torch.rand(2, 3, model.image_height, model.image_width)

    latents = model.encode(images, deterministic=True)
    reconstructed = model.decode(latents)

    assert latents.shape == (2, model.latent_channels, latent_height, latent_width)
    assert reconstructed.shape == (2, 3, model.image_height, model.image_width)


def test_wan_upsample_preserves_input_dtype() -> None:
    """The custom upsample wrapper should not silently widen activations."""

    layer = Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact")

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        images = torch.randn(2, 4, 8, 8, dtype=dtype)

        upsampled = layer(images)

        assert upsampled.dtype == dtype


def test_latent_normalization_stats_round_trip_for_image_and_video_latents() -> None:
    """Configured latent normalization stats should round-trip both image and video tensors."""

    channels = TEST_WAN_CONFIG.z_dim
    stats = LatentNormalizationStats(
        img_mean=tuple(float(index) for index in range(channels)),
        img_std=tuple(float(index + 1) for index in range(channels)),
        video_mean=tuple(float(index + 2) for index in range(channels)),
        video_std=tuple(float(index + 3) for index in range(channels)),
    )
    model = _build_small_world_model(latent_normalization_stats=stats)
    image_latents = torch.randn(2, channels, 4, 4)
    video_latents = torch.randn(2, channels, 4, 4, 4)

    normalized_images = model._normalize_image_latents(image_latents)
    normalized_videos = model._normalize_video_latents(video_latents)

    assert torch.allclose(
        model._unnormalize_image_latents(normalized_images),
        image_latents,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        model._unnormalize_video_latents(normalized_videos),
        video_latents,
        atol=1e-5,
        rtol=1e-5,
    )
    assert model.autoencoder_config()["normalization_stats"] == stats.to_dict()
