"""Tests for the Wan2.1-style temporal VAE helpers and frame mappings."""

from __future__ import annotations

import torch

from world_model_v2.model import LatentNormalizationStats, WorldModel
from world_model_v2.wan_vae import WanVAEConfig


def test_wan_vae_config_uses_temporal_four_x_mapping() -> None:
    """The default Wan config should expose the expected 4x temporal helper formulas."""

    cfg = WanVAEConfig()

    assert cfg.temporal_downsample_factor() == 4
    assert cfg.pixel_frames_to_latent_frames(1) == 1
    assert cfg.pixel_frames_to_latent_frames(5) == 2
    assert cfg.pixel_frames_to_latent_frames(13) == 4
    assert cfg.latent_frames_to_pixel_frames(1) == 1
    assert cfg.latent_frames_to_pixel_frames(2) == 5
    assert cfg.latent_frames_to_pixel_frames(4) == 13


def test_wan_video_tokenizer_preserves_expected_temporal_shapes() -> None:
    """Encoding and decoding should follow the Wan 1/5/13 to 1/2/4 frame mapping."""

    model = WorldModel(resolution=32)

    for pixel_frames, latent_frames in ((1, 1), (5, 2), (13, 4)):
        images = torch.rand(1, pixel_frames, 3, 32, 32)
        latents = model.encode_frame_sequence(images, deterministic=True)
        reconstructed = model.decode_frame_sequence(latents)

        assert latents.shape == (1, 32, latent_frames, 8, 8)
        assert reconstructed.shape == (1, pixel_frames, 3, 32, 32)


def test_wan_image_wrappers_still_support_single_frames() -> None:
    """The image-facing encode/decode wrappers should still work for one RGB frame batch."""

    model = WorldModel(resolution=32)
    images = torch.rand(2, 3, 32, 32)

    latents = model.encode(images, deterministic=True)
    reconstructed = model.decode(latents)

    assert latents.shape == (2, 32, 8, 8)
    assert reconstructed.shape == (2, 3, 32, 32)


def test_latent_normalization_stats_round_trip_for_image_and_video_latents() -> None:
    """Configured latent normalization stats should round-trip both image and video tensors."""

    stats = LatentNormalizationStats(
        img_mean=tuple(float(index) for index in range(32)),
        img_std=tuple(float(index + 1) for index in range(32)),
        video_mean=tuple(float(index + 2) for index in range(32)),
        video_std=tuple(float(index + 3) for index in range(32)),
    )
    model = WorldModel(resolution=32, latent_normalization_stats=stats)
    image_latents = torch.randn(2, 32, 8, 8)
    video_latents = torch.randn(2, 32, 4, 8, 8)

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
