"""Tests for the DreamDojo-compatible Wan 2.2 tokenizer helpers and frame mappings."""

from __future__ import annotations

import torch

from world_model_v2.model import LatentNormalizationStats, WorldModel
from world_model_v2.wan_vae import (
    DEFAULT_WAN_DEC_DIM,
    DEFAULT_WAN_DIM,
    DEFAULT_WAN_NUM_RES_BLOCKS,
    DEFAULT_WAN_Z_DIM,
    Upsample,
    WanPosteriorEncoder,
    WanVAEConfig,
    WanVideoDecoder,
    patchify,
    unpatchify,
)

TEST_WAN_CONFIG = WanVAEConfig(dim=16, dec_dim=16, z_dim=DEFAULT_WAN_Z_DIM, num_res_blocks=1)


def _build_small_world_model(
    *,
    resolution: int = 32,
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


def test_wan_vae_config_uses_dreamdojo_wan_2pt2_defaults() -> None:
    """The default Wan config should expose the expected DreamDojo Wan 2.2 defaults."""

    cfg = WanVAEConfig()

    assert cfg.dim == DEFAULT_WAN_DIM
    assert cfg.dec_dim == DEFAULT_WAN_DEC_DIM
    assert cfg.z_dim == DEFAULT_WAN_Z_DIM
    assert cfg.num_res_blocks == DEFAULT_WAN_NUM_RES_BLOCKS
    assert cfg.spatial_downsample_factor() == 16
    assert cfg.temporal_downsample_factor() == 4
    assert cfg.pixel_frames_to_latent_frames(1) == 1
    assert cfg.pixel_frames_to_latent_frames(5) == 2
    assert cfg.pixel_frames_to_latent_frames(13) == 4
    assert cfg.latent_frames_to_pixel_frames(1) == 1
    assert cfg.latent_frames_to_pixel_frames(2) == 5
    assert cfg.latent_frames_to_pixel_frames(4) == 13


def test_patchify_and_unpatchify_round_trip_for_images_and_videos() -> None:
    """Patchify and unpatchify should invert each other for image and video tensors."""

    images = torch.randn(2, 3, 8, 10)
    videos = torch.randn(2, 3, 5, 8, 10)

    assert torch.equal(unpatchify(patchify(images, 2), 2), images)
    assert torch.equal(unpatchify(patchify(videos, 2), 2), videos)


def test_wan_video_tokenizer_preserves_expected_temporal_shapes() -> None:
    """Encoding and decoding should follow the Wan 1/5/13 to 1/2/4 frame mapping."""

    model = _build_small_world_model()
    latent_height = model.image_height // model.spatial_downsample_factor
    latent_width = model.image_width // model.spatial_downsample_factor

    for pixel_frames, latent_frames in ((1, 1), (5, 2), (13, 4)):
        images = torch.rand(1, pixel_frames, 3, model.image_height, model.image_width)
        latents = model.encode_frame_sequence(images, deterministic=True)
        reconstructed = model.decode_frame_sequence(latents)

        assert latents.shape == (1, model.latent_channels, latent_frames, latent_height, latent_width)
        assert reconstructed.shape == (1, pixel_frames, 3, model.image_height, model.image_width)


def test_chunked_encoder_matches_explicit_chunk_replay_on_supported_lengths() -> None:
    """Chunked Wan 2.2 encoding should match an explicit DreamDojo chunk replay."""

    encoder = WanPosteriorEncoder(TEST_WAN_CONFIG)

    for frames in (1, 5, 13):
        video = torch.randn(1, 3, frames, 32, 32)
        chunked_mu, chunked_log_var = encoder(video)
        patchified = patchify(video, TEST_WAN_CONFIG.patch_size)
        encoder.clear_cache()
        iter_count = 1 + (patchified.shape[2] - 1) // TEST_WAN_CONFIG.temporal_window
        manual_encoded = encoder.backbone(
            patchified[:, :, :1],
            feat_cache=encoder._enc_feat_map,
            feat_idx=[0],
        )
        for index in range(1, iter_count):
            start = 1 + TEST_WAN_CONFIG.temporal_window * (index - 1)
            stop = 1 + TEST_WAN_CONFIG.temporal_window * index
            manual_encoded = torch.cat(
                [
                    manual_encoded,
                    encoder.backbone(
                        patchified[:, :, start:stop],
                        feat_cache=encoder._enc_feat_map,
                        feat_idx=[0],
                    ),
                ],
                dim=2,
            )
        if (patchified.shape[2] - 1) % TEST_WAN_CONFIG.temporal_window:
            start = 1 + TEST_WAN_CONFIG.temporal_window * (iter_count - 1)
            manual_encoded = torch.cat(
                [
                    manual_encoded,
                    encoder.backbone(
                        patchified[:, :, start:],
                        feat_cache=encoder._enc_feat_map,
                        feat_idx=[0],
                    ),
                ],
                dim=2,
            )
        full_mu, full_log_var = encoder.moments_conv(manual_encoded).chunk(2, dim=1)
        encoder.clear_cache()

        assert torch.allclose(chunked_mu, full_mu, atol=1e-5, rtol=1e-5)
        assert torch.allclose(chunked_log_var, full_log_var, atol=1e-5, rtol=1e-5)


def test_chunked_decoder_matches_full_pass_across_chunk_boundaries() -> None:
    """Chunked Wan 2.2 decoding should match an explicit DreamDojo chunk replay."""

    decoder = WanVideoDecoder(TEST_WAN_CONFIG)

    for latent_frames in (1, 2, 3):
        latents = torch.randn(1, TEST_WAN_CONFIG.z_dim, latent_frames, 2, 2)
        chunked = decoder(latents)
        projected = decoder.pre_decode_conv(latents)
        decoder.clear_cache()
        manual_decoded = decoder.backbone(
            projected[:, :, :1],
            feat_cache=decoder._feat_map,
            feat_idx=[0],
            first_chunk=True,
        )
        for index in range(1, int(projected.shape[2])):
            manual_decoded = torch.cat(
                [
                    manual_decoded,
                    decoder.backbone(
                        projected[:, :, index : index + 1],
                        feat_cache=decoder._feat_map,
                        feat_idx=[0],
                    ),
                ],
                dim=2,
            )
        full = unpatchify(manual_decoded, TEST_WAN_CONFIG.patch_size)
        decoder.clear_cache()

        assert torch.allclose(chunked, full, atol=1e-5, rtol=1e-5)


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
