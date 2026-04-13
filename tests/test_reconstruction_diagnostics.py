"""Tests for spatial-transform reconstruction diagnostics helpers."""

from __future__ import annotations

import torch

from world_model_v2.reconstruction_diagnostics import (
    SpatialTransformSpec,
    apply_spatial_transform,
    compute_motion_mask,
    compute_reconstruction_metrics,
    parse_transform_spec,
    translate_frames,
)


def test_parse_transform_spec_supports_horizontal_shift() -> None:
    """Horizontal shift specs should parse into structured transforms."""

    parsed = parse_transform_spec("hshift:-16")
    assert parsed == SpatialTransformSpec(name="hshift_x_neg16", shift_x=-16)


def test_translate_frames_shifts_right_with_replicated_edges() -> None:
    """Positive horizontal shifts should pad the exposed edge instead of wrapping."""

    frames = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    shifted = translate_frames(frames, shift_x=2)
    expected = torch.tensor([[[[1.0, 1.0, 1.0, 2.0]]]])
    assert torch.equal(shifted, expected)


def test_apply_spatial_transform_flips_then_shifts() -> None:
    """Combined transforms should preserve the requested operation order."""

    frames = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    transformed = apply_spatial_transform(
        frames,
        SpatialTransformSpec(name="combo", horizontal_flip=True, shift_x=-1),
    )
    expected = torch.tensor([[[[3.0, 2.0, 1.0, 1.0]]]])
    assert torch.equal(transformed, expected)


def test_compute_motion_mask_marks_and_dilates_active_region() -> None:
    """Motion masks should cover the moving patch and its requested dilation halo."""

    frames = torch.zeros(3, 1, 5, 5)
    frames[1:, :, 2, 2] = 1.0
    mask = compute_motion_mask(frames, threshold=0.1, dilation_radius=1)
    assert bool(mask[2, 2].item())
    assert bool(mask[1, 2].item())
    assert bool(mask[2, 1].item())
    assert not bool(mask[0, 0].item())


def test_compute_reconstruction_metrics_separates_motion_and_static_regions() -> None:
    """Motion-region metrics should isolate errors that only occur inside the mask."""

    original = torch.zeros(2, 1, 2, 2)
    reconstructed = original.clone()
    reconstructed[:, :, 0, 0] = 0.5
    mask = torch.tensor([[True, False], [False, False]])

    stats = compute_reconstruction_metrics(original, reconstructed, motion_mask=mask)

    assert stats["motion_pixel_count"] == 1
    assert stats["motion_mse"] == 0.25
    assert stats["static_mse"] == 0.0
    assert stats["motion_bbox"] == {
        "x_min": 0,
        "x_max": 0,
        "y_min": 0,
        "y_max": 0,
        "width": 1,
        "height": 1,
    }
