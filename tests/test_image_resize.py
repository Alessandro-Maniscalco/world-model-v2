"""Tests for shared image-resize helpers used by dataset preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from world_model_v2.image_resize import (
    DEFAULT_RESIZE_FILTER,
    normalize_resize_filter,
    rgb_array_to_tensor,
    supported_resize_filters,
)


def _checkerboard_rgb(height: int, width: int) -> np.ndarray:
    """Create one high-frequency RGB checkerboard for resize-filter comparisons."""

    rows, cols = np.indices((height, width))
    board = ((rows + cols) % 2).astype(np.uint8) * 255
    rgb = np.stack(
        [
            board,
            np.roll(board, shift=1, axis=1),
            np.roll(board, shift=1, axis=0),
        ],
        axis=-1,
    )
    return rgb


def test_supported_resize_filters_expose_expected_default() -> None:
    """The resize helper should default to bicubic and expose the supported names."""

    assert DEFAULT_RESIZE_FILTER == "bicubic"
    assert supported_resize_filters() == ("bilinear", "bicubic", "lanczos")
    assert normalize_resize_filter(None) == "bicubic"
    assert normalize_resize_filter("  LANCZOS ") == "lanczos"


def test_normalize_resize_filter_rejects_unknown_values() -> None:
    """The resize helper should fail fast on unsupported filter names."""

    with pytest.raises(ValueError, match="Unsupported resize_filter"):
        normalize_resize_filter("nearest")


def test_rgb_array_to_tensor_preserves_pixels_without_resizing() -> None:
    """The helper should preserve exact pixel values when the size is unchanged."""

    rgb = _checkerboard_rgb(6, 8)
    tensor = rgb_array_to_tensor(rgb, height=6, width=8)

    assert tensor.shape == (3, 6, 8)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert torch.equal(tensor, torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)


def test_rgb_array_to_tensor_produces_distinct_filtered_outputs() -> None:
    """Different resize filters should produce measurably different resized tensors."""

    rgb = _checkerboard_rgb(9, 13)
    bilinear = rgb_array_to_tensor(rgb, height=24, width=32, resize_filter="bilinear")
    bicubic = rgb_array_to_tensor(rgb, height=24, width=32, resize_filter="bicubic")
    lanczos = rgb_array_to_tensor(rgb, height=24, width=32, resize_filter="lanczos")

    assert bilinear.shape == (3, 24, 32)
    assert bicubic.shape == bilinear.shape
    assert lanczos.shape == bilinear.shape
    assert not torch.allclose(bilinear, bicubic)
    assert not torch.allclose(bicubic, lanczos)
