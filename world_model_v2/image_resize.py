"""Shared image-resize helpers for dataset preprocessing."""

from __future__ import annotations

from typing import Final

import numpy as np
import torch
from PIL import Image


DEFAULT_RESIZE_FILTER: Final[str] = "bicubic"
SUPPORTED_RESIZE_FILTERS: Final[tuple[str, ...]] = ("bilinear", "bicubic", "lanczos")


def supported_resize_filters() -> tuple[str, ...]:
    """Return the supported resize filter names."""

    return SUPPORTED_RESIZE_FILTERS


def normalize_resize_filter(resize_filter: str | None) -> str:
    """Normalize one optional resize filter name and validate it."""

    resolved = DEFAULT_RESIZE_FILTER if resize_filter is None else str(resize_filter).strip().lower()
    if resolved not in SUPPORTED_RESIZE_FILTERS:
        supported = ", ".join(SUPPORTED_RESIZE_FILTERS)
        raise ValueError(f"Unsupported resize_filter={resize_filter!r}. Expected one of: {supported}.")
    return resolved


def pil_resample(resize_filter: str | None) -> Image.Resampling:
    """Return the PIL resampling enum for one validated filter name."""

    resolved = normalize_resize_filter(resize_filter)
    if resolved == "bilinear":
        return Image.Resampling.BILINEAR
    if resolved == "bicubic":
        return Image.Resampling.BICUBIC
    return Image.Resampling.LANCZOS


def rgb_array_to_tensor(
    rgb: np.ndarray,
    height: int,
    width: int,
    *,
    resize_filter: str | None = DEFAULT_RESIZE_FILTER,
) -> torch.Tensor:
    """Resize one RGB array and return a normalized CHW float tensor."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, received shape={rgb.shape!r}.")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers.")
    rgb_uint8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_uint8.shape[0] == height and rgb_uint8.shape[1] == width:
        pixels = rgb_uint8.astype(np.float32) / 255.0
        return torch.from_numpy(np.transpose(np.ascontiguousarray(pixels), (2, 0, 1))).contiguous()
    image = Image.fromarray(rgb_uint8, mode="RGB")
    resized = image.resize((width, height), resample=pil_resample(resize_filter))
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(np.ascontiguousarray(pixels), (2, 0, 1))).contiguous()
