"""Utilities for diagnosing VAE reconstructions under spatial perturbations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialTransformSpec:
    """Describe one deterministic spatial transform for a frame sequence."""

    name: str
    horizontal_flip: bool = False
    shift_x: int = 0
    shift_y: int = 0


def _format_shift_suffix(prefix: str, value: int) -> str:
    """Return one filesystem-friendly signed integer suffix."""

    if value == 0:
        return ""
    sign = "pos" if value > 0 else "neg"
    return f"_{prefix}_{sign}{abs(int(value))}"


def parse_transform_spec(spec: str) -> SpatialTransformSpec:
    """Parse one CLI transform spec into a structured transform."""

    normalized = spec.strip().lower()
    if not normalized:
        raise ValueError("Transform spec must not be empty.")
    if normalized == "identity":
        return SpatialTransformSpec(name="identity")
    if normalized == "hflip":
        return SpatialTransformSpec(name="hflip", horizontal_flip=True)
    if normalized.startswith("hshift:"):
        shift_x = int(normalized.split(":", maxsplit=1)[1])
        return SpatialTransformSpec(
            name=f"hshift{_format_shift_suffix('x', shift_x)}",
            shift_x=shift_x,
        )
    if normalized.startswith("vshift:"):
        shift_y = int(normalized.split(":", maxsplit=1)[1])
        return SpatialTransformSpec(
            name=f"vshift{_format_shift_suffix('y', shift_y)}",
            shift_y=shift_y,
        )
    if normalized.startswith("shift:"):
        parts = normalized.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Expected shift specs in the form 'shift:<x>:<y>', "
                f"received {spec!r}."
            )
        shift_x = int(parts[1])
        shift_y = int(parts[2])
        return SpatialTransformSpec(
            name=f"shift{_format_shift_suffix('x', shift_x)}{_format_shift_suffix('y', shift_y)}",
            shift_x=shift_x,
            shift_y=shift_y,
        )
    raise ValueError(
        "Unsupported transform spec. Use one of: "
        "'identity', 'hflip', 'hshift:<pixels>', 'vshift:<pixels>', or 'shift:<x>:<y>'."
    )


def translate_frames(
    frames: torch.Tensor,
    *,
    shift_x: int = 0,
    shift_y: int = 0,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Translate a `TCHW` frame tensor using padded edges instead of wraparound."""

    if frames.ndim != 4:
        raise ValueError(f"Expected `frames` with shape [T, C, H, W], received {tuple(frames.shape)}.")
    if pad_mode not in {"replicate", "reflect", "constant"}:
        raise ValueError(f"Unsupported pad_mode {pad_mode!r}.")
    if shift_x == 0 and shift_y == 0:
        return frames.clone()

    pad_left = max(int(shift_x), 0)
    pad_right = max(-int(shift_x), 0)
    pad_top = max(int(shift_y), 0)
    pad_bottom = max(-int(shift_y), 0)
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    if pad_mode == "constant":
        padded = F.pad(frames, pad, mode=pad_mode, value=0.0)
    else:
        padded = F.pad(frames, pad, mode=pad_mode)
    start_x = max(-int(shift_x), 0)
    start_y = max(-int(shift_y), 0)
    stop_y = start_y + int(frames.shape[-2])
    stop_x = start_x + int(frames.shape[-1])
    return padded[:, :, start_y:stop_y, start_x:stop_x]


def apply_spatial_transform(
    frames: torch.Tensor,
    transform: SpatialTransformSpec,
    *,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Apply one parsed spatial transform to a `TCHW` frame tensor."""

    transformed = frames
    if transform.horizontal_flip:
        transformed = torch.flip(transformed, dims=(-1,))
    if transform.shift_x != 0 or transform.shift_y != 0:
        transformed = translate_frames(
            transformed,
            shift_x=transform.shift_x,
            shift_y=transform.shift_y,
            pad_mode=pad_mode,
        )
    else:
        transformed = transformed.clone()
    return transformed


def compute_motion_mask(
    frames: torch.Tensor,
    *,
    threshold: float = 0.03,
    dilation_radius: int = 4,
) -> torch.Tensor:
    """Return a dilated `HxW` boolean mask covering the clip's moving regions."""

    if frames.ndim != 4:
        raise ValueError(f"Expected `frames` with shape [T, C, H, W], received {tuple(frames.shape)}.")
    if frames.shape[0] < 2:
        return torch.zeros(frames.shape[-2:], dtype=torch.bool, device=frames.device)
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative.")

    per_frame_motion = torch.abs(frames[1:] - frames[:-1]).mean(dim=1)
    mask = per_frame_motion.amax(dim=0) >= float(threshold)
    if dilation_radius == 0:
        return mask
    pooled = F.max_pool2d(
        mask.to(dtype=frames.dtype).unsqueeze(0).unsqueeze(0),
        kernel_size=2 * dilation_radius + 1,
        stride=1,
        padding=dilation_radius,
    )
    return pooled[0, 0] > 0


def motion_bbox(mask: torch.Tensor) -> dict[str, int] | None:
    """Return the tight bounding box for one `HxW` boolean motion mask."""

    if mask.ndim != 2:
        raise ValueError(f"Expected `mask` with shape [H, W], received {tuple(mask.shape)}.")
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        return None
    y_min = int(nonzero[:, 0].min().item())
    y_max = int(nonzero[:, 0].max().item())
    x_min = int(nonzero[:, 1].min().item())
    x_max = int(nonzero[:, 1].max().item())
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
    }


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    """Return the mean over the masked values or `None` when the mask is empty."""

    if mask.ndim != 2:
        raise ValueError(f"Expected `mask` with shape [H, W], received {tuple(mask.shape)}.")
    expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand_as(values)
    if not bool(expanded_mask.any().item()):
        return None
    return float(values.masked_select(expanded_mask).mean().item())


def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    motion_mask: torch.Tensor | None = None,
) -> dict[str, float | int | dict[str, int] | None]:
    """Compute full-frame and motion-region reconstruction quality metrics."""

    if original.shape != reconstructed.shape:
        raise ValueError(
            "Expected `original` and `reconstructed` to share the same shape, "
            f"received {tuple(original.shape)} and {tuple(reconstructed.shape)}."
        )
    if original.ndim != 4:
        raise ValueError(
            f"Expected tensors with shape [T, C, H, W], received {tuple(original.shape)}."
        )

    squared_error = (reconstructed - original) ** 2
    absolute_error = torch.abs(reconstructed - original)
    global_mse = float(squared_error.mean().item())
    global_l1 = float(absolute_error.mean().item())
    psnr = float("inf") if global_mse <= 0.0 else float(-10.0 * math.log10(global_mse))
    stats: dict[str, float | int | dict[str, int] | None] = {
        "input_frame_count": int(original.shape[0]),
        "global_mse": global_mse,
        "global_l1": global_l1,
        "psnr_db": psnr,
    }
    if motion_mask is None:
        return stats

    if motion_mask.shape != original.shape[-2:]:
        raise ValueError(
            "Expected `motion_mask` to match the frame spatial shape, "
            f"received {tuple(motion_mask.shape)} for frames {tuple(original.shape[-2:])}."
        )
    motion_mask = motion_mask.to(dtype=torch.bool, device=original.device)
    static_mask = ~motion_mask
    stats["motion_pixel_count"] = int(motion_mask.sum().item())
    stats["motion_bbox"] = motion_bbox(motion_mask)
    stats["motion_mse"] = _masked_average(squared_error, motion_mask)
    stats["motion_l1"] = _masked_average(absolute_error, motion_mask)
    stats["static_mse"] = _masked_average(squared_error, static_mask)
    stats["static_l1"] = _masked_average(absolute_error, static_mask)
    return stats
