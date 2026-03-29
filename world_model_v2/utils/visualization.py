"""Visualization helpers for reconstruction grids and GIF exports."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


def tensor_to_uint8_images(images: torch.Tensor) -> np.ndarray:
    """Convert BCHW float images in [0, 1] to BHWC uint8 arrays."""

    array = images.detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8)
    return array.permute(0, 2, 3, 1).numpy()


def annotate_frame(frame: np.ndarray, label: str) -> Image.Image:
    """Draw a small top-left label onto a frame."""

    image = Image.fromarray(frame)
    image = ImageOps.expand(image, border=4, fill="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, min(240, image.width - 6), 32), fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255))
    return image


def build_side_by_side_grid(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_frames: int = 12,
) -> Image.Image:
    """Create a two-column contact sheet of original and reconstructed frames."""

    original_np = tensor_to_uint8_images(original[:max_frames])
    reconstructed_np = tensor_to_uint8_images(reconstructed[:max_frames])
    frames: list[Image.Image] = []
    for frame_idx, (orig_frame, recon_frame) in enumerate(zip(original_np, reconstructed_np, strict=False)):
        frames.append(annotate_frame(orig_frame, f"gt {frame_idx}"))
        frames.append(annotate_frame(recon_frame, f"recon {frame_idx}"))
    width, height = frames[0].size
    canvas = Image.new("RGB", (width * 2, height * len(original_np)), "black")
    for row, frame in enumerate(frames):
        x = 0 if row % 2 == 0 else width
        y = (row // 2) * height
        canvas.paste(frame, (x, y))
    return canvas


def write_side_by_side_gif(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_path: str | Path,
    duration_ms: int = 120,
) -> int:
    """Write a GIF with original and reconstructed frames next to each other."""

    original_np = tensor_to_uint8_images(original)
    reconstructed_np = tensor_to_uint8_images(reconstructed)
    rendered_frames: list[np.ndarray] = []
    for frame_idx, (orig_frame, recon_frame) in enumerate(zip(original_np, reconstructed_np, strict=False)):
        orig_img = annotate_frame(orig_frame, f"gt {frame_idx}")
        recon_img = annotate_frame(recon_frame, f"recon {frame_idx}")
        stacked = Image.new("RGB", (orig_img.width + recon_img.width, orig_img.height), "black")
        stacked.paste(orig_img, (0, 0))
        stacked.paste(recon_img, (orig_img.width, 0))
        rendered_frames.append(np.asarray(stacked))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, rendered_frames, duration=max(duration_ms / 1000.0, 0.01))
    return len(rendered_frames)
