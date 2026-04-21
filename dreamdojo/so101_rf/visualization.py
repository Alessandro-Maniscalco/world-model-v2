"""Visualization helpers for reconstruction grids and MP4 preview exports."""

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


def annotate_frame(
    frame: np.ndarray,
    label: str,
    *,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Draw a small top-left label onto a frame."""

    image = Image.fromarray(frame)
    image = ImageOps.expand(image, border=4, fill=border_color)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), label, fill=(255, 0, 0))
    return image


def _render_side_by_side_frames(
    original: np.ndarray,
    reconstructed: np.ndarray,
    context_frames: int,
) -> list[Image.Image]:
    """Render annotated source/reconstruction frame pairs for preview exports."""

    frames: list[Image.Image] = []
    for frame_idx, (orig_frame, recon_frame) in enumerate(zip(original, reconstructed, strict=False)):
        border_color = (255, 0, 0) if frame_idx < context_frames else (255, 255, 255)
        orig_img = annotate_frame(orig_frame, f"gt {frame_idx}", border_color=border_color)
        recon_img = annotate_frame(recon_frame, f"recon {frame_idx}", border_color=border_color)
        stacked = Image.new("RGB", (orig_img.width + recon_img.width, orig_img.height), "black")
        stacked.paste(orig_img, (0, 0))
        stacked.paste(recon_img, (orig_img.width, 0))
        frames.append(stacked)
    return frames


def build_side_by_side_grid(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_frames: int = 12,
    context_frames: int = 0,
) -> Image.Image:
    """Create a two-column contact sheet of original and reconstructed frames."""

    rendered_rows = _render_side_by_side_frames(
        tensor_to_uint8_images(original[:max_frames]),
        tensor_to_uint8_images(reconstructed[:max_frames]),
        context_frames,
    )
    width, height = rendered_rows[0].size
    canvas = Image.new("RGB", (width, height * len(rendered_rows)), "black")
    for row, frame in enumerate(rendered_rows):
        canvas.paste(frame, (0, row * height))
    return canvas


def frames_per_second_from_duration(duration_ms: int) -> float:
    """Convert a frame duration in milliseconds into frames per second."""

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive.")
    return 1000.0 / float(duration_ms)


def _to_uint8_rgb_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    """Convert one frame into a uint8 RGB array suitable for video encoding."""

    array = np.asarray(frame) if isinstance(frame, Image.Image) else frame
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Video frames must have shape HxWx3.")
    if array.dtype == np.uint8:
        return array
    return np.clip(array, 0, 255).astype(np.uint8)


def _pad_frame_to_even_size(frame: np.ndarray) -> np.ndarray:
    """Pad one frame to even width/height for H.264-compatible MP4 export."""

    pad_height = frame.shape[0] % 2
    pad_width = frame.shape[1] % 2
    if pad_height == 0 and pad_width == 0:
        return frame
    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def write_mp4_frames(
    frames: list[Image.Image | np.ndarray],
    output_path: str | Path,
    *,
    fps: float,
) -> int:
    """Write RGB preview frames to an MP4 file and return the frame count."""

    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    if not frames:
        raise ValueError("Cannot write an MP4 without any frames.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output,
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv444p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        ffmpeg_params=["-crf", "12", "-preset", "medium", "-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(_pad_frame_to_even_size(_to_uint8_rgb_array(frame)))
    return len(frames)


def write_side_by_side_mp4(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_path: str | Path,
    duration_ms: int = 120,
    context_frames: int = 0,
) -> int:
    """Write an MP4 with original and reconstructed frames next to each other."""

    rendered_frames = _render_side_by_side_frames(
        tensor_to_uint8_images(original),
        tensor_to_uint8_images(reconstructed),
        context_frames,
    )
    return write_mp4_frames(
        rendered_frames,
        output_path,
        fps=frames_per_second_from_duration(duration_ms),
    )
