"""Tests for reconstruction visualization helpers."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


def read_mp4_frames(output_path: Path) -> tuple[list[np.ndarray], dict[str, object]]:
    """Load decoded MP4 frames and metadata from one preview file."""

    reader = imageio.get_reader(output_path, format="FFMPEG")
    try:
        metadata = dict(reader.get_meta_data())
        frames = [frame for frame in reader]
    finally:
        reader.close()
    return frames, metadata


def test_build_side_by_side_grid_returns_image() -> None:
    """Grid creation should render a non-empty PIL image."""

    images = torch.rand(3, 3, 16, 16)
    reconstructed = torch.rand(3, 3, 16, 16)
    grid = build_side_by_side_grid(images, reconstructed, max_frames=3)
    assert grid.size[0] > 0
    assert grid.size[1] > 0


def test_write_side_by_side_mp4_exports_all_frames(tmp_path: Path) -> None:
    """MP4 export should return the number of rendered frames."""

    images = torch.rand(4, 3, 16, 16)
    reconstructed = torch.rand(4, 3, 16, 16)
    output_path = tmp_path / "preview.mp4"
    frame_count = write_side_by_side_mp4(images, reconstructed, output_path, duration_ms=30)
    decoded_frames, _ = read_mp4_frames(output_path)
    assert output_path.exists()
    assert frame_count == 4
    assert len(decoded_frames) == 4


def test_write_side_by_side_mp4_uses_requested_frame_rate_and_even_dimensions(tmp_path: Path) -> None:
    """MP4 export should preserve frame rate and pad odd frame sizes for the codec."""

    images = torch.rand(3, 3, 15, 17)
    reconstructed = torch.rand(3, 3, 15, 17)
    output_path = tmp_path / "preview.mp4"
    write_side_by_side_mp4(images, reconstructed, output_path, duration_ms=45)
    decoded_frames, metadata = read_mp4_frames(output_path)
    assert metadata["fps"] == pytest.approx(1000.0 / 45.0, abs=0.1)
    assert decoded_frames[0].shape[0] % 2 == 0
    assert decoded_frames[0].shape[1] % 2 == 0


def test_write_side_by_side_mp4_keeps_static_reconstruction_region_stable(tmp_path: Path) -> None:
    """A static reconstruction region should not visibly flicker across MP4 frames."""

    frame_count = 4
    height = 64
    width = 64
    images = torch.zeros(frame_count, 3, height, width, dtype=torch.float32)
    reconstructed = torch.zeros(frame_count, 3, height, width, dtype=torch.float32)

    for frame_index in range(frame_count):
        images[frame_index, 0].fill_(0.10 * frame_index)
        images[frame_index, 1].fill_(0.85 - 0.10 * frame_index)
        images[frame_index, 2].fill_(0.20 + 0.05 * frame_index)
        reconstructed[frame_index, 0, 16:48, 16:48] = 0.60
        reconstructed[frame_index, 1, 16:48, 16:48] = 0.55
        reconstructed[frame_index, 2, 16:48, 16:48] = 0.48
        reconstructed[frame_index, :, 24:40, 24:40] = torch.tensor([0.72, 0.62, 0.52]).view(3, 1, 1)

    output_path = tmp_path / "preview.mp4"
    write_side_by_side_mp4(images, reconstructed, output_path, duration_ms=40)

    decoded_frames, _ = read_mp4_frames(output_path)
    stacked = np.stack(decoded_frames, axis=0)
    crop = stacked[:, 40:68, 92:120]
    diffs = np.abs(crop[1:].astype(np.int16) - crop[:-1].astype(np.int16))
    assert float(diffs.mean()) < 1.0
