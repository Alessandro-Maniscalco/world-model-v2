"""Tests for reconstruction visualization helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_gif


def test_build_side_by_side_grid_returns_image() -> None:
    """Grid creation should render a non-empty PIL image."""

    images = torch.rand(3, 3, 16, 16)
    reconstructed = torch.rand(3, 3, 16, 16)
    grid = build_side_by_side_grid(images, reconstructed, max_frames=3)
    assert grid.size[0] > 0
    assert grid.size[1] > 0


def test_write_side_by_side_gif_exports_all_frames(tmp_path: Path) -> None:
    """GIF export should return the number of rendered frames."""

    images = torch.rand(4, 3, 16, 16)
    reconstructed = torch.rand(4, 3, 16, 16)
    output_path = tmp_path / "preview.gif"
    frame_count = write_side_by_side_gif(images, reconstructed, output_path, duration_ms=30)
    assert output_path.exists()
    assert frame_count == 4
