"""Tests for checkpoint loading and episode reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

from world_model_v2.infer.reconstruct_episode import reconstruct_episode


def test_reconstruct_episode_writes_outputs(
    fake_dataset_root: Path,
    saved_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Reconstruction should save a grid, a GIF, and matching frame-count stats."""

    output_dir = tmp_path / "recon"
    result = reconstruct_episode(
        checkpoint_path=saved_checkpoint,
        data_root=fake_dataset_root,
        task="single_grasp",
        split="val",
        camera="camera_1_color",
        episode=0,
        resolution=32,
        num_steps=2,
        start_mode="noisy-input",
        max_frames=4,
        duration_ms=40,
        output_dir=output_dir,
        device="cpu",
    )
    stats = json.loads((output_dir / "episode_0_stats.json").read_text())
    assert Path(result["grid_path"]).exists()
    assert Path(result["gif_path"]).exists()
    assert stats["input_frame_count"] == 6
    assert stats["decoded_frame_count"] == 6
    assert stats["exported_gif_frame_count"] == 6
