"""Tests for Stage-2/Stage-3 rollout inference."""

from __future__ import annotations

import json
from pathlib import Path

from world_model_v2.infer.predict_rollout import predict_rollout


def test_predict_rollout_writes_outputs(
    fake_dataset_root: Path,
    saved_stage2_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Rollout prediction should save a grid, an MP4, and matching frame-count stats."""

    output_dir = tmp_path / "rollout"
    result = predict_rollout(
        checkpoint_path=saved_stage2_checkpoint,
        data_root=fake_dataset_root,
        task="single_grasp",
        split="val",
        camera="camera_1_color",
        episode=0,
        resolution=32,
        context_size=2,
        num_steps=1,
        max_frames=4,
        duration_ms=40,
        output_dir=output_dir,
        device="cpu",
    )
    stats = json.loads((output_dir / "episode_0_stats.json").read_text())
    assert Path(result["grid_path"]).exists()
    assert Path(result["video_path"]).exists()
    assert stats["predicted_frame_count"] == 6
    assert stats["decoded_frame_count"] == 6
    assert stats["exported_video_frame_count"] == 6
