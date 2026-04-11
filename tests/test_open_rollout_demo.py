"""Tests for the standalone open-rollout demo helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.check.open_rollout_demo import build_rollout_stats


def test_build_rollout_stats_records_open_rollout_boundary_fields() -> None:
    """Rollout stats should expose the same boundary counters used by validation artifacts."""

    stats = build_rollout_stats(
        checkpoint_path=Path("checkpoints/best.pt"),
        device=torch.device("cpu"),
        infer_steps=32,
        input_frame_count=20,
        decoded_frame_count=20,
        seed_frames=1,
        stride_frames=None,
        initial_stride_frames=3,
        frame_mse=0.01,
        frame_l1=0.02,
        motion_ratio=1.25,
        predicted_motion_l1=0.05,
        ground_truth_motion_l1=0.04,
    )

    assert stats["input_frame_count"] == 20
    assert stats["decoded_frame_count"] == 20
    assert stats["predicted_frame_count"] == 20
    assert stats["exported_video_frame_count"] == 20
    assert stats["seed_frames"] == 1
    assert stats["loss_frames"] == 19
    assert stats["open_rollout_context_frames"] == 1
    assert stats["open_rollout_seed_frames"] == 1
    assert stats["open_rollout_loss_frames"] == 19
    assert stats["open_rollout_decoded_frame_count"] == 20
    assert stats["open_rollout_predicted_frame_count"] == 20
    assert stats["open_rollout_stride_frames"] is None
    assert stats["open_rollout_initial_stride_frames"] == 3
    assert stats["validation_style"] == "open_rollout_autoregressive"
    assert stats["open_rollout_validation_style"] == "open_rollout_autoregressive"
    assert stats["open_rollout_predicted_target_motion_l1"] == pytest.approx(0.05)
    assert stats["open_rollout_ground_truth_target_motion_l1"] == pytest.approx(0.04)
    assert stats["open_rollout_target_motion_ratio"] == pytest.approx(1.25)


def test_build_rollout_stats_skips_motion_fields_when_ratio_is_missing() -> None:
    """Motion-specific fields should be omitted when no motion ratio is available."""

    stats = build_rollout_stats(
        checkpoint_path="checkpoints/best.pt",
        device=torch.device("cpu"),
        infer_steps=16,
        input_frame_count=2,
        decoded_frame_count=2,
        seed_frames=1,
        stride_frames=1,
        initial_stride_frames=1,
        frame_mse=0.5,
        frame_l1=0.25,
        motion_ratio=None,
        predicted_motion_l1=0.0,
        ground_truth_motion_l1=0.0,
    )

    assert stats["open_rollout_stride_frames"] == 1
    assert stats["open_rollout_loss_frames"] == 1
    assert "open_rollout_predicted_target_motion_l1" not in stats
    assert "open_rollout_ground_truth_target_motion_l1" not in stats
    assert "open_rollout_target_motion_ratio" not in stats
    assert "open_rollout_motion_log_error" not in stats
