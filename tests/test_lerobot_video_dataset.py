"""Tests for the episode-sharded LeRobot video dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.dynamics_transformer import DynamicsFrameLayout
from world_model_v2.lerobot_video_dataset import (
    LeRobotVideoFrameDataset,
    LeRobotVideoTransitionDataset,
    LeRobotVideoValidationClipDataset,
    SO101_BASE_SIM_PICKPLACE_ACTION_DIM,
    load_lerobot_video_clip,
    resolve_lerobot_video_split,
)


def test_load_lerobot_video_clip_returns_requested_slice(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """Loading one LeRobot video clip should preserve the requested slice and actions."""

    clip = load_lerobot_video_clip(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        episode=0,
        frame_start=1,
        frame_end=2,
        resolution=8,
        load_actions=True,
    )
    assert clip["frames"].shape == (2, 3, 8, 8)
    assert clip["actions"].shape == (1, SO101_BASE_SIM_PICKPLACE_ACTION_DIM)
    assert torch.equal(
        clip["actions"][0],
        torch.tensor(
            [20.0 + offset for offset in range(SO101_BASE_SIM_PICKPLACE_ACTION_DIM)],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(clip["frame_idx"], torch.tensor([1, 2], dtype=torch.long))
    assert clip["episode_idx"].item() == 0
    assert float(clip["frames"][1, 0].mean()) > float(clip["frames"][0, 0].mean())


def test_lerobot_video_frame_dataset_flattens_all_episodes(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """All-episode LeRobot video training should flatten frames across every episode."""

    dataset = LeRobotVideoFrameDataset(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        resolution=8,
        all_episodes=True,
    )
    assert len(dataset) == 8
    first = dataset[0]
    last = dataset[7]
    assert first["episode_idx"].item() == 0
    assert first["frame_idx"].item() == 0
    assert last["episode_idx"].item() == 1
    assert last["frame_idx"].item() == 2
    sampler = dataset.training_sampler()
    assert sampler is not None
    assert sorted(list(sampler)) == list(range(len(dataset)))


def test_lerobot_video_transition_dataset_returns_sliding_windows(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """The LeRobot video transition dataset should expose sliding dynamics windows."""

    dataset = LeRobotVideoTransitionDataset(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        episode=0,
        resolution=8,
        frame_layout=DynamicsFrameLayout(
            context_frames=1,
            target_frames=1,
            temporal_compression_ratio=1,
        ),
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["context_frames"].shape == (1, 3, 8, 8)
    assert sample["target_frames"].shape == (1, 3, 8, 8)
    assert sample["actions"].shape == (1, SO101_BASE_SIM_PICKPLACE_ACTION_DIM)
    assert torch.equal(
        sample["actions"][0],
        torch.tensor(
            [10.0 + offset for offset in range(SO101_BASE_SIM_PICKPLACE_ACTION_DIM)],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(sample["context_frame_idx"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(sample["target_frame_idx"], torch.tensor([1], dtype=torch.long))


def test_lerobot_video_frame_dataset_can_include_motion_neighbors(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """LeRobot frame samples should optionally include adjacent GT frames for motion losses."""

    dataset = LeRobotVideoFrameDataset(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        resolution=8,
        all_episodes=True,
        include_motion_neighbors=True,
    )
    sample = dataset[1]
    assert sample["prev_frame"].shape == (3, 8, 8)
    assert sample["next_frame"].shape == (3, 8, 8)
    assert float(sample["next_frame"][0].mean()) > float(sample["prev_frame"][0].mean())


def test_lerobot_video_transition_dataset_exposes_future_rollout_targets(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """The LeRobot video transition dataset should expose future rollout targets when requested."""

    dataset = LeRobotVideoTransitionDataset(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        episode=0,
        resolution=8,
        frame_start=0,
        frame_end=4,
        frame_layout=DynamicsFrameLayout(
            context_frames=1,
            target_frames=2,
            temporal_compression_ratio=1,
        ),
        rollout_context_frames=1,
        rollout_chunks=1,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["future_target_frames"].shape == (2, 3, 8, 8)
    assert sample["future_actions"].shape == (2, SO101_BASE_SIM_PICKPLACE_ACTION_DIM)
    assert torch.equal(sample["future_target_frame_idx"], torch.tensor([3, 4], dtype=torch.long))
    assert torch.equal(
        sample["future_actions"][:, :4],
        torch.tensor(
            [
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_lerobot_video_validation_dataset_returns_one_clip(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """The LeRobot video validation dataset should return a single cached episode clip."""

    dataset = LeRobotVideoValidationClipDataset(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="val",
        episode=1,
        resolution=8,
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["frames"].shape == (3, 3, 8, 8)
    assert sample["actions"].shape == (2, SO101_BASE_SIM_PICKPLACE_ACTION_DIM)
    assert sample["episode_idx"].item() == 1
    assert torch.equal(sample["frame_idx"], torch.tensor([0, 1, 2], dtype=torch.long))


def test_resolve_lerobot_video_split_maps_val_to_train() -> None:
    """The LeRobot video helper should map validation requests onto the train alias."""

    assert resolve_lerobot_video_split("train") == "train"
    assert resolve_lerobot_video_split("val") == "train"
