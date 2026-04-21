"""Tests for the episode-sharded LeRobot video dataset helpers."""

from __future__ import annotations

from pathlib import Path
import pickle

import torch

from world_model_v2.dynamics_transformer import DynamicsFrameLayout
from world_model_v2.lerobot_video_dataset import (
    LeRobotEpisodeVideoRepository,
    LeRobotVideoFrameDataset,
    LeRobotVideoFrameRecord,
    LeRobotVideoTransitionDataset,
    LeRobotVideoValidationClipDataset,
    SO101_BASE_SIM_PICKPLACE_ACTION_DIM,
    SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    SO101_RELATIVE_ACTION_SCALE,
    bgr_frame_to_exact_area_downsampled_tensor,
    crop_bgr_frame,
    load_lerobot_video_clip,
    resolve_exact_downsample_crop_bounds,
    resolve_default_lerobot_crop_bounds,
    resolve_lerobot_video_split,
    so101_relative_actions_from_absolute_targets,
)
from world_model_v2.metaworld_dataset import bgr_frame_to_tensor


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
            [30.0 + offset for offset in range(SO101_BASE_SIM_PICKPLACE_ACTION_DIM)],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(clip["frame_idx"], torch.tensor([1, 2], dtype=torch.long))
    assert clip["episode_idx"].item() == 0
    assert float(clip["frames"][1, 0].mean()) > float(clip["frames"][0, 0].mean())


def test_load_lerobot_video_clip_applies_default_so101_crop(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """SO101 clips should apply the default crop before resizing."""

    repository = LeRobotEpisodeVideoRepository(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    )
    record = repository.episode_record(0)
    raw_frame = repository._read_video_frame(record, 0)
    crop_bounds = resolve_default_lerobot_crop_bounds(
        SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        raw_frame.shape[0],
        raw_frame.shape[1],
    )

    assert crop_bounds == (0, 14, 1, 15)

    manual = bgr_frame_to_tensor(
        crop_bgr_frame(raw_frame, crop_bounds),
        height=16,
        width=16,
    )
    clip = load_lerobot_video_clip(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        episode=0,
        frame_start=0,
        frame_end=0,
        resolution=16,
        height=16,
        width=16,
    )

    assert torch.allclose(clip["frames"][0], manual)


def test_resolve_exact_downsample_crop_bounds_prefers_largest_factor() -> None:
    """Exact downsampling should trim only the sides and bottom while keeping the largest factor."""

    crop_bounds = resolve_default_lerobot_crop_bounds(
        SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        frame_height=480,
        frame_width=640,
    )

    assert crop_bounds == (0, 416, 46, 594)
    assert resolve_exact_downsample_crop_bounds(
        crop_bounds,
        target_height=208,
        target_width=272,
    ) == ((0, 416, 48, 592), 2)
    assert resolve_exact_downsample_crop_bounds(
        crop_bounds,
        target_height=96,
        target_width=128,
    ) == ((0, 384, 64, 576), 4)


def test_load_lerobot_video_clip_uses_exact_area_downsample_when_possible(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """SO101 clips should use exact area downsampling when the target size fits an integer factor."""

    repository = LeRobotEpisodeVideoRepository(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    )
    record = repository.episode_record(0)
    raw_frame = repository._read_video_frame(record, 0)
    crop_bounds = resolve_default_lerobot_crop_bounds(
        SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        raw_frame.shape[0],
        raw_frame.shape[1],
    )
    exact_crop_bounds, factor = resolve_exact_downsample_crop_bounds(
        crop_bounds,
        target_height=4,
        target_width=4,
    )

    assert factor == 3
    manual = bgr_frame_to_exact_area_downsampled_tensor(
        raw_frame,
        crop_bounds=exact_crop_bounds,
        target_height=4,
        target_width=4,
    )
    bicubic_manual = bgr_frame_to_tensor(
        crop_bgr_frame(raw_frame, crop_bounds),
        height=4,
        width=4,
    )
    clip = load_lerobot_video_clip(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        episode=0,
        resolution=4,
        height=4,
        width=4,
    )

    assert torch.allclose(clip["frames"][0], manual)
    assert not torch.allclose(clip["frames"][0], bicubic_manual)


def test_lerobot_video_repository_pickle_drops_cached_video_handles(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """Pickling should clear cached OpenCV handles so Windows workers can spawn safely."""

    repository = LeRobotEpisodeVideoRepository(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    )
    record = repository.episode_record(0)
    _ = repository._read_video_frame(record, 0)

    round_tripped = pickle.loads(pickle.dumps(repository))

    assert repository._video_captures
    assert round_tripped._video_captures == {}
    frame = round_tripped.load_frame_tensor(
        frame=LeRobotVideoFrameRecord(episode=record, frame_index=0),
        resolution=8,
        height=None,
        width=None,
    )
    assert frame.shape == (3, 8, 8)


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
            [20.0 + offset for offset in range(SO101_BASE_SIM_PICKPLACE_ACTION_DIM)],
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
                [40.0, 41.0, 42.0, 43.0],
                [50.0, 51.0, 52.0, 53.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_so101_relative_actions_match_next_state_deltas() -> None:
    """SO101 relative actions should equal scaled next-state deltas."""

    actions = torch.tensor(
        [
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        ],
        dtype=torch.float32,
    )
    states = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 100.0],
            [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 101.0],
        ],
        dtype=torch.float32,
    )

    relative = so101_relative_actions_from_absolute_targets(
        actions,
        states,
        scale=1.0,
    )

    assert torch.equal(relative, torch.full((2, 6), 10.0, dtype=torch.float32))


def test_load_lerobot_video_clip_can_return_relative_so101_actions(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """SO101 clip loading should optionally expose relative next-state deltas."""

    clip = load_lerobot_video_clip(
        data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
        split="train",
        episode=0,
        frame_start=0,
        frame_end=2,
        resolution=8,
        load_actions=True,
        action_representation="relative_delta",
        action_scale=1.0,
    )

    assert clip["actions"].shape == (2, SO101_BASE_SIM_PICKPLACE_ACTION_DIM)
    assert torch.equal(
        clip["actions"],
        torch.full((2, SO101_BASE_SIM_PICKPLACE_ACTION_DIM), 10.0, dtype=torch.float32),
    )


def test_lerobot_video_transition_dataset_can_return_relative_so101_actions(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
) -> None:
    """SO101 dynamics windows should support scaled relative actions."""

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
        action_representation="relative_delta",
        action_scale=SO101_RELATIVE_ACTION_SCALE,
    )

    sample = dataset[0]

    assert torch.equal(
        sample["actions"],
        torch.full(
            (1, SO101_BASE_SIM_PICKPLACE_ACTION_DIM),
            10.0 * SO101_RELATIVE_ACTION_SCALE,
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
