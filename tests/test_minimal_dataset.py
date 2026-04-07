"""Tests for the minimal single-clip dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.minimal.dataset import (
    MinimalFrameDataset,
    MinimalTransitionDataset,
    MinimalValidationClipDataset,
    list_episode_indices,
)


def test_minimal_frame_dataset_defaults_to_full_episode(fake_long_dataset_root: Path) -> None:
    """The reconstruction dataset should default to the full episode."""

    dataset = MinimalFrameDataset(data_root=str(fake_long_dataset_root))
    assert len(dataset) == 130
    first = dataset[0]
    last = dataset[129]
    assert first["frame"].shape == (3, 128, 128)
    assert first["frame_idx"].item() == 0
    assert last["frame_idx"].item() == 129


def test_minimal_transition_dataset_defaults_to_full_episode_pairs(
    fake_long_dataset_root: Path,
) -> None:
    """The transition dataset should default to all 5-frame training windows."""

    dataset = MinimalTransitionDataset(data_root=str(fake_long_dataset_root))
    assert len(dataset) == 126
    first = dataset[0]
    last = dataset[125]
    assert first["context_frames"].shape == (3, 3, 128, 128)
    assert first["target_frames"].shape == (2, 3, 128, 128)
    assert first["actions"].shape == (4, 4)
    assert torch.equal(
        first["actions"],
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        ),
    )
    assert torch.equal(first["context_frame_idx"], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(first["target_frame_idx"], torch.tensor([3, 4], dtype=torch.long))
    assert last["actions"].shape == (4, 4)
    assert torch.equal(last["context_frame_idx"], torch.tensor([125, 126, 127], dtype=torch.long))
    assert torch.equal(last["target_frame_idx"], torch.tensor([128, 129], dtype=torch.long))


def test_minimal_validation_clip_defaults_to_full_episode(fake_long_dataset_root: Path) -> None:
    """The validation dataset should return the full episode once by default."""

    dataset = MinimalValidationClipDataset(data_root=str(fake_long_dataset_root))
    sample = dataset[0]
    assert sample["frames"].shape == (130, 3, 128, 128)
    assert sample["actions"].shape == (129, 4)
    assert torch.equal(sample["frame_idx"], torch.arange(130))


def test_minimal_datasets_preserve_explicit_frame_slice(fake_long_dataset_root: Path) -> None:
    """Explicit frame bounds should still produce the requested slice."""

    frame_dataset = MinimalFrameDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=116,
    )
    transition_dataset = MinimalTransitionDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=116,
    )
    validation_dataset = MinimalValidationClipDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=116,
    )

    assert len(frame_dataset) == 6
    assert len(transition_dataset) == 2
    assert torch.equal(
        transition_dataset[0]["actions"],
        torch.tensor(
            [
                [111.0, 112.0, 113.0, 114.0],
                [112.0, 113.0, 114.0, 115.0],
                [113.0, 114.0, 115.0, 116.0],
                [114.0, 115.0, 116.0, 117.0],
            ]
        ),
    )
    assert validation_dataset[0]["actions"].shape == (5, 4)
    assert torch.equal(validation_dataset[0]["frame_idx"], torch.arange(111, 117))


def test_minimal_dataset_supports_rectangular_resize(fake_long_dataset_root: Path) -> None:
    """The minimal datasets should preserve non-square aspect ratios when requested."""

    dataset = MinimalFrameDataset(
        data_root=str(fake_long_dataset_root),
        height=240,
        width=320,
    )
    sample = dataset[0]
    assert sample["frame"].shape == (3, 240, 320)


def test_list_episode_indices_returns_sorted_episode_numbers(
    fake_multi_episode_dataset_root: Path,
) -> None:
    """Episode discovery should return the sorted indices present on disk."""

    assert list_episode_indices(
        data_root=str(fake_multi_episode_dataset_root),
        task="single_grasp",
        split="train",
    ) == [0, 1]


def test_minimal_frame_dataset_can_span_all_episodes(
    fake_multi_episode_dataset_root: Path,
) -> None:
    """The AE dataset should flatten the full available range across all episodes."""

    dataset = MinimalFrameDataset(
        data_root=str(fake_multi_episode_dataset_root),
        split="train",
        all_episodes=True,
    )
    assert len(dataset) == 245
    first = dataset[0]
    second_episode_first = dataset[130]
    last = dataset[244]
    assert first["episode_idx"].item() == 0
    assert first["frame_idx"].item() == 0
    assert second_episode_first["episode_idx"].item() == 1
    assert second_episode_first["frame_idx"].item() == 0
    assert last["episode_idx"].item() == 1
    assert last["frame_idx"].item() == 114
