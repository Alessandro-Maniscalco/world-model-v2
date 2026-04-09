"""Tests for the root clip dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout
from world_model_v2.dataset import (
    FrameDataset,
    TransitionDataset,
    ValidationClipDataset,
    list_episode_indices,
)


def test_frame_dataset_defaults_to_full_episode(fake_long_dataset_root: Path) -> None:
    """The reconstruction dataset should default to the full episode."""

    dataset = FrameDataset(data_root=str(fake_long_dataset_root))
    assert len(dataset) == 130
    first = dataset[0]
    last = dataset[129]
    assert first["frame"].shape == (3, 128, 128)
    assert first["frame_idx"].item() == 0
    assert last["frame_idx"].item() == 129


def test_transition_dataset_defaults_to_full_episode_pairs(
    fake_long_dataset_root: Path,
) -> None:
    """The transition dataset should default to all 5-frame training windows."""

    dataset = TransitionDataset(data_root=str(fake_long_dataset_root))
    assert len(dataset) == 130 - DYNAMICS_FRAME_LAYOUT.max_frames + 1
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    assert first["context_frames"].shape == (DYNAMICS_FRAME_LAYOUT.context_frames, 3, 128, 128)
    assert first["target_frames"].shape == (DYNAMICS_FRAME_LAYOUT.target_frames, 3, 128, 128)
    assert first["actions"].shape == (DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
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
    assert torch.equal(
        first["context_frame_idx"],
        torch.arange(DYNAMICS_FRAME_LAYOUT.context_frames, dtype=torch.long),
    )
    assert torch.equal(
        first["target_frame_idx"],
        torch.arange(
            DYNAMICS_FRAME_LAYOUT.context_frames,
            DYNAMICS_FRAME_LAYOUT.max_frames,
            dtype=torch.long,
        ),
    )
    assert last["actions"].shape == (DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    last_start = len(dataset) - 1
    assert torch.equal(
        last["context_frame_idx"],
        torch.arange(
            last_start,
            last_start + DYNAMICS_FRAME_LAYOUT.context_frames,
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        last["target_frame_idx"],
        torch.arange(
            last_start + DYNAMICS_FRAME_LAYOUT.context_frames,
            last_start + DYNAMICS_FRAME_LAYOUT.max_frames,
            dtype=torch.long,
        ),
    )


def test_validation_clip_defaults_to_full_episode(fake_long_dataset_root: Path) -> None:
    """The validation dataset should return the full episode once by default."""

    dataset = ValidationClipDataset(data_root=str(fake_long_dataset_root))
    sample = dataset[0]
    assert sample["frames"].shape == (130, 3, 128, 128)
    assert sample["actions"].shape == (129, 4)
    assert torch.equal(sample["frame_idx"], torch.arange(130))


def test_datasets_preserve_explicit_frame_slice(fake_long_dataset_root: Path) -> None:
    """Explicit frame bounds should still produce the requested slice."""

    frame_dataset = FrameDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=116,
    )
    transition_dataset = TransitionDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=116,
    )
    validation_dataset = ValidationClipDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=116,
    )

    assert len(frame_dataset) == 6
    assert len(transition_dataset) == 6 - DYNAMICS_FRAME_LAYOUT.max_frames + 1
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


def test_dataset_supports_rectangular_resize(fake_long_dataset_root: Path) -> None:
    """The datasets should preserve non-square aspect ratios when requested."""

    dataset = FrameDataset(
        data_root=str(fake_long_dataset_root),
        height=240,
        width=320,
    )
    sample = dataset[0]
    assert sample["frame"].shape == (3, 240, 320)


def test_transition_dataset_supports_custom_frame_layout(
    fake_long_dataset_root: Path,
) -> None:
    """TransitionDataset should expose custom context and target window sizes."""

    dataset = TransitionDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=112,
        frame_layout=DynamicsFrameLayout(context_frames=1, target_frames=1),
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["context_frames"].shape == (1, 3, 128, 128)
    assert sample["target_frames"].shape == (1, 3, 128, 128)
    assert sample["actions"].shape == (1, 4)
    assert torch.equal(sample["context_frame_idx"], torch.tensor([111]))
    assert torch.equal(sample["target_frame_idx"], torch.tensor([112]))


def test_transition_dataset_exposes_future_rollout_targets_when_requested(
    fake_long_dataset_root: Path,
) -> None:
    """TransitionDataset should expose extra rollout targets and actions for same-context self-forcing."""

    dataset = TransitionDataset(
        data_root=str(fake_long_dataset_root),
        frame_start=111,
        frame_end=115,
        frame_layout=DynamicsFrameLayout(context_frames=1, target_frames=2),
        rollout_context_frames=1,
        rollout_chunks=1,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["future_target_frames"].shape == (2, 3, 128, 128)
    assert sample["future_actions"].shape == (2, 4)
    assert torch.equal(sample["future_target_frame_idx"], torch.tensor([114, 115]))
    assert torch.equal(
        sample["future_actions"],
        torch.tensor(
            [
                [113.0, 114.0, 115.0, 116.0],
                [114.0, 115.0, 116.0, 117.0],
            ]
        ),
    )


def test_list_episode_indices_returns_sorted_episode_numbers(
    fake_multi_episode_dataset_root: Path,
) -> None:
    """Episode discovery should return the sorted indices present on disk."""

    assert list_episode_indices(
        data_root=str(fake_multi_episode_dataset_root),
        task="single_grasp",
        split="train",
    ) == [0, 1]


def test_frame_dataset_can_span_all_episodes(
    fake_multi_episode_dataset_root: Path,
) -> None:
    """The AE dataset should flatten the full available range across all episodes."""

    dataset = FrameDataset(
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
