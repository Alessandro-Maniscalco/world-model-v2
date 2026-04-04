"""Tests for the minimal single-clip dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.minimal.dataset import (
    MinimalFrameDataset,
    MinimalTransitionDataset,
    MinimalValidationClipDataset,
)


def test_minimal_frame_dataset_returns_six_frames(fake_long_dataset_root: Path) -> None:
    """The reconstruction dataset should expose frames 111 through 116."""

    dataset = MinimalFrameDataset(data_root=str(fake_long_dataset_root))
    assert len(dataset) == 6
    first = dataset[0]
    last = dataset[5]
    assert first["frame"].shape == (3, 128, 128)
    assert first["frame_idx"].item() == 111
    assert last["frame_idx"].item() == 116


def test_minimal_transition_dataset_returns_five_pairs(fake_long_dataset_root: Path) -> None:
    """The transition dataset should expose five consecutive frame pairs."""

    dataset = MinimalTransitionDataset(data_root=str(fake_long_dataset_root))
    assert len(dataset) == 5
    first = dataset[0]
    last = dataset[4]
    assert first["current_frame"].shape == (3, 128, 128)
    assert first["next_frame"].shape == (3, 128, 128)
    assert first["current_frame_idx"].item() == 111
    assert first["next_frame_idx"].item() == 112
    assert last["current_frame_idx"].item() == 115
    assert last["next_frame_idx"].item() == 116


def test_minimal_validation_clip_returns_exact_indices(fake_long_dataset_root: Path) -> None:
    """The validation dataset should return the full `111:116` clip once."""

    dataset = MinimalValidationClipDataset(data_root=str(fake_long_dataset_root))
    sample = dataset[0]
    assert sample["frames"].shape == (6, 3, 128, 128)
    assert torch.equal(sample["frame_idx"], torch.arange(111, 117))
