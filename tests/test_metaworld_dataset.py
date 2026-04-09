"""Tests for the parquet-backed MetaWorld dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout
from world_model_v2.metaworld_dataset import (
    MetaWorldFrameDataset,
    MetaWorldTransitionDataset,
    MetaWorldValidationClipDataset,
    load_metaworld_clip,
    resolve_metaworld_split,
)


def test_load_metaworld_clip_returns_requested_slice(
    fake_metaworld_dataset_root: Path,
) -> None:
    """Loading one MT50 clip should preserve the requested episode-local slice."""

    clip = load_metaworld_clip(
        data_root=str(fake_metaworld_dataset_root),
        split="train",
        task_index=0,
        episode=0,
        frame_start=1,
        frame_end=2,
        resolution=8,
        load_actions=True,
    )
    assert clip["frames"].shape == (2, 3, 8, 8)
    assert clip["actions"].shape == (1, 4)
    assert torch.equal(clip["actions"][0], torch.tensor([20.0, 21.0, 22.0, 23.0]))
    assert torch.equal(clip["frame_idx"], torch.tensor([1, 2], dtype=torch.long))
    assert clip["episode_idx"].item() == 0


def test_metaworld_frame_dataset_flattens_all_task_episodes(
    fake_metaworld_dataset_root: Path,
) -> None:
    """All-episode MT50 training should flatten frames across the filtered task subset."""

    dataset = MetaWorldFrameDataset(
        data_root=str(fake_metaworld_dataset_root),
        split="train",
        task_index=0,
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


def test_metaworld_transition_dataset_returns_consecutive_pairs(
    fake_metaworld_dataset_root: Path,
) -> None:
    """The MT50 transition dataset should expose sliding 5-frame windows."""

    dataset = MetaWorldTransitionDataset(
        data_root=str(fake_metaworld_dataset_root),
        split="train",
        task_index=0,
        episode=0,
        resolution=8,
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["context_frames"].shape == (DYNAMICS_FRAME_LAYOUT.context_frames, 3, 8, 8)
    assert sample["target_frames"].shape == (DYNAMICS_FRAME_LAYOUT.target_frames, 3, 8, 8)
    assert sample["actions"].shape == (DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    assert torch.equal(sample["actions"][0], torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert torch.equal(
        sample["context_frame_idx"],
        torch.arange(DYNAMICS_FRAME_LAYOUT.context_frames, dtype=torch.long),
    )
    assert torch.equal(
        sample["target_frame_idx"],
        torch.arange(
            DYNAMICS_FRAME_LAYOUT.context_frames,
            DYNAMICS_FRAME_LAYOUT.max_frames,
            dtype=torch.long,
        ),
    )
    assert sample["episode_idx"].item() == 0


def test_metaworld_validation_dataset_returns_one_clip(
    fake_metaworld_dataset_root: Path,
) -> None:
    """The MT50 validation clip dataset should return a single cached episode clip."""

    dataset = MetaWorldValidationClipDataset(
        data_root=str(fake_metaworld_dataset_root),
        split="val",
        task_index=1,
        episode=0,
        resolution=8,
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["frames"].shape == (2, 3, 8, 8)
    assert sample["actions"].shape == (1, 4)
    assert torch.equal(sample["actions"][0], torch.tensor([90.0, 91.0, 92.0, 93.0]))
    assert torch.equal(sample["frame_idx"], torch.tensor([0, 1], dtype=torch.long))


def test_metaworld_transition_dataset_supports_custom_frame_layout(
    fake_metaworld_dataset_root: Path,
) -> None:
    """The MT50 transition dataset should support custom context and target windows."""

    dataset = MetaWorldTransitionDataset(
        data_root=str(fake_metaworld_dataset_root),
        split="train",
        task_index=0,
        episode=0,
        resolution=8,
        frame_start=0,
        frame_end=1,
        frame_layout=DynamicsFrameLayout(context_frames=1, target_frames=1),
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["context_frames"].shape == (1, 3, 8, 8)
    assert sample["target_frames"].shape == (1, 3, 8, 8)
    assert sample["actions"].shape == (1, 4)
    assert torch.equal(sample["context_frame_idx"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(sample["target_frame_idx"], torch.tensor([1], dtype=torch.long))


def test_resolve_metaworld_split_maps_val_to_train() -> None:
    """MT50 should treat `val` as a request for the train-only split."""

    assert resolve_metaworld_split("val") == "train"
