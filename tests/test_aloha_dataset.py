"""Tests for the MP4-backed ALOHA LeRobot dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.dynamics_transformer import DynamicsFrameLayout
from world_model_v2.metaworld_dataset import (
    AlohaFrameDataset,
    AlohaTransitionDataset,
    AlohaValidationClipDataset,
    load_aloha_clip,
    resolve_aloha_split,
)


def test_load_aloha_clip_returns_requested_slice(fake_aloha_dataset_root: Path) -> None:
    """Loading one ALOHA clip should preserve the requested slice and actions."""

    clip = load_aloha_clip(
        data_root=str(fake_aloha_dataset_root),
        split="train",
        episode=0,
        frame_start=1,
        frame_end=2,
        resolution=8,
        load_actions=True,
    )
    assert clip["frames"].shape == (2, 3, 8, 8)
    assert clip["actions"].shape == (1, 14)
    assert torch.equal(
        clip["actions"][0],
        torch.tensor([20.0 + offset for offset in range(14)], dtype=torch.float32),
    )
    assert torch.equal(clip["frame_idx"], torch.tensor([1, 2], dtype=torch.long))
    assert clip["episode_idx"].item() == 0
    assert float(clip["frames"][1, 0].mean()) > float(clip["frames"][0, 0].mean())


def test_aloha_frame_dataset_flattens_all_episodes(fake_aloha_dataset_root: Path) -> None:
    """All-episode ALOHA training should flatten frames across every episode."""

    dataset = AlohaFrameDataset(
        data_root=str(fake_aloha_dataset_root),
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


def test_aloha_frame_dataset_excludes_selected_episodes(
    fake_aloha_dataset_root: Path,
) -> None:
    """All-episode ALOHA frame training should exclude selected validation episodes."""

    dataset = AlohaFrameDataset(
        data_root=str(fake_aloha_dataset_root),
        split="train",
        resolution=8,
        all_episodes=True,
        exclude_episodes=(0,),
    )
    assert len(dataset) == 3
    first = dataset[0]
    last = dataset[2]
    assert first["episode_idx"].item() == 1
    assert first["frame_idx"].item() == 0
    assert last["episode_idx"].item() == 1
    assert last["frame_idx"].item() == 2


def test_aloha_transition_dataset_returns_sliding_windows(
    fake_aloha_dataset_root: Path,
) -> None:
    """The ALOHA transition dataset should expose sliding dynamics windows."""

    dataset = AlohaTransitionDataset(
        data_root=str(fake_aloha_dataset_root),
        split="train",
        episode=0,
        resolution=8,
        frame_layout=DynamicsFrameLayout(context_frames=1, target_frames=1),
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["context_frames"].shape == (1, 3, 8, 8)
    assert sample["target_frames"].shape == (1, 3, 8, 8)
    assert sample["actions"].shape == (1, 14)
    assert torch.equal(
        sample["actions"][0],
        torch.tensor([10.0 + offset for offset in range(14)], dtype=torch.float32),
    )
    assert torch.equal(sample["context_frame_idx"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(sample["target_frame_idx"], torch.tensor([1], dtype=torch.long))


def test_aloha_transition_dataset_exposes_future_rollout_targets(
    fake_aloha_dataset_root: Path,
) -> None:
    """The ALOHA transition dataset should expose future rollout targets when requested."""

    dataset = AlohaTransitionDataset(
        data_root=str(fake_aloha_dataset_root),
        split="train",
        episode=0,
        resolution=8,
        frame_start=0,
        frame_end=4,
        frame_layout=DynamicsFrameLayout(context_frames=1, target_frames=2),
        rollout_context_frames=1,
        rollout_chunks=1,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["future_target_frames"].shape == (2, 3, 8, 8)
    assert sample["future_actions"].shape == (2, 14)
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


def test_aloha_validation_dataset_returns_one_clip(fake_aloha_dataset_root: Path) -> None:
    """The ALOHA validation dataset should return a single cached episode clip."""

    dataset = AlohaValidationClipDataset(
        data_root=str(fake_aloha_dataset_root),
        split="val",
        episode=1,
        resolution=8,
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["frames"].shape == (3, 3, 8, 8)
    assert sample["actions"].shape == (2, 14)
    assert sample["episode_idx"].item() == 1
    assert torch.equal(sample["frame_idx"], torch.tensor([0, 1, 2], dtype=torch.long))


def test_resolve_aloha_split_maps_val_to_train() -> None:
    """The ALOHA helper should map validation requests onto the train-only split."""

    assert resolve_aloha_split("train") == "train"
    assert resolve_aloha_split("val") == "train"
