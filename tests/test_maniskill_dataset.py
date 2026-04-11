"""Tests for the replayed ManiSkill dataset helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.dynamics_transformer import DynamicsFrameLayout
from world_model_v2.maniskill_dataset import (
    MANISKILL_DEFAULT_ACTION_DIM,
    ManiSkillFrameDataset,
    ManiSkillTransitionDataset,
    ManiSkillValidationClipDataset,
    load_maniskill_clip,
    resolve_maniskill_split,
)


def test_load_maniskill_clip_returns_requested_slice(
    fake_maniskill_replay_root: Path,
) -> None:
    """Loading one ManiSkill clip should preserve the requested episode-local slice."""

    clip = load_maniskill_clip(
        data_root=str(fake_maniskill_replay_root),
        split="train",
        episode=0,
        frame_start=1,
        frame_end=2,
        resolution=8,
        load_actions=True,
    )
    assert clip["frames"].shape == (2, 3, 8, 8)
    assert clip["actions"].shape == (1, MANISKILL_DEFAULT_ACTION_DIM)
    assert torch.equal(
        clip["actions"][0],
        torch.tensor([20.0 + offset for offset in range(MANISKILL_DEFAULT_ACTION_DIM)]),
    )
    assert torch.equal(clip["frame_idx"], torch.tensor([1, 2], dtype=torch.long))
    assert clip["episode_idx"].item() == 0


def test_maniskill_frame_dataset_flattens_all_episodes(
    fake_maniskill_replay_root: Path,
) -> None:
    """All-episode ManiSkill training should flatten frames across every episode."""

    dataset = ManiSkillFrameDataset(
        data_root=str(fake_maniskill_replay_root),
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


def test_maniskill_transition_dataset_returns_sliding_windows(
    fake_maniskill_replay_root: Path,
) -> None:
    """The ManiSkill transition dataset should expose sliding dynamics windows."""

    dataset = ManiSkillTransitionDataset(
        data_root=str(fake_maniskill_replay_root),
        split="train",
        episode=0,
        resolution=8,
        frame_layout=DynamicsFrameLayout(context_frames=1, target_frames=1),
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["context_frames"].shape == (1, 3, 8, 8)
    assert sample["target_frames"].shape == (1, 3, 8, 8)
    assert sample["actions"].shape == (1, MANISKILL_DEFAULT_ACTION_DIM)
    assert torch.equal(
        sample["actions"][0],
        torch.tensor([10.0 + offset for offset in range(MANISKILL_DEFAULT_ACTION_DIM)]),
    )


def test_maniskill_transition_dataset_exposes_future_rollout_targets(
    fake_maniskill_replay_root: Path,
) -> None:
    """The ManiSkill transition dataset should expose future rollout targets when requested."""

    dataset = ManiSkillTransitionDataset(
        data_root=str(fake_maniskill_replay_root),
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
    assert sample["future_actions"].shape == (2, MANISKILL_DEFAULT_ACTION_DIM)
    assert torch.equal(sample["future_target_frame_idx"], torch.tensor([3, 4], dtype=torch.long))


def test_maniskill_validation_dataset_returns_one_clip(
    fake_maniskill_replay_root: Path,
) -> None:
    """The ManiSkill validation dataset should return a single cached episode clip."""

    dataset = ManiSkillValidationClipDataset(
        data_root=str(fake_maniskill_replay_root),
        split="val",
        episode=1,
        resolution=8,
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["frames"].shape == (3, 3, 8, 8)
    assert sample["actions"].shape == (2, MANISKILL_DEFAULT_ACTION_DIM)
    assert sample["episode_idx"].item() == 1


def test_resolve_maniskill_split_maps_val_to_train() -> None:
    """The ManiSkill helper should map validation requests onto the train alias."""

    assert resolve_maniskill_split("train") == "train"
    assert resolve_maniskill_split("val") == "train"
