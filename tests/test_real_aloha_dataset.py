"""Tests for the upstream-shaped raw-HDF5 dataset loader."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model_v2.config import DatasetConfig
from world_model_v2.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset


def test_train_dataset_returns_sequence_shaped_obs(fake_dataset_root: Path) -> None:
    """Train samples should expose `obs` dicts and sequence-shaped actions."""

    dataset = RealAlohaDataset(
        DatasetConfig(
            data_root=str(fake_dataset_root),
            resolution=16,
            horizon=1,
            val_horizon=6,
        )
    )
    sample = dataset[2]
    assert sample["obs"]["camera_1_color"].shape == (1, 3, 16, 16)
    assert sample["action"].shape == (1, 4)
    assert sample["episode_idx"].item() == 0
    assert sample["start_idx"].item() == 2


def test_validation_dataset_returns_episode_length_sequence(fake_dataset_root: Path) -> None:
    """Validation samples should return deterministic episode-sized sequences."""

    dataset = RealAlohaDataset(
        DatasetConfig(
            data_root=str(fake_dataset_root),
            split="val",
            resolution=20,
            horizon=1,
            val_horizon=6,
        )
    )
    sample = dataset[0]
    assert sample["obs"]["camera_1_color"].shape == (6, 3, 20, 20)
    assert sample["action"].shape == (6, 4)
    assert torch.equal(sample["frame_idx"], torch.arange(6))


def test_load_episode_sequence_uses_full_episode_length(fake_dataset_root: Path) -> None:
    """Explicit episode loading should return the untruncated full episode."""

    dataset = RealAlohaDataset(
        DatasetConfig(
            data_root=str(fake_dataset_root),
            split="val",
            resolution=12,
            horizon=1,
            val_horizon=4,
        )
    )
    sample = dataset.load_episode_sequence(0)
    assert sample["obs"]["camera_1_color"].shape == (6, 3, 12, 12)
    assert sample["action"].shape == (6, 4)

