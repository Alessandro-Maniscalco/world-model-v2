"""Shared pytest fixtures for the upstream-shaped Stage-1 test suite."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from world_model_v2.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig
from world_model_v2.utils.checkpointing import save_checkpoint


def write_episode(path: Path, frames: int = 6, height: int = 32, width: int = 32) -> None:
    """Create a small synthetic HDF5 episode file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image_stack = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for frame_idx in range(frames):
        image_stack[frame_idx, :, :, 0] = frame_idx * 20
        image_stack[frame_idx, :, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
        image_stack[frame_idx, :, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
    actions = np.stack(
        [np.array([frame_idx, frame_idx + 1, frame_idx + 2, frame_idx + 3], dtype=np.float32) for frame_idx in range(frames)],
        axis=0,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=actions)
        obs = handle.create_group("obs")
        images = obs.create_group("images")
        images.create_dataset("camera_1_color", data=image_stack)


@pytest.fixture()
def fake_dataset_root(tmp_path: Path) -> Path:
    """Create a tiny train/val dataset tree."""

    root = tmp_path / "data"
    write_episode(root / "single_grasp" / "train" / "episode_0.hdf5")
    write_episode(root / "single_grasp" / "val" / "episode_0.hdf5")
    return root


@pytest.fixture()
def saved_checkpoint(tmp_path: Path) -> Path:
    """Create a small valid checkpoint for inference tests."""

    checkpoint_path = tmp_path / "checkpoint.pt"
    run_config = RunConfig(
        dataset=DatasetConfig(
            data_root=str(tmp_path / "data"),
            split="train",
            obs_keys=("camera_1_color",),
            resolution=32,
            horizon=1,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=1,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            sigma_min=0.01,
            sigma_max=0.5,
            infer_steps=2,
        ),
        experiment=ExperimentConfig(run_name="fixture", device="cpu"),
    )
    model = LatentWorldModel(run_config.algorithm, obs_keys=run_config.dataset.obs_keys)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    normalization_stats = {"image_range": [0.0, 1.0], "action_min": [0, 0, 0, 0], "action_max": [1, 1, 1, 1]}
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        3,
        run_config.to_dict(),
        normalization_stats,
    )
    return checkpoint_path
