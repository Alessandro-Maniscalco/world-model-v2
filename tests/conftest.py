"""Shared pytest fixtures for the upstream-shaped Stage-1 test suite."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from world_model_v2.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig
from world_model_v2.minimal.experiment import (
    MinimalExperimentConfig,
    save_minimal_checkpoint,
)
from world_model_v2.minimal.model import MinimalWorldModel
from world_model_v2.utils.checkpointing import save_checkpoint


def write_episode(path: Path, frames: int = 6, height: int = 32, width: int = 32) -> None:
    """Create a small synthetic HDF5 episode file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image_stack = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for frame_idx in range(frames):
        image_stack[frame_idx, :, :, 0] = (frame_idx * 20) % 256
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
def fake_long_dataset_root(tmp_path: Path) -> Path:
    """Create a longer train/val dataset tree for the default `111:116` slice."""

    root = tmp_path / "long_data"
    write_episode(root / "single_grasp" / "train" / "episode_0.hdf5", frames=130)
    write_episode(root / "single_grasp" / "val" / "episode_0.hdf5", frames=130)
    return root


@pytest.fixture()
def saved_stage1_checkpoint(fake_dataset_root: Path, tmp_path: Path) -> Path:
    """Create a small valid Stage-1 checkpoint for inference and bootstrap tests."""

    checkpoint_path = tmp_path / "stage1_checkpoint.pt"
    run_config = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
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
        None,
        3,
        run_config.to_dict(),
        normalization_stats,
    )
    return checkpoint_path


@pytest.fixture()
def saved_stage2_checkpoint(
    fake_dataset_root: Path,
    saved_stage1_checkpoint: Path,
    tmp_path: Path,
) -> Path:
    """Create a small valid Stage-2 checkpoint bootstrapped from Stage 1."""

    checkpoint_path = tmp_path / "stage2_checkpoint.pt"
    run_config = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
            split="train",
            obs_keys=("camera_1_color",),
            resolution=32,
            horizon=4,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            sigma_min=0.01,
            sigma_max=0.5,
            infer_steps=2,
            dyn_infer_steps=1,
            load_ae=str(saved_stage1_checkpoint),
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        experiment=ExperimentConfig(run_name="fixture_stage2", device="cpu"),
    )
    model = LatentWorldModel(run_config.algorithm, obs_keys=run_config.dataset.obs_keys)
    model.bootstrap_from_checkpoint(str(saved_stage1_checkpoint), device="cpu")
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-3,
    )
    normalization_stats = {"image_range": [0.0, 1.0], "action_min": [0, 0, 0, 0], "action_max": [1, 1, 1, 1]}
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        None,
        2,
        run_config.to_dict(),
        normalization_stats,
    )
    return checkpoint_path


@pytest.fixture()
def saved_checkpoint(saved_stage1_checkpoint: Path) -> Path:
    """Preserve the legacy Stage-1 checkpoint fixture name for existing tests."""

    return saved_stage1_checkpoint


@pytest.fixture()
def saved_minimal_joint_checkpoint(fake_long_dataset_root: Path, tmp_path: Path) -> Path:
    """Create a deterministic minimal joint checkpoint for loading tests."""

    checkpoint_path = tmp_path / "minimal_joint.pt"
    model = MinimalWorldModel()
    for parameter in model.encoder.parameters():
        torch.nn.init.constant_(parameter, 0.25)
    for parameter in model.decoder.parameters():
        torch.nn.init.constant_(parameter, 0.5)
    for parameter in model.dynamics.parameters():
        torch.nn.init.constant_(parameter, 0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = MinimalExperimentConfig(
        mode="joint",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="fixture_joint",
        device="cpu",
    )
    save_minimal_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=5,
        config=config.to_dict(),
        mode=config.mode,
        clip_metadata=config.clip_metadata(),
        best_metric=0.123,
    )
    return checkpoint_path
