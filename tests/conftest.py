"""Shared pytest fixtures for the root world-model test suite."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from world_model_v2.experiment import ExperimentConfig, save_training_checkpoint
from world_model_v2.model import WorldModel


def write_episode(path: Path, frames: int = 6, height: int = 32, width: int = 32) -> None:
    """Create a small synthetic HDF5 episode file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image_stack = np.zeros((frames, height, width, 3), dtype=np.uint8)
    for frame_idx in range(frames):
        image_stack[frame_idx, :, :, 0] = (frame_idx * 20) % 256
        image_stack[frame_idx, :, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
        image_stack[frame_idx, :, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
    actions = np.stack(
        [
            np.array(
                [frame_idx, frame_idx + 1, frame_idx + 2, frame_idx + 3],
                dtype=np.float32,
            )
            for frame_idx in range(frames)
        ],
        axis=0,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=actions)
        obs = handle.create_group("obs")
        images = obs.create_group("images")
        images.create_dataset("camera_1_color", data=image_stack)


def encode_test_image(frame_value: int, height: int = 12, width: int = 12) -> bytes:
    """Encode one synthetic RGB image as PNG bytes for parquet-backed tests."""

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = frame_value % 256
    image[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
    image[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def write_metaworld_data_file(path: Path, frame_values: list[int]) -> None:
    """Create one small MetaWorld-style parquet shard with embedded image bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image_array = pa.StructArray.from_arrays(
        arrays=[
            pa.array(
                [encode_test_image(frame_value) for frame_value in frame_values],
                type=pa.binary(),
            ),
            pa.array(
                [f"episode_frame_{index:03d}.png" for index in range(len(frame_values))],
                type=pa.string(),
            ),
        ],
        names=["bytes", "path"],
    )
    action_array = pa.array(
        [
            [
                float(frame_value),
                float(frame_value + 1),
                float(frame_value + 2),
                float(frame_value + 3),
            ]
            for frame_value in frame_values
        ],
        type=pa.list_(pa.float32()),
    )
    pq.write_table(pa.table({"observation.image": image_array, "action": action_array}), path)


def write_metaworld_metadata(path: Path) -> None:
    """Create the MT50 metadata tables required by the parquet loader tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    meta_root = path.parents[2]
    tasks_table = pa.table(
        {
            "task_index": pa.array([0, 1], type=pa.int64()),
            "__index_level_0__": pa.array(
                ["Pick up a nut and place it onto a peg", "Open a drawer"],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(tasks_table, meta_root / "tasks.parquet")
    episodes_table = pa.table(
        {
            "episode_index": pa.array([0, 1, 2], type=pa.int64()),
            "data/chunk_index": pa.array([0, 0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0, 1], type=pa.int64()),
            "dataset_from_index": pa.array([0, 5, 8], type=pa.int64()),
            "dataset_to_index": pa.array([5, 8, 10], type=pa.int64()),
            "length": pa.array([5, 3, 2], type=pa.int64()),
            "stats/task_index/min": pa.array([[0], [0], [1]], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(episodes_table, path)


def write_metaworld_info(path: Path) -> None:
    """Create a tiny `meta/info.json` marker for local MT50 tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"robot_type": "metaworld", "splits": {"train": "0:3"}}')


@pytest.fixture()
def fake_dataset_root(tmp_path: Path) -> Path:
    """Create a tiny train and val dataset tree."""

    root = tmp_path / "data"
    write_episode(root / "single_grasp" / "train" / "episode_0.hdf5")
    write_episode(root / "single_grasp" / "val" / "episode_0.hdf5")
    return root


@pytest.fixture()
def fake_long_dataset_root(tmp_path: Path) -> Path:
    """Create a longer dataset tree for full-episode clip tests."""

    root = tmp_path / "long_data"
    write_episode(root / "single_grasp" / "train" / "episode_0.hdf5", frames=130)
    write_episode(root / "single_grasp" / "val" / "episode_0.hdf5", frames=130)
    return root


@pytest.fixture()
def fake_multi_episode_dataset_root(tmp_path: Path) -> Path:
    """Create a multi-episode dataset tree for all-episode tests."""

    root = tmp_path / "multi_episode_data"
    write_episode(root / "single_grasp" / "train" / "episode_0.hdf5", frames=130)
    write_episode(root / "single_grasp" / "train" / "episode_1.hdf5", frames=115)
    write_episode(root / "single_grasp" / "val" / "episode_0.hdf5", frames=130)
    return root


@pytest.fixture()
def fake_metaworld_dataset_root(tmp_path: Path) -> Path:
    """Create a tiny local MetaWorld-style dataset mirror."""

    root = tmp_path / "metaworld_mt50"
    write_metaworld_info(root / "meta" / "info.json")
    write_metaworld_metadata(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    write_metaworld_data_file(
        root / "data" / "chunk-000" / "file-000.parquet",
        [10, 20, 30, 40, 50, 60, 70, 80],
    )
    write_metaworld_data_file(
        root / "data" / "chunk-000" / "file-001.parquet",
        [90, 100],
    )
    return root


@pytest.fixture()
def saved_world_model_ae_checkpoint(fake_long_dataset_root: Path, tmp_path: Path) -> Path:
    """Create a deterministic Wan-AE checkpoint for loading tests."""

    checkpoint_path = tmp_path / "world_model_wan_ae.pt"
    model = WorldModel(ae_backend="wan")
    for parameter in model.encoder.parameters():
        torch.nn.init.constant_(parameter, 0.25)
    for parameter in model.decoder.parameters():
        torch.nn.init.constant_(parameter, 0.5)
    for parameter in model.dynamics.parameters():
        torch.nn.init.constant_(parameter, 0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = ExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="fixture_wan_ae",
        ae_backend="wan",
        device="cpu",
    )
    save_training_checkpoint(
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
