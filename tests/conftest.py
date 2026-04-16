"""Shared pytest fixtures for the root world-model test suite."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from world_model_v2.experiment import ExperimentConfig, save_training_checkpoint
from world_model_v2.model import WorldModel
from world_model_v2.wan_vae import (
    DEFAULT_WAN_DIM,
    DEFAULT_WAN_NUM_RES_BLOCKS,
    DEFAULT_WAN_Z_DIM,
    WanVAEConfig,
)


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


def write_aloha_video_file(
    path: Path,
    frame_values: list[int],
    fps: int = 10,
    height: int = 16,
    width: int = 16,
) -> None:
    """Create one small MP4 shard for the fake ALOHA fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for frame_value in frame_values:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = frame_value % 256
        image[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :]
        image[:, :, 2] = np.arange(height, dtype=np.uint8)[:, None]
        frames.append(image)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def write_aloha_data_file(path: Path, frame_values: list[int]) -> None:
    """Create one small ALOHA-style parquet shard with 14D actions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    action_array = pa.array(
        [
            [float(frame_value + offset) for offset in range(14)]
            for frame_value in frame_values
        ],
        type=pa.list_(pa.float32()),
    )
    state_array = pa.array(
        [
            [float(frame_value + offset) for offset in range(14)]
            for frame_value in frame_values
        ],
        type=pa.list_(pa.float32()),
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": state_array,
                "action": action_array,
                "episode_index": pa.array([0] * len(frame_values), type=pa.int64()),
                "frame_index": pa.array(list(range(len(frame_values))), type=pa.int64()),
                "timestamp": pa.array([float(index) / 10.0 for index in range(len(frame_values))]),
                "next.done": pa.array(
                    [False] * (len(frame_values) - 1) + [True],
                    type=pa.bool_(),
                ),
                "index": pa.array(list(range(len(frame_values))), type=pa.int64()),
                "task_index": pa.array([0] * len(frame_values), type=pa.int64()),
            }
        ),
        path,
    )


def write_aloha_metadata(path: Path) -> None:
    """Create the metadata tables required by the fake ALOHA dataset loader tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    meta_root = path.parents[2]
    tasks_table = pa.table(
        {
            "task_index": pa.array([0], type=pa.int64()),
            "__index_level_0__": pa.array(
                ["Pick up the cube with the right arm and transfer it to the left arm."],
                type=pa.string(),
            ),
        }
    )
    pq.write_table(tasks_table, meta_root / "tasks.parquet")
    episodes_table = pa.table(
        {
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "data/chunk_index": pa.array([0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0], type=pa.int64()),
            "dataset_from_index": pa.array([0, 5], type=pa.int64()),
            "dataset_to_index": pa.array([5, 8], type=pa.int64()),
            "videos/observation.images.top/chunk_index": pa.array([0, 0], type=pa.int64()),
            "videos/observation.images.top/file_index": pa.array([0, 0], type=pa.int64()),
            "videos/observation.images.top/from_timestamp": pa.array([0.0, 0.5], type=pa.float32()),
            "videos/observation.images.top/to_timestamp": pa.array([0.5, 0.8], type=pa.float32()),
            "length": pa.array([5, 3], type=pa.int64()),
            "stats/task_index/min": pa.array([[0], [0]], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(episodes_table, path)


def write_aloha_info(path: Path) -> None:
    """Create a tiny `meta/info.json` marker for local ALOHA tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
{
  "robot_type": "aloha",
  "fps": 10,
  "splits": {"train": "0:2"},
  "features": {
    "observation.images.top": {
      "dtype": "video",
      "video_info": {"video.fps": 10}
    },
    "observation.state": {"dtype": "float32", "shape": [14]},
    "action": {"dtype": "float32", "shape": [14]}
  }
}
""".strip()
    )


def write_maniskill_replay_h5(path: Path) -> None:
    """Create a tiny replayed ManiSkill HDF5 file with RGB frames and joint actions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for episode_index, frame_values in enumerate(([10, 20, 30, 40, 50], [60, 70, 80])):
            group = handle.create_group(f"traj_{episode_index}")
            obs = group.create_group("obs")
            sensor_data = obs.create_group("sensor_data")
            camera = sensor_data.create_group("base_camera")
            rgb = np.stack(
                [
                    np.stack(
                        [
                            np.full((16, 16), frame_value, dtype=np.uint8),
                            np.tile(np.arange(16, dtype=np.uint8)[None, :], (16, 1)),
                            np.tile(np.arange(16, dtype=np.uint8)[:, None], (1, 16)),
                        ],
                        axis=-1,
                    )
                    for frame_value in frame_values
                ],
                axis=0,
            )
            camera.create_dataset("rgb", data=rgb)
            actions = np.stack(
                [
                    np.array([float(frame_value + offset) for offset in range(8)], dtype=np.float32)
                    for frame_value in frame_values[:-1]
                ],
                axis=0,
            )
            group.create_dataset("actions", data=actions)
            group.create_dataset("terminated", data=np.zeros((len(actions),), dtype=bool))
            group.create_dataset("truncated", data=np.zeros((len(actions),), dtype=bool))
            group.create_dataset("success", data=np.ones((len(actions),), dtype=bool))
            group.create_dataset("env_states", data=np.zeros((len(frame_values), 1), dtype=np.float32))


def write_maniskill_replay_json(path: Path) -> None:
    """Create a tiny replayed ManiSkill JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "PickCube-v1",
                    "env_kwargs": {
                        "obs_mode": "rgb",
                        "control_mode": "pd_joint_pos",
                        "render_mode": "rgb_array",
                        "sim_backend": "physx_cpu",
                    },
                },
                "episodes": [
                    {
                        "episode_id": 0,
                        "elapsed_steps": 4,
                        "success": True,
                    },
                    {
                        "episode_id": 1,
                        "elapsed_steps": 2,
                        "success": True,
                    },
                ],
            }
        )
    )


def write_lerobot_video_episode(path: Path, frame_values: list[int], episode_index: int) -> None:
    """Create one small episode-sharded LeRobot parquet file with 6D actions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    action_array = pa.array(
        [
            [float(frame_value + offset) for offset in range(6)]
            for frame_value in frame_values
        ],
        type=pa.list_(pa.float32()),
    )
    state_array = pa.array(
        [
            [float(frame_value + offset) for offset in range(12)]
            for frame_value in frame_values
        ],
        type=pa.list_(pa.float32()),
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": state_array,
                "action": action_array,
                "episode_index": pa.array([episode_index] * len(frame_values), type=pa.int64()),
                "frame_index": pa.array(list(range(len(frame_values))), type=pa.int64()),
                "timestamp": pa.array([float(index) / 10.0 for index in range(len(frame_values))]),
                "index": pa.array(list(range(len(frame_values))), type=pa.int64()),
                "task_index": pa.array([0] * len(frame_values), type=pa.int64()),
            }
        ),
        path,
    )


def write_lerobot_video_metadata(root: Path, episode_lengths: list[int]) -> None:
    """Create the JSONL metadata required by the episode-sharded LeRobot loader."""

    meta_root = root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    tasks_lines = ['{"task_index": 0, "task": "pick and place"}\n']
    (meta_root / "tasks.jsonl").write_text("".join(tasks_lines), encoding="utf-8")
    episode_lines = [
        json.dumps(
            {
                "episode_index": episode_index,
                "tasks": [0],
                "length": length,
            }
        )
        + "\n"
        for episode_index, length in enumerate(episode_lengths)
    ]
    (meta_root / "episodes.jsonl").write_text("".join(episode_lines), encoding="utf-8")


def write_lerobot_video_info(path: Path, height: int = 16, width: int = 16) -> None:
    """Create a tiny `meta/info.json` marker for episode-sharded LeRobot tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "so101",
                "total_episodes": 2,
                "total_frames": 8,
                "total_tasks": 1,
                "total_videos": 2,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "splits": {"train": "0:2"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [12]},
                    "observation.images.front": {
                        "dtype": "video",
                        "shape": [height, width, 3],
                        "info": {"video.fps": 10},
                    },
                    "action": {"dtype": "float32", "shape": [6]},
                },
            }
        ),
        encoding="utf-8",
    )


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
def fake_aloha_dataset_root(tmp_path: Path) -> Path:
    """Create a tiny local ALOHA-style dataset mirror."""

    root = tmp_path / "aloha_sim_transfer_cube_scripted"
    write_aloha_info(root / "meta" / "info.json")
    write_aloha_metadata(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    write_aloha_data_file(
        root / "data" / "chunk-000" / "file-000.parquet",
        [10, 20, 30, 40, 50, 60, 70, 80],
    )
    write_aloha_video_file(
        root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4",
        [10, 20, 30, 40, 50, 60, 70, 80],
        fps=10,
    )
    return root


@pytest.fixture()
def fake_maniskill_replay_root(tmp_path: Path) -> Path:
    """Create a tiny replayed ManiSkill dataset directory."""

    root = tmp_path / "maniskill_replay"
    write_maniskill_replay_h5(root / "trajectory.rgb.pd_joint_pos.physx_cpu.h5")
    write_maniskill_replay_json(root / "trajectory.rgb.pd_joint_pos.physx_cpu.json")
    return root


@pytest.fixture()
def fake_lerobot_so101_base_sim_pickplace_root(tmp_path: Path) -> Path:
    """Create a tiny local episode-sharded SO-101 LeRobot dataset mirror."""

    root = tmp_path / "lerobot_so101_base_sim_pickplace"
    write_lerobot_video_info(root / "meta" / "info.json")
    write_lerobot_video_metadata(root, [5, 3])
    write_lerobot_video_episode(
        root / "data" / "chunk-000" / "episode_000000.parquet",
        [10, 20, 30, 40, 50],
        episode_index=0,
    )
    write_lerobot_video_episode(
        root / "data" / "chunk-000" / "episode_000001.parquet",
        [60, 70, 80],
        episode_index=1,
    )
    write_aloha_video_file(
        root / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4",
        [10, 20, 30, 40, 50],
        fps=10,
    )
    write_aloha_video_file(
        root / "videos" / "chunk-000" / "observation.images.front" / "episode_000001.mp4",
        [60, 70, 80],
        fps=10,
    )
    return root


@pytest.fixture()
def saved_world_model_ae_checkpoint(fake_long_dataset_root: Path, tmp_path: Path) -> Path:
    """Create a deterministic Wan-AE checkpoint for loading tests."""

    checkpoint_path = tmp_path / "world_model_wan_ae.pt"
    model = WorldModel(
        ae_backend="wan",
        latent_channels=DEFAULT_WAN_Z_DIM,
        wan_config=WanVAEConfig(
            dim=DEFAULT_WAN_DIM,
            z_dim=DEFAULT_WAN_Z_DIM,
            num_res_blocks=DEFAULT_WAN_NUM_RES_BLOCKS,
        ),
    )
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
