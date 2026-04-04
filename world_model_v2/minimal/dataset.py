"""Minimal dataset helpers for the single-clip world-model debug pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def resolve_episode_path(
    data_root: str | Path,
    task: str,
    split: str,
    episode: int,
) -> Path:
    """Return the on-disk HDF5 path for one episode."""

    return Path(data_root) / task / split / f"episode_{episode}.hdf5"


def resize_frame_to_tensor(frame: np.ndarray, resolution: int) -> torch.Tensor:
    """Resize one RGB frame and return a float tensor in `[0, 1]`."""

    image = Image.fromarray(frame)
    resized = image.resize((resolution, resolution), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))


def load_minimal_clip(
    data_root: str | Path,
    task: str = "single_grasp",
    split: str = "val",
    episode: int = 0,
    camera: str = "camera_1_color",
    frame_start: int = 111,
    frame_end: int = 116,
    resolution: int = 128,
) -> dict[str, Any]:
    """Load one resized frame slice from a single episode."""

    if frame_end < frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start.")

    episode_path = resolve_episode_path(data_root, task, split, episode)
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    with h5py.File(episode_path, "r") as handle:
        if "obs" not in handle or "images" not in handle["obs"]:
            raise KeyError(f"Episode is missing obs/images groups: {episode_path}")
        images_group = handle["obs"]["images"]
        if camera not in images_group:
            raise KeyError(f"Camera {camera} not found in {episode_path}")
        frames = np.asarray(images_group[camera])

    if frame_start < 0 or frame_end >= frames.shape[0]:
        raise ValueError(
            f"Requested frames {frame_start}:{frame_end} exceed episode length {frames.shape[0]}."
        )

    selected = frames[frame_start : frame_end + 1]
    clip_tensor = torch.stack(
        [resize_frame_to_tensor(frame, resolution) for frame in selected],
        dim=0,
    )
    frame_idx = torch.arange(frame_start, frame_end + 1, dtype=torch.long)
    return {
        "frames": clip_tensor,
        "frame_idx": frame_idx,
        "episode_idx": torch.tensor(episode, dtype=torch.long),
        "camera": camera,
        "episode_path": episode_path,
    }


class MinimalFrameDataset(Dataset[dict[str, Any]]):
    """Expose one clip as independent per-frame reconstruction samples."""

    def __init__(
        self,
        data_root: str | Path,
        task: str = "single_grasp",
        split: str = "val",
        episode: int = 0,
        camera: str = "camera_1_color",
        frame_start: int = 111,
        frame_end: int = 116,
        resolution: int = 128,
    ) -> None:
        """Cache the requested clip for reconstruction training."""

        self.clip = load_minimal_clip(
            data_root=data_root,
            task=task,
            split=split,
            episode=episode,
            camera=camera,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
        )

    def __len__(self) -> int:
        """Return the number of cached frames."""

        return int(self.clip["frames"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one reconstruction sample from the cached clip."""

        return {
            "frame": self.clip["frames"][index],
            "frame_idx": self.clip["frame_idx"][index],
            "episode_idx": self.clip["episode_idx"],
        }


class MinimalTransitionDataset(Dataset[dict[str, Any]]):
    """Expose one clip as consecutive frame transitions."""

    def __init__(
        self,
        data_root: str | Path,
        task: str = "single_grasp",
        split: str = "val",
        episode: int = 0,
        camera: str = "camera_1_color",
        frame_start: int = 111,
        frame_end: int = 116,
        resolution: int = 128,
    ) -> None:
        """Cache the requested clip for one-step transition training."""

        self.clip = load_minimal_clip(
            data_root=data_root,
            task=task,
            split=split,
            episode=episode,
            camera=camera,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
        )

    def __len__(self) -> int:
        """Return the number of consecutive frame pairs."""

        return max(int(self.clip["frames"].shape[0]) - 1, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one `(current, next)` transition sample."""

        return {
            "current_frame": self.clip["frames"][index],
            "next_frame": self.clip["frames"][index + 1],
            "current_frame_idx": self.clip["frame_idx"][index],
            "next_frame_idx": self.clip["frame_idx"][index + 1],
            "episode_idx": self.clip["episode_idx"],
        }


class MinimalValidationClipDataset(Dataset[dict[str, Any]]):
    """Expose the full cached clip as a single validation example."""

    def __init__(
        self,
        data_root: str | Path,
        task: str = "single_grasp",
        split: str = "val",
        episode: int = 0,
        camera: str = "camera_1_color",
        frame_start: int = 111,
        frame_end: int = 116,
        resolution: int = 128,
    ) -> None:
        """Cache one clip for validation preview generation."""

        self.clip = load_minimal_clip(
            data_root=data_root,
            task=task,
            split=split,
            episode=episode,
            camera=camera,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
        )

    def __len__(self) -> int:
        """Return the number of validation clips."""

        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the cached validation clip."""

        if index != 0:
            raise IndexError("MinimalValidationClipDataset only contains one clip.")
        return {
            "frames": self.clip["frames"],
            "frame_idx": self.clip["frame_idx"],
            "episode_idx": self.clip["episode_idx"],
        }
