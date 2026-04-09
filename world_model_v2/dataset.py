"""Dataset helpers for interactive-world-sim training and validation clips."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout


def resolve_episode_path(
    data_root: str | Path,
    task: str,
    split: str,
    episode: int,
) -> Path:
    """Return the on-disk HDF5 path for one episode."""

    return Path(data_root) / task / split / f"episode_{episode}.hdf5"


def list_episode_indices(
    data_root: str | Path,
    task: str,
    split: str,
) -> list[int]:
    """Return sorted episode indices available for one task split."""

    split_dir = Path(data_root) / task / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    pattern = re.compile(r"episode_(\d+)\.hdf5$")
    episode_indices: list[int] = []
    for path in sorted(split_dir.glob("episode_*.hdf5")):
        match = pattern.match(path.name)
        if match is not None:
            episode_indices.append(int(match.group(1)))
    if not episode_indices:
        raise FileNotFoundError(f"No episode files found under {split_dir}")
    return sorted(episode_indices)


def resize_frame_to_tensor(frame: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Resize one RGB frame and return a float tensor in `[0, 1]`."""

    image = Image.fromarray(frame)
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))


def resolve_resize_shape(
    resolution: int,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    """Resolve the requested resize shape from square or explicit dimensions."""

    resolved_height = resolution if height is None else height
    resolved_width = resolution if width is None else width
    if resolved_height <= 0 or resolved_width <= 0:
        raise ValueError("height and width must be positive integers.")
    return resolved_height, resolved_width


def load_clip(
    data_root: str | Path,
    task: str = "single_grasp",
    split: str = "val",
    episode: int = 0,
    camera: str = "camera_1_color",
    frame_start: int | None = None,
    frame_end: int | None = None,
    resolution: int = 128,
    height: int | None = None,
    width: int | None = None,
    load_actions: bool = False,
    clamp_frame_end: bool = False,
) -> dict[str, Any]:
    """Load one resized frame slice from a single episode."""

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
        action_values = None
        if load_actions:
            if "action" not in handle:
                raise KeyError(f"Episode is missing action dataset: {episode_path}")
            action_values = np.asarray(handle["action"], dtype=np.float32)

    resolved_frame_start = 0 if frame_start is None else frame_start
    resolved_frame_end = int(frames.shape[0]) - 1 if frame_end is None else frame_end

    if resolved_frame_end < resolved_frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start.")
    if resolved_frame_start < 0:
        raise ValueError("frame_start must be greater than or equal to zero.")
    if resolved_frame_start >= frames.shape[0]:
        raise ValueError(
            f"Requested frame_start {resolved_frame_start} exceeds episode length {frames.shape[0]}."
        )
    effective_frame_end = (
        min(resolved_frame_end, int(frames.shape[0]) - 1) if clamp_frame_end else resolved_frame_end
    )
    if effective_frame_end >= frames.shape[0]:
        raise ValueError(
            f"Requested frames {resolved_frame_start}:{resolved_frame_end} exceed episode length "
            f"{frames.shape[0]}."
        )

    selected = frames[resolved_frame_start : effective_frame_end + 1]
    resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
    clip_tensor = torch.stack(
        [resize_frame_to_tensor(frame, resolved_height, resolved_width) for frame in selected],
        dim=0,
    )
    frame_idx = torch.arange(resolved_frame_start, effective_frame_end + 1, dtype=torch.long)
    clip = {
        "frames": clip_tensor,
        "frame_idx": frame_idx,
        "episode_idx": torch.tensor(episode, dtype=torch.long),
        "camera": camera,
        "episode_path": episode_path,
    }
    if action_values is not None:
        required_action_stop = effective_frame_end
        if required_action_stop > int(action_values.shape[0]):
            raise ValueError(
                f"Requested frames {resolved_frame_start}:{resolved_frame_end} require action rows through "
                f"{required_action_stop - 1}, but episode only has {action_values.shape[0]} action rows."
            )
        clip["actions"] = torch.as_tensor(
            action_values[resolved_frame_start:required_action_stop],
            dtype=torch.float32,
        )
    return clip


class FrameDataset(Dataset[dict[str, Any]]):
    """Expose one clip as independent per-frame reconstruction samples."""

    def __init__(
        self,
        data_root: str | Path,
        task: str = "single_grasp",
        split: str = "val",
        episode: int = 0,
        camera: str = "camera_1_color",
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        all_episodes: bool = False,
    ) -> None:
        """Cache one clip or all split clips for reconstruction training."""

        self.all_episodes = all_episodes
        if all_episodes:
            self.clips = []
            self.cumulative_lengths: list[int] = []
            running_total = 0
            for episode_idx in list_episode_indices(data_root=data_root, task=task, split=split):
                try:
                    clip = load_clip(
                        data_root=data_root,
                        task=task,
                        split=split,
                        episode=episode_idx,
                        camera=camera,
                        frame_start=frame_start,
                        frame_end=frame_end,
                        resolution=resolution,
                        height=height,
                        width=width,
                        clamp_frame_end=True,
                    )
                except ValueError:
                    continue
                self.clips.append(clip)
                running_total += int(clip["frames"].shape[0])
                self.cumulative_lengths.append(running_total)
            if not self.clips:
                raise ValueError(
                    "No episodes in the requested split contain any frames in the requested range."
                )
        else:
            self.clip = load_clip(
                data_root=data_root,
                task=task,
                split=split,
                episode=episode,
                camera=camera,
                frame_start=frame_start,
                frame_end=frame_end,
                resolution=resolution,
                height=height,
                width=width,
            )

    def __len__(self) -> int:
        """Return the number of cached frames."""

        if self.all_episodes:
            return self.cumulative_lengths[-1]
        return int(self.clip["frames"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one reconstruction sample from the cached clip."""

        if self.all_episodes:
            if index < 0 or index >= len(self):
                raise IndexError("FrameDataset index out of range.")
            clip_index = bisect_right(self.cumulative_lengths, index)
            clip_start = 0 if clip_index == 0 else self.cumulative_lengths[clip_index - 1]
            frame_index = index - clip_start
            clip = self.clips[clip_index]
            return {
                "frame": clip["frames"][frame_index],
                "frame_idx": clip["frame_idx"][frame_index],
                "episode_idx": clip["episode_idx"],
            }
        return {
            "frame": self.clip["frames"][index],
            "frame_idx": self.clip["frame_idx"][index],
            "episode_idx": self.clip["episode_idx"],
        }


class TransitionDataset(Dataset[dict[str, Any]]):
    """Expose one clip as sliding dynamics training windows."""

    def __init__(
        self,
        data_root: str | Path,
        task: str = "single_grasp",
        split: str = "val",
        episode: int = 0,
        camera: str = "camera_1_color",
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
        rollout_context_frames: int | None = None,
        rollout_chunks: int = 0,
    ) -> None:
        """Cache the requested clip for the configured dynamics training windows."""

        self.frame_layout = frame_layout
        self.rollout_context_frames = (
            frame_layout.context_frames
            if rollout_context_frames is None
            else int(rollout_context_frames)
        )
        if self.rollout_context_frames < 1 or self.rollout_context_frames >= self.frame_layout.max_frames:
            raise ValueError("rollout_context_frames must stay within [1, max_frames - 1].")
        if rollout_chunks < 0:
            raise ValueError("rollout_chunks must be non-negative.")
        self.rollout_chunks = int(rollout_chunks)
        self.rollout_target_frames = self.frame_layout.max_frames - self.rollout_context_frames
        self.required_frames = self.frame_layout.max_frames + self.rollout_chunks * self.rollout_target_frames
        self.clip = load_clip(
            data_root=data_root,
            task=task,
            split=split,
            episode=episode,
            camera=camera,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
            height=height,
            width=width,
            load_actions=True,
        )

    def __len__(self) -> int:
        """Return the number of available dynamics windows for the configured layout."""

        return max(
            int(self.clip["frames"].shape[0]) - self.required_frames + 1,
            0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one configured `(context, target)` training sample."""

        context_stop = index + self.frame_layout.context_frames
        target_stop = context_stop + self.frame_layout.target_frames
        future_target_stop = target_stop + self.rollout_chunks * self.rollout_target_frames
        action_stop = index + self.frame_layout.num_action_per_chunk
        future_action_stop = action_stop + self.rollout_chunks * self.rollout_target_frames
        return {
            "context_frames": self.clip["frames"][index:context_stop],
            "target_frames": self.clip["frames"][context_stop:target_stop],
            "future_target_frames": self.clip["frames"][target_stop:future_target_stop],
            "actions": self.clip["actions"][index:action_stop],
            "future_actions": self.clip["actions"][action_stop:future_action_stop],
            "context_frame_idx": self.clip["frame_idx"][index:context_stop],
            "target_frame_idx": self.clip["frame_idx"][context_stop:target_stop],
            "future_target_frame_idx": self.clip["frame_idx"][target_stop:future_target_stop],
            "episode_idx": self.clip["episode_idx"],
        }


class ValidationClipDataset(Dataset[dict[str, Any]]):
    """Expose the full cached clip as a single validation example."""

    def __init__(
        self,
        data_root: str | Path,
        task: str = "single_grasp",
        split: str = "val",
        episode: int = 0,
        camera: str = "camera_1_color",
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        """Cache one clip for validation preview generation."""

        self.clip = load_clip(
            data_root=data_root,
            task=task,
            split=split,
            episode=episode,
            camera=camera,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
            height=height,
            width=width,
            load_actions=True,
        )

    def __len__(self) -> int:
        """Return the number of validation clips."""

        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the cached validation clip."""

        if index != 0:
            raise IndexError("ValidationClipDataset only contains one clip.")
        return {
            "frames": self.clip["frames"],
            "actions": self.clip["actions"],
            "frame_idx": self.clip["frame_idx"],
            "episode_idx": self.clip["episode_idx"],
        }
