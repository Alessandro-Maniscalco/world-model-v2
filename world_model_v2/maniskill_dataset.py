"""ManiSkill replayed-demo dataset helpers for the root world-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout
from world_model_v2.metaworld_dataset import MetaWorldGroupedFrameSampler


MANISKILL_DEFAULT_TRAJ_H5 = "trajectory.rgb.pd_joint_pos.physx_cpu.h5"
MANISKILL_DEFAULT_TRAJ_JSON = "trajectory.rgb.pd_joint_pos.physx_cpu.json"
MANISKILL_DEFAULT_CAMERA = "base_camera"
MANISKILL_DEFAULT_ACTION_DIM = 8


def resolve_maniskill_split(split: str) -> str:
    """Resolve the requested split against the single-file ManiSkill replay layout."""

    if split not in {"train", "val"}:
        raise ValueError(
            "ManiSkill replay datasets only support train/val aliases, "
            f"but received split={split!r}."
        )
    return "train"


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


def rgb_array_to_tensor(rgb: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Resize one RGB array and return it as a normalized CHW tensor."""

    image = Image.fromarray(rgb)
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def read_nested_h5(group: h5py.Group, path: tuple[str, ...]) -> h5py.Dataset:
    """Read one nested HDF5 dataset by walking the requested key path."""

    node: h5py.Group | h5py.Dataset = group
    for key in path:
        if not isinstance(node, h5py.Group):
            raise KeyError(f"Encountered dataset before path ended at key={key!r}.")
        if key not in node:
            raise KeyError(f"Missing HDF5 key {key!r} while resolving {'/'.join(path)}.")
        node = node[key]
    if not isinstance(node, h5py.Dataset):
        raise KeyError(f"Expected dataset at {'/'.join(path)}, found group.")
    return node


@dataclass(frozen=True)
class ManiSkillEpisodeRecord:
    """Describe one replayed ManiSkill episode stored in one trajectory HDF5 file."""

    episode_index: int
    h5_group_name: str
    elapsed_steps: int
    frame_count: int
    action_count: int
    success: bool


@dataclass(frozen=True)
class ManiSkillFrameRecord:
    """Describe one frame within a specific replayed ManiSkill episode."""

    episode_index: int
    h5_group_name: str
    frame_index: int


@dataclass(frozen=True)
class ManiSkillTransitionRecord:
    """Describe one dynamics window inside a replayed ManiSkill episode."""

    episode: ManiSkillEpisodeRecord
    start_frame_index: int


class ManiSkillReplayRepository:
    """Resolve replayed ManiSkill metadata and lazily read RGB/action tensors."""

    def __init__(
        self,
        data_root: str | Path,
        traj_h5: str = MANISKILL_DEFAULT_TRAJ_H5,
        traj_json: str = MANISKILL_DEFAULT_TRAJ_JSON,
        camera: str = MANISKILL_DEFAULT_CAMERA,
    ) -> None:
        """Create a repository accessor rooted at one replayed ManiSkill directory."""

        self.data_root = Path(data_root)
        self.traj_h5_path = self._resolve_file(traj_h5)
        self.traj_json_path = self._resolve_file(traj_json)
        self.camera = camera
        self._handle: h5py.File | None = None
        self._metadata: dict[str, Any] | None = None

    def __del__(self) -> None:
        """Close the cached HDF5 handle when the repository is destroyed."""

        if self._handle is not None:
            self._handle.close()

    def _resolve_file(self, filename: str) -> Path:
        """Resolve one required dataset file inside the configured root."""

        candidate = self.data_root / filename
        if not candidate.exists():
            raise FileNotFoundError(f"Missing ManiSkill replay file: {candidate}")
        return candidate

    def handle(self) -> h5py.File:
        """Return the lazily opened HDF5 handle."""

        if self._handle is None:
            self._handle = h5py.File(self.traj_h5_path, "r")
        return self._handle

    def metadata(self) -> dict[str, Any]:
        """Load the replayed ManiSkill JSON metadata."""

        if self._metadata is None:
            with self.traj_json_path.open("r", encoding="utf-8") as handle:
                self._metadata = json.load(handle)
        return self._metadata

    def env_id(self) -> str:
        """Return the environment id recorded in the replay metadata."""

        metadata = self.metadata()
        env_info = metadata.get("env_info", {})
        return str(env_info.get("env_id", self.data_root.name))

    def action_dim(self) -> int:
        """Return the action dimension recorded in the first episode."""

        records = self.episode_records()
        if not records:
            raise ValueError("Replay dataset does not contain any episodes.")
        return int(self.handle()[records[0].h5_group_name]["actions"].shape[1])

    def _frame_dataset(self, group_name: str) -> h5py.Dataset:
        """Return the nested RGB dataset for one trajectory group."""

        return read_nested_h5(
            self.handle()[group_name],
            ("obs", "sensor_data", self.camera, "rgb"),
        )

    def _action_dataset(self, group_name: str) -> h5py.Dataset:
        """Return the action dataset for one trajectory group."""

        return read_nested_h5(self.handle()[group_name], ("actions",))

    def episode_records(self) -> list[ManiSkillEpisodeRecord]:
        """Return the replayed ManiSkill episode list from JSON + HDF5."""

        metadata = self.metadata()
        episodes = metadata.get("episodes", [])
        records: list[ManiSkillEpisodeRecord] = []
        for episode_position, episode in enumerate(episodes):
            h5_group_name = f"traj_{int(episode['episode_id'])}"
            frame_count = int(self._frame_dataset(h5_group_name).shape[0])
            action_count = int(self._action_dataset(h5_group_name).shape[0])
            records.append(
                ManiSkillEpisodeRecord(
                    episode_index=episode_position,
                    h5_group_name=h5_group_name,
                    elapsed_steps=int(episode.get("elapsed_steps", action_count)),
                    frame_count=frame_count,
                    action_count=action_count,
                    success=bool(episode.get("success", False)),
                )
            )
        return records

    def episode_record(self, episode: int) -> ManiSkillEpisodeRecord:
        """Return one episode record from the replayed trajectory file."""

        records = self.episode_records()
        if episode < 0 or episode >= len(records):
            raise IndexError(
                f"ManiSkill episode selection {episode} is out of range for "
                f"{len(records)} available episodes."
            )
        return records[episode]

    def load_frame_tensor(
        self,
        frame: ManiSkillFrameRecord,
        resolution: int,
        height: int | None,
        width: int | None,
    ) -> torch.Tensor:
        """Load and resize one RGB frame from the replayed HDF5 file."""

        resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
        rgb = np.asarray(self._frame_dataset(frame.h5_group_name)[frame.frame_index], dtype=np.uint8)
        return rgb_array_to_tensor(rgb, height=resolved_height, width=resolved_width)

    def load_action_tensor(self, group_name: str, action_index: int) -> torch.Tensor:
        """Load one action vector from the replayed HDF5 file."""

        action = torch.as_tensor(
            np.asarray(self._action_dataset(group_name)[action_index], dtype=np.float32),
            dtype=torch.float32,
        )
        if action.ndim != 1:
            raise ValueError(f"Expected 1D ManiSkill action, received {tuple(action.shape)}.")
        return action

    def load_clip(
        self,
        record: ManiSkillEpisodeRecord,
        frame_start: int | None,
        frame_end: int | None,
        resolution: int,
        height: int | None,
        width: int | None,
        load_actions: bool = False,
        clamp_frame_end: bool = False,
    ) -> dict[str, Any]:
        """Load one resized frame slice from a replayed ManiSkill episode."""

        resolved_frame_start = 0 if frame_start is None else frame_start
        resolved_frame_end = record.frame_count - 1 if frame_end is None else frame_end
        if resolved_frame_end < resolved_frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start.")
        if resolved_frame_start < 0:
            raise ValueError("frame_start must be greater than or equal to zero.")
        if resolved_frame_start >= record.frame_count:
            raise ValueError(
                f"Requested frame_start {resolved_frame_start} exceeds episode length "
                f"{record.frame_count}."
            )
        effective_frame_end = (
            min(resolved_frame_end, record.frame_count - 1) if clamp_frame_end else resolved_frame_end
        )
        if effective_frame_end >= record.frame_count:
            raise ValueError(
                f"Requested frames {resolved_frame_start}:{resolved_frame_end} exceed episode length "
                f"{record.frame_count}."
            )

        frames: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        for local_frame_index in range(resolved_frame_start, effective_frame_end + 1):
            frame_record = ManiSkillFrameRecord(
                episode_index=record.episode_index,
                h5_group_name=record.h5_group_name,
                frame_index=local_frame_index,
            )
            frames.append(
                self.load_frame_tensor(
                    frame_record,
                    resolution=resolution,
                    height=height,
                    width=width,
                )
            )
            if load_actions and local_frame_index < effective_frame_end:
                actions.append(self.load_action_tensor(record.h5_group_name, local_frame_index))
        action_dim = self.action_dim()
        clip = {
            "frames": torch.stack(frames, dim=0),
            "frame_idx": torch.arange(
                resolved_frame_start,
                effective_frame_end + 1,
                dtype=torch.long,
            ),
            "episode_idx": torch.tensor(record.episode_index, dtype=torch.long),
            "task_name": self.env_id(),
        }
        if load_actions:
            clip["actions"] = (
                torch.stack(actions, dim=0)
                if actions
                else torch.zeros((0, action_dim), dtype=torch.float32)
            )
        return clip


def load_maniskill_clip(
    data_root: str | Path,
    split: str = "train",
    episode: int = 0,
    resolution: int = 128,
    height: int | None = None,
    width: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    traj_h5: str = MANISKILL_DEFAULT_TRAJ_H5,
    traj_json: str = MANISKILL_DEFAULT_TRAJ_JSON,
    camera: str = MANISKILL_DEFAULT_CAMERA,
    load_actions: bool = False,
    clamp_frame_end: bool = False,
) -> dict[str, Any]:
    """Load one resized frame slice from a replayed ManiSkill trajectory file."""

    resolve_maniskill_split(split)
    repository = ManiSkillReplayRepository(
        data_root=data_root,
        traj_h5=traj_h5,
        traj_json=traj_json,
        camera=camera,
    )
    record = repository.episode_record(episode=episode)
    return repository.load_clip(
        record=record,
        frame_start=frame_start,
        frame_end=frame_end,
        resolution=resolution,
        height=height,
        width=width,
        load_actions=load_actions,
        clamp_frame_end=clamp_frame_end,
    )


class ManiSkillFrameDataset(Dataset[dict[str, Any]]):
    """Expose replayed ManiSkill frames as reconstruction samples for Wan-VAE training."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
        include_motion_neighbors: bool = False,
        traj_h5: str = MANISKILL_DEFAULT_TRAJ_H5,
        traj_json: str = MANISKILL_DEFAULT_TRAJ_JSON,
        camera: str = MANISKILL_DEFAULT_CAMERA,
    ) -> None:
        """Build either one cached clip or a lazy all-episode frame index."""

        resolve_maniskill_split(split)
        self.repository = ManiSkillReplayRepository(
            data_root=data_root,
            traj_h5=traj_h5,
            traj_json=traj_json,
            camera=camera,
        )
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.include_motion_neighbors = include_motion_neighbors
        self._episode_records_by_index: dict[int, ManiSkillEpisodeRecord] = {}
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.frames: list[ManiSkillFrameRecord] = []
            buckets: list[list[int]] = []
            for episode_position, episode_record in enumerate(self.repository.episode_records()):
                if episode_position in excluded_episodes:
                    continue
                self._episode_records_by_index[episode_record.episode_index] = episode_record
                if frame_start is not None and frame_start >= episode_record.frame_count:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = episode_record.frame_count - 1 if frame_end is None else min(
                    frame_end,
                    episode_record.frame_count - 1,
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                bucket: list[int] = []
                for local_frame_index in range(resolved_frame_start, resolved_frame_end + 1):
                    dataset_index = len(self.frames)
                    self.frames.append(
                        ManiSkillFrameRecord(
                            episode_index=episode_record.episode_index,
                            h5_group_name=episode_record.h5_group_name,
                            frame_index=local_frame_index,
                        )
                    )
                    bucket.append(dataset_index)
                buckets.append(bucket)
            if not self.frames:
                raise ValueError("No ManiSkill episodes contain any frames in the requested range.")
            self._sampler = MetaWorldGroupedFrameSampler(buckets)
        else:
            self.clip = load_maniskill_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                traj_h5=traj_h5,
                traj_json=traj_json,
                camera=camera,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def __len__(self) -> int:
        """Return the number of available reconstruction frames."""

        if self.all_episodes:
            return len(self.frames)
        return int(self.clip["frames"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one reconstruction frame sample."""

        if self.all_episodes:
            if index < 0 or index >= len(self.frames):
                raise IndexError("ManiSkillFrameDataset index out of range.")
            frame_record = self.frames[index]
            sample = {
                "frame": self.repository.load_frame_tensor(
                    frame_record,
                    resolution=self.resolution,
                    height=self.height,
                    width=self.width,
                ),
                "frame_idx": torch.tensor(frame_record.frame_index, dtype=torch.long),
                "episode_idx": torch.tensor(frame_record.episode_index, dtype=torch.long),
            }
            if self.include_motion_neighbors:
                episode_record = self._episode_records_by_index[frame_record.episode_index]
                prev_index = max(frame_record.frame_index - 1, 0)
                next_index = min(frame_record.frame_index + 1, episode_record.frame_count - 1)
                prev_record = ManiSkillFrameRecord(
                    episode_index=episode_record.episode_index,
                    h5_group_name=episode_record.h5_group_name,
                    frame_index=prev_index,
                )
                next_record = ManiSkillFrameRecord(
                    episode_index=episode_record.episode_index,
                    h5_group_name=episode_record.h5_group_name,
                    frame_index=next_index,
                )
                sample["prev_frame"] = self.repository.load_frame_tensor(
                    prev_record,
                    resolution=self.resolution,
                    height=self.height,
                    width=self.width,
                )
                sample["next_frame"] = self.repository.load_frame_tensor(
                    next_record,
                    resolution=self.resolution,
                    height=self.height,
                    width=self.width,
                )
            return sample
        sample = {
            "frame": self.clip["frames"][index],
            "frame_idx": self.clip["frame_idx"][index],
            "episode_idx": self.clip["episode_idx"],
        }
        if self.include_motion_neighbors:
            prev_index = max(index - 1, 0)
            next_index = min(index + 1, int(self.clip["frames"].shape[0]) - 1)
            sample["prev_frame"] = self.clip["frames"][prev_index]
            sample["next_frame"] = self.clip["frames"][next_index]
        return sample


class ManiSkillTransitionDataset(Dataset[dict[str, Any]]):
    """Expose replayed ManiSkill clips as sliding dynamics windows."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        traj_h5: str = MANISKILL_DEFAULT_TRAJ_H5,
        traj_json: str = MANISKILL_DEFAULT_TRAJ_JSON,
        camera: str = MANISKILL_DEFAULT_CAMERA,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
        rollout_context_frames: int | None = None,
        rollout_chunks: int = 0,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
    ) -> None:
        """Cache one ManiSkill clip or flatten all episodes for dynamics windows."""

        resolve_maniskill_split(split)
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
        self.all_episodes = all_episodes
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.repository = ManiSkillReplayRepository(
                data_root=data_root,
                traj_h5=traj_h5,
                traj_json=traj_json,
                camera=camera,
            )
            self.resolution = resolution
            self.height = height
            self.width = width
            self.window_records: list[ManiSkillTransitionRecord] = []
            buckets: list[list[int]] = []
            for episode_position, episode_record in enumerate(self.repository.episode_records()):
                if episode_position in excluded_episodes:
                    continue
                if frame_start is not None and frame_start >= episode_record.frame_count:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = (
                    episode_record.frame_count - 1
                    if frame_end is None
                    else min(frame_end, episode_record.frame_count - 1)
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                clip_length = resolved_frame_end - resolved_frame_start + 1
                available_windows = max(clip_length - self.required_frames + 1, 0)
                if available_windows < 1:
                    continue
                bucket: list[int] = []
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        ManiSkillTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
                buckets.append(bucket)
            if not self.window_records:
                raise ValueError(
                    "No ManiSkill episodes contain any valid dynamics windows in the requested range."
                )
            self._sampler = MetaWorldGroupedFrameSampler(buckets)
        else:
            self.clip = load_maniskill_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                traj_h5=traj_h5,
                traj_json=traj_json,
                camera=camera,
                load_actions=True,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one configured ManiSkill `(context, target)` training sample."""

        context_stop = index + self.frame_layout.context_frames
        target_stop = context_stop + self.frame_layout.target_frames
        future_target_stop = target_stop + self.rollout_chunks * self.rollout_target_frames
        action_stop = index + self.frame_layout.num_action_per_chunk
        future_action_stop = action_stop + self.rollout_chunks * self.rollout_target_frames
        return {
            "context_frames": clip["frames"][index:context_stop],
            "target_frames": clip["frames"][context_stop:target_stop],
            "future_target_frames": clip["frames"][target_stop:future_target_stop],
            "actions": clip["actions"][index:action_stop],
            "future_actions": clip["actions"][action_stop:future_action_stop],
            "context_frame_idx": clip["frame_idx"][index:context_stop],
            "target_frame_idx": clip["frame_idx"][context_stop:target_stop],
            "future_target_frame_idx": clip["frame_idx"][target_stop:future_target_stop],
            "episode_idx": clip["episode_idx"],
        }

    def _load_transition_clip(self, record: ManiSkillTransitionRecord) -> dict[str, Any]:
        """Load the minimal frame/action slice needed for one ManiSkill dynamics window."""

        stop_frame_index = record.start_frame_index + self.required_frames
        return self.repository.load_clip(
            record=record.episode,
            frame_start=record.start_frame_index,
            frame_end=stop_frame_index - 1,
            resolution=self.resolution,
            height=self.height,
            width=self.width,
            load_actions=True,
        )

    def __len__(self) -> int:
        """Return the number of available dynamics windows for the configured layout."""

        if self.all_episodes:
            return len(self.window_records)
        return max(int(self.clip["frames"].shape[0]) - self.required_frames + 1, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one configured ManiSkill `(context, target)` training sample."""

        if self.all_episodes:
            if index < 0 or index >= len(self.window_records):
                raise IndexError("ManiSkillTransitionDataset index out of range.")
            return self._sample_from_clip(self._load_transition_clip(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class ManiSkillValidationClipDataset(Dataset[dict[str, Any]]):
    """Expose one cached ManiSkill clip as a single validation example."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        traj_h5: str = MANISKILL_DEFAULT_TRAJ_H5,
        traj_json: str = MANISKILL_DEFAULT_TRAJ_JSON,
        camera: str = MANISKILL_DEFAULT_CAMERA,
    ) -> None:
        """Cache one ManiSkill clip for validation preview generation."""

        self.clip = load_maniskill_clip(
            data_root=data_root,
            split=split,
            episode=episode,
            resolution=resolution,
            height=height,
            width=width,
            frame_start=frame_start,
            frame_end=frame_end,
            traj_h5=traj_h5,
            traj_json=traj_json,
            camera=camera,
            load_actions=True,
        )

    def __len__(self) -> int:
        """Return the number of validation clips."""

        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the cached validation clip."""

        if index != 0:
            raise IndexError("ManiSkillValidationClipDataset only contains one clip.")
        return {
            "frames": self.clip["frames"],
            "actions": self.clip["actions"],
            "frame_idx": self.clip["frame_idx"],
            "episode_idx": self.clip["episode_idx"],
        }
