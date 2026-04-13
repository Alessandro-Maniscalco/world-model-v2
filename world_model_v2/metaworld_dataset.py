"""LeRobot dataset helpers for the root world-model pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import random
import subprocess
from typing import Any

import cv2
from huggingface_hub import hf_hub_download
import imageio_ffmpeg
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout


METAWORLD_DATASET_ID = "lerobot/metaworld_mt50"
METAWORLD_TASKS_PATH = "meta/tasks.parquet"
METAWORLD_EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"
METAWORLD_IMAGE_COLUMN = "observation.image"
METAWORLD_ACTION_COLUMN_CANDIDATES = ("action", "actions")
METAWORLD_ACTION_DIM = 4

ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID = "lerobot/aloha_sim_transfer_cube_scripted"
ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN = "observation.images.top"
ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM = 14


@dataclass(frozen=True)
class MetaWorldEpisodeRecord:
    """Describe one MT50 episode and the parquet shard rows that store it."""

    episode_index: int
    task_index: int
    task_name: str
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    length: int


@dataclass(frozen=True)
class MetaWorldFrameRecord:
    """Describe one frame inside a specific MT50 parquet shard."""

    episode_index: int
    task_index: int
    data_chunk_index: int
    data_file_index: int
    row_index_in_file: int
    frame_index: int


@dataclass(frozen=True)
class MetaWorldTransitionRecord:
    """Describe one dynamics window inside a specific MT50 episode."""

    episode: MetaWorldEpisodeRecord
    start_frame_index: int


def resolve_metaworld_split(split: str) -> str:
    """Resolve the requested split against MT50's train-only layout."""

    if split not in {"train", "val"}:
        raise ValueError(
            f"MetaWorld MT50 only exposes a train split, but received split={split!r}."
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


def image_bytes_to_tensor(image_bytes: bytes, height: int, width: int) -> torch.Tensor:
    """Decode one encoded RGB image and resize it to a float tensor."""

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def bgr_frame_to_tensor(frame_bgr: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Convert one OpenCV BGR frame into a resized RGB float tensor."""

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).contiguous()


class MetaWorldRepository:
    """Resolve MT50 metadata and lazily read individual parquet shards."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        repo_id: str = METAWORLD_DATASET_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Create a dataset accessor from a local mirror or the Hugging Face Hub."""

        self.repo_id = repo_id
        self.data_root = Path(data_root) if data_root not in {None, ""} else None
        self.local_root = self._resolve_local_root(self.data_root)
        if cache_dir not in {None, ""}:
            self.cache_dir = str(cache_dir)
        elif self.local_root is None and self.data_root is not None:
            self.cache_dir = str(self.data_root)
        else:
            self.cache_dir = None

    def _resolve_local_root(self, candidate: Path | None) -> Path | None:
        """Return the local dataset root when the expected metadata tree exists."""

        if candidate is None:
            return None
        if (candidate / "meta" / "info.json").exists():
            return candidate
        return None

    def resolve_file(self, relative_path: str) -> Path:
        """Return the local path for one metadata or shard file."""

        if self.local_root is not None:
            resolved = self.local_root / relative_path
            if not resolved.exists():
                raise FileNotFoundError(f"Missing MetaWorld file: {resolved}")
            return resolved
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=relative_path,
                cache_dir=self.cache_dir,
            )
        )

    @lru_cache(maxsize=1)
    def task_names(self) -> dict[int, str]:
        """Load the task-index to task-name mapping."""

        table = pq.read_table(self.resolve_file(METAWORLD_TASKS_PATH))
        task_index_column = table.column("task_index")
        task_name_column = table.column("__index_level_0__")
        return {
            int(task_index_column[index].as_py()): str(task_name_column[index].as_py())
            for index in range(table.num_rows)
        }

    @lru_cache(maxsize=1)
    def all_episode_records(self) -> tuple[MetaWorldEpisodeRecord, ...]:
        """Load all episode metadata records with only the columns we need."""

        columns = [
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "length",
            "stats/task_index/min",
        ]
        table = pq.read_table(self.resolve_file(METAWORLD_EPISODES_PATH), columns=columns)
        task_names = self.task_names()
        records: list[MetaWorldEpisodeRecord] = []
        for row_index in range(table.num_rows):
            task_index = int(table.column("stats/task_index/min")[row_index].as_py()[0])
            records.append(
                MetaWorldEpisodeRecord(
                    episode_index=int(table.column("episode_index")[row_index].as_py()),
                    task_index=task_index,
                    task_name=task_names[task_index],
                    data_chunk_index=int(table.column("data/chunk_index")[row_index].as_py()),
                    data_file_index=int(table.column("data/file_index")[row_index].as_py()),
                    dataset_from_index=int(table.column("dataset_from_index")[row_index].as_py()),
                    dataset_to_index=int(table.column("dataset_to_index")[row_index].as_py()),
                    length=int(table.column("length")[row_index].as_py()),
                )
            )
        return tuple(records)

    @lru_cache(maxsize=1)
    def file_start_indices(self) -> dict[tuple[int, int], int]:
        """Return the global dataset row offset where each parquet shard starts."""

        starts: dict[tuple[int, int], int] = {}
        for record in self.all_episode_records():
            key = (record.data_chunk_index, record.data_file_index)
            current = starts.get(key)
            if current is None or record.dataset_from_index < current:
                starts[key] = record.dataset_from_index
        return starts

    def episode_records(self, task_index: int | None = None) -> list[MetaWorldEpisodeRecord]:
        """Return the episode list, optionally restricted to one task index."""

        records = list(self.all_episode_records())
        if task_index is None:
            return records
        filtered = [record for record in records if record.task_index == task_index]
        if not filtered:
            raise ValueError(f"No MetaWorld episodes found for task_index={task_index}.")
        return filtered

    def episode_record(
        self,
        episode: int,
        task_index: int | None = None,
    ) -> MetaWorldEpisodeRecord:
        """Return one episode record from the optionally filtered episode list."""

        records = self.episode_records(task_index=task_index)
        if episode < 0 or episode >= len(records):
            raise IndexError(
                f"MetaWorld episode selection {episode} is out of range for "
                f"{len(records)} available episodes."
            )
        return records[episode]

    def _data_file_relative_path(self, chunk_index: int, file_index: int) -> str:
        """Return the dataset-relative path for one MT50 data parquet shard."""

        return f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    @lru_cache(maxsize=2)
    def image_table(self, chunk_index: int, file_index: int) -> Any:
        """Load and cache one parquet shard's image column."""

        return pq.read_table(
            self.resolve_file(self._data_file_relative_path(chunk_index, file_index)),
            columns=[METAWORLD_IMAGE_COLUMN],
        )

    @lru_cache(maxsize=8)
    def action_column_name(self, chunk_index: int, file_index: int) -> str:
        """Return the parquet action column name for one shard."""

        schema_names = pq.read_schema(
            self.resolve_file(self._data_file_relative_path(chunk_index, file_index))
        ).names
        for column_name in METAWORLD_ACTION_COLUMN_CANDIDATES:
            if column_name in schema_names:
                return column_name
        raise KeyError(
            "MetaWorld parquet shard is missing an action column. "
            f"Checked {METAWORLD_ACTION_COLUMN_CANDIDATES}."
        )

    @lru_cache(maxsize=2)
    def action_table(self, chunk_index: int, file_index: int) -> Any:
        """Load and cache one parquet shard's action column."""

        column_name = self.action_column_name(chunk_index, file_index)
        return pq.read_table(
            self.resolve_file(self._data_file_relative_path(chunk_index, file_index)),
            columns=[column_name],
        )

    def frame_row_index_in_file(
        self,
        record: MetaWorldEpisodeRecord,
        frame_index: int,
    ) -> int:
        """Translate an episode-local frame index into a shard-local row index."""

        if frame_index < 0 or frame_index >= record.length:
            raise IndexError(
                f"Requested frame index {frame_index} is out of bounds for episode "
                f"{record.episode_index} with length {record.length}."
            )
        file_start = self.file_start_indices()[(record.data_chunk_index, record.data_file_index)]
        return record.dataset_from_index - file_start + frame_index

    def load_frame_bytes(self, frame: MetaWorldFrameRecord) -> bytes:
        """Load the encoded image bytes for one frame record."""

        cell = self.image_table(frame.data_chunk_index, frame.data_file_index)[METAWORLD_IMAGE_COLUMN][
            frame.row_index_in_file
        ].as_py()
        image_bytes = cell.get("bytes")
        if image_bytes is None:
            raise ValueError("MetaWorld parquet row is missing embedded image bytes.")
        return bytes(image_bytes)

    def load_frame_tensor(
        self,
        frame: MetaWorldFrameRecord,
        resolution: int,
        height: int | None,
        width: int | None,
    ) -> torch.Tensor:
        """Load and resize one frame tensor from the backing parquet shard."""

        resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
        return image_bytes_to_tensor(
            self.load_frame_bytes(frame),
            height=resolved_height,
            width=resolved_width,
        )

    def load_action_tensor(self, frame: MetaWorldFrameRecord) -> torch.Tensor:
        """Load one action vector from the backing parquet shard."""

        column_name = self.action_column_name(frame.data_chunk_index, frame.data_file_index)
        value = self.action_table(frame.data_chunk_index, frame.data_file_index)[column_name][
            frame.row_index_in_file
        ].as_py()
        action = torch.as_tensor(value, dtype=torch.float32)
        if action.ndim != 1 or action.shape[0] != METAWORLD_ACTION_DIM:
            raise ValueError(
                f"Expected MetaWorld action shape ({METAWORLD_ACTION_DIM},), "
                f"received {tuple(action.shape)}."
            )
        return action

    def load_clip(
        self,
        record: MetaWorldEpisodeRecord,
        frame_start: int | None,
        frame_end: int | None,
        resolution: int,
        height: int | None,
        width: int | None,
        load_actions: bool = False,
        clamp_frame_end: bool = False,
    ) -> dict[str, Any]:
        """Load one resized frame slice from a specific MT50 episode."""

        resolved_frame_start = 0 if frame_start is None else frame_start
        resolved_frame_end = record.length - 1 if frame_end is None else frame_end
        if resolved_frame_end < resolved_frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start.")
        if resolved_frame_start < 0:
            raise ValueError("frame_start must be greater than or equal to zero.")
        if resolved_frame_start >= record.length:
            raise ValueError(
                f"Requested frame_start {resolved_frame_start} exceeds episode length {record.length}."
            )
        effective_frame_end = (
            min(resolved_frame_end, record.length - 1) if clamp_frame_end else resolved_frame_end
        )
        if effective_frame_end >= record.length:
            raise ValueError(
                f"Requested frames {resolved_frame_start}:{resolved_frame_end} exceed episode length "
                f"{record.length}."
            )

        resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
        frames: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        for local_frame_index in range(resolved_frame_start, effective_frame_end + 1):
            frame_record = MetaWorldFrameRecord(
                episode_index=record.episode_index,
                task_index=record.task_index,
                data_chunk_index=record.data_chunk_index,
                data_file_index=record.data_file_index,
                row_index_in_file=self.frame_row_index_in_file(record, local_frame_index),
                frame_index=local_frame_index,
            )
            frames.append(
                image_bytes_to_tensor(
                    self.load_frame_bytes(frame_record),
                    height=resolved_height,
                    width=resolved_width,
                )
            )
            if load_actions and local_frame_index < effective_frame_end:
                actions.append(self.load_action_tensor(frame_record))
        clip = {
            "frames": torch.stack(frames, dim=0),
            "frame_idx": torch.arange(
                resolved_frame_start,
                effective_frame_end + 1,
                dtype=torch.long,
            ),
            "episode_idx": torch.tensor(record.episode_index, dtype=torch.long),
            "task_idx": torch.tensor(record.task_index, dtype=torch.long),
            "task_name": record.task_name,
        }
        if load_actions:
            clip["actions"] = (
                torch.stack(actions, dim=0)
                if actions
                else torch.zeros((0, METAWORLD_ACTION_DIM), dtype=torch.float32)
            )
        return clip


def load_metaworld_clip(
    data_root: str | Path | None = None,
    split: str = "train",
    episode: int = 0,
    task_index: int | None = None,
    resolution: int = 128,
    height: int | None = None,
    width: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    repo_id: str = METAWORLD_DATASET_ID,
    cache_dir: str | Path | None = None,
    load_actions: bool = False,
    clamp_frame_end: bool = False,
) -> dict[str, Any]:
    """Load one resized frame slice from the MT50 dataset."""

    resolve_metaworld_split(split)
    repository = MetaWorldRepository(data_root=data_root, repo_id=repo_id, cache_dir=cache_dir)
    record = repository.episode_record(episode=episode, task_index=task_index)
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


class MetaWorldGroupedFrameSampler(Sampler[int]):
    """Shuffle cached MT50 dataset indices while keeping batches shard-local when possible."""

    def __init__(self, indices_by_shard: list[list[int]], seed: int = 7) -> None:
        """Store per-shard index buckets for randomized iteration."""

        self.indices_by_shard = [list(bucket) for bucket in indices_by_shard]
        self.seed = seed
        self.iteration = 0

    def __iter__(self) -> Any:
        """Yield frame indices grouped by shard with per-epoch randomization."""

        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        bucket_order = list(range(len(self.indices_by_shard)))
        rng.shuffle(bucket_order)
        for bucket_index in bucket_order:
            bucket = list(self.indices_by_shard[bucket_index])
            rng.shuffle(bucket)
            yield from bucket

    def __len__(self) -> int:
        """Return the number of frame indices available to iterate."""

        return sum(len(bucket) for bucket in self.indices_by_shard)


class MetaWorldFrameDataset(Dataset[dict[str, Any]]):
    """Expose MT50 frames as reconstruction samples for Wan-VAE training."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        task_index: int | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
        include_motion_neighbors: bool = False,
        repo_id: str = METAWORLD_DATASET_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Build either one cached clip or a lazy all-episode frame index."""

        resolve_metaworld_split(split)
        self.repository = MetaWorldRepository(data_root=data_root, repo_id=repo_id, cache_dir=cache_dir)
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.include_motion_neighbors = include_motion_neighbors
        self.task_index = task_index
        self._episode_records_by_index: dict[int, MetaWorldEpisodeRecord] = {}
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.frames: list[MetaWorldFrameRecord] = []
            shard_buckets: dict[tuple[int, int], list[int]] = {}
            for episode_position, episode_record in enumerate(
                self.repository.episode_records(task_index=task_index)
            ):
                if episode_position in excluded_episodes:
                    continue
                self._episode_records_by_index[episode_record.episode_index] = episode_record
                if frame_start is not None and frame_start >= episode_record.length:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = episode_record.length - 1 if frame_end is None else min(
                    frame_end,
                    episode_record.length - 1,
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                shard_key = (episode_record.data_chunk_index, episode_record.data_file_index)
                bucket = shard_buckets.setdefault(shard_key, [])
                for local_frame_index in range(resolved_frame_start, resolved_frame_end + 1):
                    dataset_index = len(self.frames)
                    self.frames.append(
                        MetaWorldFrameRecord(
                            episode_index=episode_record.episode_index,
                            task_index=episode_record.task_index,
                            data_chunk_index=episode_record.data_chunk_index,
                            data_file_index=episode_record.data_file_index,
                            row_index_in_file=self.repository.frame_row_index_in_file(
                                episode_record,
                                local_frame_index,
                            ),
                            frame_index=local_frame_index,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.frames:
                raise ValueError(
                    "No MetaWorld episodes contain any frames in the requested range."
                )
            self._sampler = MetaWorldGroupedFrameSampler(list(shard_buckets.values()))
        else:
            self.clip = load_metaworld_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                task_index=task_index,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                cache_dir=cache_dir,
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
                raise IndexError("MetaWorldFrameDataset index out of range.")
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
                next_index = min(frame_record.frame_index + 1, episode_record.length - 1)
                prev_record = MetaWorldFrameRecord(
                    episode_index=episode_record.episode_index,
                    task_index=episode_record.task_index,
                    data_chunk_index=episode_record.data_chunk_index,
                    data_file_index=episode_record.data_file_index,
                    row_index_in_file=self.repository.frame_row_index_in_file(
                        episode_record,
                        prev_index,
                    ),
                    frame_index=prev_index,
                )
                next_record = MetaWorldFrameRecord(
                    episode_index=episode_record.episode_index,
                    task_index=episode_record.task_index,
                    data_chunk_index=episode_record.data_chunk_index,
                    data_file_index=episode_record.data_file_index,
                    row_index_in_file=self.repository.frame_row_index_in_file(
                        episode_record,
                        next_index,
                    ),
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


class MetaWorldAutoencoderClipDataset(Dataset[dict[str, Any]]):
    """Expose MT50 episodes as sliding AE clip windows."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        task_index: int | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
        repo_id: str = METAWORLD_DATASET_ID,
        cache_dir: str | Path | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
    ) -> None:
        """Build either one cached clip or a lazy all-episode AE clip index."""

        resolve_metaworld_split(split)
        self.repository = MetaWorldRepository(data_root=data_root, repo_id=repo_id, cache_dir=cache_dir)
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.required_frames = frame_layout.max_pixel_frames
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.window_records: list[MetaWorldTransitionRecord] = []
            shard_buckets: dict[tuple[int, int], list[int]] = {}
            for episode_position, episode_record in enumerate(
                self.repository.episode_records(task_index=task_index)
            ):
                if episode_position in excluded_episodes:
                    continue
                if frame_start is not None and frame_start >= episode_record.length:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = (
                    episode_record.length - 1
                    if frame_end is None
                    else min(frame_end, episode_record.length - 1)
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                clip_length = resolved_frame_end - resolved_frame_start + 1
                available_windows = max(clip_length - self.required_frames + 1, 0)
                if available_windows < 1:
                    continue
                shard_key = (episode_record.data_chunk_index, episode_record.data_file_index)
                bucket = shard_buckets.setdefault(shard_key, [])
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        MetaWorldTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.window_records:
                raise ValueError("No MetaWorld episodes contain valid AE clip windows in the requested range.")
            self._sampler = MetaWorldGroupedFrameSampler(list(shard_buckets.values()))
        else:
            self.clip = load_metaworld_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                task_index=task_index,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                cache_dir=cache_dir,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one AE clip window from one cached MT50 clip."""

        stop = index + self.required_frames
        return {
            "frames": clip["frames"][index:stop],
            "frame_idx": clip["frame_idx"][index:stop],
            "episode_idx": clip["episode_idx"],
        }

    def _load_clip_window(self, record: MetaWorldTransitionRecord) -> dict[str, Any]:
        """Load the minimal frame slice needed for one lazy AE clip window."""

        stop_frame_index = record.start_frame_index + self.required_frames
        return self.repository.load_clip(
            record=record.episode,
            frame_start=record.start_frame_index,
            frame_end=stop_frame_index - 1,
            resolution=self.resolution,
            height=self.height,
            width=self.width,
        )

    def __len__(self) -> int:
        """Return the number of available AE clip windows."""

        if self.all_episodes:
            return len(self.window_records)
        return max(int(self.clip["frames"].shape[0]) - self.required_frames + 1, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one cached AE clip window."""

        if self.all_episodes:
            if index < 0 or index >= len(self.window_records):
                raise IndexError("MetaWorldAutoencoderClipDataset index out of range.")
            return self._sample_from_clip(self._load_clip_window(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class MetaWorldTransitionDataset(Dataset[dict[str, Any]]):
    """Expose one MT50 episode as sliding dynamics windows."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        task_index: int | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        repo_id: str = METAWORLD_DATASET_ID,
        cache_dir: str | Path | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
        rollout_context_frames: int | None = None,
        rollout_chunks: int = 0,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
    ) -> None:
        """Cache one MT50 clip or flatten all task episodes for dynamics windows."""

        resolve_metaworld_split(split)
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
        self.context_pixel_frames = self.frame_layout.context_pixel_frames
        self.target_pixel_frames = self.frame_layout.target_pixel_frames
        self.rollout_target_pixel_frames = (
            self.frame_layout.temporal_compression_ratio * self.rollout_target_frames
        )
        total_latent_frames = self.frame_layout.max_frames + self.rollout_chunks * self.rollout_target_frames
        self.required_frames = self.frame_layout.pixel_frames_for_latent_frames(total_latent_frames)
        self.all_episodes = all_episodes
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.repository = MetaWorldRepository(
                data_root=data_root,
                repo_id=repo_id,
                cache_dir=cache_dir,
            )
            self.resolution = resolution
            self.height = height
            self.width = width
            self.window_records: list[MetaWorldTransitionRecord] = []
            shard_buckets: dict[tuple[int, int], list[int]] = {}
            for episode_position, episode_record in enumerate(
                self.repository.episode_records(task_index=task_index)
            ):
                if episode_position in excluded_episodes:
                    continue
                if frame_start is not None and frame_start >= episode_record.length:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = (
                    episode_record.length - 1
                    if frame_end is None
                    else min(frame_end, episode_record.length - 1)
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                clip_length = resolved_frame_end - resolved_frame_start + 1
                available_windows = max(clip_length - self.required_frames + 1, 0)
                if available_windows < 1:
                    continue
                shard_key = (episode_record.data_chunk_index, episode_record.data_file_index)
                bucket = shard_buckets.setdefault(shard_key, [])
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        MetaWorldTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.window_records:
                raise ValueError(
                    "No MetaWorld episodes contain any valid dynamics windows in the requested range."
                )
            self._sampler = MetaWorldGroupedFrameSampler(list(shard_buckets.values()))
        else:
            self.clip = load_metaworld_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                task_index=task_index,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                cache_dir=cache_dir,
                load_actions=True,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one configured MT50 `(context, target)` training sample from one clip."""

        context_stop = index + self.context_pixel_frames
        target_stop = context_stop + self.target_pixel_frames
        future_target_stop = target_stop + self.rollout_chunks * self.rollout_target_pixel_frames
        action_stop = index + self.frame_layout.num_action_per_chunk
        future_action_stop = action_stop + self.rollout_chunks * self.rollout_target_pixel_frames
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

    def _load_transition_clip(self, record: MetaWorldTransitionRecord) -> dict[str, Any]:
        """Load the minimal frame/action slice needed for one lazy MT50 dynamics window."""

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
        return max(
            int(self.clip["frames"].shape[0]) - self.required_frames + 1,
            0,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one configured MT50 `(context, target)` training sample."""

        if self.all_episodes:
            if index < 0 or index >= len(self.window_records):
                raise IndexError("MetaWorldTransitionDataset index out of range.")
            return self._sample_from_clip(self._load_transition_clip(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class MetaWorldValidationClipDataset(Dataset[dict[str, Any]]):
    """Expose one cached MT50 clip as a single validation example."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        task_index: int | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        repo_id: str = METAWORLD_DATASET_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Cache one MT50 clip for validation preview generation."""

        self.clip = load_metaworld_clip(
            data_root=data_root,
            split=split,
            episode=episode,
            task_index=task_index,
            resolution=resolution,
            height=height,
            width=width,
            frame_start=frame_start,
            frame_end=frame_end,
            repo_id=repo_id,
            cache_dir=cache_dir,
            load_actions=True,
        )

    def __len__(self) -> int:
        """Return the number of validation clips."""

        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the cached validation clip."""

        if index != 0:
            raise IndexError("MetaWorldValidationClipDataset only contains one clip.")
        return {
            "frames": self.clip["frames"],
            "actions": self.clip["actions"],
            "frame_idx": self.clip["frame_idx"],
            "episode_idx": self.clip["episode_idx"],
        }


def resolve_aloha_split(split: str) -> str:
    """Resolve the requested split against the train-only ALOHA sim layout."""

    if split not in {"train", "val"}:
        raise ValueError(
            "ALOHA sim transfer cube scripted only exposes a train split, "
            f"but received split={split!r}."
        )
    return "train"


def _first_scalar(value: Any) -> int:
    """Return the first integer from a scalar-or-sequence parquet value."""

    if isinstance(value, list):
        if not value:
            raise ValueError("Expected a non-empty sequence in parquet metadata.")
        return int(value[0])
    return int(value)


@dataclass(frozen=True)
class AlohaEpisodeRecord:
    """Describe one ALOHA sim episode and the shard rows that store it."""

    episode_index: int
    task_index: int
    task_name: str
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    video_chunk_index: int
    video_file_index: int
    video_from_timestamp: float
    video_to_timestamp: float
    length: int


@dataclass(frozen=True)
class AlohaFrameRecord:
    """Describe one ALOHA frame within its parquet row and MP4 shard."""

    episode_index: int
    task_index: int
    data_chunk_index: int
    data_file_index: int
    row_index_in_file: int
    frame_index: int
    video_chunk_index: int
    video_file_index: int
    video_frame_index_in_file: int


@dataclass(frozen=True)
class AlohaTransitionRecord:
    """Describe one dynamics window inside a specific ALOHA episode."""

    episode: AlohaEpisodeRecord
    start_frame_index: int


class AlohaSimRepository:
    """Resolve ALOHA sim metadata and lazily read MP4-backed frame shards."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
        cache_dir: str | Path | None = None,
        image_column: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN,
    ) -> None:
        """Create a dataset accessor from a local mirror or the Hugging Face Hub."""

        self.repo_id = repo_id
        self.image_column = image_column
        self.data_root = Path(data_root) if data_root not in {None, ""} else None
        self.local_root = self._resolve_local_root(self.data_root)
        if cache_dir not in {None, ""}:
            self.cache_dir = str(cache_dir)
        elif self.local_root is None and self.data_root is not None:
            self.cache_dir = str(self.data_root)
        else:
            self.cache_dir = None
        self._video_captures: dict[tuple[int, int], cv2.VideoCapture] = {}

    def __del__(self) -> None:
        """Release cached OpenCV video handles when the repository is destroyed."""

        for capture in self._video_captures.values():
            capture.release()

    def _resolve_local_root(self, candidate: Path | None) -> Path | None:
        """Return the local dataset root when the expected metadata tree exists."""

        if candidate is None:
            return None
        if (candidate / "meta" / "info.json").exists():
            return candidate
        return None

    def resolve_file(self, relative_path: str) -> Path:
        """Return the local path for one metadata, parquet, or video shard."""

        if self.local_root is not None:
            resolved = self.local_root / relative_path
            if not resolved.exists():
                raise FileNotFoundError(f"Missing ALOHA dataset file: {resolved}")
            return resolved
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=relative_path,
                cache_dir=self.cache_dir,
            )
        )

    @lru_cache(maxsize=1)
    def info(self) -> dict[str, Any]:
        """Load and cache the LeRobot `meta/info.json` payload."""

        with self.resolve_file("meta/info.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @lru_cache(maxsize=1)
    def video_fps(self) -> float:
        """Return the configured FPS for the selected video stream."""

        features = self.info().get("features", {})
        image_feature = features.get(self.image_column, {})
        video_info = image_feature.get("video_info", {})
        raw_fps = video_info.get("video.fps", self.info().get("fps", 0.0))
        return float(raw_fps)

    @lru_cache(maxsize=1)
    def task_names(self) -> dict[int, str]:
        """Load the task-index to task-name mapping."""

        table = pq.read_table(self.resolve_file(METAWORLD_TASKS_PATH))
        task_index_column = table.column("task_index")
        task_name_column = table.column("__index_level_0__")
        return {
            int(task_index_column[index].as_py()): str(task_name_column[index].as_py())
            for index in range(table.num_rows)
        }

    @lru_cache(maxsize=1)
    def all_episode_records(self) -> tuple[AlohaEpisodeRecord, ...]:
        """Load all ALOHA episode metadata records with only the columns we need."""

        video_prefix = f"videos/{self.image_column}"
        columns = [
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            f"{video_prefix}/chunk_index",
            f"{video_prefix}/file_index",
            f"{video_prefix}/from_timestamp",
            f"{video_prefix}/to_timestamp",
            "length",
            "stats/task_index/min",
        ]
        table = pq.read_table(self.resolve_file(METAWORLD_EPISODES_PATH), columns=columns)
        task_names = self.task_names()
        records: list[AlohaEpisodeRecord] = []
        for row_index in range(table.num_rows):
            raw_task_index = table.column("stats/task_index/min")[row_index].as_py()
            task_index = _first_scalar(raw_task_index)
            records.append(
                AlohaEpisodeRecord(
                    episode_index=int(table.column("episode_index")[row_index].as_py()),
                    task_index=task_index,
                    task_name=task_names[task_index],
                    data_chunk_index=int(table.column("data/chunk_index")[row_index].as_py()),
                    data_file_index=int(table.column("data/file_index")[row_index].as_py()),
                    dataset_from_index=int(table.column("dataset_from_index")[row_index].as_py()),
                    dataset_to_index=int(table.column("dataset_to_index")[row_index].as_py()),
                    video_chunk_index=int(table.column(f"{video_prefix}/chunk_index")[row_index].as_py()),
                    video_file_index=int(table.column(f"{video_prefix}/file_index")[row_index].as_py()),
                    video_from_timestamp=float(
                        table.column(f"{video_prefix}/from_timestamp")[row_index].as_py()
                    ),
                    video_to_timestamp=float(
                        table.column(f"{video_prefix}/to_timestamp")[row_index].as_py()
                    ),
                    length=int(table.column("length")[row_index].as_py()),
                )
            )
        return tuple(records)

    @lru_cache(maxsize=1)
    def file_start_indices(self) -> dict[tuple[int, int], int]:
        """Return the global dataset row offset where each parquet shard starts."""

        starts: dict[tuple[int, int], int] = {}
        for record in self.all_episode_records():
            key = (record.data_chunk_index, record.data_file_index)
            current = starts.get(key)
            if current is None or record.dataset_from_index < current:
                starts[key] = record.dataset_from_index
        return starts

    def episode_records(self) -> list[AlohaEpisodeRecord]:
        """Return all episode records for the single-task ALOHA dataset."""

        return list(self.all_episode_records())

    def episode_record(self, episode: int) -> AlohaEpisodeRecord:
        """Return one episode record from the dataset."""

        records = self.episode_records()
        if episode < 0 or episode >= len(records):
            raise IndexError(
                f"ALOHA episode selection {episode} is out of range for "
                f"{len(records)} available episodes."
            )
        return records[episode]

    def _data_file_relative_path(self, chunk_index: int, file_index: int) -> str:
        """Return the dataset-relative path for one ALOHA data parquet shard."""

        return f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

    def _video_file_relative_path(self, chunk_index: int, file_index: int) -> str:
        """Return the dataset-relative path for one ALOHA MP4 shard."""

        return f"videos/{self.image_column}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

    def _decoded_video_root(self) -> Path:
        """Return the cache directory used for transcoded video shards."""

        if self.cache_dir not in {None, ""}:
            base_root = Path(self.cache_dir)
        elif self.local_root is not None:
            base_root = self.local_root / ".decoded_video_cache"
        else:
            base_root = Path(".cache") / "decoded_video_cache"
        return base_root / self.repo_id.replace("/", "--")

    def _ensure_decoded_video(self, chunk_index: int, file_index: int) -> Path:
        """Transcode one AV1 shard into a broadly decodable H.264 cache file."""

        source = self.resolve_file(self._video_file_relative_path(chunk_index, file_index))
        target = self._decoded_video_root() / f"chunk-{chunk_index:03d}-file-{file_index:03d}.mp4"
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ],
            check=True,
        )
        return target

    @lru_cache(maxsize=4)
    def action_table(self, chunk_index: int, file_index: int) -> Any:
        """Load and cache one parquet shard's action column."""

        return pq.read_table(
            self.resolve_file(self._data_file_relative_path(chunk_index, file_index)),
            columns=["action"],
        )

    def frame_row_index_in_file(
        self,
        record: AlohaEpisodeRecord,
        frame_index: int,
    ) -> int:
        """Translate an episode-local frame index into a shard-local row index."""

        if frame_index < 0 or frame_index >= record.length:
            raise IndexError(
                f"Requested frame index {frame_index} is out of bounds for episode "
                f"{record.episode_index} with length {record.length}."
            )
        file_start = self.file_start_indices()[(record.data_chunk_index, record.data_file_index)]
        return record.dataset_from_index - file_start + frame_index

    def video_frame_index_in_file(self, record: AlohaEpisodeRecord, frame_index: int) -> int:
        """Translate an episode-local frame index into an MP4-local frame index."""

        video_start_frame = int(round(record.video_from_timestamp * self.video_fps()))
        return video_start_frame + frame_index

    def _video_capture(self, chunk_index: int, file_index: int) -> cv2.VideoCapture:
        """Open and cache one MP4 shard for random frame access."""

        key = (chunk_index, file_index)
        capture = self._video_captures.get(key)
        if capture is not None and capture.isOpened():
            return capture
        decoded_path = self._ensure_decoded_video(*key)
        capture = cv2.VideoCapture(str(decoded_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open decoded ALOHA video shard {key}.")
        self._video_captures[key] = capture
        return capture

    def _read_video_frame(
        self,
        chunk_index: int,
        file_index: int,
        frame_index_in_file: int,
    ) -> np.ndarray:
        """Seek to and decode one MP4 frame from the selected shard."""

        capture = self._video_capture(chunk_index, file_index)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index_in_file)
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        capture.release()
        self._video_captures.pop((chunk_index, file_index), None)
        capture = self._video_capture(chunk_index, file_index)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index_in_file)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(
                "Failed to decode ALOHA frame "
                f"{frame_index_in_file} from video shard {(chunk_index, file_index)}."
            )
        return frame

    def load_frame_tensor(
        self,
        frame: AlohaFrameRecord,
        resolution: int,
        height: int | None,
        width: int | None,
    ) -> torch.Tensor:
        """Load and resize one frame tensor from the backing MP4 shard."""

        resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
        return bgr_frame_to_tensor(
            self._read_video_frame(
                frame.video_chunk_index,
                frame.video_file_index,
                frame.video_frame_index_in_file,
            ),
            height=resolved_height,
            width=resolved_width,
        )

    def load_action_tensor(self, frame: AlohaFrameRecord) -> torch.Tensor:
        """Load one action vector from the backing parquet shard."""

        value = self.action_table(frame.data_chunk_index, frame.data_file_index)["action"][
            frame.row_index_in_file
        ].as_py()
        action = torch.as_tensor(value, dtype=torch.float32)
        if action.ndim != 1 or action.shape[0] != ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM:
            raise ValueError(
                "Expected ALOHA action shape "
                f"({ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM},), "
                f"received {tuple(action.shape)}."
            )
        return action

    def load_clip(
        self,
        record: AlohaEpisodeRecord,
        frame_start: int | None,
        frame_end: int | None,
        resolution: int,
        height: int | None,
        width: int | None,
        load_actions: bool = False,
        clamp_frame_end: bool = False,
    ) -> dict[str, Any]:
        """Load one resized frame slice from a specific ALOHA episode."""

        resolved_frame_start = 0 if frame_start is None else frame_start
        resolved_frame_end = record.length - 1 if frame_end is None else frame_end
        if resolved_frame_end < resolved_frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start.")
        if resolved_frame_start < 0:
            raise ValueError("frame_start must be greater than or equal to zero.")
        if resolved_frame_start >= record.length:
            raise ValueError(
                f"Requested frame_start {resolved_frame_start} exceeds episode length {record.length}."
            )
        effective_frame_end = (
            min(resolved_frame_end, record.length - 1) if clamp_frame_end else resolved_frame_end
        )
        if effective_frame_end >= record.length:
            raise ValueError(
                f"Requested frames {resolved_frame_start}:{resolved_frame_end} exceed episode length "
                f"{record.length}."
            )

        frames: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        for local_frame_index in range(resolved_frame_start, effective_frame_end + 1):
            frame_record = AlohaFrameRecord(
                episode_index=record.episode_index,
                task_index=record.task_index,
                data_chunk_index=record.data_chunk_index,
                data_file_index=record.data_file_index,
                row_index_in_file=self.frame_row_index_in_file(record, local_frame_index),
                frame_index=local_frame_index,
                video_chunk_index=record.video_chunk_index,
                video_file_index=record.video_file_index,
                video_frame_index_in_file=self.video_frame_index_in_file(record, local_frame_index),
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
                actions.append(self.load_action_tensor(frame_record))
        clip = {
            "frames": torch.stack(frames, dim=0),
            "frame_idx": torch.arange(
                resolved_frame_start,
                effective_frame_end + 1,
                dtype=torch.long,
            ),
            "episode_idx": torch.tensor(record.episode_index, dtype=torch.long),
            "task_idx": torch.tensor(record.task_index, dtype=torch.long),
            "task_name": record.task_name,
        }
        if load_actions:
            clip["actions"] = (
                torch.stack(actions, dim=0)
                if actions
                else torch.zeros(
                    (0, ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM),
                    dtype=torch.float32,
                )
            )
        return clip


def load_aloha_clip(
    data_root: str | Path | None = None,
    split: str = "train",
    episode: int = 0,
    resolution: int = 128,
    height: int | None = None,
    width: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
    cache_dir: str | Path | None = None,
    load_actions: bool = False,
    clamp_frame_end: bool = False,
) -> dict[str, Any]:
    """Load one resized frame slice from the ALOHA sim transfer-cube dataset."""

    resolve_aloha_split(split)
    repository = AlohaSimRepository(data_root=data_root, repo_id=repo_id, cache_dir=cache_dir)
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


class AlohaFrameDataset(Dataset[dict[str, Any]]):
    """Expose ALOHA sim frames as reconstruction samples for Wan-VAE training."""

    def __init__(
        self,
        data_root: str | Path | None = None,
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
        repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Build either one cached clip or a lazy all-episode frame index."""

        resolve_aloha_split(split)
        self.repository = AlohaSimRepository(data_root=data_root, repo_id=repo_id, cache_dir=cache_dir)
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.include_motion_neighbors = include_motion_neighbors
        self._episode_records_by_index: dict[int, AlohaEpisodeRecord] = {}
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.frames: list[AlohaFrameRecord] = []
            shard_buckets: dict[tuple[int, int], list[int]] = {}
            for episode_position, episode_record in enumerate(self.repository.episode_records()):
                if episode_position in excluded_episodes:
                    continue
                self._episode_records_by_index[episode_record.episode_index] = episode_record
                if frame_start is not None and frame_start >= episode_record.length:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = episode_record.length - 1 if frame_end is None else min(
                    frame_end,
                    episode_record.length - 1,
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                shard_key = (episode_record.video_chunk_index, episode_record.video_file_index)
                bucket = shard_buckets.setdefault(shard_key, [])
                for local_frame_index in range(resolved_frame_start, resolved_frame_end + 1):
                    dataset_index = len(self.frames)
                    self.frames.append(
                        AlohaFrameRecord(
                            episode_index=episode_record.episode_index,
                            task_index=episode_record.task_index,
                            data_chunk_index=episode_record.data_chunk_index,
                            data_file_index=episode_record.data_file_index,
                            row_index_in_file=self.repository.frame_row_index_in_file(
                                episode_record,
                                local_frame_index,
                            ),
                            frame_index=local_frame_index,
                            video_chunk_index=episode_record.video_chunk_index,
                            video_file_index=episode_record.video_file_index,
                            video_frame_index_in_file=self.repository.video_frame_index_in_file(
                                episode_record,
                                local_frame_index,
                            ),
                        )
                    )
                    bucket.append(dataset_index)
            if not self.frames:
                raise ValueError("No ALOHA episodes contain any frames in the requested range.")
            self._sampler = MetaWorldGroupedFrameSampler(list(shard_buckets.values()))
        else:
            self.clip = load_aloha_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                cache_dir=cache_dir,
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
                raise IndexError("AlohaFrameDataset index out of range.")
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
                next_index = min(frame_record.frame_index + 1, episode_record.length - 1)
                prev_record = AlohaFrameRecord(
                    episode_index=episode_record.episode_index,
                    task_index=episode_record.task_index,
                    data_chunk_index=episode_record.data_chunk_index,
                    data_file_index=episode_record.data_file_index,
                    row_index_in_file=self.repository.frame_row_index_in_file(
                        episode_record,
                        prev_index,
                    ),
                    frame_index=prev_index,
                    video_chunk_index=episode_record.video_chunk_index,
                    video_file_index=episode_record.video_file_index,
                    video_frame_index_in_file=self.repository.video_frame_index_in_file(
                        episode_record,
                        prev_index,
                    ),
                )
                next_record = AlohaFrameRecord(
                    episode_index=episode_record.episode_index,
                    task_index=episode_record.task_index,
                    data_chunk_index=episode_record.data_chunk_index,
                    data_file_index=episode_record.data_file_index,
                    row_index_in_file=self.repository.frame_row_index_in_file(
                        episode_record,
                        next_index,
                    ),
                    frame_index=next_index,
                    video_chunk_index=episode_record.video_chunk_index,
                    video_file_index=episode_record.video_file_index,
                    video_frame_index_in_file=self.repository.video_frame_index_in_file(
                        episode_record,
                        next_index,
                    ),
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


class AlohaAutoencoderClipDataset(Dataset[dict[str, Any]]):
    """Expose ALOHA simulation episodes as sliding AE clip windows."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
        repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
        cache_dir: str | Path | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
    ) -> None:
        """Build either one cached clip or a lazy all-episode AE clip index."""

        resolve_aloha_split(split)
        self.repository = AlohaSimRepository(data_root=data_root, repo_id=repo_id, cache_dir=cache_dir)
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.required_frames = frame_layout.max_pixel_frames
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.window_records: list[AlohaTransitionRecord] = []
            shard_buckets: dict[tuple[int, int], list[int]] = {}
            for episode_position, episode_record in enumerate(self.repository.episode_records()):
                if episode_position in excluded_episodes:
                    continue
                if frame_start is not None and frame_start >= episode_record.length:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = (
                    episode_record.length - 1
                    if frame_end is None
                    else min(frame_end, episode_record.length - 1)
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                clip_length = resolved_frame_end - resolved_frame_start + 1
                available_windows = max(clip_length - self.required_frames + 1, 0)
                if available_windows < 1:
                    continue
                shard_key = (episode_record.data_chunk_index, episode_record.data_file_index)
                bucket = shard_buckets.setdefault(shard_key, [])
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        AlohaTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.window_records:
                raise ValueError("No ALOHA episodes contain valid AE clip windows in the requested range.")
            self._sampler = MetaWorldGroupedFrameSampler(list(shard_buckets.values()))
        else:
            self.clip = load_aloha_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                cache_dir=cache_dir,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one AE clip window from one cached ALOHA clip."""

        stop = index + self.required_frames
        return {
            "frames": clip["frames"][index:stop],
            "frame_idx": clip["frame_idx"][index:stop],
            "episode_idx": clip["episode_idx"],
        }

    def _load_clip_window(self, record: AlohaTransitionRecord) -> dict[str, Any]:
        """Load the minimal frame slice needed for one lazy AE clip window."""

        stop_frame_index = record.start_frame_index + self.required_frames
        return self.repository.load_clip(
            record=record.episode,
            frame_start=record.start_frame_index,
            frame_end=stop_frame_index - 1,
            resolution=self.resolution,
            height=self.height,
            width=self.width,
        )

    def __len__(self) -> int:
        """Return the number of available AE clip windows."""

        if self.all_episodes:
            return len(self.window_records)
        return max(int(self.clip["frames"].shape[0]) - self.required_frames + 1, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one cached AE clip window."""

        if self.all_episodes:
            if index < 0 or index >= len(self.window_records):
                raise IndexError("AlohaAutoencoderClipDataset index out of range.")
            return self._sample_from_clip(self._load_clip_window(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class AlohaTransitionDataset(Dataset[dict[str, Any]]):
    """Expose ALOHA sim clips as sliding dynamics windows."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
        cache_dir: str | Path | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
        rollout_context_frames: int | None = None,
        rollout_chunks: int = 0,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
    ) -> None:
        """Cache one ALOHA clip or flatten all episodes for dynamics windows."""

        resolve_aloha_split(split)
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
        self.context_pixel_frames = self.frame_layout.context_pixel_frames
        self.target_pixel_frames = self.frame_layout.target_pixel_frames
        self.rollout_target_pixel_frames = (
            self.frame_layout.temporal_compression_ratio * self.rollout_target_frames
        )
        total_latent_frames = self.frame_layout.max_frames + self.rollout_chunks * self.rollout_target_frames
        self.required_frames = self.frame_layout.pixel_frames_for_latent_frames(total_latent_frames)
        self.all_episodes = all_episodes
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.repository = AlohaSimRepository(
                data_root=data_root,
                repo_id=repo_id,
                cache_dir=cache_dir,
            )
            self.resolution = resolution
            self.height = height
            self.width = width
            self.window_records: list[AlohaTransitionRecord] = []
            shard_buckets: dict[tuple[int, int], list[int]] = {}
            for episode_position, episode_record in enumerate(self.repository.episode_records()):
                if episode_position in excluded_episodes:
                    continue
                if frame_start is not None and frame_start >= episode_record.length:
                    continue
                resolved_frame_start = 0 if frame_start is None else frame_start
                resolved_frame_end = (
                    episode_record.length - 1
                    if frame_end is None
                    else min(frame_end, episode_record.length - 1)
                )
                if resolved_frame_end < resolved_frame_start:
                    continue
                clip_length = resolved_frame_end - resolved_frame_start + 1
                available_windows = max(clip_length - self.required_frames + 1, 0)
                if available_windows < 1:
                    continue
                shard_key = (episode_record.video_chunk_index, episode_record.video_file_index)
                bucket = shard_buckets.setdefault(shard_key, [])
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        AlohaTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.window_records:
                raise ValueError(
                    "No ALOHA episodes contain any valid dynamics windows in the requested range."
                )
            self._sampler = MetaWorldGroupedFrameSampler(list(shard_buckets.values()))
        else:
            self.clip = load_aloha_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                cache_dir=cache_dir,
                load_actions=True,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one configured ALOHA `(context, target)` training sample from one clip."""

        context_stop = index + self.context_pixel_frames
        target_stop = context_stop + self.target_pixel_frames
        future_target_stop = target_stop + self.rollout_chunks * self.rollout_target_pixel_frames
        action_stop = index + self.frame_layout.num_action_per_chunk
        future_action_stop = action_stop + self.rollout_chunks * self.rollout_target_pixel_frames
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

    def _load_transition_clip(self, record: AlohaTransitionRecord) -> dict[str, Any]:
        """Load the minimal frame/action slice needed for one lazy ALOHA dynamics window."""

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
        """Return one configured ALOHA `(context, target)` training sample."""

        if self.all_episodes:
            if index < 0 or index >= len(self.window_records):
                raise IndexError("AlohaTransitionDataset index out of range.")
            return self._sample_from_clip(self._load_transition_clip(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class AlohaValidationClipDataset(Dataset[dict[str, Any]]):
    """Expose one cached ALOHA clip as a single validation example."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 128,
        height: int | None = None,
        width: int | None = None,
        repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Cache one ALOHA clip for validation preview generation."""

        self.clip = load_aloha_clip(
            data_root=data_root,
            split=split,
            episode=episode,
            resolution=resolution,
            height=height,
            width=width,
            frame_start=frame_start,
            frame_end=frame_end,
            repo_id=repo_id,
            cache_dir=cache_dir,
            load_actions=True,
        )

    def __len__(self) -> int:
        """Return the number of validation clips."""

        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the cached validation clip."""

        if index != 0:
            raise IndexError("AlohaValidationClipDataset only contains one clip.")
        return {
            "frames": self.clip["frames"],
            "actions": self.clip["actions"],
            "frame_idx": self.clip["frame_idx"],
            "episode_idx": self.clip["episode_idx"],
        }
