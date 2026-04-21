"""Standalone SO101 video dataset helpers for DreamDojo-style RF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any

import cv2
from huggingface_hub import hf_hub_download
import imageio_ffmpeg
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from dreamdojo.so101_rf.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout


SO101_BASE_SIM_PICKPLACE_DATASET_ID = "davidlinjiahao/lerobot_so101_base_sim_pickplace"
SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN = "observation.images.front"
SO101_BASE_SIM_PICKPLACE_ACTION_DIM = 6
SO101_STATE_COLUMN = "observation.state"
SO101_RELATIVE_ACTION_SCALE = 20.0
SO101_DEFAULT_CROP_LEFT_KEEP_FRACTION = 23.0 / 320.0
SO101_DEFAULT_CROP_RIGHT_KEEP_FRACTION = 297.0 / 320.0
SO101_DEFAULT_CROP_BOTTOM_KEEP_FRACTION = 208.0 / 240.0
LEROBOT_VIDEO_EPISODES_PATH = "meta/episodes.jsonl"
LEROBOT_VIDEO_TASKS_PATH = "meta/tasks.jsonl"
LEROBOT_VIDEO_ACTION_COLUMN_CANDIDATES = ("action", "actions")
LEROBOT_ACTION_REPRESENTATIONS = ("absolute", "relative_delta")


def resolve_resize_shape(
    resolution: int,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    """Resolve one square-or-explicit resize request into `(height, width)`."""

    resolved_height = resolution if height is None else int(height)
    resolved_width = resolution if width is None else int(width)
    if resolved_height <= 0 or resolved_width <= 0:
        raise ValueError("height and width must be positive integers.")
    return resolved_height, resolved_width


def bgr_frame_to_tensor(
    frame_bgr: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert one decoded BGR frame into a resized normalized RGB tensor."""

    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(np.ascontiguousarray(pixels), (2, 0, 1))).contiguous()


class GroupedFrameSampler(Sampler[int]):
    """Shuffle grouped dataset indices while keeping nearby items bucket-local."""

    def __init__(self, indices_by_group: list[list[int]], seed: int = 7) -> None:
        """Store one list of index buckets for grouped randomized iteration."""

        self.indices_by_group = [list(bucket) for bucket in indices_by_group]
        self.seed = int(seed)
        self.iteration = 0

    def __iter__(self) -> Any:
        """Yield dataset indices with per-epoch bucket and intra-bucket shuffling."""

        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        bucket_order = list(range(len(self.indices_by_group)))
        rng.shuffle(bucket_order)
        for bucket_index in bucket_order:
            bucket = list(self.indices_by_group[bucket_index])
            rng.shuffle(bucket)
            yield from bucket

    def __len__(self) -> int:
        """Return the total number of iterable dataset indices."""

        return sum(len(bucket) for bucket in self.indices_by_group)


def resolve_lerobot_video_split(split: str) -> str:
    """Resolve the requested split against the train-only LeRobot sim layout."""

    if split not in {"train", "val"}:
        raise ValueError(
            "Episode-sharded LeRobot sim datasets only expose train/val aliases, "
            f"but received split={split!r}."
        )
    return "train"


def resolve_default_lerobot_crop_bounds(
    repo_id: str,
    frame_height: int,
    frame_width: int,
) -> tuple[int, int, int, int] | None:
    """Return one dataset-specific `(top, bottom, left, right)` crop box if configured."""

    if repo_id != SO101_BASE_SIM_PICKPLACE_DATASET_ID:
        return None
    top = 0
    bottom = min(max(math.ceil(frame_height * SO101_DEFAULT_CROP_BOTTOM_KEEP_FRACTION), 1), frame_height)
    left = min(max(math.floor(frame_width * SO101_DEFAULT_CROP_LEFT_KEEP_FRACTION), 0), frame_width - 1)
    right = min(
        max(math.ceil(frame_width * SO101_DEFAULT_CROP_RIGHT_KEEP_FRACTION), left + 1),
        frame_width,
    )
    if bottom <= top or right <= left:
        return None
    return top, bottom, left, right


def crop_bgr_frame(
    frame_bgr: np.ndarray,
    crop_bounds: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Crop one OpenCV BGR frame to the requested keep bounds."""

    if crop_bounds is None:
        return frame_bgr
    top, bottom, left, right = crop_bounds
    return frame_bgr[top:bottom, left:right]


def resolve_exact_downsample_crop_bounds(
    crop_bounds: tuple[int, int, int, int] | None,
    *,
    target_height: int,
    target_width: int,
    min_downsample_factor: int = 2,
) -> tuple[tuple[int, int, int, int], int] | None:
    """Return one centered crop that exactly area-downsamples into the target size."""

    if crop_bounds is None:
        return None
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_height and target_width must be positive integers.")
    if min_downsample_factor < 1:
        raise ValueError("min_downsample_factor must be at least 1.")
    top, bottom, left, right = crop_bounds
    crop_height = int(bottom - top)
    crop_width = int(right - left)
    max_factor = min(crop_height // target_height, crop_width // target_width)
    if max_factor < min_downsample_factor:
        return None
    downsample_factor = int(max_factor)
    aligned_height = int(target_height) * downsample_factor
    aligned_width = int(target_width) * downsample_factor
    trim_height = crop_height - aligned_height
    trim_width = crop_width - aligned_width
    aligned_bottom = bottom - trim_height
    aligned_left = left + (trim_width // 2)
    aligned_right = right - (trim_width - (trim_width // 2))
    return (top, aligned_bottom, aligned_left, aligned_right), downsample_factor


def bgr_frame_to_exact_area_downsampled_tensor(
    frame_bgr: np.ndarray,
    *,
    crop_bounds: tuple[int, int, int, int],
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Crop one BGR frame and exact-downsample it into a normalized RGB tensor."""

    cropped = crop_bgr_frame(frame_bgr, crop_bounds)
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(np.ascontiguousarray(pixels), (2, 0, 1))).contiguous()


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL file into a list of dictionaries."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def render_episode_relative_path(
    template: str,
    *,
    episode_index: int,
    chunk_index: int,
    video_key: str,
) -> str:
    """Render one LeRobot path template for a specific episode."""

    return str(
        template.format(
            episode_index=int(episode_index),
            episode_chunk=int(chunk_index),
            chunk_index=int(chunk_index),
            file_index=int(chunk_index),
            video_key=video_key,
        )
    )


def validate_lerobot_action_representation(
    repo_id: str,
    action_representation: str,
) -> None:
    """Reject unsupported action representations for the selected LeRobot dataset."""

    if action_representation not in LEROBOT_ACTION_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported LeRobot action_representation={action_representation!r}. "
            f"Expected one of {LEROBOT_ACTION_REPRESENTATIONS}."
        )
    if (
        action_representation != "absolute"
        and repo_id != SO101_BASE_SIM_PICKPLACE_DATASET_ID
    ):
        raise ValueError(
            "Relative LeRobot actions are currently only implemented for the SO101 base sim "
            f"pickplace dataset, received repo_id={repo_id!r}."
        )


def so101_relative_actions_from_absolute_targets(
    actions: torch.Tensor,
    states: torch.Tensor,
    *,
    scale: float = SO101_RELATIVE_ACTION_SCALE,
) -> torch.Tensor:
    """Convert SO101 next-state targets into scaled relative next-step deltas."""

    if actions.ndim != 2:
        raise ValueError(
            f"Expected SO101 actions with shape (T, A), received {tuple(actions.shape)}."
        )
    if states.ndim != 2 or states.shape[0] != actions.shape[0]:
        raise ValueError(
            "SO101 states must align one-to-one with actions; received "
            f"actions={tuple(actions.shape)} states={tuple(states.shape)}."
        )
    if states.shape[1] < actions.shape[1]:
        raise ValueError(
            "SO101 states must expose at least as many channels as actions; received "
            f"actions={tuple(actions.shape)} states={tuple(states.shape)}."
        )
    resolved_scale = float(scale)
    if resolved_scale <= 0.0:
        raise ValueError("SO101 relative action scale must be positive.")
    return (actions - states[:, : actions.shape[1]]) * resolved_scale


@dataclass(frozen=True)
class LeRobotVideoEpisodeRecord:
    """Describe one episode-sharded LeRobot video episode."""

    episode_index: int
    task_index: int
    task_name: str
    length: int
    chunk_index: int
    data_relative_path: str
    video_relative_path: str


@dataclass(frozen=True)
class LeRobotVideoFrameRecord:
    """Describe one frame inside a specific episode-sharded video dataset."""

    episode: LeRobotVideoEpisodeRecord
    frame_index: int


@dataclass(frozen=True)
class LeRobotVideoTransitionRecord:
    """Describe one dynamics window inside a specific LeRobot video episode."""

    episode: LeRobotVideoEpisodeRecord
    start_frame_index: int


class LeRobotEpisodeVideoRepository:
    """Resolve episode-sharded LeRobot metadata and lazily read video/parquet episodes."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        repo_id: str = SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column: str = SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        cache_dir: str | Path | None = None,
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
        self._video_captures: dict[int, cv2.VideoCapture] = {}

    def __getstate__(self) -> dict[str, Any]:
        """Drop unpicklable OpenCV state before the repository is serialized."""

        state = dict(self.__dict__)
        state["_video_captures"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore one repository after worker-process deserialization."""

        self.__dict__.update(state)
        self._video_captures = {}

    def __del__(self) -> None:
        """Release cached OpenCV captures when the repository is destroyed."""

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
        """Return the local path for one metadata, parquet, or video asset."""

        if self.local_root is not None:
            resolved = self.local_root / relative_path
            if not resolved.exists():
                raise FileNotFoundError(f"Missing LeRobot video dataset file: {resolved}")
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
    def episode_chunk_size(self) -> int:
        """Return the chunk size declared by the dataset metadata."""

        return int(self.info().get("chunks_size", 1000))

    @lru_cache(maxsize=1)
    def data_path_template(self) -> str:
        """Return the dataset-relative parquet template for one episode."""

        template = self.info().get("data_path")
        if not isinstance(template, str):
            raise ValueError("LeRobot video dataset is missing data_path in meta/info.json.")
        return template

    @lru_cache(maxsize=1)
    def video_path_template(self) -> str:
        """Return the dataset-relative MP4 template for one episode."""

        template = self.info().get("video_path")
        if not isinstance(template, str):
            raise ValueError("LeRobot video dataset is missing video_path in meta/info.json.")
        return template

    @lru_cache(maxsize=1)
    def action_dim(self) -> int:
        """Return the configured action dimension from the dataset metadata."""

        features = self.info().get("features", {})
        action_feature = features.get("action", {})
        shape = action_feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 1:
            raise ValueError("LeRobot video dataset action metadata is missing a 1D shape.")
        return int(shape[0])

    @lru_cache(maxsize=1)
    def video_fps(self) -> float:
        """Return the configured FPS for the selected video stream."""

        features = self.info().get("features", {})
        image_feature = features.get(self.image_column, {})
        video_info = image_feature.get("video_info")
        if isinstance(video_info, dict):
            return float(video_info.get("video.fps", self.info().get("fps", 0.0)))
        legacy_info = image_feature.get("info")
        if isinstance(legacy_info, dict):
            return float(legacy_info.get("video.fps", self.info().get("fps", 0.0)))
        return float(self.info().get("fps", 0.0))

    @lru_cache(maxsize=1)
    def task_names(self) -> dict[int, str]:
        """Load the task-index to task-name mapping from JSONL metadata."""

        path = self.resolve_file(LEROBOT_VIDEO_TASKS_PATH)
        mapping: dict[int, str] = {}
        for record in read_jsonl_records(path):
            mapping[int(record["task_index"])] = str(record.get("task", record["task_index"]))
        return mapping

    @lru_cache(maxsize=1)
    def all_episode_records(self) -> tuple[LeRobotVideoEpisodeRecord, ...]:
        """Load all episode metadata records from JSONL metadata."""

        records: list[LeRobotVideoEpisodeRecord] = []
        task_names = self.task_names()
        for record in read_jsonl_records(self.resolve_file(LEROBOT_VIDEO_EPISODES_PATH)):
            episode_index = int(record["episode_index"])
            task_indices = record.get("tasks", [0])
            if not isinstance(task_indices, list) or len(task_indices) < 1:
                raise ValueError("Episode-sharded LeRobot metadata must provide at least one task.")
            task_index = int(task_indices[0])
            chunk_index = episode_index // self.episode_chunk_size()
            records.append(
                LeRobotVideoEpisodeRecord(
                    episode_index=episode_index,
                    task_index=task_index,
                    task_name=task_names.get(task_index, str(task_index)),
                    length=int(record["length"]),
                    chunk_index=chunk_index,
                    data_relative_path=render_episode_relative_path(
                        self.data_path_template(),
                        episode_index=episode_index,
                        chunk_index=chunk_index,
                        video_key=self.image_column,
                    ),
                    video_relative_path=render_episode_relative_path(
                        self.video_path_template(),
                        episode_index=episode_index,
                        chunk_index=chunk_index,
                        video_key=self.image_column,
                    ),
                )
            )
        return tuple(records)

    def episode_records(self) -> list[LeRobotVideoEpisodeRecord]:
        """Return all episodes for the configured dataset."""

        return list(self.all_episode_records())

    def episode_record(self, episode: int) -> LeRobotVideoEpisodeRecord:
        """Return one episode record by position in the sorted episode list."""

        records = self.episode_records()
        if episode < 0 or episode >= len(records):
            raise IndexError(
                f"LeRobot episode selection {episode} is out of range for "
                f"{len(records)} available episodes."
            )
        return records[episode]

    @lru_cache(maxsize=16)
    def episode_schema_names(self, relative_path: str) -> tuple[str, ...]:
        """Return the parquet schema names for one episode shard."""

        return tuple(pq.read_schema(self.resolve_file(relative_path)).names)

    @lru_cache(maxsize=16)
    def action_column_name(self, relative_path: str) -> str:
        """Return the action column name for one episode parquet shard."""

        for column_name in LEROBOT_VIDEO_ACTION_COLUMN_CANDIDATES:
            if column_name in self.episode_schema_names(relative_path):
                return column_name
        raise KeyError(
            "LeRobot episode parquet is missing an action column. "
            f"Checked {LEROBOT_VIDEO_ACTION_COLUMN_CANDIDATES}."
        )

    @lru_cache(maxsize=16)
    def state_column_name(self, relative_path: str) -> str:
        """Return the state column name required for relative SO101 actions."""

        if SO101_STATE_COLUMN not in self.episode_schema_names(relative_path):
            raise KeyError(
                "LeRobot episode parquet is missing the observation.state column required "
                "for SO101 relative actions."
            )
        return SO101_STATE_COLUMN

    @lru_cache(maxsize=8)
    def episode_table(self, relative_path: str) -> Any:
        """Load and cache one episode parquet table."""

        return pq.read_table(self.resolve_file(relative_path))

    def _decoded_video_root(self) -> Path:
        """Return the cache directory used for transcoded episode MP4s."""

        if self.cache_dir not in {None, ""}:
            base_root = Path(self.cache_dir)
        elif self.local_root is not None:
            base_root = self.local_root / ".decoded_video_cache"
        else:
            base_root = Path(".cache") / "decoded_video_cache"
        return base_root / self.repo_id.replace("/", "--")

    def _ensure_decoded_video(self, record: LeRobotVideoEpisodeRecord) -> Path:
        """Transcode one AV1 episode MP4 into a broadly decodable H.264 cache file."""

        source = self.resolve_file(record.video_relative_path)
        target = self._decoded_video_root() / f"episode_{record.episode_index:06d}.mp4"
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

    def _video_capture(self, record: LeRobotVideoEpisodeRecord) -> cv2.VideoCapture:
        """Open and cache the decoded episode MP4 for random frame access."""

        capture = self._video_captures.get(record.episode_index)
        if capture is not None and capture.isOpened():
            return capture
        decoded_path = self._ensure_decoded_video(record)
        capture = cv2.VideoCapture(str(decoded_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open decoded LeRobot video for episode {record.episode_index}.")
        self._video_captures[record.episode_index] = capture
        return capture

    def _read_video_frame(self, record: LeRobotVideoEpisodeRecord, frame_index: int) -> Any:
        """Seek to and decode one frame from the selected episode video."""

        capture = self._video_capture(record)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        capture.release()
        self._video_captures.pop(record.episode_index, None)
        capture = self._video_capture(record)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(
                f"Failed to decode frame {frame_index} from episode {record.episode_index}."
            )
        return frame

    def _frame_tensor_from_bgr(
        self,
        frame_bgr: Any,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Crop and resize one decoded BGR frame into the training tensor format."""

        crop_bounds = resolve_default_lerobot_crop_bounds(
            self.repo_id,
            frame_bgr.shape[0],
            frame_bgr.shape[1],
        )
        exact_downsample = resolve_exact_downsample_crop_bounds(
            crop_bounds,
            target_height=height,
            target_width=width,
        )
        if exact_downsample is not None:
            aligned_crop_bounds, _ = exact_downsample
            return bgr_frame_to_exact_area_downsampled_tensor(
                frame_bgr,
                crop_bounds=aligned_crop_bounds,
                target_height=height,
                target_width=width,
            )
        cropped_frame_bgr = crop_bgr_frame(frame_bgr, crop_bounds)
        return bgr_frame_to_tensor(
            cropped_frame_bgr,
            height=height,
            width=width,
        )

    def load_frame_tensors(
        self,
        record: LeRobotVideoEpisodeRecord,
        frame_start: int,
        frame_end: int,
        resolution: int,
        height: int | None,
        width: int | None,
    ) -> list[torch.Tensor]:
        """Load one contiguous frame range with a single sequential decode pass."""

        if frame_end < frame_start:
            return []
        resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
        capture = self._video_capture(record)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        frames: list[torch.Tensor] = []
        next_frame_index = frame_start
        while next_frame_index <= frame_end:
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                self._video_captures.pop(record.episode_index, None)
                capture = self._video_capture(record)
                capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"Failed to decode frame {next_frame_index} from episode "
                        f"{record.episode_index}."
                    )
            frames.append(
                self._frame_tensor_from_bgr(
                    frame,
                    height=resolved_height,
                    width=resolved_width,
                )
            )
            next_frame_index += 1
        return frames

    def load_frame_tensor(
        self,
        frame: LeRobotVideoFrameRecord,
        resolution: int,
        height: int | None,
        width: int | None,
    ) -> torch.Tensor:
        """Load and resize one frame tensor from the backing episode MP4."""

        resolved_height, resolved_width = resolve_resize_shape(resolution, height, width)
        frame_bgr = self._read_video_frame(frame.episode, frame.frame_index)
        return self._frame_tensor_from_bgr(
            frame_bgr,
            height=resolved_height,
            width=resolved_width,
        )

    def load_action_tensor(
        self,
        frame: LeRobotVideoFrameRecord,
        *,
        action_representation: str = "absolute",
        action_scale: float = 1.0,
    ) -> torch.Tensor:
        """Load one action vector from the backing episode parquet shard."""

        actions = self.load_action_tensors(
            frame.episode,
            frame.frame_index,
            frame.frame_index + 1,
            action_representation=action_representation,
            action_scale=action_scale,
        )
        return actions[0]

    def load_action_tensors(
        self,
        record: LeRobotVideoEpisodeRecord,
        frame_start: int,
        frame_end_exclusive: int,
        *,
        action_representation: str = "absolute",
        action_scale: float = 1.0,
    ) -> torch.Tensor:
        """Load a contiguous action slice for the requested episode frame range."""

        validate_lerobot_action_representation(self.repo_id, action_representation)
        expected_dim = self.action_dim()
        if frame_end_exclusive <= frame_start:
            return torch.zeros((0, expected_dim), dtype=torch.float32)
        column_name = self.action_column_name(record.data_relative_path)
        table = self.episode_table(record.data_relative_path)
        if frame_end_exclusive > table.num_rows:
            raise IndexError(
                f"Requested action rows {frame_start}:{frame_end_exclusive} exceed parquet rows for "
                f"episode {record.episode_index}."
            )
        actions = torch.as_tensor(
            table[column_name].slice(frame_start, frame_end_exclusive - frame_start).to_pylist(),
            dtype=torch.float32,
        )
        if actions.ndim != 2 or actions.shape[1] != expected_dim:
            raise ValueError(
                f"Expected LeRobot action shape (N, {expected_dim}), received {tuple(actions.shape)}."
            )
        if action_representation == "absolute":
            return actions
        state_column_name = self.state_column_name(record.data_relative_path)
        states = torch.as_tensor(
            table[state_column_name]
            .slice(frame_start, frame_end_exclusive - frame_start)
            .to_pylist(),
            dtype=torch.float32,
        )
        return so101_relative_actions_from_absolute_targets(
            actions,
            states,
            scale=action_scale,
        )

    def load_clip(
        self,
        record: LeRobotVideoEpisodeRecord,
        frame_start: int | None,
        frame_end: int | None,
        resolution: int,
        height: int | None,
        width: int | None,
        load_actions: bool = False,
        action_representation: str = "absolute",
        action_scale: float = 1.0,
        clamp_frame_end: bool = False,
    ) -> dict[str, Any]:
        """Load one resized frame slice from a specific episode video dataset."""

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
        frames = self.load_frame_tensors(
            record,
            resolved_frame_start,
            effective_frame_end,
            resolution=resolution,
            height=height,
            width=width,
        )
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
            clip["actions"] = self.load_action_tensors(
                record,
                resolved_frame_start,
                effective_frame_end,
                action_representation=action_representation,
                action_scale=action_scale,
            )
        return clip


def load_lerobot_video_clip(
    data_root: str | Path | None = None,
    split: str = "train",
    episode: int = 0,
    resolution: int = 128,
    height: int | None = None,
    width: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    repo_id: str = SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    image_column: str = SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    cache_dir: str | Path | None = None,
    load_actions: bool = False,
    action_representation: str = "absolute",
    action_scale: float = 1.0,
    clamp_frame_end: bool = False,
) -> dict[str, Any]:
    """Load one resized frame slice from an episode-sharded LeRobot video dataset."""

    resolve_lerobot_video_split(split)
    repository = LeRobotEpisodeVideoRepository(
        data_root=data_root,
        repo_id=repo_id,
        image_column=image_column,
        cache_dir=cache_dir,
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
        action_representation=action_representation,
        action_scale=action_scale,
        clamp_frame_end=clamp_frame_end,
    )


class LeRobotVideoFrameDataset(Dataset[dict[str, Any]]):
    """Expose episode-sharded LeRobot video frames as reconstruction samples."""

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
        repo_id: str = SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column: str = SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Build either one cached clip or a lazy all-episode frame index."""

        resolve_lerobot_video_split(split)
        self.repository = LeRobotEpisodeVideoRepository(
            data_root=data_root,
            repo_id=repo_id,
            image_column=image_column,
            cache_dir=cache_dir,
        )
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.include_motion_neighbors = include_motion_neighbors
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.frames: list[LeRobotVideoFrameRecord] = []
            episode_buckets: dict[int, list[int]] = {}
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
                bucket = episode_buckets.setdefault(episode_record.episode_index, [])
                for local_frame_index in range(resolved_frame_start, resolved_frame_end + 1):
                    dataset_index = len(self.frames)
                    self.frames.append(
                        LeRobotVideoFrameRecord(
                            episode=episode_record,
                            frame_index=local_frame_index,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.frames:
                raise ValueError(
                    "No LeRobot episode-video episodes contain any frames in the requested range."
                )
            self._sampler = GroupedFrameSampler(list(episode_buckets.values()))
        else:
            self.clip = load_lerobot_video_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                image_column=image_column,
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
                raise IndexError("LeRobotVideoFrameDataset index out of range.")
            frame_record = self.frames[index]
            sample = {
                "frame": self.repository.load_frame_tensor(
                    frame_record,
                    resolution=self.resolution,
                    height=self.height,
                    width=self.width,
                ),
                "frame_idx": torch.tensor(frame_record.frame_index, dtype=torch.long),
                "episode_idx": torch.tensor(frame_record.episode.episode_index, dtype=torch.long),
            }
            if self.include_motion_neighbors:
                prev_record = LeRobotVideoFrameRecord(
                    episode=frame_record.episode,
                    frame_index=max(frame_record.frame_index - 1, 0),
                )
                next_record = LeRobotVideoFrameRecord(
                    episode=frame_record.episode,
                    frame_index=min(frame_record.frame_index + 1, frame_record.episode.length - 1),
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


class LeRobotVideoAutoencoderClipDataset(Dataset[dict[str, Any]]):
    """Expose episode-sharded LeRobot videos as sliding AE clip windows."""

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
        repo_id: str = SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column: str = SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        cache_dir: str | Path | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
    ) -> None:
        """Build either one cached clip or a lazy all-episode AE clip index."""

        resolve_lerobot_video_split(split)
        self.repository = LeRobotEpisodeVideoRepository(
            data_root=data_root,
            repo_id=repo_id,
            image_column=image_column,
            cache_dir=cache_dir,
        )
        self.resolution = resolution
        self.height = height
        self.width = width
        self.all_episodes = all_episodes
        self.required_frames = frame_layout.max_pixel_frames
        excluded_episodes = set(int(episode_index) for episode_index in exclude_episodes)
        if all_episodes:
            self.window_records: list[LeRobotVideoTransitionRecord] = []
            episode_buckets: dict[int, list[int]] = {}
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
                bucket = episode_buckets.setdefault(episode_record.episode_index, [])
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        LeRobotVideoTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.window_records:
                raise ValueError(
                    "No LeRobot episode-video episodes contain valid AE clip windows in the requested range."
                )
            self._sampler = GroupedFrameSampler(list(episode_buckets.values()))
        else:
            self.clip = load_lerobot_video_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                image_column=image_column,
                cache_dir=cache_dir,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one AE clip window from one cached LeRobot clip."""

        stop = index + self.required_frames
        return {
            "frames": clip["frames"][index:stop],
            "frame_idx": clip["frame_idx"][index:stop],
            "episode_idx": clip["episode_idx"],
        }

    def _load_clip_window(self, record: LeRobotVideoTransitionRecord) -> dict[str, Any]:
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
                raise IndexError("LeRobotVideoAutoencoderClipDataset index out of range.")
            return self._sample_from_clip(self._load_clip_window(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class LeRobotVideoTransitionDataset(Dataset[dict[str, Any]]):
    """Expose episode-sharded LeRobot video clips as sliding dynamics windows."""

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
        repo_id: str = SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column: str = SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        cache_dir: str | Path | None = None,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
        rollout_context_frames: int | None = None,
        rollout_chunks: int = 0,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
        action_representation: str = "absolute",
        action_scale: float = 1.0,
    ) -> None:
        """Cache one clip or flatten all episodes for dynamics windows."""

        resolve_lerobot_video_split(split)
        validate_lerobot_action_representation(repo_id, action_representation)
        self.frame_layout = frame_layout
        self.action_representation = action_representation
        self.action_scale = float(action_scale)
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
            self.repository = LeRobotEpisodeVideoRepository(
                data_root=data_root,
                repo_id=repo_id,
                image_column=image_column,
                cache_dir=cache_dir,
            )
            self.resolution = resolution
            self.height = height
            self.width = width
            self.window_records: list[LeRobotVideoTransitionRecord] = []
            episode_buckets: dict[int, list[int]] = {}
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
                bucket = episode_buckets.setdefault(episode_record.episode_index, [])
                for offset in range(available_windows):
                    dataset_index = len(self.window_records)
                    self.window_records.append(
                        LeRobotVideoTransitionRecord(
                            episode=episode_record,
                            start_frame_index=resolved_frame_start + offset,
                        )
                    )
                    bucket.append(dataset_index)
            if not self.window_records:
                raise ValueError(
                    "No LeRobot episode-video episodes contain valid dynamics windows in the requested range."
                )
            self._sampler = GroupedFrameSampler(list(episode_buckets.values()))
        else:
            self.clip = load_lerobot_video_clip(
                data_root=data_root,
                split=split,
                episode=episode,
                resolution=resolution,
                height=height,
                width=width,
                frame_start=frame_start,
                frame_end=frame_end,
                repo_id=repo_id,
                image_column=image_column,
                cache_dir=cache_dir,
                load_actions=True,
                action_representation=self.action_representation,
                action_scale=self.action_scale,
            )
            self._sampler = None

    def training_sampler(self) -> Sampler[int] | None:
        """Return the preferred training sampler for the current dataset mode."""

        return self._sampler

    def _sample_from_clip(self, clip: dict[str, Any], index: int) -> dict[str, Any]:
        """Return one configured `(context, target)` training sample from one clip."""

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

    def _load_transition_clip(self, record: LeRobotVideoTransitionRecord) -> dict[str, Any]:
        """Load the minimal frame/action slice needed for one lazy dynamics window."""

        stop_frame_index = record.start_frame_index + self.required_frames
        return self.repository.load_clip(
            record=record.episode,
            frame_start=record.start_frame_index,
            frame_end=stop_frame_index - 1,
            resolution=self.resolution,
            height=self.height,
            width=self.width,
            load_actions=True,
            action_representation=self.action_representation,
            action_scale=self.action_scale,
        )

    def __len__(self) -> int:
        """Return the number of available dynamics windows for the configured layout."""

        if self.all_episodes:
            return len(self.window_records)
        return max(int(self.clip["frames"].shape[0]) - self.required_frames + 1, 0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one configured `(context, target)` training sample."""

        if self.all_episodes:
            if index < 0 or index >= len(self.window_records):
                raise IndexError("LeRobotVideoTransitionDataset index out of range.")
            return self._sample_from_clip(self._load_transition_clip(self.window_records[index]), 0)
        return self._sample_from_clip(self.clip, index)


class LeRobotVideoValidationClipDataset(Dataset[dict[str, Any]]):
    """Expose one cached episode-sharded LeRobot clip as a single validation example."""

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
        repo_id: str = SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column: str = SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        cache_dir: str | Path | None = None,
        action_representation: str = "absolute",
        action_scale: float = 1.0,
    ) -> None:
        """Cache one clip for validation preview generation."""

        validate_lerobot_action_representation(repo_id, action_representation)
        self.clip = load_lerobot_video_clip(
            data_root=data_root,
            split=split,
            episode=episode,
            resolution=resolution,
            height=height,
            width=width,
            frame_start=frame_start,
            frame_end=frame_end,
            repo_id=repo_id,
            image_column=image_column,
            cache_dir=cache_dir,
            load_actions=True,
            action_representation=action_representation,
            action_scale=action_scale,
        )

    def __len__(self) -> int:
        """Return the number of validation clips."""

        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the cached validation clip."""

        if index != 0:
            raise IndexError("LeRobotVideoValidationClipDataset only contains one clip.")
        return {
            "frames": self.clip["frames"],
            "actions": self.clip["actions"],
            "frame_idx": self.clip["frame_idx"],
            "episode_idx": self.clip["episode_idx"],
        }


def load_so101_video_clip(
    data_root: str | Path | None = None,
    split: str = "train",
    episode: int = 0,
    resolution: int = 96,
    height: int | None = 96,
    width: int | None = 128,
    frame_start: int | None = None,
    frame_end: int | None = None,
    load_actions: bool = False,
    action_representation: str = "relative_delta",
    action_scale: float = SO101_RELATIVE_ACTION_SCALE,
    clamp_frame_end: bool = False,
) -> dict[str, Any]:
    """Load one SO101 clip with the standalone defaults applied."""

    return load_lerobot_video_clip(
        data_root=data_root,
        split=split,
        episode=episode,
        resolution=resolution,
        height=height,
        width=width,
        frame_start=frame_start,
        frame_end=frame_end,
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        load_actions=load_actions,
        action_representation=action_representation,
        action_scale=action_scale,
        clamp_frame_end=clamp_frame_end,
    )


class So101TransitionDataset(LeRobotVideoTransitionDataset):
    """Expose SO101 clip windows for standalone DreamDojo-style RF dynamics training."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 96,
        height: int | None = 96,
        width: int | None = 128,
        frame_layout: DynamicsFrameLayout = DYNAMICS_FRAME_LAYOUT,
        rollout_context_frames: int | None = None,
        rollout_chunks: int = 0,
        all_episodes: bool = False,
        exclude_episodes: tuple[int, ...] = (),
        action_representation: str = "relative_delta",
        action_scale: float = SO101_RELATIVE_ACTION_SCALE,
    ) -> None:
        """Create one SO101 transition dataset using the standalone defaults."""

        super().__init__(
            data_root=data_root,
            split=split,
            episode=episode,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
            height=height,
            width=width,
            repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
            image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
            frame_layout=frame_layout,
            rollout_context_frames=rollout_context_frames,
            rollout_chunks=rollout_chunks,
            all_episodes=all_episodes,
            exclude_episodes=exclude_episodes,
            action_representation=action_representation,
            action_scale=action_scale,
        )


class So101ValidationClipDataset(LeRobotVideoValidationClipDataset):
    """Expose one SO101 clip as a standalone DreamDojo-style validation sample."""

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        episode: int = 0,
        frame_start: int | None = None,
        frame_end: int | None = None,
        resolution: int = 96,
        height: int | None = 96,
        width: int | None = 128,
        action_representation: str = "relative_delta",
        action_scale: float = SO101_RELATIVE_ACTION_SCALE,
    ) -> None:
        """Create one SO101 validation clip dataset using the standalone defaults."""

        super().__init__(
            data_root=data_root,
            split=split,
            episode=episode,
            frame_start=frame_start,
            frame_end=frame_end,
            resolution=resolution,
            height=height,
            width=width,
            repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
            image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
            action_representation=action_representation,
            action_scale=action_scale,
        )
