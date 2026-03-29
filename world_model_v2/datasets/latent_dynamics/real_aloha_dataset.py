"""Sequence-shaped raw-HDF5 dataset loader for Interactive World Sim data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from world_model_v2.config import DatasetConfig


@dataclass(frozen=True)
class SequenceRecord:
    """Describe one indexed sequence inside an episode file."""

    episode_idx: int
    start_idx: int
    episode_length: int
    path: Path


class RealAlohaDataset(Dataset[dict[str, Any]]):
    """Load train windows or validation episodes from raw Interactive World Sim data."""

    def __init__(self, cfg: DatasetConfig) -> None:
        """Index the configured task split and expose sequence-shaped samples."""

        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        self.split_dir = self.data_root / cfg.task / cfg.split
        self.episode_files = sorted(self.split_dir.glob("episode_*.hdf5"))
        if not self.episode_files:
            raise FileNotFoundError(f"No episode files found in {self.split_dir}")
        self.is_val = cfg.split == "val"
        self.sequence_length = cfg.val_horizon if self.is_val else cfg.horizon
        self.records = self._build_index()

    def _build_index(self) -> list[SequenceRecord]:
        """Build the train-window or validation-episode index."""

        records: list[SequenceRecord] = []
        for episode_idx, episode_path in enumerate(self.episode_files):
            with h5py.File(episode_path, "r") as handle:
                episode_length = int(handle["action"].shape[0])
                for key in self.cfg.obs_keys:
                    if key not in handle["obs"]["images"]:
                        raise KeyError(f"Camera {key} not found in {episode_path}")
            if self.is_val:
                records.append(
                    SequenceRecord(
                        episode_idx=episode_idx,
                        start_idx=0,
                        episode_length=episode_length,
                        path=episode_path,
                    )
                )
            else:
                max_start = max(episode_length - self.sequence_length, 0)
                for start_idx in range(max_start + 1):
                    records.append(
                        SequenceRecord(
                            episode_idx=episode_idx,
                            start_idx=start_idx,
                            episode_length=episode_length,
                            path=episode_path,
                        )
                    )
        return records

    def __len__(self) -> int:
        """Return the number of indexed sequences."""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sequence-shaped sample with `obs` and `action` tensors."""

        record = self.records[index]
        return self._load_sequence(record, target_length=self.sequence_length)

    def _load_sequence(
        self,
        record: SequenceRecord,
        target_length: int | None,
    ) -> dict[str, Any]:
        """Load one sequence and pad it when the requested length is longer."""

        requested_length = target_length or record.episode_length
        with h5py.File(record.path, "r") as handle:
            end_idx = min(record.start_idx + requested_length, record.episode_length)
            frames_by_key = {
                key: np.asarray(handle["obs"]["images"][key][record.start_idx:end_idx])
                for key in self.cfg.obs_keys
            }
            actions = np.asarray(handle["action"][record.start_idx:end_idx], dtype=np.float32)

        valid_length = actions.shape[0]
        if valid_length == 0:
            raise ValueError(f"Empty sequence loaded from {record.path}")

        obs = {
            key: torch.stack(
                [resize_frame_to_tensor(frame, self.cfg.resolution) for frame in frame_array],
                dim=0,
            )
            for key, frame_array in frames_by_key.items()
        }
        action_tensor = torch.from_numpy(actions)

        if valid_length < requested_length:
            pad_frames = requested_length - valid_length
            for key, tensor in obs.items():
                obs[key] = torch.cat([tensor, tensor[-1:].repeat(pad_frames, 1, 1, 1)], dim=0)
            action_tensor = torch.cat(
                [action_tensor, action_tensor[-1:].repeat(pad_frames, 1)],
                dim=0,
            )

        frame_idx = torch.arange(record.start_idx, record.start_idx + requested_length, dtype=torch.long)
        return {
            "obs": obs,
            "action": action_tensor,
            "episode_idx": torch.tensor(record.episode_idx, dtype=torch.long),
            "start_idx": torch.tensor(record.start_idx, dtype=torch.long),
            "frame_idx": frame_idx,
            "valid_length": torch.tensor(valid_length, dtype=torch.long),
        }

    def load_episode_sequence(self, episode_idx: int) -> dict[str, Any]:
        """Load one full episode from the configured split without truncation."""

        episode_path = self.episode_files[episode_idx]
        with h5py.File(episode_path, "r") as handle:
            episode_length = int(handle["action"].shape[0])
        record = SequenceRecord(
            episode_idx=episode_idx,
            start_idx=0,
            episode_length=episode_length,
            path=episode_path,
        )
        return self._load_sequence(record, target_length=None)

    def get_validation_dataset(self) -> "RealAlohaDataset":
        """Return the matching validation dataset."""

        return RealAlohaDataset(self.cfg.validation_copy())

    def compute_action_stats(self) -> dict[str, list[float]]:
        """Compute action min/max statistics for checkpoint metadata."""

        mins: np.ndarray | None = None
        maxs: np.ndarray | None = None
        for episode_path in self.episode_files:
            with h5py.File(episode_path, "r") as handle:
                actions = np.asarray(handle["action"][()], dtype=np.float32)
            current_min = actions.min(axis=0)
            current_max = actions.max(axis=0)
            mins = current_min if mins is None else np.minimum(mins, current_min)
            maxs = current_max if maxs is None else np.maximum(maxs, current_max)
        assert mins is not None and maxs is not None
        return {"action_min": mins.tolist(), "action_max": maxs.tolist()}


def resize_frame_to_tensor(frame: np.ndarray, resolution: int) -> torch.Tensor:
    """Resize a uint8 RGB frame and convert it to a float tensor in `[0, 1]`."""

    image = Image.fromarray(frame)
    resized = image.resize((resolution, resolution), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)))
