"""Dataclass-based run configuration for the upstream-shaped Stage-1 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Dataset settings for the latent-dynamics package."""

    data_root: str = "data/full"
    task: str = "single_grasp"
    split: str = "train"
    obs_keys: tuple[str, ...] = ("camera_1_color",)
    resolution: int = 128
    horizon: int = 1
    val_horizon: int = 200
    action_mode: str = "single_grasp"

    def to_dict(self) -> dict[str, Any]:
        """Convert the config into a JSON-serializable dictionary."""

        payload = asdict(self)
        payload["obs_keys"] = list(self.obs_keys)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetConfig":
        """Build a dataset config from checkpoint JSON data."""

        return cls(
            data_root=payload["data_root"],
            task=payload["task"],
            split=payload["split"],
            obs_keys=tuple(payload["obs_keys"]),
            resolution=payload["resolution"],
            horizon=payload["horizon"],
            val_horizon=payload["val_horizon"],
            action_mode=payload["action_mode"],
        )

    def validation_copy(self) -> "DatasetConfig":
        """Return the matching validation split config."""

        return DatasetConfig(
            data_root=self.data_root,
            task=self.task,
            split="val",
            obs_keys=self.obs_keys,
            resolution=self.resolution,
            horizon=self.horizon,
            val_horizon=self.val_horizon,
            action_mode=self.action_mode,
        )


@dataclass
class AlgorithmConfig:
    """Algorithm settings for the Stage-1 latent world model."""

    training_stage: int = 1
    latent_channels: int = 4
    latent_dim: int = 128
    hidden_channels: int = 64
    timesteps: int = 32
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    infer_steps: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert the config into a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AlgorithmConfig":
        """Build an algorithm config from checkpoint JSON data."""

        return cls(**payload)

    @classmethod
    def upstream_stage1(cls, num_views: int) -> "AlgorithmConfig":
        """Return upstream-sized Stage-1 dimensions for the overlapping config fields."""

        if num_views < 1:
            raise ValueError("num_views must be at least 1")
        return cls(
            training_stage=1,
            latent_channels=4 * num_views,
            latent_dim=512,
            hidden_channels=64,
            timesteps=1000,
            infer_steps=3,
        )


@dataclass
class ExperimentConfig:
    """Experiment runner settings for Stage-1 training."""

    run_name: str = ""
    output_dir: str = "outputs/stage1"
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    max_steps: int = 50
    validation_interval: int = 25
    checkpoint_interval: int = 25
    save_preview_initial_minutes: float = 0.0
    save_preview_late_minutes: float = 0.0
    save_preview_switch_minutes: float = 0.0
    early_stop_window_size: int = 0
    early_stop_patience_windows: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_warmup_steps: int = 0
    log_interval: int = 5
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 7
    resume: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the config into a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        """Build an experiment config from checkpoint JSON data."""

        return cls(**payload)


@dataclass
class RunConfig:
    """Top-level configuration grouping dataset, algorithm, and experiment."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert the grouped config into a nested dictionary."""

        return {
            "dataset": self.dataset.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "experiment": self.experiment.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        """Build a grouped config from checkpoint JSON data."""

        return cls(
            dataset=DatasetConfig.from_dict(payload["dataset"]),
            algorithm=AlgorithmConfig.from_dict(payload["algorithm"]),
            experiment=ExperimentConfig.from_dict(payload["experiment"]),
        )
