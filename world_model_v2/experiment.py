"""Experiment runner for the Wan-VAE plus RF-DiT world-model pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from world_model_v2.dataset import (
    FrameDataset,
    TransitionDataset,
    ValidationClipDataset,
)
from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT, DynamicsFrameLayout
from world_model_v2.dynamics_transformer import DynamicsTrainingInputs
from world_model_v2.metaworld_dataset import (
    METAWORLD_DATASET_ID,
    MetaWorldFrameDataset,
    MetaWorldTransitionDataset,
    MetaWorldValidationClipDataset,
)
from world_model_v2.model import WorldModel
from world_model_v2.utils.checkpointing import append_jsonl, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


WORLD_MODEL_CHECKPOINT_KIND = "world_model_v2_v1"
LEGACY_WORLD_MODEL_CHECKPOINT_KINDS = frozenset(
    {
        "world_model_v2_minimal_v1",
        "world_model_v2_minimal_v2",
    }
)
AUTO_BATCH_SIZE_BACKOFF_DIVISOR = 2


@dataclass
class ExperimentConfig:
    """Group CLI-configurable settings for the world-model experiment."""

    mode: str = "ae_only"
    dataset_format: str = "interactive_world_sim"
    data_root: str = "data/full"
    task: str = "single_grasp"
    metaworld_task_index: int | None = None
    metaworld_repo_id: str = METAWORLD_DATASET_ID
    metaworld_cache_dir: str = ""
    split: str = "val"
    episode: int = 0
    train_all_episodes: bool = False
    validation_split: str = ""
    validation_episode: int = 0
    camera: str = "camera_1_color"
    frame_start: int | None = None
    frame_end: int | None = None
    resolution: int = 128
    height: int | None = None
    width: int | None = None
    latent_channels: int = 16
    hidden_channels: int = 64
    ae_backend: str = "wan"
    dynamics_infer_steps: int = 16
    dynamics_train_timesteps: int = 1000
    dynamics_rf_shift: float = 5.0
    conditional_frame_timestep: float = -1.0
    dynamics_video_condition_dropout: float = 0.0
    dynamics_guidance_scale: float = 0.0
    dynamics_self_forcing_loss_weight: float = 0.0
    dynamics_context_frames: int = DYNAMICS_FRAME_LAYOUT.context_frames
    dynamics_target_frames: int = DYNAMICS_FRAME_LAYOUT.target_frames
    dynamics_conditioning_frame_choices: tuple[int, ...] | None = None
    dynamics_conditioning_frame_probabilities: tuple[float, ...] | None = None
    dynamics_validation_conditioning_frame_choices: tuple[int, ...] | None = None
    dynamics_open_rollout_context_frames: int | None = None
    dynamics_model_channels: int = 256
    dynamics_num_blocks: int = 4
    dynamics_num_heads: int = 4
    dynamics_action_conditioning_mode: str = "chunk_per_frame"
    dynamics_zero_init_action_embedder: bool = False
    dynamics_use_adaln_lora: bool = False
    dynamics_adaln_lora_dim: int = 64
    dynamics_rope_t_extrapolation_ratio: float = 1.0
    dynamics_validation_metric: str = "next_frame_mse"
    kl_beta: float = 1e-4
    recon_mse_weight: float = 1.0
    recon_l1_weight: float = 0.0
    recon_edge_weight: float = 0.0
    batch_size: int = 32
    auto_batch_size: bool = False
    lr: float = 1e-4
    max_steps: int = 3000
    validation_interval: int = 100
    checkpoint_interval: int = 100
    early_stop_window_size: int = 1
    early_stop_patience_windows: int = 5
    early_stop_min_delta: float = 1e-10
    early_stop_warmup_steps: int = 300
    log_interval: int = 10
    output_dir: str = "outputs"
    run_name: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume: str = ""
    load_encoder_decoder: str = ""
    load_dynamics: str = ""
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)

    def clip_metadata(self) -> dict[str, Any]:
        """Return the clip settings that define the debug dataset."""

        return {
            "dataset_format": self.dataset_format,
            "task": self.task,
            "metaworld_task_index": self.metaworld_task_index,
            "metaworld_repo_id": self.metaworld_repo_id,
            "split": self.resolved_split(),
            "episode": self.episode,
            "camera": self.resolved_camera(),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "resolution": self.resolution,
            "height": self.resolved_height(),
            "width": self.resolved_width(),
        }

    def resolved_split(self) -> str:
        """Return the effective training split for the selected dataset format."""

        if self.dataset_format == "lerobot_metaworld":
            return "train"
        return self.split

    def resolved_validation_split(self) -> str:
        """Return the validation split after applying the optional override."""

        if self.dataset_format == "lerobot_metaworld":
            return "train"
        return self.validation_split or self.split

    def resolved_camera(self) -> str:
        """Return the effective image stream key for the selected dataset format."""

        if self.dataset_format == "lerobot_metaworld":
            return "observation.image"
        return self.camera

    def resolved_height(self) -> int:
        """Return the configured image height after applying square fallback."""

        return self.resolution if self.height is None else self.height

    def resolved_width(self) -> int:
        """Return the configured image width after applying square fallback."""

        return self.resolution if self.width is None else self.width

    def dynamics_frame_layout(self) -> DynamicsFrameLayout:
        """Return the configured dynamics chunk layout."""

        return DynamicsFrameLayout(
            context_frames=self.dynamics_context_frames,
            target_frames=self.dynamics_target_frames,
        )


def finite_difference_gradients(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal and vertical finite-difference image gradients."""

    grad_x = images[..., :, 1:] - images[..., :, :-1]
    grad_y = images[..., 1:, :] - images[..., :-1, :]
    return grad_x, grad_y


def reconstruction_loss_terms(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    l1_weight: float,
    edge_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute mixed pixel and edge reconstruction losses for one image batch."""

    if mse_weight < 0.0 or l1_weight < 0.0 or edge_weight < 0.0:
        raise ValueError("Reconstruction loss weights must be non-negative.")
    total_weight = mse_weight + l1_weight + edge_weight
    if total_weight <= 0.0:
        raise ValueError("At least one reconstruction loss weight must be positive.")
    recon_mse = F.mse_loss(predicted, target)
    recon_l1 = F.l1_loss(predicted, target)
    predicted_grad_x, predicted_grad_y = finite_difference_gradients(predicted)
    target_grad_x, target_grad_y = finite_difference_gradients(target)
    edge_l1 = 0.5 * (
        F.l1_loss(predicted_grad_x, target_grad_x) + F.l1_loss(predicted_grad_y, target_grad_y)
    )
    recon_loss = (
        mse_weight * recon_mse
        + l1_weight * recon_l1
        + edge_weight * edge_l1
    ) / total_weight
    return {
        "recon_loss": recon_loss,
        "recon_mse": recon_mse.detach(),
        "recon_l1": recon_l1.detach(),
        "edge_l1": edge_l1.detach(),
    }


def save_training_checkpoint(
    path: str | Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    step: int,
    config: dict[str, Any],
    mode: str,
    clip_metadata: dict[str, Any],
    best_metric: float | None,
) -> None:
    """Save a world-model checkpoint with reproducibility metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": WORLD_MODEL_CHECKPOINT_KIND,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": config,
        "mode": mode,
        "clip_metadata": clip_metadata,
        "best_metric": best_metric,
        "ae_backend": model.ae_backend,
        "autoencoder": model.autoencoder_config(),
        "dynamics_backend": model.dynamics_backend,
        "dynamics": model.dynamics_config(),
    }
    torch.save(payload, output_path)


def load_training_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load and validate one world-model checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    accepted_kinds = {WORLD_MODEL_CHECKPOINT_KIND, *LEGACY_WORLD_MODEL_CHECKPOINT_KINDS}
    if checkpoint.get("kind") not in accepted_kinds:
        raise ValueError(f"{path} is not a supported world-model checkpoint.")
    return checkpoint


def checkpoint_ae_backend(checkpoint: dict[str, Any]) -> str:
    """Return the checkpoint's recorded autoencoder backend."""

    backend = checkpoint.get("ae_backend")
    if isinstance(backend, str):
        return backend
    autoencoder = checkpoint.get("autoencoder")
    if isinstance(autoencoder, dict) and isinstance(autoencoder.get("backend"), str):
        return str(autoencoder["backend"])
    config = checkpoint.get("config")
    if isinstance(config, dict) and isinstance(config.get("ae_backend"), str):
        return str(config["ae_backend"])
    raise ValueError("Checkpoint is missing autoencoder backend metadata.")


def checkpoint_dynamics_backend(checkpoint: dict[str, Any]) -> str:
    """Return the checkpoint's recorded dynamics backend."""

    backend = checkpoint.get("dynamics_backend")
    if isinstance(backend, str):
        return backend
    dynamics = checkpoint.get("dynamics")
    if isinstance(dynamics, dict) and isinstance(dynamics.get("backend"), str):
        return str(dynamics["backend"])
    return "legacy_conv"


class Experiment:
    """Train and validate the world model in AE-only or dynamics-only mode."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        """Build datasets, model, optimizer, and run directories."""

        self.cfg = cfg
        self._validate_config()
        self._set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self._validate_device()
        self.run_name = cfg.run_name or self._default_run_name(cfg.mode)
        self.run_dir = Path(cfg.output_dir) / self.run_name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.model = WorldModel(
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
            ae_backend=cfg.ae_backend,
            resolution=cfg.resolution,
            height=cfg.height,
            width=cfg.width,
            dynamics_infer_steps=cfg.dynamics_infer_steps,
            dynamics_train_timesteps=cfg.dynamics_train_timesteps,
            dynamics_rf_shift=cfg.dynamics_rf_shift,
            conditional_frame_timestep=cfg.conditional_frame_timestep,
            dynamics_video_condition_dropout=cfg.dynamics_video_condition_dropout,
            dynamics_guidance_scale=cfg.dynamics_guidance_scale,
            dynamics_context_frames=cfg.dynamics_context_frames,
            dynamics_target_frames=cfg.dynamics_target_frames,
            dynamics_conditioning_frame_choices=cfg.dynamics_conditioning_frame_choices,
            dynamics_conditioning_frame_probabilities=cfg.dynamics_conditioning_frame_probabilities,
            dynamics_validation_conditioning_frame_choices=cfg.dynamics_validation_conditioning_frame_choices,
            dynamics_open_rollout_context_frames=cfg.dynamics_open_rollout_context_frames,
            dynamics_model_channels=cfg.dynamics_model_channels,
            dynamics_num_blocks=cfg.dynamics_num_blocks,
            dynamics_num_heads=cfg.dynamics_num_heads,
            dynamics_action_conditioning_mode=cfg.dynamics_action_conditioning_mode,
            dynamics_zero_init_action_embedder=cfg.dynamics_zero_init_action_embedder,
            dynamics_use_adaln_lora=cfg.dynamics_use_adaln_lora,
            dynamics_adaln_lora_dim=cfg.dynamics_adaln_lora_dim,
            dynamics_rope_t_extrapolation_ratio=cfg.dynamics_rope_t_extrapolation_ratio,
        ).to(self.device)
        self._load_requested_pretrained_weights()
        self.model.configure_trainability(cfg.mode)
        self.train_dataset = self._build_train_dataset()
        self._validate_train_dataset()
        self.cfg.batch_size = self._resolve_train_batch_size(self.train_dataset)
        self.train_loader = self._build_train_loader(self.train_dataset)
        self.val_loader = self._build_val_loader()
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.best_metric: float | None = None
        self.early_stop_window_losses: deque[float] = deque(
            maxlen=max(cfg.early_stop_window_size, 1)
        )
        self.best_window_loss: float | None = None
        self.non_improving_windows = 0
        self.early_stop_observations = 0
        self.current_step = 0
        self.run_started_at_monotonic: float | None = None
        if cfg.resume:
            self.current_step = self._load_resume()

    def _set_seed(self, seed: int) -> None:
        """Seed Python, NumPy, and PyTorch RNGs."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_device(self) -> None:
        """Fail fast when the requested CUDA device is unavailable."""

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")

    def _validate_config(self) -> None:
        """Validate mode-specific flag combinations before work begins."""

        if self.cfg.mode not in {"ae_only", "dynamics_only"}:
            raise ValueError(f"Unsupported mode: {self.cfg.mode}")
        if self.cfg.dataset_format not in {"interactive_world_sim", "lerobot_metaworld"}:
            raise ValueError(f"Unsupported dataset format: {self.cfg.dataset_format}")
        if self.cfg.ae_backend != "wan":
            raise ValueError(
                f"Unsupported autoencoder backend: {self.cfg.ae_backend}. "
                "This codepath now only supports the Wan VAE."
            )
        if self.cfg.dynamics_infer_steps < 1:
            raise ValueError("dynamics_infer_steps must be positive.")
        if self.cfg.dynamics_train_timesteps < 2:
            raise ValueError("dynamics_train_timesteps must be at least 2.")
        if self.cfg.dynamics_rf_shift <= 0.0:
            raise ValueError("dynamics_rf_shift must be positive.")
        if self.cfg.conditional_frame_timestep < -1.0:
            raise ValueError("conditional_frame_timestep must be -1 or a non-negative value.")
        if not 0.0 <= self.cfg.dynamics_video_condition_dropout <= 1.0:
            raise ValueError("dynamics_video_condition_dropout must be between 0 and 1.")
        if self.cfg.dynamics_guidance_scale < 0.0:
            raise ValueError("dynamics_guidance_scale must be non-negative.")
        if self.cfg.dynamics_self_forcing_loss_weight < 0.0:
            raise ValueError("dynamics_self_forcing_loss_weight must be non-negative.")
        if self.cfg.dynamics_context_frames < 1:
            raise ValueError("dynamics_context_frames must be positive.")
        if self.cfg.dynamics_target_frames < 1:
            raise ValueError("dynamics_target_frames must be positive.")
        if self.cfg.dynamics_model_channels < 1:
            raise ValueError("dynamics_model_channels must be positive.")
        if self.cfg.dynamics_num_blocks < 1:
            raise ValueError("dynamics_num_blocks must be positive.")
        if self.cfg.dynamics_num_heads < 1:
            raise ValueError("dynamics_num_heads must be positive.")
        if self.cfg.dynamics_action_conditioning_mode not in {"chunk_per_frame", "global_chunk"}:
            raise ValueError(
                "dynamics_action_conditioning_mode must be 'chunk_per_frame' or 'global_chunk'."
            )
        if self.cfg.dynamics_adaln_lora_dim < 1:
            raise ValueError("dynamics_adaln_lora_dim must be positive.")
        if self.cfg.dynamics_rope_t_extrapolation_ratio <= 0.0:
            raise ValueError("dynamics_rope_t_extrapolation_ratio must be positive.")
        if self.cfg.dynamics_validation_metric not in {"next_frame_mse", "open_rollout_frame_mse"}:
            raise ValueError(
                "dynamics_validation_metric must be 'next_frame_mse' or 'open_rollout_frame_mse'."
            )
        if (
            self.cfg.frame_start is not None
            and self.cfg.frame_end is not None
            and self.cfg.frame_end < self.cfg.frame_start
        ):
            raise ValueError("frame_end must be greater than or equal to frame_start.")
        if self.cfg.train_all_episodes and self.cfg.mode != "ae_only":
            raise ValueError("--train-all-episodes is only supported for mode ae_only.")
        if self.cfg.resume and (self.cfg.load_encoder_decoder or self.cfg.load_dynamics):
            raise ValueError("Do not combine --resume with --load-encoder-decoder/--load-dynamics.")
        if self.cfg.mode == "ae_only" and self.cfg.load_dynamics:
            raise ValueError("--load-dynamics is invalid for mode ae_only.")
        if (
            self.cfg.mode == "dynamics_only"
            and not self.cfg.resume
            and not self.cfg.load_encoder_decoder
        ):
            raise ValueError("--load-encoder-decoder is required for mode dynamics_only.")
        if self._early_stop_enabled() and self.cfg.validation_interval <= 0:
            raise ValueError("Validation-based early stopping requires validation_interval > 0.")
        if self.cfg.dataset_format == "lerobot_metaworld":
            if self.cfg.split not in {"train", "val"}:
                raise ValueError("MetaWorld MT50 only supports train/val aliases for split.")
            if self.cfg.validation_split not in {"", "train", "val"}:
                raise ValueError(
                    "MetaWorld MT50 only supports train/val aliases for validation_split."
                )
            if self.cfg.metaworld_task_index is not None and self.cfg.metaworld_task_index < 0:
                raise ValueError("metaworld_task_index must be greater than or equal to zero.")

    def _validate_train_dataset(self) -> None:
        """Reject dynamics-only runs that do not contain any valid dynamics windows."""

        if self.cfg.mode == "dynamics_only" and len(self.train_dataset) < 1:
            raise ValueError(
                f"dynamics_only requires at least {self.model.dynamics.cfg.max_frames} "
                "frames in the selected clip so one valid dynamics window exists."
            )

    def _default_run_name(self, mode: str) -> str:
        """Return a timestamped default run name."""

        return f"world_model_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _elapsed_run_seconds(self) -> float:
        """Return the elapsed wall-clock seconds since `run()` began."""

        if self.run_started_at_monotonic is None:
            return 0.0
        return time.monotonic() - self.run_started_at_monotonic

    def _early_stop_enabled(self) -> bool:
        """Return whether validation plateau stopping is enabled."""

        return self.cfg.early_stop_window_size > 0 and self.cfg.early_stop_patience_windows > 0

    def _build_train_dataset(self) -> Dataset[dict[str, Any]]:
        """Build the mode-specific training dataset."""

        frame_layout = self.cfg.dynamics_frame_layout()
        if self.cfg.dataset_format == "lerobot_metaworld":
            dataset_kwargs = {
                "data_root": self.cfg.data_root,
                "split": self.cfg.resolved_split(),
                "episode": self.cfg.episode,
                "task_index": self.cfg.metaworld_task_index,
                "frame_start": self.cfg.frame_start,
                "frame_end": self.cfg.frame_end,
                "resolution": self.cfg.resolution,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "repo_id": self.cfg.metaworld_repo_id,
                "cache_dir": self.cfg.metaworld_cache_dir or None,
            }
            if self.cfg.mode == "ae_only":
                dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
                return MetaWorldFrameDataset(**dataset_kwargs)
            dataset_kwargs["frame_layout"] = frame_layout
            return MetaWorldTransitionDataset(**dataset_kwargs)
        dataset_kwargs = {
            "data_root": self.cfg.data_root,
            "task": self.cfg.task,
            "split": self.cfg.split,
            "episode": self.cfg.episode,
            "camera": self.cfg.camera,
            "frame_start": self.cfg.frame_start,
            "frame_end": self.cfg.frame_end,
            "resolution": self.cfg.resolution,
            "height": self.cfg.height,
            "width": self.cfg.width,
        }
        if self.cfg.mode == "ae_only":
            dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
            return FrameDataset(**dataset_kwargs)
        dataset_kwargs["frame_layout"] = frame_layout
        return TransitionDataset(**dataset_kwargs)

    def _build_train_loader(self, dataset: Dataset[dict[str, Any]]) -> DataLoader[Any]:
        """Build the training dataloader for the resolved batch size."""

        sampler_factory = getattr(dataset, "training_sampler", None)
        if callable(sampler_factory):
            sampler = sampler_factory()
            if sampler is not None:
                return DataLoader(
                    dataset,
                    batch_size=self.cfg.batch_size,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=0,
                )
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0)

    def _build_val_loader(self) -> DataLoader[Any]:
        """Build the full-clip validation dataloader."""

        if self.cfg.dataset_format == "lerobot_metaworld":
            dataset = MetaWorldValidationClipDataset(
                data_root=self.cfg.data_root,
                split=self.cfg.resolved_validation_split(),
                episode=self.cfg.validation_episode,
                task_index=self.cfg.metaworld_task_index,
                frame_start=self.cfg.frame_start,
                frame_end=self.cfg.frame_end,
                resolution=self.cfg.resolution,
                height=self.cfg.height,
                width=self.cfg.width,
                repo_id=self.cfg.metaworld_repo_id,
                cache_dir=self.cfg.metaworld_cache_dir or None,
            )
            return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        dataset = ValidationClipDataset(
            data_root=self.cfg.data_root,
            task=self.cfg.task,
            split=self.cfg.resolved_validation_split(),
            episode=self.cfg.validation_episode,
            camera=self.cfg.camera,
            frame_start=self.cfg.frame_start,
            frame_end=self.cfg.frame_end,
            resolution=self.cfg.resolution,
            height=self.cfg.height,
            width=self.cfg.width,
        )
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer over trainable parameters."""

        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters were selected for the current mode.")
        return torch.optim.AdamW(
            parameters,
            lr=self.cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    def _resolve_train_batch_size(self, dataset: Dataset[dict[str, Any]]) -> int:
        """Return the configured or automatically probed training batch size."""

        max_dataset_batch = max(len(dataset), 1)
        if not self.cfg.auto_batch_size:
            return max(self.cfg.batch_size, 1)
        if self.cfg.dataset_format == "lerobot_metaworld":
            return max(self.cfg.batch_size, 1)
        if self.device.type != "cuda":
            return max_dataset_batch
        return self._probe_cuda_batch_size(dataset, max_dataset_batch)

    def _probe_cuda_batch_size(
        self,
        dataset: Dataset[dict[str, Any]],
        max_batch_size: int,
    ) -> int:
        """Probe the largest CUDA batch size that fits for one training step."""

        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        left = 1
        right = max_batch_size
        best = 1
        try:
            while left <= right:
                candidate = (left + right) // 2
                if self._batch_size_fits(dataset, candidate):
                    best = candidate
                    left = candidate + 1
                else:
                    right = candidate - 1
        finally:
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        return best

    def _batch_size_fits(
        self,
        dataset: Dataset[dict[str, Any]],
        batch_size: int,
    ) -> bool:
        """Return whether one dry-run training step fits in CUDA memory."""

        batch = default_collate([dataset[index] for index in range(min(batch_size, len(dataset)))])
        moved_batch = self._move_batch_to_device(batch)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        probe_optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        parameter_backup = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        try:
            loss_dict = self._train_step(moved_batch)
            loss_dict["loss"].backward()
            probe_optimizer.step()
            return True
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            return False
        finally:
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in parameter_backup:
                        parameter.copy_(parameter_backup[name].to(parameter.device))
            self.model.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            del probe_optimizer
            del parameter_backup
            del moved_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _is_cuda_oom(self, error: RuntimeError) -> bool:
        """Return whether the runtime error looks like a CUDA OOM."""

        return "out of memory" in str(error).lower()

    def _reduce_batch_size_after_oom(self) -> bool:
        """Shrink the train batch size after a CUDA OOM and rebuild the loader."""

        if not self.cfg.auto_batch_size or self.device.type != "cuda" or self.cfg.batch_size <= 1:
            return False
        new_batch_size = max(1, self.cfg.batch_size // AUTO_BATCH_SIZE_BACKOFF_DIVISOR)
        if new_batch_size == self.cfg.batch_size:
            return False
        self.cfg.batch_size = new_batch_size
        self.train_loader = self._build_train_loader(self.train_dataset)
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "auto_batch_size_backoff",
                    "reason": "cuda_oom",
                    "new_batch_size": self.cfg.batch_size,
                },
                sort_keys=True,
            )
        )
        return True

    def _cleanup_after_cuda_oom(self) -> None:
        """Aggressively release state after a failed CUDA training step."""

        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _execute_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one optimizer step and return the detached metric tensors."""

        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self._train_step(batch)
        loss_dict["loss"].backward()
        self.optimizer.step()
        return loss_dict

    def _assert_checkpoint_backend(self, checkpoint: dict[str, Any], path: str | Path) -> None:
        """Ensure the checkpoint backend matches the currently requested backend."""

        backend = checkpoint_ae_backend(checkpoint)
        if backend != self.cfg.ae_backend:
            raise ValueError(
                f"Checkpoint backend {backend} from {path} does not match requested "
                f"backend {self.cfg.ae_backend}."
            )

    def _assert_checkpoint_dynamics_backend(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
    ) -> None:
        """Ensure the checkpoint dynamics backend matches the RF DiT architecture."""

        backend = checkpoint_dynamics_backend(checkpoint)
        if backend != self.model.dynamics_backend:
            raise ValueError(
                f"Checkpoint dynamics backend {backend} from {path} does not match requested "
                f"backend {self.model.dynamics_backend}. Old conv-dynamics checkpoints are not "
                "compatible with the current RF DiT."
            )
        checkpoint_dynamics = checkpoint.get("dynamics")
        if not isinstance(checkpoint_dynamics, dict):
            return
        checkpoint_config = checkpoint_dynamics.get("config")
        if not isinstance(checkpoint_config, dict):
            return
        checkpoint_max_frames = checkpoint_config.get("max_frames")
        if checkpoint_max_frames != self.model.dynamics.cfg.max_frames:
            raise ValueError(
                f"Checkpoint dynamics config max_frames={checkpoint_max_frames} from {path} "
                f"does not match the requested RF DiT max_frames="
                f"{self.model.dynamics.cfg.max_frames}. This run expects "
                f"{self.model.dynamics.cfg.context_frames} context frames plus "
                f"{self.model.dynamics.cfg.target_frames} target frame(s)."
            )
        checkpoint_context_frames = checkpoint_config.get("context_frames")
        if checkpoint_context_frames != self.model.dynamics.cfg.context_frames:
            raise ValueError(
                f"Checkpoint dynamics config context_frames={checkpoint_context_frames} from {path} "
                f"does not match the requested context_frames={self.model.dynamics.cfg.context_frames}."
            )
        checkpoint_target_frames = checkpoint_config.get("target_frames")
        if checkpoint_target_frames != self.model.dynamics.cfg.target_frames:
            raise ValueError(
                f"Checkpoint dynamics config target_frames={checkpoint_target_frames} from {path} "
                f"does not match the requested target_frames={self.model.dynamics.cfg.target_frames}."
            )
        checkpoint_action_steps = checkpoint_config.get("num_action_per_chunk")
        if checkpoint_action_steps != self.model.dynamics.cfg.num_action_per_chunk:
            raise ValueError(
                f"Checkpoint dynamics config num_action_per_chunk={checkpoint_action_steps} from {path} "
                "does not match the active DreamDojo-style action-conditioned RF DiT. "
                "Older stub-action checkpoints are not load-compatible; start a fresh dynamics run "
                "or skip loading dynamics weights."
            )
        checkpoint_action_dim = checkpoint_config.get("action_dim")
        if checkpoint_action_dim != self.model.dynamics.cfg.action_dim:
            raise ValueError(
                f"Checkpoint dynamics config action_dim={checkpoint_action_dim} from {path} "
                f"does not match the requested action_dim={self.model.dynamics.cfg.action_dim}."
            )
        checkpoint_action_conditioning_mode = checkpoint_config.get("action_conditioning_mode")
        checkpoint_action_input_features = None
        model_state = checkpoint.get("model_state")
        if isinstance(model_state, dict):
            action_projection = model_state.get("dynamics.net.action_embedder_B_D.fc1.weight")
            if isinstance(action_projection, torch.Tensor) and action_projection.ndim == 2:
                checkpoint_action_input_features = int(action_projection.shape[1])
        if checkpoint_action_conditioning_mode is None:
            if checkpoint_action_input_features == self.model.dynamics.cfg.action_dim:
                checkpoint_action_conditioning_mode = "chunk_per_frame"
            elif checkpoint_action_input_features == (
                self.model.dynamics.cfg.action_dim * self.model.dynamics.cfg.num_action_per_chunk
            ):
                checkpoint_action_conditioning_mode = "global_chunk"
        if checkpoint_action_conditioning_mode != self.model.dynamics.cfg.action_conditioning_mode:
            raise ValueError(
                f"Checkpoint dynamics config action_conditioning_mode={checkpoint_action_conditioning_mode} "
                f"from {path} does not match the requested action_conditioning_mode="
                f"{self.model.dynamics.cfg.action_conditioning_mode}."
            )

    def _load_requested_pretrained_weights(self) -> None:
        """Load any optional encoder/decoder or dynamics checkpoint weights."""

        if self.cfg.load_encoder_decoder:
            checkpoint = load_training_checkpoint(self.cfg.load_encoder_decoder, self.device)
            self._assert_checkpoint_backend(checkpoint, self.cfg.load_encoder_decoder)
            self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
            self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
        if self.cfg.load_dynamics:
            checkpoint = load_training_checkpoint(self.cfg.load_dynamics, self.device)
            self._assert_checkpoint_backend(checkpoint, self.cfg.load_dynamics)
            self._assert_checkpoint_dynamics_backend(checkpoint, self.cfg.load_dynamics)
            self._load_submodule_state("dynamics", self.model.dynamics, checkpoint["model_state"])

    def _load_submodule_state(
        self,
        prefix: str,
        module: torch.nn.Module,
        model_state: dict[str, torch.Tensor],
    ) -> None:
        """Load one named submodule from a saved state dictionary."""

        prefix_with_dot = f"{prefix}."
        submodule_state = {
            key.removeprefix(prefix_with_dot): value
            for key, value in model_state.items()
            if key.startswith(prefix_with_dot)
        }
        if not submodule_state:
            raise KeyError(f"Checkpoint is missing weights for {prefix}.")
        module.load_state_dict(submodule_state, strict=True)

    def _load_resume(self) -> int:
        """Restore a previous training checkpoint and return its step."""

        checkpoint = load_training_checkpoint(self.cfg.resume, self.device)
        self._assert_checkpoint_backend(checkpoint, self.cfg.resume)
        if self.cfg.mode == "ae_only":
            self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
            self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
        else:
            self._assert_checkpoint_dynamics_backend(checkpoint, self.cfg.resume)
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
        if checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        restored_best_metric = self._restore_best_metric_from_metrics()
        self.best_metric = (
            checkpoint.get("best_metric")
            if restored_best_metric is None
            else restored_best_metric
        )
        return int(checkpoint["step"])

    def _record_metric_for_early_stop(
        self,
        step: int,
        metric_value: float,
    ) -> dict[str, Any] | None:
        """Update validation plateau state and return an evaluation record when due."""

        if not self._early_stop_enabled():
            return None

        self.early_stop_observations += 1
        self.early_stop_window_losses.append(metric_value)
        if len(self.early_stop_window_losses) < self.cfg.early_stop_window_size:
            return None
        if self.early_stop_observations % self.cfg.early_stop_window_size != 0:
            return None

        window_loss = float(sum(self.early_stop_window_losses) / len(self.early_stop_window_losses))
        improved = False
        if step < self.cfg.early_stop_warmup_steps:
            if self.best_window_loss is None or window_loss < self.best_window_loss:
                self.best_window_loss = window_loss
            self.non_improving_windows = 0
        elif self.best_window_loss is None or (
            window_loss < self.best_window_loss - self.cfg.early_stop_min_delta
        ):
            self.best_window_loss = window_loss
            self.non_improving_windows = 0
            improved = True
        else:
            self.non_improving_windows += 1

        should_stop = (
            step >= self.cfg.early_stop_warmup_steps
            and self.non_improving_windows >= self.cfg.early_stop_patience_windows
        )
        return {
            "step": step,
            "metric": self._validation_metric_name(),
            "window_loss": window_loss,
            "best_window_loss": self.best_window_loss,
            "improved": improved,
            "non_improving_windows": self.non_improving_windows,
            "patience_windows": self.cfg.early_stop_patience_windows,
            "min_delta": self.cfg.early_stop_min_delta,
            "warmup_steps": self.cfg.early_stop_warmup_steps,
            "should_stop": should_stop,
        }

    def _restore_metrics_path(self) -> Path | None:
        """Find the metrics log that should seed validation plateau state."""

        if self.cfg.resume:
            resume_run_dir = Path(self.cfg.resume).resolve().parent.parent
            if resume_run_dir != self.run_dir.resolve():
                resume_metrics = resume_run_dir / "metrics.jsonl"
                if resume_metrics.exists():
                    return resume_metrics
        if self.metrics_path.exists():
            return self.metrics_path
        return None

    def _restore_best_metric_from_metrics(self) -> float | None:
        """Rebuild the active best metric from saved validation records."""

        restore_path = self._restore_metrics_path()
        if restore_path is None:
            return None
        metric_name = self._validation_metric_name()
        best_metric: float | None = None
        for line in restore_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            validation = record.get("validation")
            if not isinstance(validation, dict) or metric_name not in validation:
                continue
            metric_value = float(validation[metric_name])
            if best_metric is None or metric_value < best_metric:
                best_metric = metric_value
        return best_metric

    def _extract_early_stop_metric_value(self, record: dict[str, Any]) -> float | None:
        """Extract the current mode's validation metric from one metrics record."""

        validation = record.get("validation")
        if not isinstance(validation, dict):
            return None
        metric_name = self._validation_metric_name()
        if metric_name not in validation:
            return None
        return float(validation[metric_name])

    def _restore_early_stop_state(self, step: int) -> None:
        """Replay prior validation metrics to rebuild plateau state for resumes."""

        if not self._early_stop_enabled() or step <= 0:
            return
        restore_path = self._restore_metrics_path()
        if restore_path is None:
            return

        self.early_stop_window_losses.clear()
        self.best_window_loss = None
        self.non_improving_windows = 0
        self.early_stop_observations = 0
        with restore_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_step = int(record.get("step", 0))
                if record_step > step:
                    break
                metric_value = self._extract_early_stop_metric_value(record)
                if metric_value is None:
                    continue
                self._record_metric_for_early_stop(record_step, metric_value)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor leaves in one batch to the configured device."""

        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return moved

    def _validation_metric_name(self) -> str:
        """Return the metric name used for best-checkpoint selection."""

        if self.cfg.mode == "ae_only":
            return "ae_loss"
        return self.cfg.dynamics_validation_metric

    def _train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Dispatch to the mode-specific training objective."""

        if self.cfg.mode == "ae_only":
            return self._ae_only_training_step(batch)
        return self._dynamics_only_training_step(batch)

    def _ae_only_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one KL-regularized autoencoder reconstruction step."""

        frames = batch["frame"]
        output = self.model.autoencode(frames, sample_posterior=True)
        recon_terms = reconstruction_loss_terms(
            output.reconstructed,
            frames,
            mse_weight=self.cfg.recon_mse_weight,
            l1_weight=self.cfg.recon_l1_weight,
            edge_weight=self.cfg.recon_edge_weight,
        )
        ae_loss = recon_terms["recon_loss"] + self.cfg.kl_beta * output.kl_loss
        return {
            "loss": ae_loss,
            "recon_loss": recon_terms["recon_loss"].detach(),
            "recon_mse": recon_terms["recon_mse"],
            "recon_l1": recon_terms["recon_l1"],
            "edge_l1": recon_terms["edge_l1"],
            "kl_loss": output.kl_loss.detach(),
            "ae_loss": ae_loss.detach(),
        }

    def _dynamics_only_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one frozen-autoencoder mixed-conditioning RF dynamics step."""

        context_frames = batch["context_frames"]
        target_frames = batch["target_frames"]
        actions = batch["actions"]
        with torch.no_grad():
            context_latent_video = self.model.encode_context_frames(context_frames, deterministic=True)
            target_latent_video = self.model.encode_frame_sequence(target_frames, deterministic=True)
        clean_latent_video = torch.cat([context_latent_video, target_latent_video], dim=2)
        dynamics_inputs = self.model.dynamics.prepare_training_inputs(clean_latent_video, actions=actions)
        predicted_velocity = self.model.dynamics(
            noisy_latent_video=dynamics_inputs.noisy_latent_video,
            timesteps=dynamics_inputs.timesteps,
            condition_mask=dynamics_inputs.condition_mask,
            actions=dynamics_inputs.actions,
            conditioning_latent_video=dynamics_inputs.conditioning_latent_video,
            target_velocity=dynamics_inputs.target_velocity,
            use_video_condition=dynamics_inputs.use_video_condition,
        )
        latent_rf_mse = F.mse_loss(predicted_velocity, dynamics_inputs.target_velocity)
        latent_rf_self_forcing_mse, self_forcing_stats = self._dynamics_self_forcing_loss(
            clean_latent_video=clean_latent_video,
            predicted_velocity=predicted_velocity,
            dynamics_inputs=dynamics_inputs,
        )
        weighted_self_forcing_loss = (
            self.cfg.dynamics_self_forcing_loss_weight * latent_rf_self_forcing_mse
        )
        total_loss = latent_rf_mse + weighted_self_forcing_loss
        metrics = {
            "loss": total_loss,
            "latent_rf_mse": latent_rf_mse.detach(),
            "target_sigma": dynamics_inputs.target_sigmas.mean().detach(),
            "conditioning_frames_mean": dynamics_inputs.num_conditional_frames.float().mean().detach(),
            "use_video_condition_mean": dynamics_inputs.use_video_condition.float().mean().detach(),
        }
        if self.cfg.dynamics_self_forcing_loss_weight > 0.0:
            metrics.update(
                {
                    "latent_rf_self_forcing_mse": latent_rf_self_forcing_mse.detach(),
                    "latent_rf_self_forcing_weighted_loss": weighted_self_forcing_loss.detach(),
                    "latent_rf_total_loss": total_loss.detach(),
                    **self_forcing_stats,
                }
            )
        return metrics

    def _dynamics_self_forcing_loss(
        self,
        *,
        clean_latent_video: torch.Tensor,
        predicted_velocity: torch.Tensor,
        dynamics_inputs: DynamicsTrainingInputs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a DreamDojo-inspired causal self-forcing loss for later target frames."""

        zero = predicted_velocity.new_zeros(())
        if (
            self.cfg.dynamics_self_forcing_loss_weight <= 0.0
            or self.model.dynamics.cfg.target_frames < 2
        ):
            return zero, {}

        base_reference_noise = (
            dynamics_inputs.target_velocity + dynamics_inputs.conditioning_latent_video
        )
        recursive_predicted_clean = self.model.dynamics.recover_clean_latent_video(
            noisy_latent_video=dynamics_inputs.noisy_latent_video,
            predicted_velocity=predicted_velocity,
            target_sigmas=dynamics_inputs.target_sigmas,
        ).detach()
        total_self_forcing_loss = zero
        total_self_forcing_steps = 0
        stats: dict[str, torch.Tensor] = {}
        for conditioning_frames in range(
            self.model.dynamics.cfg.context_frames + 1,
            self.model.dynamics.cfg.max_frames,
        ):
            auxiliary_clean_latent_video = clean_latent_video.clone()
            auxiliary_clean_latent_video[:, :, :conditioning_frames] = recursive_predicted_clean[
                :, :, :conditioning_frames
            ]
            auxiliary_noisy_latent_video, auxiliary_target_velocity = (
                self.model.dynamics.flow.interpolate(
                    noise=base_reference_noise,
                    clean=auxiliary_clean_latent_video,
                    sigmas=dynamics_inputs.target_sigmas,
                )
            )
            auxiliary_condition_mask = self.model.dynamics.make_condition_mask(
                auxiliary_clean_latent_video,
                num_conditional_frames=conditioning_frames,
                allow_unregistered=True,
            )
            auxiliary_predicted_velocity = self.model.dynamics(
                noisy_latent_video=auxiliary_noisy_latent_video,
                timesteps=dynamics_inputs.timesteps,
                condition_mask=auxiliary_condition_mask,
                actions=dynamics_inputs.actions,
                conditioning_latent_video=auxiliary_clean_latent_video,
                target_velocity=auxiliary_target_velocity,
                use_video_condition=True,
            )
            step_loss = F.mse_loss(
                auxiliary_predicted_velocity[:, :, conditioning_frames:],
                auxiliary_target_velocity[:, :, conditioning_frames:],
            )
            total_self_forcing_loss = total_self_forcing_loss + step_loss
            total_self_forcing_steps += 1
            stats[f"latent_rf_self_forcing_mse_ctx{conditioning_frames}"] = step_loss.detach()
            recursive_predicted_clean = self.model.dynamics.recover_clean_latent_video(
                noisy_latent_video=auxiliary_noisy_latent_video,
                predicted_velocity=auxiliary_predicted_velocity,
                target_sigmas=dynamics_inputs.target_sigmas,
            ).detach()
        return total_self_forcing_loss / total_self_forcing_steps, stats

    @torch.no_grad()
    def _validate_ae_only_frames(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float, float, float, float]:
        """Validate AE reconstructions in chunks to avoid full-clip CUDA OOMs."""

        chunk_size = max(1, min(int(self.cfg.batch_size), int(frames.shape[0])))
        reconstructed_chunks: list[torch.Tensor] = []
        total_recon_loss = 0.0
        total_recon_mse = 0.0
        total_recon_l1 = 0.0
        total_edge_l1 = 0.0
        total_kl = 0.0
        total_frames = int(frames.shape[0])
        for start in range(0, total_frames, chunk_size):
            stop = min(start + chunk_size, total_frames)
            frame_chunk = frames[start:stop]
            output = self.model.autoencode(frame_chunk, sample_posterior=False)
            chunk_frames = int(frame_chunk.shape[0])
            recon_terms = reconstruction_loss_terms(
                output.reconstructed,
                frame_chunk,
                mse_weight=self.cfg.recon_mse_weight,
                l1_weight=self.cfg.recon_l1_weight,
                edge_weight=self.cfg.recon_edge_weight,
            )
            reconstructed_chunks.append(output.reconstructed.detach().cpu())
            total_recon_loss += float(recon_terms["recon_loss"].item()) * chunk_frames
            total_recon_mse += float(recon_terms["recon_mse"].item()) * chunk_frames
            total_recon_l1 += float(recon_terms["recon_l1"].item()) * chunk_frames
            total_edge_l1 += float(recon_terms["edge_l1"].item()) * chunk_frames
            total_kl += float(output.kl_loss.item()) * chunk_frames
        reconstructed = torch.cat(reconstructed_chunks, dim=0)
        recon_loss = total_recon_loss / total_frames
        recon_mse = total_recon_mse / total_frames
        recon_l1 = total_recon_l1 / total_frames
        edge_l1 = total_edge_l1 / total_frames
        kl_loss = total_kl / total_frames
        ae_loss = recon_loss + self.cfg.kl_beta * kl_loss
        return reconstructed, recon_loss, recon_mse, recon_l1, edge_l1, kl_loss, ae_loss

    @torch.no_grad()
    def _validate_dynamics_one_step(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor | None = None,
        *,
        num_conditional_frames: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Validate dynamics with teacher-forced predictions for one conditioning length."""

        supported_counts = self.model.dynamics.cfg.conditioning_frame_choices
        context_frames = (
            min(supported_counts)
            if num_conditional_frames is None
            else num_conditional_frames
        )
        if context_frames not in supported_counts:
            raise ValueError(
                f"Expected num_conditional_frames from {supported_counts}, received {context_frames}."
            )
        target_frames = self.model.dynamics.cfg.max_frames - context_frames
        if frames.shape[0] < self.model.dynamics.cfg.max_frames:
            raise ValueError(
                f"Dynamics validation requires at least {self.model.dynamics.cfg.max_frames} frames."
            )
        predicted_frames = [frames[:context_frames].detach().cpu()]
        total_frame_squared_error = 0.0
        total_frame_values = 0
        total_latent_squared_error = 0.0
        total_latent_values = 0
        per_target_frame_squared_error = [0.0 for _ in range(target_frames)]
        per_target_frame_values = [0 for _ in range(target_frames)]
        per_target_latent_squared_error = [0.0 for _ in range(target_frames)]
        per_target_latent_values = [0 for _ in range(target_frames)]
        predicted_motion_l1_total = 0.0
        ground_truth_motion_l1_total = 0.0
        motion_value_count = 0
        for target_start in range(context_frames, int(frames.shape[0]), target_frames):
            current_frames = frames[target_start - context_frames : target_start].unsqueeze(0)
            target_stop = min(target_start + target_frames, int(frames.shape[0]))
            target_chunk = frames[target_start:target_stop]
            padded_target_chunk = target_chunk
            if target_chunk.shape[0] < target_frames:
                pad_frame = target_chunk[-1:].expand(target_frames - target_chunk.shape[0], -1, -1, -1)
                padded_target_chunk = torch.cat([target_chunk, pad_frame], dim=0)
            current_latent = self.model.encode_context_frames(current_frames, deterministic=True)
            clean_chunk_frames = torch.cat([current_frames[0], padded_target_chunk], dim=0).unsqueeze(0)
            clean_chunk_latent = self.model.encode_frame_sequence(clean_chunk_frames, deterministic=True)
            target_latent = clean_chunk_latent[:, :, context_frames:]
            action_window = None
            if actions is not None:
                action_start = target_start - context_frames
                action_stop = action_start + self.model.dynamics.cfg.num_action_per_chunk
                action_window = actions[action_start:min(action_stop, int(actions.shape[0]))]
                if action_window.shape[0] < self.model.dynamics.cfg.num_action_per_chunk:
                    pad_actions = torch.zeros(
                        self.model.dynamics.cfg.num_action_per_chunk - action_window.shape[0],
                        self.model.dynamics.cfg.action_dim,
                        device=actions.device,
                        dtype=actions.dtype,
                    )
                    action_window = torch.cat([action_window, pad_actions], dim=0)
                action_window = action_window.unsqueeze(0)
            generator = torch.Generator(device=current_latent.device.type)
            generator.manual_seed(target_start + context_frames * 1000)
            predicted_latent = self.model.predict_next_latent(
                current_latent,
                actions=action_window,
                generator=generator,
            )
            predicted_frame = self.model.decode_frame_sequence(predicted_latent)[0, : target_chunk.shape[0]]
            predicted_frames.append(predicted_frame.detach().cpu())
            total_frame_squared_error += float(
                F.mse_loss(predicted_frame, target_chunk, reduction="sum").item()
            )
            total_frame_values += int(target_chunk.numel())
            total_latent_squared_error += float(
                F.mse_loss(
                    predicted_latent[:, :, : target_chunk.shape[0]],
                    target_latent[:, :, : target_chunk.shape[0]],
                    reduction="sum",
                ).item()
            )
            total_latent_values += int(target_latent[:, :, : target_chunk.shape[0]].numel())
            for offset in range(int(target_chunk.shape[0])):
                per_target_frame_squared_error[offset] += float(
                    F.mse_loss(
                        predicted_frame[offset:offset + 1],
                        target_chunk[offset:offset + 1],
                        reduction="sum",
                    ).item()
                )
                per_target_frame_values[offset] += int(target_chunk[offset:offset + 1].numel())
                per_target_latent_squared_error[offset] += float(
                    F.mse_loss(
                        predicted_latent[:, :, offset:offset + 1],
                        target_latent[:, :, offset:offset + 1],
                        reduction="sum",
                    ).item()
                )
                per_target_latent_values[offset] += int(target_latent[:, :, offset:offset + 1].numel())
            if target_chunk.shape[0] > 1:
                predicted_motion_l1_total += float(
                    torch.abs(predicted_frame[1:] - predicted_frame[:-1]).sum().item()
                )
                ground_truth_motion_l1_total += float(
                    torch.abs(target_chunk[1:] - target_chunk[:-1]).sum().item()
                )
                motion_value_count += int((target_chunk.shape[0] - 1) * target_chunk[0].numel())
        preview_frames = torch.cat(predicted_frames, dim=0)
        stats = {
            "input_frame_count": int(frames.shape[0]),
            "decoded_frame_count": int(preview_frames.shape[0]),
            "predicted_frame_count": int(preview_frames.shape[0]),
            "seed_frames": int(context_frames),
            "loss_frames": int(frames.shape[0] - context_frames),
            "next_frame_mse": total_frame_squared_error / max(total_frame_values, 1),
            "next_latent_mse": total_latent_squared_error / max(total_latent_values, 1),
            "validation_style": (
                f"teacher_forced_{context_frames}_context_{target_frames}_target"
            ),
        }
        for offset in range(target_frames):
            if per_target_frame_values[offset] > 0:
                stats[f"next_frame_mse_target_{offset}"] = (
                    per_target_frame_squared_error[offset] / per_target_frame_values[offset]
                )
            if per_target_latent_values[offset] > 0:
                stats[f"next_latent_mse_target_{offset}"] = (
                    per_target_latent_squared_error[offset] / per_target_latent_values[offset]
                )
        if motion_value_count > 0:
            predicted_motion_l1 = predicted_motion_l1_total / motion_value_count
            ground_truth_motion_l1 = ground_truth_motion_l1_total / motion_value_count
            stats["predicted_target_motion_l1"] = predicted_motion_l1
            stats["ground_truth_target_motion_l1"] = ground_truth_motion_l1
            stats["target_motion_ratio"] = predicted_motion_l1 / max(ground_truth_motion_l1, 1e-12)
        return preview_frames, stats

    @torch.no_grad()
    def _validate_dynamics_open_rollout(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Validate dynamics with a fully autoregressive open rollout."""

        context_frames = self.model.dynamics.cfg.open_rollout_context_frames
        if frames.shape[0] <= context_frames:
            raise ValueError(
                f"Open-rollout validation requires more than {context_frames} frames."
            )
        rollout_steps = int(frames.shape[0]) - context_frames
        seed_frames = frames[:context_frames].unsqueeze(0)
        rollout_actions = None if actions is None else actions.unsqueeze(0)
        predicted = self.model.rollout(
            seed_frames,
            steps=rollout_steps,
            actions=rollout_actions,
        )[0]
        predicted_targets = predicted[context_frames:]
        target_frames = frames[context_frames:]
        predicted_motion_l1 = 0.0
        ground_truth_motion_l1 = 0.0
        if predicted_targets.shape[0] > 1:
            predicted_motion_l1 = float(
                torch.abs(predicted_targets[1:] - predicted_targets[:-1]).mean().item()
            )
            ground_truth_motion_l1 = float(
                torch.abs(target_frames[1:] - target_frames[:-1]).mean().item()
            )
        stats = {
            "open_rollout_seed_frames": int(context_frames),
            "open_rollout_loss_frames": int(rollout_steps),
            "open_rollout_decoded_frame_count": int(predicted.shape[0]),
            "open_rollout_predicted_frame_count": int(predicted.shape[0]),
            "open_rollout_frame_mse": float(F.mse_loss(predicted_targets, target_frames).item()),
            "open_rollout_frame_l1": float(F.l1_loss(predicted_targets, target_frames).item()),
            "open_rollout_validation_style": "open_rollout_autoregressive",
        }
        if predicted_targets.shape[0] > 1:
            stats["open_rollout_predicted_target_motion_l1"] = predicted_motion_l1
            stats["open_rollout_ground_truth_target_motion_l1"] = ground_truth_motion_l1
            stats["open_rollout_target_motion_ratio"] = (
                predicted_motion_l1 / max(ground_truth_motion_l1, 1e-12)
            )
        return predicted.detach().cpu(), stats

    def _suffix_validation_stats(
        self,
        stats: dict[str, Any],
        suffix: str,
        *,
        excluded_keys: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Return one validation dictionary with every metric renamed by suffix."""

        excluded = frozenset() if excluded_keys is None else excluded_keys
        return {
            f"{key}_{suffix}": value
            for key, value in stats.items()
            if key not in excluded
        }

    @torch.no_grad()
    def _validate(self, step: int) -> dict[str, Any]:
        """Run validation, export artifacts, and update the best checkpoint."""

        self.model.eval()
        batch = self._move_batch_to_device(next(iter(self.val_loader)))
        frames = batch["frames"][0]
        if self.cfg.mode == "ae_only":
            reconstructed, recon_loss, recon_mse, recon_l1, edge_l1, kl_loss, ae_loss = self._validate_ae_only_frames(frames)
            stats = {
                "episode": int(batch["episode_idx"].reshape(-1)[0].item()),
                "input_frame_count": int(frames.shape[0]),
                "decoded_frame_count": int(reconstructed.shape[0]),
                "recon_loss": float(recon_loss),
                "recon_mse": float(recon_mse),
                "recon_l1": float(recon_l1),
                "edge_l1": float(edge_l1),
                "kl_loss": kl_loss,
                "ae_loss": ae_loss,
                "mode": self.cfg.mode,
                "ae_backend": self.cfg.ae_backend,
                "dynamics_backend": self.model.dynamics_backend,
            }
            preview_frames = reconstructed
            context_frames = 0
        else:
            clip_actions = batch.get("actions")
            validation_context_choices = self.model.dynamics.cfg.validation_conditioning_frame_choices
            primary_context_frames = validation_context_choices[0]
            preview_frames, dynamics_stats = self._validate_dynamics_one_step(
                frames,
                actions=None if clip_actions is None else clip_actions[0],
                num_conditional_frames=primary_context_frames,
            )
            auxiliary_stats = {}
            for validation_context_frames in validation_context_choices[1:]:
                _, extra_context_stats = self._validate_dynamics_one_step(
                    frames,
                    actions=None if clip_actions is None else clip_actions[0],
                    num_conditional_frames=validation_context_frames,
                )
                suffix = (
                    f"{validation_context_frames}"
                    f"to{self.model.dynamics.cfg.max_frames - validation_context_frames}"
                )
                auxiliary_stats.update(
                    self._suffix_validation_stats(
                        extra_context_stats,
                        suffix,
                        excluded_keys=frozenset(
                            {"input_frame_count", "decoded_frame_count", "predicted_frame_count"}
                        ),
                    )
                )
                auxiliary_stats[f"validation_style_{suffix}"] = extra_context_stats["validation_style"]
            _, open_rollout_stats = self._validate_dynamics_open_rollout(
                frames,
                actions=None if clip_actions is None else clip_actions[0],
            )
            stats = {
                "episode": int(batch["episode_idx"].reshape(-1)[0].item()),
                **dynamics_stats,
                **self._suffix_validation_stats(dynamics_stats, f"{primary_context_frames}to{self.model.dynamics.cfg.max_frames - primary_context_frames}"),
                **auxiliary_stats,
                **open_rollout_stats,
                "conditioning_frame_choices": list(self.model.dynamics.cfg.conditioning_frame_choices),
                "conditioning_frame_probabilities": (
                    None
                    if self.model.dynamics.cfg.conditioning_frame_probabilities is None
                    else list(self.model.dynamics.cfg.conditioning_frame_probabilities)
                ),
                "validation_conditioning_frame_choices": list(
                    self.model.dynamics.cfg.validation_conditioning_frame_choices
                ),
                "open_rollout_context_frames": int(self.model.dynamics.cfg.open_rollout_context_frames),
                "mode": self.cfg.mode,
                "ae_backend": self.cfg.ae_backend,
                "dynamics_backend": self.model.dynamics_backend,
            }
            context_frames = primary_context_frames

        output_dir = self.run_dir / "samples" / f"step_{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "episode_0_grid.png"
        video_path = output_dir / "episode_0.mp4"
        stats_path = output_dir / "episode_0_stats.json"
        build_side_by_side_grid(
            original=frames.detach().cpu(),
            reconstructed=preview_frames.detach().cpu(),
            max_frames=int(frames.shape[0]),
            context_frames=context_frames,
        ).save(grid_path)
        exported_frame_count = write_side_by_side_mp4(
            original=frames.detach().cpu(),
            reconstructed=preview_frames.detach().cpu(),
            output_path=video_path,
            duration_ms=120,
            context_frames=context_frames,
        )
        stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
        stats["elapsed_run_seconds"] = self._elapsed_run_seconds()
        stats["exported_video_frame_count"] = int(exported_frame_count)
        save_json(stats_path, stats)
        metric_name = self._validation_metric_name()
        metric_value = float(stats[metric_name])
        if self.best_metric is None or metric_value < self.best_metric:
            self.best_metric = metric_value
            self._save_checkpoint(self.checkpoints_dir / "best.pt", step)
        append_jsonl(self.metrics_path, {"step": step, "validation": stats})
        return stats

    def _save_checkpoint(self, path: str | Path, step: int) -> None:
        """Save one world-model checkpoint to disk."""

        save_training_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            config=self.cfg.to_dict(),
            mode=self.cfg.mode,
            clip_metadata=self.cfg.clip_metadata(),
            best_metric=self.best_metric,
        )

    def _handle_validation_early_stop(
        self,
        step: int,
        validation_stats: dict[str, Any],
    ) -> bool:
        """Record one validation observation and return whether training should stop."""

        early_stop_record = self._record_metric_for_early_stop(
            step,
            float(validation_stats[self._validation_metric_name()]),
        )
        if early_stop_record is None:
            return False

        append_jsonl(
            self.metrics_path,
            {"step": step, "early_stop": early_stop_record},
        )
        if early_stop_record["improved"] or early_stop_record["should_stop"]:
            print(
                json.dumps(
                    {"step": step, "early_stop": early_stop_record},
                    sort_keys=True,
                )
            )
        if not early_stop_record["should_stop"]:
            return False

        self._save_checkpoint(self.checkpoints_dir / "last.pt", step)
        stopped_record = {
            "step": step,
            "stopped": {
                "reason": "plateau",
                "best_window_loss": early_stop_record["best_window_loss"],
                "window_loss": early_stop_record["window_loss"],
                "non_improving_windows": early_stop_record["non_improving_windows"],
            },
        }
        append_jsonl(self.metrics_path, stopped_record)
        print(json.dumps(stopped_record, sort_keys=True))
        return True

    def run(self) -> None:
        """Execute the configured training loop end to end."""

        self.run_started_at_monotonic = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.run_dir / "config.json", self.cfg.to_dict())
        self._restore_early_stop_state(self.current_step)
        append_jsonl(
            self.metrics_path,
            {
                "run_start": {
                    "config": self.cfg.to_dict(),
                    "mode": self.cfg.mode,
                    "clip_metadata": self.cfg.clip_metadata(),
                }
            },
        )

        train_iterator = iter(self.train_loader)
        last_validation_step = -1
        while self.current_step < self.cfg.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)
            batch = self._move_batch_to_device(batch)
            self.model.train()
            loss_dict: dict[str, torch.Tensor] | None = None
            try:
                loss_dict = self._execute_training_step(batch)
            except RuntimeError as error:
                if not self._is_cuda_oom(error):
                    raise
                recovered = self._reduce_batch_size_after_oom()
                batch = None
                loss_dict = None
                error = None
                self._cleanup_after_cuda_oom()
                if not recovered:
                    raise
                train_iterator = iter(self.train_loader)
                continue
            batch = None
            self.current_step += 1

            metric_record = {"step": self.current_step, "loss": float(loss_dict["loss"].detach().cpu())}
            for key, value in loss_dict.items():
                if key == "loss":
                    continue
                metric_record[key] = float(value.detach().cpu())
            append_jsonl(self.metrics_path, metric_record)
            if self.current_step == 1 or (
                self.cfg.log_interval > 0 and self.current_step % self.cfg.log_interval == 0
            ):
                print(json.dumps(metric_record, sort_keys=True))

            if (
                self.cfg.validation_interval > 0
                and self.current_step % self.cfg.validation_interval == 0
            ):
                validation_stats = self._validate(self.current_step)
                last_validation_step = self.current_step
                print(json.dumps({"step": self.current_step, "validation": validation_stats}, sort_keys=True))
                if self._handle_validation_early_stop(self.current_step, validation_stats):
                    return

            if (
                self.cfg.checkpoint_interval > 0
                and self.current_step % self.cfg.checkpoint_interval == 0
            ):
                self._save_checkpoint(self.checkpoints_dir / "last.pt", self.current_step)

        if last_validation_step != self.current_step:
            validation_stats = self._validate(self.current_step)
            print(json.dumps({"step": self.current_step, "validation": validation_stats}, sort_keys=True))
            self._handle_validation_early_stop(self.current_step, validation_stats)
        self._save_checkpoint(self.checkpoints_dir / "last.pt", self.current_step)
