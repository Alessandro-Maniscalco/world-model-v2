"""Minimal experiment runner for the single-clip VAE-plus-dynamics world model."""

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

from world_model_v2.minimal.dataset import (
    MinimalFrameDataset,
    MinimalTransitionDataset,
    MinimalValidationClipDataset,
)
from world_model_v2.minimal.metaworld_dataset import (
    METAWORLD_DATASET_ID,
    MetaWorldFrameDataset,
    MetaWorldTransitionDataset,
    MetaWorldValidationClipDataset,
)
from world_model_v2.minimal.model import MinimalWorldModel
from world_model_v2.utils.checkpointing import append_jsonl, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


MINIMAL_CHECKPOINT_KIND = "world_model_v2_minimal_v2"
LEGACY_MINIMAL_CHECKPOINT_KIND = "world_model_v2_minimal_v1"
AUTO_BATCH_SIZE_BACKOFF_DIVISOR = 2


@dataclass
class MinimalExperimentConfig:
    """Group CLI-configurable settings for the minimal experiment."""

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
    output_dir: str = "outputs/minimal"
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


def save_minimal_checkpoint(
    path: str | Path,
    model: MinimalWorldModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    step: int,
    config: dict[str, Any],
    mode: str,
    clip_metadata: dict[str, Any],
    best_metric: float | None,
) -> None:
    """Save a minimal-pipeline checkpoint with reproducibility metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": MINIMAL_CHECKPOINT_KIND,
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
    }
    torch.save(payload, output_path)


def load_minimal_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load and validate one minimal-pipeline checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("kind") not in {MINIMAL_CHECKPOINT_KIND, LEGACY_MINIMAL_CHECKPOINT_KIND}:
        raise ValueError(f"{path} is not a minimal world-model checkpoint.")
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


class MinimalExperiment:
    """Train and validate the minimal world model in AE-only or dynamics-only mode."""

    def __init__(self, cfg: MinimalExperimentConfig) -> None:
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
        self.model = MinimalWorldModel(
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
            ae_backend=cfg.ae_backend,
            resolution=cfg.resolution,
            height=cfg.height,
            width=cfg.width,
        ).to(self.device)
        self._load_requested_pretrained_weights()
        self.model.configure_trainability(cfg.mode)
        self.train_dataset = self._build_train_dataset()
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
                "The minimal path now only supports the Wan VAE."
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

    def _default_run_name(self, mode: str) -> str:
        """Return a timestamped default run name."""

        return f"minimal_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _early_stop_enabled(self) -> bool:
        """Return whether validation plateau stopping is enabled."""

        return self.cfg.early_stop_window_size > 0 and self.cfg.early_stop_patience_windows > 0

    def _build_train_dataset(self) -> Dataset[dict[str, Any]]:
        """Build the mode-specific training dataset."""

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
            return MinimalFrameDataset(**dataset_kwargs)
        return MinimalTransitionDataset(**dataset_kwargs)

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
        dataset = MinimalValidationClipDataset(
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

    def _load_requested_pretrained_weights(self) -> None:
        """Load any optional encoder/decoder or dynamics checkpoint weights."""

        if self.cfg.load_encoder_decoder:
            checkpoint = load_minimal_checkpoint(self.cfg.load_encoder_decoder, self.device)
            self._assert_checkpoint_backend(checkpoint, self.cfg.load_encoder_decoder)
            self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
            self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
        if self.cfg.load_dynamics:
            checkpoint = load_minimal_checkpoint(self.cfg.load_dynamics, self.device)
            self._assert_checkpoint_backend(checkpoint, self.cfg.load_dynamics)
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

        checkpoint = load_minimal_checkpoint(self.cfg.resume, self.device)
        self._assert_checkpoint_backend(checkpoint, self.cfg.resume)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        if checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_metric = checkpoint.get("best_metric")
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
        return "rollout_mse"

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
        """Run one frozen-autoencoder latent dynamics step."""

        current_frame = batch["current_frame"]
        next_frame = batch["next_frame"]
        with torch.no_grad():
            current_latent = self.model.encode(current_frame, deterministic=True)
            target_latent = self.model.encode(next_frame, deterministic=True)
        predicted_latent = self.model.predict_next_latent(current_latent)
        latent_mse = F.mse_loss(predicted_latent, target_latent)
        return {"loss": latent_mse, "latent_mse": latent_mse.detach()}

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
            }
            preview_frames = reconstructed
            context_frames = 0
        else:
            rollout = self.model.rollout(frames[:1], steps=frames.shape[0] - 1)[0]
            rollout_mse = F.mse_loss(rollout[1:], frames[1:]).item()
            stats = {
                "episode": int(batch["episode_idx"].reshape(-1)[0].item()),
                "input_frame_count": int(frames.shape[0]),
                "decoded_frame_count": int(rollout.shape[0]),
                "predicted_frame_count": int(rollout.shape[0]),
                "rollout_mse": float(rollout_mse),
                "seed_frames": 1,
                "loss_frames": int(frames.shape[0] - 1),
                "mode": self.cfg.mode,
                "ae_backend": self.cfg.ae_backend,
            }
            preview_frames = rollout
            context_frames = 1

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
        """Save one minimal checkpoint to disk."""

        save_minimal_checkpoint(
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
