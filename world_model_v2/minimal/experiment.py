"""Minimal experiment runner for the single-clip multi-mode world model."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from world_model_v2.minimal.dataset import (
    MinimalFrameDataset,
    MinimalTransitionDataset,
    MinimalValidationClipDataset,
)
from world_model_v2.minimal.model import MinimalWorldModel
from world_model_v2.utils.checkpointing import append_jsonl, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_gif


MINIMAL_CHECKPOINT_KIND = "world_model_v2_minimal_v1"


@dataclass
class MinimalExperimentConfig:
    """Group CLI-configurable settings for the minimal experiment."""

    mode: str = "joint"
    data_root: str = "data/full"
    task: str = "single_grasp"
    split: str = "val"
    episode: int = 0
    camera: str = "camera_1_color"
    frame_start: int = 111
    frame_end: int = 116
    resolution: int = 128
    latent_channels: int = 4
    hidden_channels: int = 64
    batch_size: int = 32
    lr: float = 1e-3
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
            "task": self.task,
            "split": self.split,
            "episode": self.episode,
            "camera": self.camera,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "resolution": self.resolution,
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
    }
    torch.save(payload, output_path)


def load_minimal_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load and validate one minimal-pipeline checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("kind") != MINIMAL_CHECKPOINT_KIND:
        raise ValueError(f"{path} is not a minimal world-model checkpoint.")
    return checkpoint


class MinimalExperiment:
    """Train and validate the minimal world model in one of three modes."""

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
        ).to(self.device)
        self._load_requested_pretrained_weights()
        self.model.configure_trainability(cfg.mode)
        self.train_loader = self._build_train_loader()
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

        if self.cfg.mode not in {"joint", "ae_only", "dynamics_only"}:
            raise ValueError(f"Unsupported mode: {self.cfg.mode}")
        if self.cfg.frame_end < self.cfg.frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start.")
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

    def _default_run_name(self, mode: str) -> str:
        """Return a timestamped default run name."""

        return f"minimal_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _early_stop_enabled(self) -> bool:
        """Return whether validation plateau stopping is enabled."""

        return (
            self.cfg.early_stop_window_size > 0
            and self.cfg.early_stop_patience_windows > 0
        )

    def _build_train_loader(self) -> DataLoader[Any]:
        """Build the mode-specific training dataloader."""

        dataset_kwargs = {
            "data_root": self.cfg.data_root,
            "task": self.cfg.task,
            "split": self.cfg.split,
            "episode": self.cfg.episode,
            "camera": self.cfg.camera,
            "frame_start": self.cfg.frame_start,
            "frame_end": self.cfg.frame_end,
            "resolution": self.cfg.resolution,
        }
        if self.cfg.mode == "ae_only":
            dataset = MinimalFrameDataset(**dataset_kwargs)
        else:
            dataset = MinimalTransitionDataset(**dataset_kwargs)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0)

    def _build_val_loader(self) -> DataLoader[Any]:
        """Build the full-clip validation dataloader."""

        dataset = MinimalValidationClipDataset(
            data_root=self.cfg.data_root,
            task=self.cfg.task,
            split=self.cfg.split,
            episode=self.cfg.episode,
            camera=self.cfg.camera,
            frame_start=self.cfg.frame_start,
            frame_end=self.cfg.frame_end,
            resolution=self.cfg.resolution,
        )
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer over trainable parameters."""

        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters were selected for the current mode.")
        return torch.optim.AdamW(parameters, lr=self.cfg.lr)

    def _load_requested_pretrained_weights(self) -> None:
        """Load any optional encoder/decoder or dynamics checkpoint weights."""

        if self.cfg.load_encoder_decoder:
            checkpoint = load_minimal_checkpoint(self.cfg.load_encoder_decoder, self.device)
            self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
            self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
        if self.cfg.load_dynamics:
            checkpoint = load_minimal_checkpoint(self.cfg.load_dynamics, self.device)
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
            return "recon_mse"
        return "rollout_mse"

    def _train_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Dispatch to the mode-specific training objective."""

        if self.cfg.mode == "ae_only":
            return self._ae_only_training_step(batch)
        if self.cfg.mode == "joint":
            return self._joint_training_step(batch)
        return self._dynamics_only_training_step(batch)

    def _ae_only_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one autoencoder-only reconstruction step."""

        frames = batch["frame"]
        reconstructed = self.model.reconstruct(frames)
        recon_mse = F.mse_loss(reconstructed, frames)
        return {"loss": recon_mse, "recon_mse": recon_mse.detach()}

    def _joint_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one joint encoder-dynamics-decoder training step."""

        current_frame = batch["current_frame"]
        next_frame = batch["next_frame"]
        current_latent = self.model.encode(current_frame)
        reconstructed = self.model.decode(current_latent)
        predicted_next = self.model.decode(self.model.predict_next_latent(current_latent))
        pred_mse = F.mse_loss(predicted_next, next_frame)
        recon_mse = F.mse_loss(reconstructed, current_frame)
        loss = pred_mse + 0.25 * recon_mse
        return {
            "loss": loss,
            "pred_mse": pred_mse.detach(),
            "recon_mse": recon_mse.detach(),
        }

    def _dynamics_only_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one frozen-autoencoder latent dynamics step."""

        current_frame = batch["current_frame"]
        next_frame = batch["next_frame"]
        with torch.no_grad():
            current_latent = self.model.encode(current_frame)
            target_latent = self.model.encode(next_frame)
        predicted_latent = self.model.predict_next_latent(current_latent)
        latent_mse = F.mse_loss(predicted_latent, target_latent)
        return {"loss": latent_mse, "latent_mse": latent_mse.detach()}

    @torch.no_grad()
    def _validate(self, step: int) -> dict[str, Any]:
        """Run validation, export artifacts, and update the best checkpoint."""

        self.model.eval()
        batch = self._move_batch_to_device(next(iter(self.val_loader)))
        frames = batch["frames"][0]
        if self.cfg.mode == "ae_only":
            reconstructed = self.model.reconstruct(frames)
            stats = {
                "episode": int(batch["episode_idx"].reshape(-1)[0].item()),
                "input_frame_count": int(frames.shape[0]),
                "decoded_frame_count": int(reconstructed.shape[0]),
                "recon_mse": float(F.mse_loss(reconstructed, frames).item()),
                "mode": self.cfg.mode,
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
            }
            preview_frames = rollout
            context_frames = 1

        output_dir = self.run_dir / "samples" / f"step_{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "episode_0_grid.png"
        gif_path = output_dir / "episode_0.gif"
        stats_path = output_dir / "episode_0_stats.json"
        build_side_by_side_grid(
            original=frames.detach().cpu(),
            reconstructed=preview_frames.detach().cpu(),
            max_frames=int(frames.shape[0]),
            context_frames=context_frames,
        ).save(grid_path)
        exported_frame_count = write_side_by_side_gif(
            original=frames.detach().cpu(),
            reconstructed=preview_frames.detach().cpu(),
            output_path=gif_path,
            duration_ms=120,
            context_frames=context_frames,
        )
        stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
        stats["exported_gif_frame_count"] = int(exported_frame_count)
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
            self.optimizer.zero_grad(set_to_none=True)
            loss_dict = self._train_step(batch)
            loss_dict["loss"].backward()
            self.optimizer.step()
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
