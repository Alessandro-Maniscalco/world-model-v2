"""Plain PyTorch experiment runner for the upstream-shaped Stage-1 package."""

from __future__ import annotations

from collections import deque
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model_v2.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from world_model_v2.config import RunConfig
from world_model_v2.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset
from world_model_v2.utils.checkpointing import append_jsonl, load_checkpoint, save_checkpoint, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_gif


class LatentDynamicsExperiment:
    """Run Stage-1 training, checkpointing, and validation previews."""

    def __init__(self, cfg: RunConfig) -> None:
        """Build datasets, model, optimizer, and filesystem paths."""

        self.cfg = cfg
        self._set_seed(cfg.experiment.seed)
        self.device = torch.device(cfg.experiment.device)
        self._validate_device()
        self.run_name = cfg.experiment.run_name or self._default_run_name(cfg.dataset.task)
        self.run_dir = Path(cfg.experiment.output_dir) / self.run_name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.train_dataset = RealAlohaDataset(cfg.dataset)
        self.val_dataset = self.train_dataset.get_validation_dataset()
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.experiment.batch_size,
            shuffle=True,
            num_workers=cfg.experiment.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        self.model = LatentWorldModel(cfg.algorithm, cfg.dataset.obs_keys).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.weight_decay,
        )
        self.early_stop_window_losses: deque[float] = deque(
            maxlen=max(cfg.experiment.early_stop_window_size, 1)
        )
        self.best_window_loss: float | None = None
        self.non_improving_windows = 0
        self.trainable_parameter_count = self._count_trainable_parameters()
        self.normalization_stats = {
            "image_range": [0.0, 1.0],
            **self.train_dataset.compute_action_stats(),
        }

    def _set_seed(self, seed: int) -> None:
        """Seed Python, NumPy, and PyTorch RNGs."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_device(self) -> None:
        """Fail early when the configured CUDA device is unusable."""

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available to PyTorch. Install a torch wheel "
                "compatible with the local NVIDIA driver or rerun with --device cpu."
            )

    def _default_run_name(self, task: str) -> str:
        """Generate a timestamped default run name."""

        return f"{task}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _count_trainable_parameters(self) -> int:
        """Return the number of trainable model parameters."""

        return sum(param.numel() for param in self.model.parameters() if param.requires_grad)

    def _is_step_interval_due(self, step: int, interval: int) -> bool:
        """Return whether a positive step interval is due at the current step."""

        return interval > 0 and step % interval == 0

    def _current_save_preview_interval_seconds(self, elapsed_seconds: float) -> float | None:
        """Return the active wall-clock save interval for the current elapsed time."""

        initial_seconds = self.cfg.experiment.save_preview_initial_minutes * 60.0
        late_seconds = self.cfg.experiment.save_preview_late_minutes * 60.0
        switch_seconds = self.cfg.experiment.save_preview_switch_minutes * 60.0
        if initial_seconds <= 0.0 and late_seconds <= 0.0:
            return None
        if switch_seconds > 0.0 and elapsed_seconds >= switch_seconds:
            if late_seconds > 0.0:
                return late_seconds
            return initial_seconds if initial_seconds > 0.0 else None
        if initial_seconds > 0.0:
            return initial_seconds
        return late_seconds if late_seconds > 0.0 else None

    def _is_time_save_preview_due(
        self,
        elapsed_seconds: float,
        last_save_preview_seconds: float,
    ) -> bool:
        """Return whether the wall-clock save-and-preview schedule is due."""

        interval_seconds = self._current_save_preview_interval_seconds(elapsed_seconds)
        if interval_seconds is None:
            return False
        return elapsed_seconds - last_save_preview_seconds >= interval_seconds

    def _build_startup_record(self) -> dict[str, object]:
        """Summarize the run configuration for logs and terminal output."""

        return {
            "run_name": self.run_name,
            "device": str(self.device),
            "trainable_parameters": int(self.trainable_parameter_count),
            "dataset": {
                "task": self.cfg.dataset.task,
                "split": self.cfg.dataset.split,
                "resolution": self.cfg.dataset.resolution,
                "horizon": self.cfg.dataset.horizon,
                "val_horizon": self.cfg.dataset.val_horizon,
                "obs_keys": list(self.cfg.dataset.obs_keys),
            },
            "algorithm": {
                "training_stage": self.cfg.algorithm.training_stage,
                "latent_channels": self.cfg.algorithm.latent_channels,
                "latent_dim": self.cfg.algorithm.latent_dim,
                "hidden_channels": self.cfg.algorithm.hidden_channels,
                "timesteps": self.cfg.algorithm.timesteps,
                "infer_steps": self.cfg.algorithm.infer_steps,
            },
            "experiment": {
                "batch_size": self.cfg.experiment.batch_size,
                "lr": self.cfg.experiment.lr,
                "checkpoint_interval": self.cfg.experiment.checkpoint_interval,
                "validation_interval": self.cfg.experiment.validation_interval,
                "save_preview_initial_minutes": self.cfg.experiment.save_preview_initial_minutes,
                "save_preview_late_minutes": self.cfg.experiment.save_preview_late_minutes,
                "save_preview_switch_minutes": self.cfg.experiment.save_preview_switch_minutes,
                "early_stop_window_size": self.cfg.experiment.early_stop_window_size,
                "early_stop_patience_windows": self.cfg.experiment.early_stop_patience_windows,
                "early_stop_min_delta": self.cfg.experiment.early_stop_min_delta,
                "early_stop_warmup_steps": self.cfg.experiment.early_stop_warmup_steps,
            },
        }

    def _early_stop_enabled(self) -> bool:
        """Return whether plateau-based early stopping is configured."""

        return (
            self.cfg.experiment.early_stop_window_size > 0
            and self.cfg.experiment.early_stop_patience_windows > 0
        )

    def _record_loss_for_early_stop(
        self,
        step: int,
        loss_value: float,
    ) -> dict[str, object] | None:
        """Update rolling-loss plateau state and return an evaluation record when due."""

        if not self._early_stop_enabled():
            return None
        self.early_stop_window_losses.append(loss_value)
        if len(self.early_stop_window_losses) < self.cfg.experiment.early_stop_window_size:
            return None
        if step % self.cfg.experiment.early_stop_window_size != 0:
            return None

        window_loss = float(sum(self.early_stop_window_losses) / len(self.early_stop_window_losses))
        improved = False
        if step < self.cfg.experiment.early_stop_warmup_steps:
            if self.best_window_loss is None or window_loss < self.best_window_loss:
                self.best_window_loss = window_loss
            self.non_improving_windows = 0
        elif self.best_window_loss is None or (
            window_loss < self.best_window_loss - self.cfg.experiment.early_stop_min_delta
        ):
            self.best_window_loss = window_loss
            self.non_improving_windows = 0
            improved = True
        else:
            self.non_improving_windows += 1

        should_stop = (
            step >= self.cfg.experiment.early_stop_warmup_steps
            and self.non_improving_windows >= self.cfg.experiment.early_stop_patience_windows
        )
        return {
            "step": step,
            "window_loss": window_loss,
            "best_window_loss": self.best_window_loss,
            "improved": improved,
            "non_improving_windows": self.non_improving_windows,
            "patience_windows": self.cfg.experiment.early_stop_patience_windows,
            "min_delta": self.cfg.experiment.early_stop_min_delta,
            "warmup_steps": self.cfg.experiment.early_stop_warmup_steps,
            "should_stop": should_stop,
        }

    def _restore_early_stop_state(self, step: int) -> None:
        """Rebuild plateau state from prior metric logs when resuming a run."""

        if not self._early_stop_enabled() or step <= 0:
            return
        restore_path = self._restore_metrics_path()
        if restore_path is None:
            return
        self.early_stop_window_losses.clear()
        self.best_window_loss = None
        self.non_improving_windows = 0
        with restore_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "loss" not in record:
                    continue
                record_step = int(record["step"])
                if record_step > step:
                    break
                self._record_loss_for_early_stop(record_step, float(record["loss"]))

    def _restore_metrics_path(self) -> Path | None:
        """Find the metric log that should seed plateau state for a resumed run."""

        if self.cfg.experiment.resume:
            resume_run_dir = Path(self.cfg.experiment.resume).resolve().parent.parent
            if resume_run_dir != self.run_dir.resolve():
                resume_metrics = resume_run_dir / "metrics.jsonl"
                if resume_metrics.exists():
                    return resume_metrics
        if self.metrics_path.exists():
            return self.metrics_path
        return None

    def _next_validation_batch(self) -> dict[str, object]:
        """Return the first validation batch for deterministic previews."""

        return next(iter(self.val_loader))

    def _move_batch_to_device(self, batch: dict[str, object]) -> dict[str, object]:
        """Move tensor leaves in a nested batch to the configured device."""

        obs = {
            key: value.to(self.device) for key, value in batch["obs"].items()  # type: ignore[union-attr]
        }
        moved: dict[str, object] = {"obs": obs}
        for key, value in batch.items():
            if key == "obs":
                continue
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _write_validation_preview(self, step: int) -> dict[str, object]:
        """Run validation, export artifacts, and return stats."""

        batch = self._move_batch_to_device(self._next_validation_batch())
        preview = self.model.validation_step(
            batch,
            num_steps=max(4, self.cfg.algorithm.infer_steps),
            start_mode="noisy-input",
        )
        output_dir = self.run_dir / "samples" / f"step_{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "episode_0_grid.png"
        gif_path = output_dir / "episode_0.gif"
        stats_path = output_dir / "episode_0_stats.json"

        grid = build_side_by_side_grid(preview["original"], preview["reconstructed"], max_frames=12)
        grid.save(grid_path)
        exported_frame_count = write_side_by_side_gif(
            preview["original"],
            preview["reconstructed"],
            gif_path,
            duration_ms=120,
        )
        stats = dict(preview["stats"])
        stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
        stats["exported_gif_frame_count"] = int(exported_frame_count)
        if stats["input_frame_count"] != stats["decoded_frame_count"]:
            raise RuntimeError(f"Decoded frame count mismatch: {stats}")
        if stats["decoded_frame_count"] != stats["exported_gif_frame_count"]:
            raise RuntimeError(f"Exported GIF frame count mismatch: {stats}")
        save_json(stats_path, stats)
        return stats

    def _save_checkpoint(self, step: int) -> None:
        """Write a step checkpoint and refresh `last.pt`."""

        step_path = self.checkpoints_dir / f"step_{step:06d}.pt"
        last_path = self.checkpoints_dir / "last.pt"
        payload_config = self.cfg.to_dict()
        save_checkpoint(
            step_path,
            self.model,
            self.optimizer,
            step,
            payload_config,
            self.normalization_stats,
        )
        save_checkpoint(
            last_path,
            self.model,
            self.optimizer,
            step,
            payload_config,
            self.normalization_stats,
        )
        print(f"Saved checkpoint to {last_path}")

    def _load_resume(self) -> int:
        """Restore model and optimizer state when resume is configured."""

        if not self.cfg.experiment.resume:
            return 0
        checkpoint = load_checkpoint(self.cfg.experiment.resume, self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return int(checkpoint["step"])

    def run(self) -> None:
        """Run the Stage-1 training loop and periodic validation previews."""

        step = self._load_resume()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.run_dir / "config.json", self.cfg.to_dict())
        self._restore_early_stop_state(step)
        startup_record = self._build_startup_record()
        append_jsonl(self.metrics_path, {"run_start": startup_record})
        print(json.dumps({"run_start": startup_record}, sort_keys=True))
        run_start_time = time.monotonic()
        last_save_preview_seconds = 0.0

        while step < self.cfg.experiment.max_steps:
            for batch in self.train_loader:
                step_start_time = time.monotonic()
                batch = self._move_batch_to_device(batch)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.model.training_step(batch)
                loss_dict["loss"].backward()
                self.optimizer.step()
                step += 1
                elapsed_seconds = time.monotonic() - run_start_time
                step_time_seconds = time.monotonic() - step_start_time

                metric_record = {
                    "step": step,
                    "elapsed_minutes": elapsed_seconds / 60.0,
                    "step_time_seconds": step_time_seconds,
                    "loss": float(loss_dict["loss"].detach().cpu()),
                    "recon_loss": float(loss_dict["recon_loss"].cpu()),
                    "clean_loss": float(loss_dict["clean_loss"].cpu()),
                }
                append_jsonl(self.metrics_path, metric_record)
                early_stop_record = self._record_loss_for_early_stop(
                    step,
                    metric_record["loss"],
                )
                if early_stop_record is not None:
                    append_jsonl(self.metrics_path, {"step": step, "early_stop": early_stop_record})

                if self._is_step_interval_due(step, self.cfg.experiment.log_interval) or step == 1:
                    print(json.dumps(metric_record, sort_keys=True))
                if early_stop_record is not None and (
                    early_stop_record["improved"] or early_stop_record["should_stop"]
                ):
                    print(json.dumps({"step": step, "early_stop": early_stop_record}, sort_keys=True))

                time_save_preview_due = self._is_time_save_preview_due(
                    elapsed_seconds,
                    last_save_preview_seconds,
                )
                early_stop_due = bool(
                    early_stop_record is not None and early_stop_record["should_stop"]
                )
                if (
                    self._is_step_interval_due(step, self.cfg.experiment.checkpoint_interval)
                    or time_save_preview_due
                    or early_stop_due
                    or step == self.cfg.experiment.max_steps
                ):
                    self._save_checkpoint(step)

                if (
                    self._is_step_interval_due(step, self.cfg.experiment.validation_interval)
                    or time_save_preview_due
                    or early_stop_due
                    or step == self.cfg.experiment.max_steps
                ):
                    stats = self._write_validation_preview(step)
                    append_jsonl(self.metrics_path, {"step": step, "validation": stats})
                    print(json.dumps({"step": step, "validation": stats}, sort_keys=True))
                if time_save_preview_due:
                    last_save_preview_seconds = time.monotonic() - run_start_time

                if early_stop_due:
                    append_jsonl(
                        self.metrics_path,
                        {
                            "step": step,
                            "stopped": {
                                "reason": "plateau",
                                "best_window_loss": early_stop_record["best_window_loss"],
                                "window_loss": early_stop_record["window_loss"],
                                "non_improving_windows": early_stop_record["non_improving_windows"],
                            },
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "stopped": {
                                    "reason": "plateau",
                                    "best_window_loss": early_stop_record["best_window_loss"],
                                    "window_loss": early_stop_record["window_loss"],
                                    "non_improving_windows": early_stop_record["non_improving_windows"],
                                },
                            },
                            sort_keys=True,
                        )
                    )
                    return

                if step >= self.cfg.experiment.max_steps:
                    break
