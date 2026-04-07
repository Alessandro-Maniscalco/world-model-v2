"""Plain PyTorch experiment runner for the upstream-shaped three-stage package."""

from __future__ import annotations

from collections import deque
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from world_model_v2.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from world_model_v2.config import RunConfig
from world_model_v2.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset
from world_model_v2.utils.checkpointing import append_jsonl, load_checkpoint, save_checkpoint, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


class LatentDynamicsExperiment:
    """Run Stage 1, 2, and 3 training, checkpointing, and validation previews."""

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
        self._validate_stage_requirements()
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
        self._validate_action_dim()
        self.model = LatentWorldModel(cfg.algorithm, cfg.dataset.obs_keys).to(self.device)
        if cfg.algorithm.load_ae:
            self.model.bootstrap_from_checkpoint(cfg.algorithm.load_ae, self.device)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self.early_stop_window_losses: deque[float] = deque(
            maxlen=max(cfg.experiment.early_stop_window_size, 1)
        )
        self.best_window_loss: float | None = None
        self.non_improving_windows = 0
        self.early_stop_observations = 0
        self.trainable_parameter_count = self._count_trainable_parameters()
        self.normalization_stats = {
            "image_range": [0.0, 1.0],
            **self.train_dataset.compute_action_stats(),
        }
        self.model.set_normalization_stats(self.normalization_stats)

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

    def _validate_stage_requirements(self) -> None:
        """Fail early when the selected stage configuration is incomplete."""

        if self.cfg.algorithm.training_stage == 2 and self.cfg.dataset.horizon < 2:
            raise ValueError("Stage 2 training requires dataset horizon >= 2.")
        if self.cfg.algorithm.training_stage in (2, 3) and not self.cfg.algorithm.load_ae:
            raise ValueError("Stage 2 and 3 require cfg.algorithm.load_ae.")

    def _validate_action_dim(self) -> None:
        """Check that the configured action dimension matches the dataset."""

        sample = self.train_dataset[0]
        observed_action_dim = int(torch.as_tensor(sample["action"]).shape[-1])
        if observed_action_dim != self.cfg.algorithm.action_dim:
            raise ValueError(
                f"Configured action_dim={self.cfg.algorithm.action_dim} does not match dataset "
                f"action dim={observed_action_dim}."
            )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer over the currently trainable parameter set."""

        parameters = [param for param in self.model.parameters() if param.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters were selected for the current stage.")
        return torch.optim.AdamW(
            parameters,
            lr=self.cfg.experiment.lr,
            weight_decay=self.cfg.experiment.weight_decay,
        )

    def _build_lr_scheduler(self) -> LinearLR | None:
        """Build the configured learning-rate scheduler when requested."""

        if self.cfg.experiment.lr_scheduler == "none":
            return None
        if self.cfg.experiment.lr_scheduler == "linear":
            return LinearLR(
                self.optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=max(1, self.cfg.experiment.warmup_steps),
            )
        raise ValueError(f"Unsupported lr_scheduler={self.cfg.experiment.lr_scheduler}")

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
                "dyn_infer_steps": self.cfg.algorithm.dyn_infer_steps,
                "load_ae": self.cfg.algorithm.load_ae,
                "action_dim": self.cfg.algorithm.action_dim,
                "dynamics_hidden_channels": self.cfg.algorithm.dynamics_hidden_channels,
                "action_emb_dim": self.cfg.algorithm.action_emb_dim,
                "dynamics_attention_heads": self.cfg.algorithm.dynamics_attention_heads,
                "mask_prev_action": self.cfg.algorithm.mask_prev_action,
                "sampling_strategy": self.cfg.algorithm.sampling_strategy,
                "prev_frame_noise_scale": self.cfg.algorithm.prev_frame_noise_scale,
                "last_frame_loss_only": self.cfg.algorithm.last_frame_loss_only,
                "loss_weighting": self.cfg.algorithm.loss_weighting,
                "stage3_latent_noise_std": self.cfg.algorithm.stage3_latent_noise_std,
            },
            "experiment": {
                "batch_size": self.cfg.experiment.batch_size,
                "lr": self.cfg.experiment.lr,
                "grad_clip_norm": self.cfg.experiment.grad_clip_norm,
                "lr_scheduler": self.cfg.experiment.lr_scheduler,
                "warmup_steps": self.cfg.experiment.warmup_steps,
                "checkpoint_interval": self.cfg.experiment.checkpoint_interval,
                "validation_interval": self.cfg.experiment.validation_interval,
                "save_preview_initial_minutes": self.cfg.experiment.save_preview_initial_minutes,
                "save_preview_late_minutes": self.cfg.experiment.save_preview_late_minutes,
                "save_preview_switch_minutes": self.cfg.experiment.save_preview_switch_minutes,
                "early_stop_metric": self.cfg.experiment.early_stop_metric,
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

    def _record_metric_for_early_stop(
        self,
        step: int,
        metric_value: float,
        metric_name: str,
    ) -> dict[str, object] | None:
        """Update plateau state for one metric observation and return an evaluation record when due."""

        if not self._early_stop_enabled():
            return None
        if metric_name != self.cfg.experiment.early_stop_metric:
            return None
        self.early_stop_observations += 1
        self.early_stop_window_losses.append(metric_value)
        if len(self.early_stop_window_losses) < self.cfg.experiment.early_stop_window_size:
            return None
        if self.early_stop_observations % self.cfg.experiment.early_stop_window_size != 0:
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
            "metric": metric_name,
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
                self._record_metric_for_early_stop(
                    record_step,
                    metric_value,
                    self.cfg.experiment.early_stop_metric,
                )

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

    def _extract_early_stop_metric_value(self, record: dict[str, object]) -> float | None:
        """Extract the configured early-stop metric from one JSONL record."""

        metric_name = self.cfg.experiment.early_stop_metric
        if metric_name == "training_loss":
            if "loss" not in record:
                return None
            return float(record["loss"])
        if metric_name == "validation_dyn_loss":
            validation = record.get("validation")
            if not isinstance(validation, dict) or "dyn_loss" not in validation:
                return None
            return float(validation["dyn_loss"])
        raise ValueError(f"Unsupported early_stop_metric={metric_name}")

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

        self.model.eval()
        preview: dict[str, object] | None = None
        dyn_losses: list[float] = []
        for batch_idx, batch in enumerate(self.val_loader):
            moved_batch = self._move_batch_to_device(batch)
            current_preview = self.model.validation_step(
                moved_batch,
                num_steps=max(4, self.cfg.algorithm.infer_steps),
                start_mode="noisy-input",
                rollout_context_size=self.cfg.dataset.horizon,
            )
            if batch_idx == 0:
                preview = current_preview
            current_stats = current_preview["stats"]
            if isinstance(current_stats, dict) and "dyn_loss" in current_stats:
                dyn_losses.append(float(current_stats["dyn_loss"]))
        if preview is None:
            raise RuntimeError("Validation loader produced no batches.")
        output_dir = self.run_dir / "samples" / f"step_{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "episode_0_grid.png"
        video_path = output_dir / "episode_0.mp4"
        stats_path = output_dir / "episode_0_stats.json"

        stats = dict(preview["stats"])
        if dyn_losses:
            stats["dyn_loss"] = float(sum(dyn_losses) / len(dyn_losses))
        context_frames = int(stats.get("context_frames", 0))
        grid = build_side_by_side_grid(
            preview["original"],
            preview["reconstructed"],
            max_frames=12,
            context_frames=context_frames,
        )
        grid.save(grid_path)
        exported_frame_count = write_side_by_side_mp4(
            preview["original"],
            preview["reconstructed"],
            video_path,
            duration_ms=120,
            context_frames=context_frames,
        )
        stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
        stats["exported_video_frame_count"] = int(exported_frame_count)
        if "predicted_frame_count" in stats and stats["input_frame_count"] != stats["predicted_frame_count"]:
            raise RuntimeError(f"Predicted frame count mismatch: {stats}")
        if stats["input_frame_count"] != stats["decoded_frame_count"]:
            raise RuntimeError(f"Decoded frame count mismatch: {stats}")
        if stats["decoded_frame_count"] != stats["exported_video_frame_count"]:
            raise RuntimeError(f"Exported video frame count mismatch: {stats}")
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
            self.lr_scheduler,
            step,
            payload_config,
            self.normalization_stats,
        )
        save_checkpoint(
            last_path,
            self.model,
            self.optimizer,
            self.lr_scheduler,
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
        if self.lr_scheduler is not None and checkpoint.get("scheduler_state") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        return int(checkpoint["step"])

    def run(self) -> None:
        """Run the configured stage training loop and periodic validation previews."""

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
                if self.cfg.experiment.grad_clip_norm > 0.0:
                    clip_grad_norm_(
                        [param for param in self.model.parameters() if param.requires_grad],
                        self.cfg.experiment.grad_clip_norm,
                    )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                step += 1
                elapsed_seconds = time.monotonic() - run_start_time
                step_time_seconds = time.monotonic() - step_start_time

                metric_record = {
                    "step": step,
                    "elapsed_minutes": elapsed_seconds / 60.0,
                    "step_time_seconds": step_time_seconds,
                    "loss": float(loss_dict["loss"].detach().cpu()),
                }
                for key, value in loss_dict.items():
                    if key == "loss" or not isinstance(value, torch.Tensor) or value.ndim != 0:
                        continue
                    metric_record[key] = float(value.detach().cpu())
                append_jsonl(self.metrics_path, metric_record)
                early_stop_record: dict[str, object] | None = None
                if self.cfg.experiment.early_stop_metric == "training_loss":
                    early_stop_record = self._record_metric_for_early_stop(
                        step,
                        metric_record["loss"],
                        "training_loss",
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
                checkpoint_due = (
                    self._is_step_interval_due(step, self.cfg.experiment.checkpoint_interval)
                    or time_save_preview_due
                    or step == self.cfg.experiment.max_steps
                )
                validation_due = (
                    self._is_step_interval_due(step, self.cfg.experiment.validation_interval)
                    or time_save_preview_due
                    or step == self.cfg.experiment.max_steps
                )
                if (
                    early_stop_record is not None
                    and early_stop_record["should_stop"]
                    and self.cfg.experiment.early_stop_metric == "training_loss"
                ):
                    validation_due = True

                if validation_due:
                    stats = self._write_validation_preview(step)
                    append_jsonl(self.metrics_path, {"step": step, "validation": stats})
                    print(json.dumps({"step": step, "validation": stats}, sort_keys=True))
                    if self.cfg.experiment.early_stop_metric == "validation_dyn_loss":
                        early_stop_record = self._record_metric_for_early_stop(
                            step,
                            float(stats["dyn_loss"]),
                            "validation_dyn_loss",
                        )
                        if early_stop_record is not None:
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
                early_stop_due = bool(
                    early_stop_record is not None and early_stop_record["should_stop"]
                )
                if checkpoint_due or early_stop_due:
                    self._save_checkpoint(step)
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
