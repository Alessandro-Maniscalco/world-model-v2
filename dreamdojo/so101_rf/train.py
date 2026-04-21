"""Train a standalone SO101 `1 -> 3` DreamDojo-style rectified-flow DiT."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dreamdojo.so101_rf.checkpointing import append_jsonl, save_json
from dreamdojo.so101_rf.dataset import (
    SO101_RELATIVE_ACTION_SCALE,
    So101TransitionDataset,
    So101ValidationClipDataset,
)
from dreamdojo.so101_rf.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from dreamdojo.so101_rf.model import WorldModel
from dreamdojo.so101_rf.runtime import (
    DREAMDOJO_UPSTREAM_COMMIT,
    load_wan_weights_into_model,
    resolved_training_autocast_dtype,
    run_teacher_forced_validation,
    save_training_checkpoint,
)
from dreamdojo.so101_rf.visualization import build_side_by_side_grid, write_side_by_side_mp4
from dreamdojo.so101_rf.wan_vae import (
    DEFAULT_WAN_DIM,
    DEFAULT_WAN_NUM_RES_BLOCKS,
    DEFAULT_WAN_Z_DIM,
    WanVAEConfig,
)


DEFAULT_VALIDATION_METRIC = "worst_case_next_frame_mse"


@dataclass
class TrainConfig:
    """Configure the standalone SO101 rectified-flow training loop."""

    data_root: str = "data/so101_base_sim_pickplace_cache"
    split: str = "train"
    episode: int = 0
    train_all_episodes: bool = True
    validation_split: str = "train"
    validation_episode: int = 0
    validation_max_frames: int = 30
    resolution: int = 96
    height: int = 96
    width: int = 128
    wan_dim: int = DEFAULT_WAN_DIM
    latent_channels: int = DEFAULT_WAN_Z_DIM
    wan_num_res_blocks: int = DEFAULT_WAN_NUM_RES_BLOCKS
    hidden_channels: int = 64
    dynamics_infer_steps: int = 35
    dynamics_train_timesteps: int = 1000
    dynamics_rf_shift: float = 5.0
    dynamics_context_frames: int = DYNAMICS_FRAME_LAYOUT.context_frames
    dynamics_target_frames: int = DYNAMICS_FRAME_LAYOUT.target_frames
    dynamics_patch_spatial: int = 2
    dynamics_model_channels: int = 1536
    dynamics_num_blocks: int = 20
    dynamics_num_heads: int = 12
    dynamics_action_dim: int = 6
    dynamics_action_conditioning_mode: str = "chunk_per_frame"
    dynamics_action_representation: str = "relative_delta"
    dynamics_action_scale: float = SO101_RELATIVE_ACTION_SCALE
    dynamics_adaln_lora_dim: int = 128
    dynamics_validation_metric: str = DEFAULT_VALIDATION_METRIC
    batch_size: int = 16
    auto_batch_size: bool = True
    dataloader_num_workers: int = 1
    lr: float = 3e-4
    lr_warmup_steps: int = 200
    optimizer_beta1: float = 0.95
    num_epochs: float = 2.0
    max_steps: int | None = None
    validation_interval: int = 250
    checkpoint_interval: int = 250
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "dreamdojo/outputs"
    run_name: str = ""
    seed: int = 7
    wan_vae_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""

        return asdict(self)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone SO101 trainer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/so101_base_sim_pickplace_cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--train-all-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-split", default="train")
    parser.add_argument("--validation-episode", type=int, default=0)
    parser.add_argument("--validation-max-frames", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--wan-dim", type=int, default=DEFAULT_WAN_DIM)
    parser.add_argument("--latent-channels", type=int, default=DEFAULT_WAN_Z_DIM)
    parser.add_argument("--wan-num-res-blocks", type=int, default=DEFAULT_WAN_NUM_RES_BLOCKS)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dynamics-infer-steps", type=int, default=35)
    parser.add_argument("--dynamics-train-timesteps", type=int, default=1000)
    parser.add_argument("--dynamics-rf-shift", type=float, default=5.0)
    parser.add_argument("--dynamics-context-frames", type=int, default=DYNAMICS_FRAME_LAYOUT.context_frames)
    parser.add_argument("--dynamics-target-frames", type=int, default=DYNAMICS_FRAME_LAYOUT.target_frames)
    parser.add_argument("--dynamics-patch-spatial", type=int, default=2)
    parser.add_argument("--dynamics-model-channels", type=int, default=1536)
    parser.add_argument("--dynamics-num-blocks", type=int, default=20)
    parser.add_argument("--dynamics-num-heads", type=int, default=12)
    parser.add_argument("--dynamics-action-dim", type=int, default=6)
    parser.add_argument(
        "--dynamics-action-conditioning-mode",
        choices=["chunk_per_frame"],
        default="chunk_per_frame",
    )
    parser.add_argument(
        "--dynamics-action-representation",
        choices=["relative_delta"],
        default="relative_delta",
    )
    parser.add_argument("--dynamics-action-scale", type=float, default=SO101_RELATIVE_ACTION_SCALE)
    parser.add_argument("--dynamics-adaln-lora-dim", type=int, default=128)
    parser.add_argument("--dynamics-validation-metric", default=DEFAULT_VALIDATION_METRIC)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--auto-batch-size",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--optimizer-beta1", type=float, default=0.95)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="dreamdojo/outputs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wan-vae-path", required=True)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Convert parsed CLI arguments into a standalone training config."""

    return TrainConfig(
        data_root=args.data_root,
        split=args.split,
        episode=args.episode,
        train_all_episodes=args.train_all_episodes,
        validation_split=args.validation_split,
        validation_episode=args.validation_episode,
        validation_max_frames=args.validation_max_frames,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
        wan_dim=args.wan_dim,
        latent_channels=args.latent_channels,
        wan_num_res_blocks=args.wan_num_res_blocks,
        hidden_channels=args.hidden_channels,
        dynamics_infer_steps=args.dynamics_infer_steps,
        dynamics_train_timesteps=args.dynamics_train_timesteps,
        dynamics_rf_shift=args.dynamics_rf_shift,
        dynamics_context_frames=args.dynamics_context_frames,
        dynamics_target_frames=args.dynamics_target_frames,
        dynamics_patch_spatial=args.dynamics_patch_spatial,
        dynamics_model_channels=args.dynamics_model_channels,
        dynamics_num_blocks=args.dynamics_num_blocks,
        dynamics_num_heads=args.dynamics_num_heads,
        dynamics_action_dim=args.dynamics_action_dim,
        dynamics_action_conditioning_mode=args.dynamics_action_conditioning_mode,
        dynamics_action_representation=args.dynamics_action_representation,
        dynamics_action_scale=args.dynamics_action_scale,
        dynamics_adaln_lora_dim=args.dynamics_adaln_lora_dim,
        dynamics_validation_metric=args.dynamics_validation_metric,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        optimizer_beta1=args.optimizer_beta1,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        validation_interval=args.validation_interval,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        wan_vae_path=args.wan_vae_path,
    )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig, device: torch.device) -> WorldModel:
    """Instantiate the standalone SO101 DreamDojo-style world model."""

    model = WorldModel(
        latent_channels=cfg.latent_channels,
        hidden_channels=cfg.hidden_channels,
        ae_backend="wan",
        resolution=cfg.resolution,
        height=cfg.height,
        width=cfg.width,
        dynamics_infer_steps=cfg.dynamics_infer_steps,
        dynamics_train_timesteps=cfg.dynamics_train_timesteps,
        dynamics_rf_shift=cfg.dynamics_rf_shift,
        dynamics_context_frames=cfg.dynamics_context_frames,
        dynamics_target_frames=cfg.dynamics_target_frames,
        dynamics_open_rollout_context_frames=cfg.dynamics_context_frames,
        dynamics_patch_spatial=cfg.dynamics_patch_spatial,
        dynamics_model_channels=cfg.dynamics_model_channels,
        dynamics_num_blocks=cfg.dynamics_num_blocks,
        dynamics_num_heads=cfg.dynamics_num_heads,
        dynamics_action_dim=cfg.dynamics_action_dim,
        dynamics_action_conditioning_mode=cfg.dynamics_action_conditioning_mode,
        dynamics_use_adaln_lora=True,
        dynamics_adaln_lora_dim=cfg.dynamics_adaln_lora_dim,
        wan_config=WanVAEConfig(
            dim=cfg.wan_dim,
            z_dim=cfg.latent_channels,
            num_res_blocks=cfg.wan_num_res_blocks,
        ),
    ).to(device)
    load_wan_weights_into_model(model, cfg.wan_vae_path)
    model.configure_trainability("dynamics_only")
    return model


def build_train_dataset(cfg: TrainConfig) -> So101TransitionDataset:
    """Build the standalone SO101 training dataset."""

    exclude_episodes = (
        (cfg.validation_episode,)
        if cfg.train_all_episodes and cfg.validation_split == cfg.split
        else ()
    )
    return So101TransitionDataset(
        data_root=cfg.data_root,
        split=cfg.split,
        episode=cfg.episode,
        resolution=cfg.resolution,
        height=cfg.height,
        width=cfg.width,
        all_episodes=cfg.train_all_episodes,
        exclude_episodes=exclude_episodes,
        action_representation=cfg.dynamics_action_representation,
        action_scale=cfg.dynamics_action_scale,
    )


def build_validation_dataset(cfg: TrainConfig) -> So101ValidationClipDataset:
    """Build the standalone SO101 validation dataset."""

    validation_frame_end = (
        None
        if cfg.validation_max_frames is None
        else max(int(cfg.validation_max_frames) - 1, 0)
    )
    return So101ValidationClipDataset(
        data_root=cfg.data_root,
        split=cfg.validation_split,
        episode=cfg.validation_episode,
        frame_start=0,
        frame_end=validation_frame_end,
        resolution=cfg.resolution,
        height=cfg.height,
        width=cfg.width,
        action_representation=cfg.dynamics_action_representation,
        action_scale=cfg.dynamics_action_scale,
    )


def build_train_loader(
    dataset: So101TransitionDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[Any]:
    """Build the standalone training dataloader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def build_optimizer(model: WorldModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Build the standalone optimizer over trainable dynamics parameters."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were selected for standalone training.")
    return torch.optim.AdamW(
        parameters,
        lr=cfg.lr,
        betas=(cfg.optimizer_beta1, 0.999),
        weight_decay=0.01,
    )


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move one collated batch dictionary onto the requested device."""

    moved: dict[str, Any] = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
    return moved


def resolve_max_steps(dataset_length: int, batch_size: int, cfg: TrainConfig) -> int:
    """Resolve the exact number of optimizer steps for the requested epoch budget."""

    if cfg.max_steps is not None:
        return int(cfg.max_steps)
    if dataset_length < 1:
        raise ValueError("The standalone training dataset is empty.")
    return int(math.ceil(float(cfg.num_epochs) * float(dataset_length) / float(batch_size)))


def resolve_run_name(cfg: TrainConfig, resolved_batch_size: int) -> str:
    """Return the standalone run name after batch-size probing."""

    if cfg.run_name:
        return cfg.run_name
    return (
        "so101_wan22_rf_dit_"
        f"{cfg.height}x{cfg.width}_"
        f"{cfg.dynamics_model_channels}x{cfg.dynamics_num_blocks}x{cfg.dynamics_num_heads}x"
        f"{cfg.dynamics_adaln_lora_dim}_ps{cfg.dynamics_patch_spatial}_"
        f"{cfg.dynamics_context_frames}to{cfg.dynamics_target_frames}_bs{resolved_batch_size}_"
        f"{int(cfg.num_epochs)}ep"
    )


def active_learning_rate(step: int, cfg: TrainConfig) -> float:
    """Return the learning rate that should be used for one optimizer step."""

    if cfg.lr_warmup_steps <= 1:
        return float(cfg.lr)
    return float(cfg.lr) * min((int(step) + 1) / float(cfg.lr_warmup_steps), 1.0)


def dynamics_training_step(model: WorldModel, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Run one standalone `dynamics_only` training step and return loss metrics."""

    context_frames = batch["context_frames"]
    target_frames = batch["target_frames"]
    actions = batch["actions"]
    future_target_frames = batch["future_target_frames"]
    with torch.no_grad():
        full_frame_chunk = torch.cat([context_frames, target_frames, future_target_frames], dim=1)
        extended_clean_latent_video = model.encode_frame_sequence(
            full_frame_chunk,
            deterministic=True,
        )
    clean_latent_video = extended_clean_latent_video[:, :, : model.dynamics.cfg.max_frames]
    dynamics_inputs = model.dynamics.prepare_training_inputs(clean_latent_video, actions=actions)
    predicted_velocity = model.dynamics(
        noisy_latent_video=dynamics_inputs.noisy_latent_video,
        timesteps=dynamics_inputs.timesteps,
        condition_mask=dynamics_inputs.condition_mask,
        actions=dynamics_inputs.actions,
        conditioning_latent_video=dynamics_inputs.conditioning_latent_video,
        target_velocity=dynamics_inputs.target_velocity,
        use_video_condition=dynamics_inputs.use_video_condition,
    )
    latent_rf_mse = F.mse_loss(predicted_velocity, dynamics_inputs.target_velocity)
    return {
        "loss": latent_rf_mse,
        "latent_rf_mse": latent_rf_mse.detach(),
        "target_sigma": dynamics_inputs.target_sigmas.mean().detach(),
    }


def is_cuda_oom(error: RuntimeError) -> bool:
    """Return whether one runtime error looks like a CUDA out-of-memory failure."""

    return "out of memory" in str(error).lower()


def cleanup_cuda_state() -> None:
    """Release temporary Python and CUDA allocations after one failed probe or step."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def probe_batch_size(
    cfg: TrainConfig,
    dataset: So101TransitionDataset,
    *,
    device: torch.device,
) -> int:
    """Probe the largest runnable batch size by halving on CUDA OOM."""

    candidate = min(int(cfg.batch_size), len(dataset))
    if candidate < 1:
        raise ValueError("Cannot probe batch size on an empty training dataset.")
    if not cfg.auto_batch_size or device.type != "cuda":
        return candidate
    while candidate >= 1:
        model: WorldModel | None = None
        optimizer: torch.optim.Optimizer | None = None
        batch: dict[str, Any] | None = None
        moved_batch: dict[str, Any] | None = None
        try:
            model = build_model(cfg, device)
            optimizer = build_optimizer(model, cfg)
            batch = default_collate([dataset[index] for index in range(min(candidate, len(dataset)))])
            moved_batch = move_batch_to_device(batch, device)
            autocast_dtype = resolved_training_autocast_dtype(device)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            grad_scaler = (
                torch.amp.GradScaler("cuda", enabled=True)
                if autocast_dtype == torch.float16
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                metrics = dynamics_training_step(model, moved_batch)
            if grad_scaler is not None:
                grad_scaler.scale(metrics["loss"]).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                metrics["loss"].backward()
                optimizer.step()
            print(
                json.dumps(
                    {
                        "event": "auto_batch_size_resolved",
                        "requested_batch_size": int(cfg.batch_size),
                        "resolved_batch_size": int(candidate),
                    },
                    sort_keys=True,
                )
            )
            return candidate
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            candidate //= 2
            cleanup_cuda_state()
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if model is not None:
                model.zero_grad(set_to_none=True)
            del moved_batch
            del batch
            del optimizer
            del model
            cleanup_cuda_state()
    raise RuntimeError("Automatic CUDA batch-size probing could not fit even batch size 1.")


class So101RfTrainer:
    """Train the standalone SO101 DreamDojo-style rectified-flow model."""

    def __init__(self, cfg: TrainConfig) -> None:
        """Build standalone training state from one resolved config."""

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.train_dataset = build_train_dataset(cfg)
        self.cfg.batch_size = probe_batch_size(cfg, self.train_dataset, device=self.device)
        self.cfg.run_name = resolve_run_name(cfg, self.cfg.batch_size)
        self.cfg.max_steps = resolve_max_steps(len(self.train_dataset), self.cfg.batch_size, cfg)
        self.validation_dataset = build_validation_dataset(cfg)
        self.run_dir = Path(cfg.output_dir) / self.cfg.run_name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.samples_dir = self.run_dir / "samples"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_model(self.cfg, self.device)
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.train_loader = build_train_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.dataloader_num_workers,
            device=self.device,
        )
        self.training_autocast_dtype = resolved_training_autocast_dtype(self.device)
        self.grad_scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self.training_autocast_dtype == torch.float16
            else None
        )
        self.best_metric: float | None = None
        self.current_step = 0
        self.started_at = time.perf_counter()
        save_json(self.config_path, self.cfg.to_dict())
        append_jsonl(
            self.metrics_path,
            {
                "run_start": {
                    "config": self.cfg.to_dict(),
                    "dreamdojo_upstream_commit": DREAMDOJO_UPSTREAM_COMMIT,
                    "resolved_batch_size": int(self.cfg.batch_size),
                    "resolved_max_steps": int(self.cfg.max_steps),
                    "train_dataset_windows": int(len(self.train_dataset)),
                    "validation_dataset_frames": int(self.validation_dataset[0]["frames"].shape[0]),
                }
            },
        )

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move one collated train batch onto the configured device."""

        return move_batch_to_device(batch, self.device)

    def _active_learning_rate(self) -> float:
        """Return the learning rate for the next optimizer step."""

        return active_learning_rate(self.current_step, self.cfg)

    def _prepare_learning_rate_for_update(self) -> float:
        """Apply the next learning rate to every optimizer parameter group."""

        learning_rate = self._active_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        return learning_rate

    def _next_train_batch(self, iterator: Any) -> tuple[Any, dict[str, Any]]:
        """Return the next training batch, rewinding at the end of one epoch."""

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(self.train_loader)
            batch = next(iterator)
        return iterator, self._move_batch_to_device(batch)

    def _save_checkpoint(self, path: str | Path) -> None:
        """Save one standalone training checkpoint to disk."""

        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.current_step,
            config=self.cfg.to_dict(),
            best_metric=self.best_metric,
        )

    def _execute_training_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one optimizer step and return detached scalar metrics."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        learning_rate = self._prepare_learning_rate_for_update()
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=self.training_autocast_dtype)
            if self.training_autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            metrics = dynamics_training_step(self.model, batch)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(metrics["loss"]).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            metrics["loss"].backward()
            self.optimizer.step()
        return {
            "loss": float(metrics["loss"].detach().item()),
            "latent_rf_mse": float(metrics["latent_rf_mse"].item()),
            "target_sigma": float(metrics["target_sigma"].item()),
            "learning_rate": float(learning_rate),
            "elapsed_run_seconds": float(time.perf_counter() - self.started_at),
        }

    def _validate(self) -> dict[str, Any]:
        """Run teacher-forced validation, export artifacts, and update best-checkpoint state."""

        self.model.eval()
        clip = self.validation_dataset[0]
        frames = clip["frames"].to(self.device)
        actions = clip["actions"].to(self.device)
        preview_frames, stats = run_teacher_forced_validation(self.model, frames, actions)
        stats.update(
            {
                "episode": int(clip["episode_idx"].item()),
                "mode": "dynamics_only",
                "ae_backend": self.model.ae_backend,
                "dynamics_backend": self.model.dynamics_backend,
                "conditioning_frame_choices": list(self.model.dynamics.cfg.conditioning_frame_choices),
                "validation_conditioning_frame_choices": list(self.model.dynamics.cfg.validation_conditioning_frame_choices),
                "open_rollout_context_frames": int(self.model.dynamics.cfg.open_rollout_context_frames),
                "open_rollout_stride_frames": (
                    None
                    if self.model.dynamics.cfg.open_rollout_stride_frames is None
                    else int(self.model.dynamics.cfg.open_rollout_stride_frames)
                ),
                "dreamdojo_upstream_commit": DREAMDOJO_UPSTREAM_COMMIT,
                "dynamics_action_representation": self.cfg.dynamics_action_representation,
                "dynamics_action_scale": float(self.cfg.dynamics_action_scale),
            }
        )
        metric_value = float(stats[self.cfg.dynamics_validation_metric])
        is_best_checkpoint = self.best_metric is None or metric_value < self.best_metric
        if is_best_checkpoint:
            self.best_metric = metric_value
            self._save_checkpoint(self.checkpoints_dir / "best.pt")
        output_dir = self.samples_dir / f"step_{self.current_step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / f"episode_{stats['episode']}_grid.png"
        video_path = output_dir / f"episode_{stats['episode']}.mp4"
        stats_path = output_dir / f"episode_{stats['episode']}_stats.json"
        build_side_by_side_grid(
            original=frames.detach().cpu(),
            reconstructed=preview_frames,
            max_frames=int(frames.shape[0]),
            context_frames=int(self.model.latent_frames_to_pixel_frames(self.model.dynamics.cfg.validation_conditioning_frame_choices[0])),
        ).save(grid_path)
        exported_frame_count = write_side_by_side_mp4(
            original=frames.detach().cpu(),
            reconstructed=preview_frames,
            output_path=video_path,
            context_frames=int(self.model.latent_frames_to_pixel_frames(self.model.dynamics.cfg.validation_conditioning_frame_choices[0])),
        )
        stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
        stats["best_checkpoint"] = str(self.checkpoints_dir / "best.pt")
        stats["is_best_checkpoint"] = bool(is_best_checkpoint)
        stats["exported_video_frame_count"] = int(exported_frame_count)
        save_json(stats_path, stats)
        append_jsonl(self.metrics_path, {"step": self.current_step, "validation": stats})
        print(json.dumps({"step": self.current_step, "validation": stats}, sort_keys=True))
        return stats

    def run(self) -> dict[str, Any]:
        """Execute the standalone training loop to completion."""

        train_iterator = iter(self.train_loader)
        last_train_metrics: dict[str, float] = {}
        for step in range(1, int(self.cfg.max_steps) + 1):
            self.current_step = step
            train_iterator, batch = self._next_train_batch(train_iterator)
            last_train_metrics = self._execute_training_step(batch)
            if step == 1 or step % self.cfg.log_interval == 0:
                metric_record = {"step": step, **last_train_metrics}
                append_jsonl(self.metrics_path, metric_record)
                print(json.dumps(metric_record, sort_keys=True))
            if step % self.cfg.validation_interval == 0 or step == int(self.cfg.max_steps):
                self._validate()
            if step % self.cfg.checkpoint_interval == 0 or step == int(self.cfg.max_steps):
                self._save_checkpoint(self.checkpoints_dir / "last.pt")
        return {
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.checkpoints_dir / "best.pt"),
            "last_checkpoint": str(self.checkpoints_dir / "last.pt"),
            "best_metric": self.best_metric,
            "steps": int(self.cfg.max_steps),
        }


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run standalone SO101 training."""

    cfg = build_config(parse_args(argv))
    set_random_seed(cfg.seed)
    trainer = So101RfTrainer(cfg)
    summary = trainer.run()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
