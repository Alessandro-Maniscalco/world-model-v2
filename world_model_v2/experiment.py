"""Experiment runner for the Wan-VAE plus RF-DiT world-model pipeline."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from numbers import Integral, Real
import platform
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from world_model_v2.dataset import (
    AutoencoderClipDataset,
    TransitionDataset,
    ValidationClipDataset,
)
from world_model_v2.dynamics_transformer import (
    DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION,
    DYNAMICS_FRAME_LAYOUT,
    DynamicsFrameLayout,
)
from world_model_v2.dynamics_transformer import DynamicsTrainingInputs
from world_model_v2.lerobot_video_dataset import (
    LEROBOT_ACTION_REPRESENTATIONS,
    LeRobotVideoAutoencoderClipDataset,
    LeRobotVideoTransitionDataset,
    LeRobotVideoValidationClipDataset,
    SO101_BASE_SIM_PICKPLACE_ACTION_DIM,
    SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    SO101_RELATIVE_ACTION_SCALE,
)
from world_model_v2.maniskill_dataset import (
    MANISKILL_DEFAULT_ACTION_DIM,
    MANISKILL_DEFAULT_CAMERA,
    MANISKILL_DEFAULT_TRAJ_H5,
    MANISKILL_DEFAULT_TRAJ_JSON,
    ManiSkillAutoencoderClipDataset,
    ManiSkillTransitionDataset,
    ManiSkillValidationClipDataset,
)
from world_model_v2.metaworld_dataset import (
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM,
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN,
    AlohaAutoencoderClipDataset,
    AlohaTransitionDataset,
    AlohaValidationClipDataset,
    METAWORLD_DATASET_ID,
    METAWORLD_ACTION_DIM,
    MetaWorldAutoencoderClipDataset,
    MetaWorldTransitionDataset,
    MetaWorldValidationClipDataset,
)
from world_model_v2.model import LatentNormalizationStats, WorldModel
from world_model_v2.utils.checkpointing import append_jsonl, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4
from world_model_v2.wandb_logger import WandbRunLogger
from world_model_v2.wan_vae import (
    DEFAULT_WAN_DIM,
    DEFAULT_WAN_NUM_RES_BLOCKS,
    DEFAULT_WAN_Z_DIM,
    WanVAEConfig,
    dreamdojo_wan_state_dict,
    remap_dreamdojo_wan_state_dict,
)


WORLD_MODEL_CHECKPOINT_KIND = "world_model_v2_v1"
LEGACY_WORLD_MODEL_CHECKPOINT_KINDS = frozenset(
    {
        "world_model_v2_minimal_v1",
        "world_model_v2_minimal_v2",
    }
)
AUTO_BATCH_SIZE_BACKOFF_DIVISOR = 2
AUTO_BATCH_SIZE_GROWTH_FACTOR = 2
AUTO_TRAIN_DATALOADER_BENCHMARK_BATCHES = 4
AUTO_TRAIN_DATALOADER_CANDIDATES = (0, 2, 4)
DATALOADER_AUTOTUNE_CACHE_FILENAME = ".dataloader_autotune_cache.json"
_TEACHER_FORCED_NEXT_FRAME_MSE_PATTERN = re.compile(r"^next_frame_mse(?:_\d+to\d+)?$")
_OPEN_ROLLOUT_VALIDATION_METRICS = frozenset(
    {
        "open_rollout_frame_mse",
        "open_rollout_consistency_score",
    }
)
_TORCH_SAVE_ZIP_WRITE_FAILURE_MARKERS = (
    "PytorchStreamWriter failed writing file",
    "unexpected pos",
)
_REGENERABLE_CHECKPOINT_STATE_KEYS = frozenset({"net.pos_embedder.seq"})
DYNAMICS_ACTION_REPRESENTATION_CHOICES = ("dataset_default", *LEROBOT_ACTION_REPRESENTATIONS)


def validation_metric_requires_open_rollout(metric_name: str) -> bool:
    """Return whether one validation metric depends on open-rollout stats."""

    return metric_name in _OPEN_ROLLOUT_VALIDATION_METRICS


def resolve_dynamics_action_representation(
    *,
    mode: str,
    dataset_format: str,
    action_representation: str,
) -> str:
    """Resolve one configured action representation into the effective runtime mode."""

    if action_representation not in DYNAMICS_ACTION_REPRESENTATION_CHOICES:
        raise ValueError(
            f"Unsupported dynamics_action_representation={action_representation!r}. "
            f"Expected one of {DYNAMICS_ACTION_REPRESENTATION_CHOICES}."
        )
    if action_representation != "dataset_default":
        return action_representation
    if mode == "dynamics_only" and dataset_format == "lerobot_so101_base_sim_pickplace":
        return "relative_delta"
    return "absolute"


def resolve_dynamics_action_scale(
    *,
    mode: str,
    dataset_format: str,
    action_representation: str,
    action_scale: float,
) -> float:
    """Resolve the effective action scaling after representation selection."""

    resolved_representation = resolve_dynamics_action_representation(
        mode=mode,
        dataset_format=dataset_format,
        action_representation=action_representation,
    )
    if resolved_representation == "absolute":
        return 1.0
    resolved_scale = float(action_scale)
    if resolved_scale <= 0.0:
        raise ValueError("dynamics_action_scale must be positive when using relative actions.")
    return resolved_scale


def resolved_dynamics_action_representation_from_config(
    config: Mapping[str, Any],
    *,
    mode_override: str | None = None,
) -> str:
    """Resolve the effective action representation from serialized config data."""

    resolved_mode = (
        str(mode_override)
        if mode_override is not None
        else str(config.get("mode", "ae_only"))
    )
    return resolve_dynamics_action_representation(
        mode=resolved_mode,
        dataset_format=str(config.get("dataset_format", "interactive_world_sim")),
        action_representation=str(config.get("dynamics_action_representation", "absolute")),
    )


def resolved_dynamics_action_scale_from_config(
    config: Mapping[str, Any],
    *,
    mode_override: str | None = None,
) -> float:
    """Resolve the effective action scale from serialized config data."""

    resolved_mode = (
        str(mode_override)
        if mode_override is not None
        else str(config.get("mode", "ae_only"))
    )
    return resolve_dynamics_action_scale(
        mode=resolved_mode,
        dataset_format=str(config.get("dataset_format", "interactive_world_sim")),
        action_representation=str(config.get("dynamics_action_representation", "absolute")),
        action_scale=float(config.get("dynamics_action_scale", SO101_RELATIVE_ACTION_SCALE)),
    )


def is_retryable_torch_save_zip_error(error: RuntimeError) -> bool:
    """Return whether one `torch.save` failure matches the known zip-writer write bug."""

    message = str(error)
    return any(marker in message for marker in _TORCH_SAVE_ZIP_WRITE_FAILURE_MARKERS)


def remove_file_with_retries(path: Path, *, retries: int = 5, delay_seconds: float = 0.1) -> None:
    """Delete one file while tolerating short-lived Windows file-handle lag after save failures."""

    resolved_retries = max(int(retries), 1)
    for attempt in range(resolved_retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 >= resolved_retries:
                raise
            gc.collect()
            time.sleep(delay_seconds)


def replace_file_with_retries(
    source: Path,
    destination: Path,
    *,
    retries: int = 5,
    delay_seconds: float = 0.1,
) -> None:
    """Atomically replace one file while tolerating short-lived Windows file-handle lag."""

    resolved_retries = max(int(retries), 1)
    for attempt in range(resolved_retries):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 >= resolved_retries:
                raise
            gc.collect()
            time.sleep(delay_seconds)


def temporary_checkpoint_path(path: Path) -> Path:
    """Return one same-directory temporary checkpoint path for atomic save-and-replace."""

    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def resolved_training_autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Return the preferred mixed-precision dtype for training on one device."""

    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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
    aloha_repo_id: str = ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID
    aloha_cache_dir: str = ""
    maniskill_traj_h5: str = MANISKILL_DEFAULT_TRAJ_H5
    maniskill_traj_json: str = MANISKILL_DEFAULT_TRAJ_JSON
    maniskill_camera: str = MANISKILL_DEFAULT_CAMERA
    split: str = "val"
    episode: int = 0
    train_all_episodes: bool = False
    validation_split: str = ""
    validation_episode: int = 0
    validation_episodes: tuple[int, ...] | None = None
    validation_max_frames: int | None = None
    camera: str = "camera_1_color"
    frame_start: int | None = None
    frame_end: int | None = None
    resolution: int = 128
    height: int | None = None
    width: int | None = None
    wan_dim: int = DEFAULT_WAN_DIM
    latent_channels: int = DEFAULT_WAN_Z_DIM
    wan_num_res_blocks: int = DEFAULT_WAN_NUM_RES_BLOCKS
    hidden_channels: int = 64
    ae_backend: str = "wan"
    dynamics_infer_steps: int = 16
    dynamics_train_timesteps: int = 1000
    dynamics_rf_shift: float = 5.0
    conditional_frame_timestep: float = -1.0
    conditional_frame_sigma: float = 0.0
    dynamics_video_condition_dropout: float = 0.0
    dynamics_guidance_scale: float = 0.0
    dynamics_self_forcing_loss_weight: float = 0.0
    dynamics_rollout_self_forcing_loss_weight: float = 0.0
    dynamics_self_forcing_mode: str = "expanded_context"
    dynamics_self_forcing_warmup_steps: int = 0
    dynamics_self_forcing_ramp_steps: int = 0
    dynamics_rollout_self_forcing_warmup_steps: int = 0
    dynamics_rollout_self_forcing_ramp_steps: int = 0
    dynamics_self_forcing_rollout_chunks: int = 0
    dynamics_context_frames: int = DYNAMICS_FRAME_LAYOUT.context_frames
    dynamics_target_frames: int = DYNAMICS_FRAME_LAYOUT.target_frames
    dynamics_conditioning_frame_choices: tuple[int, ...] | None = None
    dynamics_conditioning_frame_probabilities: tuple[float, ...] | None = None
    dynamics_validation_conditioning_frame_choices: tuple[int, ...] | None = None
    dynamics_open_rollout_context_frames: int | None = None
    dynamics_open_rollout_stride_frames: int | None = None
    dynamics_patch_spatial: int = 1
    dynamics_model_channels: int = 256
    dynamics_num_blocks: int = 4
    dynamics_num_heads: int = 4
    dynamics_action_conditioning_mode: str = "chunk_per_frame"
    dynamics_action_representation: str = "dataset_default"
    dynamics_action_scale: float = SO101_RELATIVE_ACTION_SCALE
    dynamics_zero_init_action_embedder: bool = False
    dynamics_use_adaln_lora: bool = True
    dynamics_adaln_lora_dim: int = 64
    dynamics_rope_t_extrapolation_ratio: float = 1.0
    dynamics_use_learned_temporal_embedding: bool = False
    dynamics_validation_metric: str = "next_frame_mse"
    dynamics_run_open_rollout_validation: bool | None = None
    kl_beta: float = 1e-4
    recon_mse_weight: float = 1.0
    recon_l1_weight: float = 0.0
    recon_edge_weight: float = 0.0
    recon_motion_weight: float = 0.0
    recon_motion_edge_weight: float = 0.0
    recon_motion_threshold: float = 0.02
    recon_motion_dilation_kernel_size: int = 5
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool | None = None
    auto_batch_size: bool = False
    lr: float = 1e-4
    lr_warmup_steps: int = 200
    optimizer_beta1: float = 0.95
    max_steps: int = 3000
    validation_interval: int = 250
    validation_start_step: int = 0
    checkpoint_interval: int = 250
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
    wandb_enabled: bool = False
    wandb_project: str = "world-model-v2"
    wandb_entity: str = ""
    wandb_group: str = ""
    wandb_name: str = ""
    wandb_tags: tuple[str, ...] | None = None
    wandb_mode: str = "online"
    wandb_run_id: str = ""
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
            "aloha_repo_id": self.aloha_repo_id,
            "maniskill_traj_h5": self.maniskill_traj_h5,
            "maniskill_traj_json": self.maniskill_traj_json,
            "split": self.resolved_split(),
            "episode": self.episode,
            "camera": self.resolved_camera(),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "resolution": self.resolution,
            "height": self.resolved_height(),
            "width": self.resolved_width(),
            "dynamics_action_representation": (
                self.resolved_dynamics_action_representation()
                if self.mode == "dynamics_only"
                else None
            ),
            "dynamics_action_scale": (
                self.resolved_dynamics_action_scale()
                if self.mode == "dynamics_only"
                else None
            ),
        }

    def resolved_split(self) -> str:
        """Return the effective training split for the selected dataset format."""

        if self.dataset_format in {
            "lerobot_metaworld",
            "lerobot_aloha_sim_transfer_cube_scripted",
            "lerobot_so101_base_sim_pickplace",
            "maniskill_replay",
        }:
            return "train"
        return self.split

    def resolved_validation_split(self) -> str:
        """Return the validation split after applying the optional override."""

        if self.dataset_format in {
            "lerobot_metaworld",
            "lerobot_aloha_sim_transfer_cube_scripted",
            "lerobot_so101_base_sim_pickplace",
            "maniskill_replay",
        }:
            return "train"
        return self.validation_split or self.split

    def resolved_validation_episodes(self) -> tuple[int, ...]:
        """Return the ordered validation episodes used for checkpoint selection."""

        if self.validation_episodes is None:
            return (self.validation_episode,)
        return self.validation_episodes

    def resolved_validation_frame_end(self) -> int | None:
        """Return the effective validation-only frame end after applying the optional cap."""

        if self.validation_max_frames is None:
            return self.frame_end
        validation_start = 0 if self.frame_start is None else int(self.frame_start)
        capped_end = validation_start + int(self.validation_max_frames) - 1
        if self.frame_end is None:
            return capped_end
        return min(int(self.frame_end), capped_end)

    def resolved_run_open_rollout_validation(self) -> bool:
        """Return whether validation should execute the expensive open rollout."""

        if self.dynamics_run_open_rollout_validation is not None:
            return self.dynamics_run_open_rollout_validation
        return validation_metric_requires_open_rollout(self.dynamics_validation_metric)

    def resolved_camera(self) -> str:
        """Return the effective image stream key for the selected dataset format."""

        if self.dataset_format == "lerobot_metaworld":
            return "observation.image"
        if self.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
            return ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN
        if self.dataset_format == "lerobot_so101_base_sim_pickplace":
            return SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN
        if self.dataset_format == "maniskill_replay":
            return self.maniskill_camera
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

    def resolved_dynamics_action_representation(self) -> str:
        """Return the effective action representation used by dynamics datasets."""

        return resolve_dynamics_action_representation(
            mode=self.mode,
            dataset_format=self.dataset_format,
            action_representation=self.dynamics_action_representation,
        )

    def resolved_dynamics_action_scale(self) -> float:
        """Return the effective action scale used by dynamics datasets."""

        return resolve_dynamics_action_scale(
            mode=self.mode,
            dataset_format=self.dataset_format,
            action_representation=self.dynamics_action_representation,
            action_scale=self.dynamics_action_scale,
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
    motion_weight: float = 0.0,
    motion_edge_weight: float = 0.0,
    motion_threshold: float = 0.02,
    motion_dilation_kernel_size: int = 5,
    prev_frame: torch.Tensor | None = None,
    next_frame: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute mixed pixel, edge, and motion-focused reconstruction losses."""

    if (
        mse_weight < 0.0
        or l1_weight < 0.0
        or edge_weight < 0.0
        or motion_weight < 0.0
        or motion_edge_weight < 0.0
    ):
        raise ValueError("Reconstruction loss weights must be non-negative.")
    if motion_threshold < 0.0:
        raise ValueError("recon_motion_threshold must be non-negative.")
    if motion_dilation_kernel_size < 1 or motion_dilation_kernel_size % 2 == 0:
        raise ValueError("recon_motion_dilation_kernel_size must be a positive odd integer.")
    if (motion_weight > 0.0 or motion_edge_weight > 0.0) and (
        prev_frame is None or next_frame is None
    ):
        raise ValueError(
            "Motion-weighted reconstruction requires both prev_frame and next_frame tensors."
        )
    total_weight = mse_weight + l1_weight + edge_weight + motion_weight + motion_edge_weight
    if total_weight <= 0.0:
        raise ValueError("At least one reconstruction loss weight must be positive.")
    recon_mse = F.mse_loss(predicted, target)
    recon_l1 = F.l1_loss(predicted, target)
    predicted_grad_x, predicted_grad_y = finite_difference_gradients(predicted)
    target_grad_x, target_grad_y = finite_difference_gradients(target)
    edge_l1 = 0.5 * (
        F.l1_loss(predicted_grad_x, target_grad_x) + F.l1_loss(predicted_grad_y, target_grad_y)
    )
    motion_l1 = predicted.new_zeros(())
    motion_edge_l1 = predicted.new_zeros(())
    motion_mask_fraction = predicted.new_zeros(())
    if motion_weight > 0.0 or motion_edge_weight > 0.0:
        motion_prev = torch.mean(torch.abs(target - prev_frame), dim=1, keepdim=True)
        motion_next = torch.mean(torch.abs(next_frame - target), dim=1, keepdim=True)
        motion_signal = torch.maximum(motion_prev, motion_next)
        motion_mask = (motion_signal > motion_threshold).to(dtype=predicted.dtype)
        if motion_dilation_kernel_size > 1:
            motion_mask = F.max_pool2d(
                motion_mask,
                kernel_size=motion_dilation_kernel_size,
                stride=1,
                padding=motion_dilation_kernel_size // 2,
            )
        expanded_motion_mask = motion_mask.expand(-1, predicted.shape[1], -1, -1)
        motion_mask_fraction = motion_mask.mean().detach()
        if motion_weight > 0.0:
            motion_l1 = torch.sum(torch.abs(predicted - target) * expanded_motion_mask) / (
                expanded_motion_mask.sum().clamp_min(1.0)
            )
        if motion_edge_weight > 0.0:
            motion_mask_x = expanded_motion_mask[..., :, 1:]
            motion_mask_y = expanded_motion_mask[..., 1:, :]
            masked_edge_x = torch.sum(torch.abs(predicted_grad_x - target_grad_x) * motion_mask_x) / (
                motion_mask_x.sum().clamp_min(1.0)
            )
            masked_edge_y = torch.sum(torch.abs(predicted_grad_y - target_grad_y) * motion_mask_y) / (
                motion_mask_y.sum().clamp_min(1.0)
            )
            motion_edge_l1 = 0.5 * (masked_edge_x + masked_edge_y)
    recon_loss = (
        mse_weight * recon_mse
        + l1_weight * recon_l1
        + edge_weight * edge_l1
        + motion_weight * motion_l1
        + motion_edge_weight * motion_edge_l1
    ) / total_weight
    return {
        "recon_loss": recon_loss,
        "recon_mse": recon_mse.detach(),
        "recon_l1": recon_l1.detach(),
        "edge_l1": edge_l1.detach(),
        "motion_l1": motion_l1.detach(),
        "motion_edge_l1": motion_edge_l1.detach(),
        "motion_mask_fraction": motion_mask_fraction,
    }


def teacher_forced_next_frame_mse_stats(stats: dict[str, Any]) -> dict[str, float]:
    """Return aggregate teacher-forced frame MSEs keyed by validated layout suffix."""

    collected: dict[str, float] = {}
    for key, value in stats.items():
        if not _TEACHER_FORCED_NEXT_FRAME_MSE_PATTERN.fullmatch(key):
            continue
        if isinstance(value, (int, float)):
            collected[key] = float(value)
    return collected


def compute_motion_ratio(predicted_motion_l1: float, ground_truth_motion_l1: float) -> float:
    """Return a stable predicted/ground-truth motion ratio."""

    if ground_truth_motion_l1 <= 1e-12:
        if predicted_motion_l1 <= 1e-12:
            return 1.0
        return math.inf
    return predicted_motion_l1 / ground_truth_motion_l1


def motion_ratio_log_error(motion_ratio: float) -> float:
    """Return a symmetric log-domain penalty for motion ratios away from `1.0`."""

    if not math.isfinite(motion_ratio) or motion_ratio <= 0.0:
        return math.inf
    return abs(math.log(motion_ratio))


def open_rollout_consistency_score(
    frame_mse: float,
    motion_ratio: float | None,
) -> float:
    """Combine open-rollout pixel error with a symmetric motion-ratio penalty."""

    if not math.isfinite(frame_mse):
        return math.inf
    if motion_ratio is None:
        return float(frame_mse)
    log_error = motion_ratio_log_error(float(motion_ratio))
    if not math.isfinite(log_error):
        return math.inf
    return float(frame_mse) * (1.0 + log_error)


def validation_metric_value_from_stats(
    metric_name: str,
    stats: dict[str, Any],
) -> float | None:
    """Return one validation metric value, deriving compatibility metrics when possible."""

    raw_value = stats.get(metric_name)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if metric_name != "open_rollout_consistency_score":
        return None
    raw_frame_mse = stats.get("open_rollout_frame_mse")
    if not isinstance(raw_frame_mse, (int, float)):
        return None
    raw_motion_ratio = stats.get("open_rollout_target_motion_ratio")
    if raw_motion_ratio is None:
        return float(raw_frame_mse)
    if not isinstance(raw_motion_ratio, (int, float)):
        return None
    return open_rollout_consistency_score(
        float(raw_frame_mse),
        float(raw_motion_ratio),
    )


def _normalized_optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Normalize one optional integer or integer-sequence config value."""

    if value is None:
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise TypeError(f"Expected an integer sequence, received {type(value).__name__}.")


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
    temp_path = temporary_checkpoint_path(output_path)
    payload = {
        "kind": WORLD_MODEL_CHECKPOINT_KIND,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
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
    temp_save_succeeded = False
    try:
        try:
            torch.save(payload, temp_path)
        except RuntimeError as error:
            if not is_retryable_torch_save_zip_error(error):
                raise
            gc.collect()
            remove_file_with_retries(temp_path)
            torch.save(payload, temp_path, _use_new_zipfile_serialization=False)
        temp_save_succeeded = True
        replace_file_with_retries(temp_path, output_path)
    except RuntimeError as error:
        if not temp_save_succeeded:
            try:
                remove_file_with_retries(temp_path)
            except PermissionError:
                pass
        raise
    except Exception:
        if not temp_save_succeeded:
            try:
                remove_file_with_retries(temp_path)
            except PermissionError:
                pass
        raise


def load_training_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load and validate one world-model checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    accepted_kinds = {WORLD_MODEL_CHECKPOINT_KIND, *LEGACY_WORLD_MODEL_CHECKPOINT_KINDS}
    if checkpoint.get("kind") not in accepted_kinds:
        raise ValueError(f"{path} is not a supported world-model checkpoint.")
    return checkpoint


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and torch RNG state for deterministic resumes."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _normalized_rng_state_tensor(state: Any) -> torch.Tensor:
    """Return one RNG-state tensor in the CPU `uint8` format PyTorch expects."""

    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def restore_rng_state(state: Any) -> None:
    """Restore a previously captured RNG state when checkpoint metadata provides it."""

    if not isinstance(state, dict):
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(_normalized_rng_state_tensor(torch_state))
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [_normalized_rng_state_tensor(cuda_state) for cuda_state in cuda_states]
        )


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
        self._cached_encoder_decoder_source: tuple[str, Any] | None = None
        self._validate_config()
        self._set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self._validate_device()
        self.run_name = cfg.run_name or self._default_run_name(cfg.mode)
        self.run_dir = Path(cfg.output_dir) / self.run_name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        wan_config, latent_normalization_stats = self._resolve_initial_autoencoder_metadata()
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
            conditional_frame_sigma=cfg.conditional_frame_sigma,
            dynamics_video_condition_dropout=cfg.dynamics_video_condition_dropout,
            dynamics_guidance_scale=cfg.dynamics_guidance_scale,
            dynamics_context_frames=cfg.dynamics_context_frames,
            dynamics_target_frames=cfg.dynamics_target_frames,
            dynamics_conditioning_frame_choices=cfg.dynamics_conditioning_frame_choices,
            dynamics_conditioning_frame_probabilities=cfg.dynamics_conditioning_frame_probabilities,
            dynamics_validation_conditioning_frame_choices=cfg.dynamics_validation_conditioning_frame_choices,
            dynamics_open_rollout_context_frames=cfg.dynamics_open_rollout_context_frames,
            dynamics_open_rollout_stride_frames=cfg.dynamics_open_rollout_stride_frames,
            dynamics_patch_spatial=cfg.dynamics_patch_spatial,
            dynamics_model_channels=cfg.dynamics_model_channels,
            dynamics_num_blocks=cfg.dynamics_num_blocks,
            dynamics_num_heads=cfg.dynamics_num_heads,
            dynamics_action_dim=self._resolved_dynamics_action_dim(),
            dynamics_action_conditioning_mode=cfg.dynamics_action_conditioning_mode,
            dynamics_zero_init_action_embedder=cfg.dynamics_zero_init_action_embedder,
            dynamics_use_adaln_lora=cfg.dynamics_use_adaln_lora,
            dynamics_adaln_lora_dim=cfg.dynamics_adaln_lora_dim,
            dynamics_rope_t_extrapolation_ratio=cfg.dynamics_rope_t_extrapolation_ratio,
            dynamics_use_learned_temporal_embedding=cfg.dynamics_use_learned_temporal_embedding,
            wan_config=wan_config,
            latent_normalization_stats=latent_normalization_stats,
        ).to(self.device)
        self._load_requested_pretrained_weights()
        self.model.configure_trainability(cfg.mode)
        self.train_dataset = self._build_train_dataset()
        self._validate_train_dataset()
        # Auto-batch probing runs a dry training step, so schedule-dependent losses
        # need a well-defined step value before the probe happens.
        self.current_step = 0
        requested_batch_size = max(int(self.cfg.batch_size), 1)
        self.cfg.batch_size = self._resolve_train_batch_size(self.train_dataset)
        if self.cfg.auto_batch_size:
            self._log_auto_batch_size_resolution(
                requested_batch_size=requested_batch_size,
                resolved_batch_size=self.cfg.batch_size,
                max_dataset_batch=max(len(self.train_dataset), 1),
            )
        self._train_loader_worker_resolution_source = "explicit"
        self._resolved_train_loader_num_workers = self._resolve_train_loader_num_workers(
            self.train_dataset
        )
        self.train_loader = self._build_train_loader(self.train_dataset)
        self.val_loader = self._build_val_loader()
        self.optimizer = self._build_optimizer()
        self.training_autocast_dtype = resolved_training_autocast_dtype(self.device)
        self.grad_scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self.training_autocast_dtype == torch.float16
            else None
        )
        self.scheduler = None
        self.best_metric: float | None = None
        self.early_stop_window_losses: deque[float] = deque(
            maxlen=max(cfg.early_stop_window_size, 1)
        )
        self.best_window_loss: float | None = None
        self.non_improving_windows = 0
        self.early_stop_observations = 0
        self._resume_validation_setup_matches = True
        self.run_started_at_monotonic: float | None = None
        self.wandb_logger: WandbRunLogger | None = None
        if cfg.resume:
            self.current_step = self._load_resume()

    def _resolved_dynamics_action_dim(self) -> int:
        """Return the action dimension implied by the selected dataset format."""

        if self.cfg.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
            return ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM
        if self.cfg.dataset_format == "lerobot_so101_base_sim_pickplace":
            return SO101_BASE_SIM_PICKPLACE_ACTION_DIM
        if self.cfg.dataset_format == "maniskill_replay":
            return MANISKILL_DEFAULT_ACTION_DIM
        return METAWORLD_ACTION_DIM

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

    def _requested_wan_config(self) -> WanVAEConfig:
        """Return the Wan 2.2 tokenizer config requested by the run."""

        return WanVAEConfig(
            dim=self.cfg.wan_dim,
            z_dim=self.cfg.latent_channels,
            num_res_blocks=self.cfg.wan_num_res_blocks,
        )

    def _load_encoder_decoder_source(self) -> tuple[str, Any]:
        """Load and cache the requested encoder-decoder source checkpoint."""

        if not self.cfg.load_encoder_decoder:
            raise ValueError("load_encoder_decoder is empty.")
        if self._cached_encoder_decoder_source is not None:
            return self._cached_encoder_decoder_source
        path = self.cfg.load_encoder_decoder
        try:
            source: tuple[str, Any] = ("world_model", load_training_checkpoint(path, self.device))
        except ValueError:
            payload = torch.load(Path(path), map_location=self.device, weights_only=False)
            state_dict = dreamdojo_wan_state_dict(payload)
            if state_dict is None:
                raise ValueError(
                    f"{path} is neither a supported world-model checkpoint nor a raw DreamDojo Wan 2.2 state dict."
                ) from None
            source = ("dreamdojo_raw", state_dict)
        self._cached_encoder_decoder_source = source
        return source

    def _assert_requested_wan_matches_dreamdojo_raw_checkpoint(self, path: str | Path) -> WanVAEConfig:
        """Require the active tokenizer config to match the raw DreamDojo Wan 2.2 checkpoint."""

        requested_config = self._requested_wan_config()
        if not requested_config.is_dreamdojo_exact():
            raise ValueError(
                f"Raw DreamDojo Wan 2.2 weights from {path} require the exact tokenizer config "
                f"{WanVAEConfig().to_dict()}, received {requested_config.to_dict()}."
            )
        return requested_config

    def _assert_checkpoint_wan_shape_matches_requested(
        self,
        checkpoint_config: WanVAEConfig,
        path: str | Path,
    ) -> None:
        """Reject checkpoints whose Wan tokenizer config differs from the request."""

        requested_config = self._requested_wan_config()
        if checkpoint_config.to_dict() != requested_config.to_dict():
            raise ValueError(
                f"Checkpoint autoencoder config from {path} does not match the requested Wan 2.2 "
                "tokenizer config. Older approximate 2x-temporal tokenizer checkpoints are not "
                "load-compatible with the DreamDojo-exact Wan 2.2 port."
            )

    def _extract_checkpoint_autoencoder_metadata(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
        *,
        require_stats: bool,
    ) -> tuple[WanVAEConfig, LatentNormalizationStats]:
        """Read the serialized Wan config and latent stats from one checkpoint."""

        backend = checkpoint_ae_backend(checkpoint)
        if backend != self.cfg.ae_backend:
            raise ValueError(
                f"Checkpoint backend {backend} from {path} does not match requested "
                f"backend {self.cfg.ae_backend}."
            )
        autoencoder = checkpoint.get("autoencoder")
        if not isinstance(autoencoder, dict):
            raise ValueError(f"Checkpoint {path} is missing autoencoder metadata.")
        raw_config = autoencoder.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError(
                f"Checkpoint {path} is missing the serialized Wan autoencoder config."
            )
        wan_config = WanVAEConfig.from_dict(raw_config)
        raw_stats = autoencoder.get("normalization_stats")
        if require_stats and not isinstance(raw_stats, dict):
            raise ValueError(
                f"Checkpoint {path} is missing autoencoder normalization_stats. "
                "Older spatial-only checkpoints are not load-compatible with the temporal Wan tokenizer."
            )
        stats_payload = raw_stats if isinstance(raw_stats, dict) else None
        return wan_config, LatentNormalizationStats.from_dict(stats_payload, wan_config.z_dim)

    def _resolve_initial_autoencoder_metadata(
        self,
    ) -> tuple[WanVAEConfig, LatentNormalizationStats]:
        """Resolve the initial Wan config and latent stats before model construction."""

        requested_config = self._requested_wan_config()
        if self.cfg.resume:
            checkpoint = load_training_checkpoint(self.cfg.resume, self.device)
            checkpoint_config, checkpoint_stats = self._extract_checkpoint_autoencoder_metadata(
                checkpoint,
                self.cfg.resume,
                require_stats=True,
            )
            self._assert_checkpoint_wan_shape_matches_requested(
                checkpoint_config,
                self.cfg.resume,
            )
            return checkpoint_config, checkpoint_stats
        if self.cfg.load_encoder_decoder:
            source_kind, source_payload = self._load_encoder_decoder_source()
            if source_kind == "world_model":
                checkpoint_config, checkpoint_stats = self._extract_checkpoint_autoencoder_metadata(
                    source_payload,
                    self.cfg.load_encoder_decoder,
                    require_stats=True,
                )
                self._assert_checkpoint_wan_shape_matches_requested(
                    checkpoint_config,
                    self.cfg.load_encoder_decoder,
                )
                return checkpoint_config, checkpoint_stats
            requested_config = self._assert_requested_wan_matches_dreamdojo_raw_checkpoint(
                self.cfg.load_encoder_decoder
            )
            return requested_config, LatentNormalizationStats.default_for_channels(requested_config.z_dim)
        return requested_config, LatentNormalizationStats.default_for_channels(requested_config.z_dim)

    def _validate_config(self) -> None:
        """Validate mode-specific flag combinations before work begins."""

        if self.cfg.mode not in {"ae_only", "dynamics_only"}:
            raise ValueError(f"Unsupported mode: {self.cfg.mode}")
        if self.cfg.dataset_format not in {
            "interactive_world_sim",
            "lerobot_metaworld",
            "lerobot_aloha_sim_transfer_cube_scripted",
            "lerobot_so101_base_sim_pickplace",
            "maniskill_replay",
        }:
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
        if not 0.0 <= self.cfg.conditional_frame_sigma <= 1.0:
            raise ValueError("conditional_frame_sigma must be between 0 and 1.")
        if (
            self.cfg.dynamics_open_rollout_stride_frames is not None
            and self.cfg.dynamics_open_rollout_stride_frames < 1
        ):
            raise ValueError("dynamics_open_rollout_stride_frames must be positive.")
        if not 0.0 <= self.cfg.dynamics_video_condition_dropout <= 1.0:
            raise ValueError("dynamics_video_condition_dropout must be between 0 and 1.")
        if self.cfg.dynamics_guidance_scale < 0.0:
            raise ValueError("dynamics_guidance_scale must be non-negative.")
        if self.cfg.dynamics_self_forcing_loss_weight < 0.0:
            raise ValueError("dynamics_self_forcing_loss_weight must be non-negative.")
        if self.cfg.dynamics_rollout_self_forcing_loss_weight < 0.0:
            raise ValueError("dynamics_rollout_self_forcing_loss_weight must be non-negative.")
        if self.cfg.dynamics_self_forcing_mode not in {"expanded_context", "rollout"}:
            raise ValueError(
                "dynamics_self_forcing_mode must be 'expanded_context' or 'rollout'."
            )
        if self.cfg.dynamics_self_forcing_warmup_steps < 0:
            raise ValueError("dynamics_self_forcing_warmup_steps must be non-negative.")
        if self.cfg.dynamics_self_forcing_ramp_steps < 0:
            raise ValueError("dynamics_self_forcing_ramp_steps must be non-negative.")
        if self.cfg.dynamics_rollout_self_forcing_warmup_steps < 0:
            raise ValueError("dynamics_rollout_self_forcing_warmup_steps must be non-negative.")
        if self.cfg.dynamics_rollout_self_forcing_ramp_steps < 0:
            raise ValueError("dynamics_rollout_self_forcing_ramp_steps must be non-negative.")
        if self.cfg.dynamics_self_forcing_rollout_chunks < 0:
            raise ValueError("dynamics_self_forcing_rollout_chunks must be non-negative.")
        if (
            self.cfg.dynamics_self_forcing_mode == "rollout"
            and self.cfg.dynamics_rollout_self_forcing_loss_weight > 0.0
        ):
            raise ValueError(
                "dynamics_rollout_self_forcing_loss_weight cannot be combined with dynamics_self_forcing_mode='rollout'."
            )
        if (
            self.cfg.dynamics_self_forcing_loss_weight > 0.0
            and self.cfg.dynamics_self_forcing_mode == "rollout"
            and self.cfg.dynamics_self_forcing_rollout_chunks < 1
        ):
            raise ValueError(
                "dynamics_self_forcing_rollout_chunks must be positive when rollout self-forcing is enabled."
            )
        if (
            self.cfg.dynamics_rollout_self_forcing_loss_weight > 0.0
            and self.cfg.dynamics_self_forcing_rollout_chunks < 1
        ):
            raise ValueError(
                "dynamics_self_forcing_rollout_chunks must be positive when rollout self-forcing auxiliary loss is enabled."
            )
        if self.cfg.dynamics_context_frames < 1:
            raise ValueError("dynamics_context_frames must be positive.")
        if self.cfg.dynamics_target_frames < 1:
            raise ValueError("dynamics_target_frames must be positive.")
        if self.cfg.dynamics_patch_spatial < 1:
            raise ValueError("dynamics_patch_spatial must be positive.")
        if self.cfg.dynamics_model_channels < 1:
            raise ValueError("dynamics_model_channels must be positive.")
        if self.cfg.dynamics_num_blocks < 1:
            raise ValueError("dynamics_num_blocks must be positive.")
        if self.cfg.dynamics_num_heads < 1:
            raise ValueError("dynamics_num_heads must be positive.")
        if self.cfg.dynamics_action_conditioning_mode != "chunk_per_frame":
            raise ValueError(
                "dynamics_action_conditioning_mode must be 'chunk_per_frame' for the "
                "DreamDojo-mechanics RF DiT."
            )
        if self.cfg.dynamics_action_representation not in DYNAMICS_ACTION_REPRESENTATION_CHOICES:
            raise ValueError(
                "dynamics_action_representation must be one of "
                f"{DYNAMICS_ACTION_REPRESENTATION_CHOICES}."
            )
        resolved_action_representation = self.cfg.resolved_dynamics_action_representation()
        resolved_action_scale = self.cfg.resolved_dynamics_action_scale()
        if self.cfg.mode != "dynamics_only" and resolved_action_representation != "absolute":
            raise ValueError(
                "Relative dynamics actions are only meaningful in mode dynamics_only."
            )
        if (
            resolved_action_representation != "absolute"
            and self.cfg.dataset_format != "lerobot_so101_base_sim_pickplace"
        ):
            raise ValueError(
                "Relative dynamics actions are currently only implemented for "
                "dataset_format='lerobot_so101_base_sim_pickplace'."
            )
        if resolved_action_representation == "relative_delta" and resolved_action_scale <= 0.0:
            raise ValueError("dynamics_action_scale must be positive for relative dynamics actions.")
        if not self.cfg.dynamics_use_adaln_lora:
            raise ValueError("dynamics_use_adaln_lora must be enabled for the DreamDojo-mechanics RF DiT.")
        if self.cfg.dynamics_adaln_lora_dim < 1:
            raise ValueError("dynamics_adaln_lora_dim must be positive.")
        if self.cfg.dynamics_rope_t_extrapolation_ratio <= 0.0:
            raise ValueError("dynamics_rope_t_extrapolation_ratio must be positive.")
        if self.cfg.dynamics_use_learned_temporal_embedding:
            raise ValueError(
                "dynamics_use_learned_temporal_embedding is unsupported in the "
                "DreamDojo-mechanics RF DiT."
            )
        if self.cfg.dynamics_validation_metric not in {
            "next_frame_mse",
            "open_rollout_frame_mse",
            "open_rollout_consistency_score",
        }:
            raise ValueError(
                "dynamics_validation_metric must be 'next_frame_mse', "
                "'open_rollout_frame_mse', or 'open_rollout_consistency_score'."
            )
        if (
            self.cfg.dynamics_run_open_rollout_validation is False
            and validation_metric_requires_open_rollout(self.cfg.dynamics_validation_metric)
        ):
            raise ValueError(
                "dynamics_run_open_rollout_validation cannot be disabled when "
                "dynamics_validation_metric requires open-rollout stats."
            )
        if self.cfg.recon_motion_weight < 0.0:
            raise ValueError("recon_motion_weight must be non-negative.")
        if self.cfg.recon_motion_edge_weight < 0.0:
            raise ValueError("recon_motion_edge_weight must be non-negative.")
        if self.cfg.recon_motion_threshold < 0.0:
            raise ValueError("recon_motion_threshold must be non-negative.")
        if (
            self.cfg.recon_motion_dilation_kernel_size < 1
            or self.cfg.recon_motion_dilation_kernel_size % 2 == 0
        ):
            raise ValueError(
                "recon_motion_dilation_kernel_size must be a positive odd integer."
            )
        if (
            self.cfg.frame_start is not None
            and self.cfg.frame_end is not None
            and self.cfg.frame_end < self.cfg.frame_start
        ):
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
        if self.cfg.validation_start_step < 0:
            raise ValueError("validation_start_step must be non-negative.")
        if self.cfg.validation_max_frames is not None and self.cfg.validation_max_frames < 1:
            raise ValueError("validation_max_frames must be positive when provided.")
        if not 0.0 <= self.cfg.optimizer_beta1 < 1.0:
            raise ValueError("optimizer_beta1 must be in [0, 1).")
        if self.cfg.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be greater than or equal to one.")
        if self.cfg.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative.")
        if self.cfg.dataloader_num_workers is not None and self.cfg.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative.")
        if (
            self.cfg.mode == "dynamics_only"
            and self.cfg.validation_max_frames is not None
            and self.cfg.validation_max_frames < self.cfg.dynamics_frame_layout().max_pixel_frames
        ):
            raise ValueError(
                "validation_max_frames must cover at least one full dynamics chunk in dynamics_only mode."
            )
        if self.cfg.dataloader_prefetch_factor < 1:
            raise ValueError("dataloader_prefetch_factor must be greater than or equal to one.")
        if self.cfg.wandb_enabled and not self.cfg.wandb_project.strip():
            raise ValueError("wandb_project must be non-empty when wandb logging is enabled.")
        if self.cfg.wandb_mode not in {"online", "offline"}:
            raise ValueError("wandb_mode must be 'online' or 'offline'.")
        if self.cfg.wandb_run_id and not self.cfg.wandb_run_id.strip():
            raise ValueError("wandb_run_id must be non-empty when provided.")
        if self.cfg.dataset_format == "lerobot_metaworld":
            if self.cfg.split not in {"train", "val"}:
                raise ValueError("MetaWorld MT50 only supports train/val aliases for split.")
            if self.cfg.validation_split not in {"", "train", "val"}:
                raise ValueError(
                    "MetaWorld MT50 only supports train/val aliases for validation_split."
                )
            if self.cfg.metaworld_task_index is not None and self.cfg.metaworld_task_index < 0:
                raise ValueError("metaworld_task_index must be greater than or equal to zero.")
        if self.cfg.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
            if self.cfg.split not in {"train", "val"}:
                raise ValueError("ALOHA sim transfer cube scripted only supports train/val aliases.")
            if self.cfg.validation_split not in {"", "train", "val"}:
                raise ValueError(
                    "ALOHA sim transfer cube scripted only supports train/val validation aliases."
                )
        if self.cfg.dataset_format == "lerobot_so101_base_sim_pickplace":
            if self.cfg.split not in {"train", "val"}:
                raise ValueError(
                    "SO101 base sim pickplace only supports train/val aliases."
                )
            if self.cfg.validation_split not in {"", "train", "val"}:
                raise ValueError(
                    "SO101 base sim pickplace only supports train/val validation aliases."
                )
        if self.cfg.dataset_format == "maniskill_replay":
            if self.cfg.split not in {"train", "val"}:
                raise ValueError("ManiSkill replay datasets only support train/val aliases.")
            if self.cfg.validation_split not in {"", "train", "val"}:
                raise ValueError(
                    "ManiSkill replay datasets only support train/val validation aliases."
                )
        validation_episodes = self.cfg.resolved_validation_episodes()
        if len(validation_episodes) < 1:
            raise ValueError("At least one validation episode must be selected.")
        if any(episode < 0 for episode in validation_episodes):
            raise ValueError("validation episodes must be greater than or equal to zero.")

    def _validate_train_dataset(self) -> None:
        """Reject dynamics-only runs that do not contain any valid dynamics windows."""

        if self.cfg.mode == "dynamics_only" and len(self.train_dataset) < 1:
            required_frames = self.model.latent_frames_to_pixel_frames(
                self.model.dynamics.cfg.max_frames
            )
            if self.cfg.dynamics_self_forcing_rollout_chunks > 0:
                rollout_target_frames = (
                    self.model.dynamics.cfg.max_frames
                    - self.model.dynamics.cfg.open_rollout_context_frames
                )
                required_frames += (
                    self.cfg.dynamics_self_forcing_rollout_chunks
                    * self.model.temporal_downsample_factor
                    * rollout_target_frames
                )
            raise ValueError(
                f"dynamics_only requires at least {required_frames} pixel frames in the selected "
                "clip so one valid dynamics window exists."
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

        frame_layout = DynamicsFrameLayout(
            context_frames=self.cfg.dynamics_context_frames,
            target_frames=self.cfg.dynamics_target_frames,
            temporal_compression_ratio=self.model.temporal_downsample_factor,
        )
        exclude_episodes = self._excluded_training_episodes()
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
                dataset_kwargs["exclude_episodes"] = exclude_episodes
                dataset_kwargs["frame_layout"] = frame_layout
                return MetaWorldAutoencoderClipDataset(**dataset_kwargs)
            dataset_kwargs["frame_layout"] = frame_layout
            dataset_kwargs["rollout_context_frames"] = self.model.dynamics.cfg.open_rollout_context_frames
            dataset_kwargs["rollout_chunks"] = self.cfg.dynamics_self_forcing_rollout_chunks
            dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
            dataset_kwargs["exclude_episodes"] = exclude_episodes
            return MetaWorldTransitionDataset(**dataset_kwargs)
        if self.cfg.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
            dataset_kwargs = {
                "data_root": self.cfg.data_root,
                "split": self.cfg.resolved_split(),
                "episode": self.cfg.episode,
                "frame_start": self.cfg.frame_start,
                "frame_end": self.cfg.frame_end,
                "resolution": self.cfg.resolution,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "repo_id": self.cfg.aloha_repo_id,
                "cache_dir": self.cfg.aloha_cache_dir or None,
            }
            if self.cfg.mode == "ae_only":
                dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
                dataset_kwargs["exclude_episodes"] = exclude_episodes
                dataset_kwargs["frame_layout"] = frame_layout
                return AlohaAutoencoderClipDataset(**dataset_kwargs)
            dataset_kwargs["frame_layout"] = frame_layout
            dataset_kwargs["rollout_context_frames"] = self.model.dynamics.cfg.open_rollout_context_frames
            dataset_kwargs["rollout_chunks"] = self.cfg.dynamics_self_forcing_rollout_chunks
            dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
            dataset_kwargs["exclude_episodes"] = exclude_episodes
            return AlohaTransitionDataset(**dataset_kwargs)
        if self.cfg.dataset_format == "lerobot_so101_base_sim_pickplace":
            dataset_kwargs = {
                "data_root": self.cfg.data_root,
                "split": self.cfg.resolved_split(),
                "episode": self.cfg.episode,
                "frame_start": self.cfg.frame_start,
                "frame_end": self.cfg.frame_end,
                "resolution": self.cfg.resolution,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "repo_id": SO101_BASE_SIM_PICKPLACE_DATASET_ID,
                "image_column": SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
            }
            if self.cfg.mode == "ae_only":
                dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
                dataset_kwargs["exclude_episodes"] = exclude_episodes
                dataset_kwargs["frame_layout"] = frame_layout
                return LeRobotVideoAutoencoderClipDataset(**dataset_kwargs)
            dataset_kwargs["frame_layout"] = frame_layout
            dataset_kwargs["rollout_context_frames"] = self.model.dynamics.cfg.open_rollout_context_frames
            dataset_kwargs["rollout_chunks"] = self.cfg.dynamics_self_forcing_rollout_chunks
            dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
            dataset_kwargs["exclude_episodes"] = exclude_episodes
            dataset_kwargs["action_representation"] = self.cfg.resolved_dynamics_action_representation()
            dataset_kwargs["action_scale"] = self.cfg.resolved_dynamics_action_scale()
            return LeRobotVideoTransitionDataset(**dataset_kwargs)
        if self.cfg.dataset_format == "maniskill_replay":
            dataset_kwargs = {
                "data_root": self.cfg.data_root,
                "split": self.cfg.resolved_split(),
                "episode": self.cfg.episode,
                "frame_start": self.cfg.frame_start,
                "frame_end": self.cfg.frame_end,
                "resolution": self.cfg.resolution,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "traj_h5": self.cfg.maniskill_traj_h5,
                "traj_json": self.cfg.maniskill_traj_json,
                "camera": self.cfg.maniskill_camera,
            }
            if self.cfg.mode == "ae_only":
                dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
                dataset_kwargs["exclude_episodes"] = exclude_episodes
                dataset_kwargs["frame_layout"] = frame_layout
                return ManiSkillAutoencoderClipDataset(**dataset_kwargs)
            dataset_kwargs["frame_layout"] = frame_layout
            dataset_kwargs["rollout_context_frames"] = self.model.dynamics.cfg.open_rollout_context_frames
            dataset_kwargs["rollout_chunks"] = self.cfg.dynamics_self_forcing_rollout_chunks
            dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
            dataset_kwargs["exclude_episodes"] = exclude_episodes
            return ManiSkillTransitionDataset(**dataset_kwargs)
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
            dataset_kwargs["exclude_episodes"] = exclude_episodes
            dataset_kwargs["frame_layout"] = frame_layout
            return AutoencoderClipDataset(**dataset_kwargs)
        dataset_kwargs["frame_layout"] = frame_layout
        dataset_kwargs["rollout_context_frames"] = self.model.dynamics.cfg.open_rollout_context_frames
        dataset_kwargs["rollout_chunks"] = self.cfg.dynamics_self_forcing_rollout_chunks
        dataset_kwargs["all_episodes"] = self.cfg.train_all_episodes
        dataset_kwargs["exclude_episodes"] = exclude_episodes
        return TransitionDataset(**dataset_kwargs)

    def _excluded_training_episodes(self) -> tuple[int, ...]:
        """Return validation episodes to exclude when all-episode training shares the same pool."""

        if not self.cfg.train_all_episodes:
            return ()
        if self.cfg.resolved_split() != self.cfg.resolved_validation_split():
            return ()
        return self.cfg.resolved_validation_episodes()

    def _build_train_loader(self, dataset: Dataset[dict[str, Any]]) -> DataLoader[Any]:
        """Build the training dataloader for the resolved batch size."""

        return self._build_train_loader_for_num_workers(
            dataset,
            num_workers=self._resolved_train_loader_num_workers,
        )

    def _build_train_loader_for_num_workers(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        num_workers: int,
    ) -> DataLoader[Any]:
        """Build one training dataloader for a specific worker-count setting."""

        sampler_factory = getattr(dataset, "training_sampler", None)
        if callable(sampler_factory):
            sampler = sampler_factory()
            if sampler is not None:
                return DataLoader(
                    dataset,
                    batch_size=self.cfg.batch_size,
                    sampler=sampler,
                    shuffle=False,
                    **self._train_loader_options(num_workers=num_workers),
                )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            **self._train_loader_options(num_workers=num_workers),
        )

    def _build_validation_dataset_for_episode(self, episode: int) -> Dataset[dict[str, Any]]:
        """Build one full-clip validation dataset for the requested episode."""

        if self.cfg.dataset_format == "lerobot_metaworld":
            return MetaWorldValidationClipDataset(
                data_root=self.cfg.data_root,
                split=self.cfg.resolved_validation_split(),
                episode=episode,
                task_index=self.cfg.metaworld_task_index,
                frame_start=self.cfg.frame_start,
                frame_end=self.cfg.resolved_validation_frame_end(),
                resolution=self.cfg.resolution,
                height=self.cfg.height,
                width=self.cfg.width,
                repo_id=self.cfg.metaworld_repo_id,
                cache_dir=self.cfg.metaworld_cache_dir or None,
            )
        if self.cfg.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
            return AlohaValidationClipDataset(
                data_root=self.cfg.data_root,
                split=self.cfg.resolved_validation_split(),
                episode=episode,
                frame_start=self.cfg.frame_start,
                frame_end=self.cfg.resolved_validation_frame_end(),
                resolution=self.cfg.resolution,
                height=self.cfg.height,
                width=self.cfg.width,
                repo_id=self.cfg.aloha_repo_id,
                cache_dir=self.cfg.aloha_cache_dir or None,
            )
        if self.cfg.dataset_format == "lerobot_so101_base_sim_pickplace":
            return LeRobotVideoValidationClipDataset(
                data_root=self.cfg.data_root,
                split=self.cfg.resolved_validation_split(),
                episode=episode,
                frame_start=self.cfg.frame_start,
                frame_end=self.cfg.resolved_validation_frame_end(),
                resolution=self.cfg.resolution,
                height=self.cfg.height,
                width=self.cfg.width,
                repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
                image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
                action_representation=self.cfg.resolved_dynamics_action_representation(),
                action_scale=self.cfg.resolved_dynamics_action_scale(),
            )
        if self.cfg.dataset_format == "maniskill_replay":
            return ManiSkillValidationClipDataset(
                data_root=self.cfg.data_root,
                split=self.cfg.resolved_validation_split(),
                episode=episode,
                frame_start=self.cfg.frame_start,
                frame_end=self.cfg.resolved_validation_frame_end(),
                resolution=self.cfg.resolution,
                height=self.cfg.height,
                width=self.cfg.width,
                traj_h5=self.cfg.maniskill_traj_h5,
                traj_json=self.cfg.maniskill_traj_json,
                camera=self.cfg.maniskill_camera,
            )
        return ValidationClipDataset(
            data_root=self.cfg.data_root,
            task=self.cfg.task,
            split=self.cfg.resolved_validation_split(),
            episode=episode,
            camera=self.cfg.camera,
            frame_start=self.cfg.frame_start,
            frame_end=self.cfg.resolved_validation_frame_end(),
            resolution=self.cfg.resolution,
            height=self.cfg.height,
            width=self.cfg.width,
        )

    def _build_val_loader(self) -> DataLoader[Any]:
        """Build the full-clip validation dataloader."""

        validation_datasets = [
            self._build_validation_dataset_for_episode(episode)
            for episode in self.cfg.resolved_validation_episodes()
        ]
        dataset: Dataset[dict[str, Any]]
        if len(validation_datasets) == 1:
            dataset = validation_datasets[0]
        else:
            dataset = ConcatDataset(validation_datasets)
        return DataLoader(dataset, batch_size=1, shuffle=False, **self._eval_loader_options())

    def _pin_memory_enabled(self) -> bool:
        """Return whether dataloaders should stage tensors in pinned host memory."""

        if self.cfg.dataloader_pin_memory is not None:
            return bool(self.cfg.dataloader_pin_memory)
        return self.device.type == "cuda"

    def _train_loader_options(self, *, num_workers: int | None = None) -> dict[str, Any]:
        """Return the training `DataLoader` options derived from the config."""

        resolved_num_workers = (
            int(self._resolved_train_loader_num_workers)
            if num_workers is None
            else int(num_workers)
        )
        options: dict[str, Any] = {
            "num_workers": resolved_num_workers,
            "pin_memory": self._pin_memory_enabled(),
        }
        if options["num_workers"] > 0:
            options["persistent_workers"] = True
            options["prefetch_factor"] = int(self.cfg.dataloader_prefetch_factor)
        return options

    def _eval_loader_options(self) -> dict[str, Any]:
        """Return the lightweight evaluation `DataLoader` options."""

        return {
            "num_workers": 0,
            "pin_memory": self._pin_memory_enabled(),
        }

    def _resolve_train_loader_num_workers(self, dataset: Dataset[dict[str, Any]]) -> int:
        """Return the explicit or auto-tuned worker count for the training loader."""

        if self.cfg.dataloader_num_workers is not None:
            resolved = int(self.cfg.dataloader_num_workers)
            self._train_loader_worker_resolution_source = "explicit"
            self._log_train_loader_num_workers_resolution(
                resolved_num_workers=resolved,
                source=self._train_loader_worker_resolution_source,
                candidates=(resolved,),
            )
            return resolved
        if self.device.type != "cuda":
            resolved = 0
            self._train_loader_worker_resolution_source = "cpu_default"
            self._log_train_loader_num_workers_resolution(
                resolved_num_workers=resolved,
                source=self._train_loader_worker_resolution_source,
                candidates=(resolved,),
            )
            return resolved
        candidates = self._auto_train_loader_worker_candidates()
        if len(candidates) == 1:
            resolved = candidates[0]
            self._train_loader_worker_resolution_source = "single_candidate"
            self._log_train_loader_num_workers_resolution(
                resolved_num_workers=resolved,
                source=self._train_loader_worker_resolution_source,
                candidates=candidates,
            )
            return resolved
        cache_key = self._dataloader_autotune_cache_key()
        cached_num_workers = self._load_cached_train_loader_num_workers(cache_key)
        if cached_num_workers in candidates:
            resolved = int(cached_num_workers)
            self._train_loader_worker_resolution_source = "cache"
            self._log_train_loader_num_workers_resolution(
                resolved_num_workers=resolved,
                source=self._train_loader_worker_resolution_source,
                candidates=candidates,
            )
            return resolved
        resolved = self._benchmark_auto_train_loader_num_workers(dataset, candidates)
        self._save_cached_train_loader_num_workers(cache_key, resolved)
        self._train_loader_worker_resolution_source = "benchmark"
        self._log_train_loader_num_workers_resolution(
            resolved_num_workers=resolved,
            source=self._train_loader_worker_resolution_source,
            candidates=candidates,
        )
        return resolved

    def _auto_train_loader_worker_candidates(self) -> tuple[int, ...]:
        """Return the capped candidate worker counts considered during autotuning."""

        cpu_count = max(int(os.cpu_count() or 1), 1)
        candidates = sorted(
            candidate
            for candidate in AUTO_TRAIN_DATALOADER_CANDIDATES
            if candidate <= cpu_count and candidate >= 0
        )
        if 0 not in candidates:
            candidates.insert(0, 0)
        return tuple(candidates)

    def _benchmark_auto_train_loader_num_workers(
        self,
        dataset: Dataset[dict[str, Any]],
        candidates: tuple[int, ...],
    ) -> int:
        """Benchmark a small worker-count set and return the fastest steady-state choice."""

        timings: dict[int, float] = {}
        failures: dict[int, str] = {}
        for num_workers in candidates:
            try:
                timings[num_workers] = self._benchmark_train_loader_num_workers(
                    dataset,
                    num_workers=num_workers,
                )
            except Exception as error:
                if num_workers == 0:
                    raise
                failures[num_workers] = f"{type(error).__name__}: {error}"
        if not timings:
            raise RuntimeError("Automatic dataloader worker tuning did not produce any usable result.")
        resolved = min(
            timings.items(),
            key=lambda item: (float(item[1]), int(item[0])),
        )[0]
        print(
            json.dumps(
                {
                    "benchmark_seconds": {str(key): value for key, value in timings.items()},
                    "candidates": list(candidates),
                    "event": "auto_dataloader_workers_benchmarked",
                    "failed_candidates": failures,
                    "resolved_num_workers": resolved,
                },
                sort_keys=True,
            )
        )
        return int(resolved)

    def _benchmark_train_loader_num_workers(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        num_workers: int,
    ) -> float:
        """Measure one worker setting using a few steady-state batch fetch timings."""

        loader = self._build_train_loader_for_num_workers(dataset, num_workers=num_workers)
        iterator = iter(loader)
        all_timings: list[float] = []
        measured_timings: list[float] = []
        try:
            for batch_index in range(AUTO_TRAIN_DATALOADER_BENCHMARK_BATCHES):
                started_at = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                elapsed = time.perf_counter() - started_at
                all_timings.append(elapsed)
                if batch_index > 0:
                    measured_timings.append(elapsed)
                del batch
        finally:
            self._shutdown_dataloader_iterator(iterator)
            del iterator
            del loader
            gc.collect()
        active_timings = measured_timings or all_timings
        if not active_timings:
            raise RuntimeError("Cannot benchmark dataloader workers on an empty training iterator.")
        return float(sum(active_timings) / len(active_timings))

    def _shutdown_dataloader_iterator(self, iterator: Any) -> None:
        """Shut down one transient dataloader iterator after an autotune benchmark."""

        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()

    def _dataloader_autotune_cache_path(self) -> Path:
        """Return the shared JSON cache path for worker autotune decisions."""

        return Path(self.cfg.output_dir) / DATALOADER_AUTOTUNE_CACHE_FILENAME

    def _dataloader_autotune_cache_key(self) -> str:
        """Return one stable cache key for the current dataset and machine signature."""

        device_name: str | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
        signature = {
            "batch_size": int(self.cfg.batch_size),
            "clip_metadata": self.cfg.clip_metadata(),
            "cpu_count": int(os.cpu_count() or 1),
            "dataset_format": self.cfg.dataset_format,
            "device_name": device_name,
            "device_type": self.device.type,
            "dynamics_context_frames": self.cfg.dynamics_context_frames,
            "dynamics_target_frames": self.cfg.dynamics_target_frames,
            "mode": self.cfg.mode,
            "platform": platform.platform(),
            "prefetch_factor": int(self.cfg.dataloader_prefetch_factor),
            "resolved_split": self.cfg.resolved_split(),
            "train_all_episodes": bool(self.cfg.train_all_episodes),
        }
        return json.dumps(signature, sort_keys=True)

    def _load_cached_train_loader_num_workers(self, cache_key: str) -> int | None:
        """Return a cached worker resolution when one exists for the current signature."""

        cache_path = self._dataloader_autotune_cache_path()
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return None
        entry = entries.get(cache_key)
        if not isinstance(entry, dict):
            return None
        cached_num_workers = entry.get("num_workers")
        if not isinstance(cached_num_workers, Integral) or isinstance(cached_num_workers, bool):
            return None
        return int(cached_num_workers)

    def _save_cached_train_loader_num_workers(self, cache_key: str, num_workers: int) -> None:
        """Persist one successful worker-count resolution for future matching runs."""

        cache_path = self._dataloader_autotune_cache_path()
        payload: dict[str, Any] = {"entries": {}, "version": 1}
        if cache_path.exists():
            try:
                existing_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                existing_payload = None
            if isinstance(existing_payload, dict):
                payload["version"] = int(existing_payload.get("version", 1))
                existing_entries = existing_payload.get("entries")
                if isinstance(existing_entries, dict):
                    payload["entries"] = dict(existing_entries)
        payload["entries"][cache_key] = {
            "num_workers": int(num_workers),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(cache_path, payload)

    def _log_train_loader_num_workers_resolution(
        self,
        *,
        resolved_num_workers: int,
        source: str,
        candidates: tuple[int, ...],
    ) -> None:
        """Emit one JSON event describing the active training-loader worker decision."""

        print(
            json.dumps(
                {
                    "candidates": list(candidates),
                    "event": "train_loader_num_workers_resolved",
                    "requested_num_workers": self.cfg.dataloader_num_workers,
                    "resolved_num_workers": int(resolved_num_workers),
                    "source": source,
                },
                sort_keys=True,
            )
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer over trainable parameters."""

        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters were selected for the current mode.")
        return torch.optim.AdamW(
            parameters,
            lr=self.cfg.lr,
            betas=(self.cfg.optimizer_beta1, 0.999),
            weight_decay=0.01,
        )

    def _active_learning_rate(self) -> float:
        """Return the learning rate that should be used for the next optimizer update."""

        if self.cfg.resume:
            return float(self.cfg.lr)
        warmup_steps = int(self.cfg.lr_warmup_steps)
        if warmup_steps <= 1:
            return float(self.cfg.lr)
        return float(self.cfg.lr) * min((self.current_step + 1) / float(warmup_steps), 1.0)

    def _set_optimizer_learning_rate(self, learning_rate: float) -> float:
        """Apply one learning rate to every optimizer parameter group."""

        resolved_learning_rate = float(learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = resolved_learning_rate
        return resolved_learning_rate

    def _prepare_learning_rate_for_update(self) -> float:
        """Resolve and apply the learning rate for the next optimizer step."""

        return self._set_optimizer_learning_rate(self._active_learning_rate())

    def _attach_learning_rate_metric(
        self,
        loss_dict: dict[str, torch.Tensor],
        learning_rate: float,
    ) -> dict[str, torch.Tensor]:
        """Attach the active optimizer learning rate to one training metric dictionary."""

        metrics = dict(loss_dict)
        metrics["learning_rate"] = metrics["loss"].detach().new_tensor(float(learning_rate))
        return metrics

    def _resolve_train_batch_size(self, dataset: Dataset[dict[str, Any]]) -> int:
        """Return the configured or automatically probed training batch size."""

        requested_batch_size = max(int(self.cfg.batch_size), 1)
        max_dataset_batch = max(len(dataset), 1)
        if not self.cfg.auto_batch_size:
            return requested_batch_size
        if self.device.type != "cuda":
            return min(max_dataset_batch, requested_batch_size)
        return self._probe_cuda_batch_size(dataset, requested_batch_size, max_dataset_batch)

    def _probe_cuda_batch_size(
        self,
        dataset: Dataset[dict[str, Any]],
        requested_batch_size: int,
        max_batch_size: int,
    ) -> int:
        """Probe the largest CUDA batch size that fits for one training step."""

        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        initial_batch_size = min(max(int(requested_batch_size), 1), max_batch_size)
        best = 0
        try:
            if self._batch_size_fits(dataset, initial_batch_size):
                best = initial_batch_size
                highest_failed_batch: int | None = None
                while best < max_batch_size:
                    next_batch_size = min(best * AUTO_BATCH_SIZE_GROWTH_FACTOR, max_batch_size)
                    if next_batch_size == best:
                        break
                    if self._batch_size_fits(dataset, next_batch_size):
                        best = next_batch_size
                        continue
                    highest_failed_batch = next_batch_size
                    break
                if best >= max_batch_size:
                    return max_batch_size
                left = best + 1
                right = (
                    max_batch_size
                    if highest_failed_batch is None
                    else max(highest_failed_batch - 1, best)
                )
            else:
                left = 1
                right = initial_batch_size - 1
            while left <= right:
                candidate = (left + right) // 2
                if self._batch_size_fits(dataset, candidate):
                    best = candidate
                    left = candidate + 1
                else:
                    right = candidate - 1
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        if best < 1:
            raise RuntimeError(
                "Automatic CUDA batch-size probing could not fit even batch size 1. "
                "Free GPU memory may be exhausted by another process."
            )
        return best

    def _log_auto_batch_size_resolution(
        self,
        requested_batch_size: int,
        resolved_batch_size: int,
        max_dataset_batch: int,
    ) -> None:
        """Emit a JSON event describing the resolved auto-batch training size."""

        print(
            json.dumps(
                {
                    "event": "auto_batch_size_resolved",
                    "device": self.device.type,
                    "requested_batch_size": requested_batch_size,
                    "resolved_batch_size": resolved_batch_size,
                    "max_dataset_batch": max_dataset_batch,
                },
                sort_keys=True,
            )
        )

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
            betas=(self.cfg.optimizer_beta1, 0.999),
            weight_decay=0.0,
        )
        parameter_backup = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        loss_dict: dict[str, torch.Tensor] | None = None
        probe_autocast_dtype = resolved_training_autocast_dtype(self.device)
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=probe_autocast_dtype)
            if probe_autocast_dtype is not None
            else nullcontext()
        )
        try:
            with autocast_context:
                loss_dict = self._train_step(moved_batch)
            loss_dict["loss"].backward()
            probe_optimizer.step()
            return True
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            return False
        finally:
            self.model.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            del loss_dict
            del moved_batch
            del batch
            del probe_optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in parameter_backup:
                        parameter.copy_(parameter_backup[name])
            del parameter_backup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

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
        active_learning_rate = self._prepare_learning_rate_for_update()
        autocast_context = (
            torch.autocast(device_type=self.device.type, dtype=self.training_autocast_dtype)
            if self.training_autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            loss_dict = self._train_step(batch)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss_dict["loss"]).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return self._attach_learning_rate_metric(loss_dict, active_learning_rate)
        loss_dict["loss"].backward()
        self.optimizer.step()
        return self._attach_learning_rate_metric(loss_dict, active_learning_rate)

    def _execute_accumulated_training_step(
        self,
        train_iterator: Any,
    ) -> tuple[dict[str, torch.Tensor], Any]:
        """Consume one or more microbatches and return one optimizer-update metric dict."""

        accumulation_steps = self._gradient_accumulation_steps()
        if accumulation_steps == 1:
            train_iterator, batch = self._next_train_batch(train_iterator)
            return self._execute_training_step(batch), train_iterator

        self.optimizer.zero_grad(set_to_none=True)
        active_learning_rate = self._prepare_learning_rate_for_update()
        metric_sums: dict[str, torch.Tensor] = {}
        for _ in range(accumulation_steps):
            train_iterator, batch = self._next_train_batch(train_iterator)
            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=self.training_autocast_dtype)
                if self.training_autocast_dtype is not None
                else nullcontext()
            )
            with autocast_context:
                loss_dict = self._train_step(batch)
            scaled_loss = loss_dict["loss"] / float(accumulation_steps)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            for key, value in loss_dict.items():
                detached_value = value.detach()
                if key not in metric_sums:
                    metric_sums[key] = detached_value.clone()
                    continue
                metric_sums[key] = metric_sums[key] + detached_value
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        averaged_metrics = {
            key: value / float(accumulation_steps) for key, value in metric_sums.items()
        }
        return self._attach_learning_rate_metric(averaged_metrics, active_learning_rate), train_iterator

    def _assert_checkpoint_backend(self, checkpoint: dict[str, Any], path: str | Path) -> None:
        """Ensure the checkpoint backend matches the currently requested backend."""

        backend = checkpoint_ae_backend(checkpoint)
        if backend != self.cfg.ae_backend:
            raise ValueError(
                f"Checkpoint backend {backend} from {path} does not match requested "
                f"backend {self.cfg.ae_backend}."
            )

    def _assert_checkpoint_autoencoder_compatible(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
    ) -> None:
        """Ensure the checkpoint autoencoder metadata matches the active temporal tokenizer."""

        self._assert_checkpoint_backend(checkpoint, path)
        checkpoint_config, checkpoint_stats = self._extract_checkpoint_autoencoder_metadata(
            checkpoint,
            path,
            require_stats=True,
        )
        active_autoencoder = self.model.autoencoder_config()
        active_config = active_autoencoder.get("config")
        if checkpoint_config.to_dict() != active_config:
            raise ValueError(
                f"Checkpoint autoencoder config from {path} does not match the active Wan "
                "temporal tokenizer config."
            )
        active_stats = active_autoencoder.get("normalization_stats")
        if checkpoint_stats.to_dict() != active_stats:
            raise ValueError(
                f"Checkpoint latent normalization stats from {path} do not match the active "
                "autoencoder statistics."
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
        root_checkpoint_config = checkpoint.get("config")
        if not isinstance(root_checkpoint_config, dict):
            root_checkpoint_config = {}
        checkpoint_mode = str(checkpoint.get("mode", root_checkpoint_config.get("mode", self.cfg.mode)))
        checkpoint_action_representation = resolved_dynamics_action_representation_from_config(
            root_checkpoint_config,
            mode_override=checkpoint_mode,
        )
        active_action_representation = self.cfg.resolved_dynamics_action_representation()
        if checkpoint_action_representation != active_action_representation:
            raise ValueError(
                "Checkpoint dynamics action representation from "
                f"{path} resolves to {checkpoint_action_representation!r}, but the active run "
                f"uses {active_action_representation!r}. Old absolute-action checkpoints are not "
                "compatible with the new relative-action SO101 dynamics path."
            )
        checkpoint_action_scale = resolved_dynamics_action_scale_from_config(
            root_checkpoint_config,
            mode_override=checkpoint_mode,
        )
        active_action_scale = self.cfg.resolved_dynamics_action_scale()
        if not math.isclose(checkpoint_action_scale, active_action_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Checkpoint dynamics action scale={checkpoint_action_scale} from {path} does not "
                f"match the requested action scale={active_action_scale}."
            )
        checkpoint_architecture_version = checkpoint_config.get("architecture_version")
        if checkpoint_architecture_version != DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION:
            raise ValueError(
                "Checkpoint dynamics backbone is not load-compatible with the current "
                f"DreamDojo-mechanics RF DiT. Expected architecture_version="
                f"{DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION!r}, received "
                f"{checkpoint_architecture_version!r} from {path}. Start a fresh dynamics run "
                "or skip loading dynamics weights."
            )
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
        if checkpoint_action_conditioning_mode != self.model.dynamics.cfg.action_conditioning_mode:
            raise ValueError(
                f"Checkpoint dynamics config action_conditioning_mode={checkpoint_action_conditioning_mode} "
                f"from {path} does not match the requested action_conditioning_mode="
                f"{self.model.dynamics.cfg.action_conditioning_mode}."
            )

    def _load_requested_pretrained_weights(self) -> None:
        """Load any optional encoder/decoder or dynamics checkpoint weights."""

        if self.cfg.load_encoder_decoder:
            source_kind, source_payload = self._load_encoder_decoder_source()
            if source_kind == "world_model":
                checkpoint = source_payload
                self._assert_checkpoint_autoencoder_compatible(
                    checkpoint,
                    self.cfg.load_encoder_decoder,
                )
                self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
                self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
            else:
                self._assert_requested_wan_matches_dreamdojo_raw_checkpoint(self.cfg.load_encoder_decoder)
                remapped_state = remap_dreamdojo_wan_state_dict(source_payload)
                self._load_submodule_state("encoder", self.model.encoder, remapped_state)
                self._load_submodule_state("decoder", self.model.decoder, remapped_state)
        if self.cfg.load_dynamics:
            checkpoint = load_training_checkpoint(self.cfg.load_dynamics, self.device)
            self._assert_checkpoint_autoencoder_compatible(checkpoint, self.cfg.load_dynamics)
            self._assert_checkpoint_dynamics_backend(checkpoint, self.cfg.load_dynamics)
            self._load_submodule_state(
                "dynamics",
                self.model.dynamics,
                checkpoint["model_state"],
                allowed_missing_keys=self._allowed_dynamics_missing_keys(checkpoint),
            )
            self._warm_start_extra_dynamics_blocks_from_checkpoint_tail(checkpoint)

    def _load_submodule_state(
        self,
        prefix: str,
        module: torch.nn.Module,
        model_state: dict[str, torch.Tensor],
        *,
        allowed_missing_keys: set[str] | None = None,
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
        expected_state = module.state_dict()
        for key in tuple(submodule_state.keys()):
            if key not in _REGENERABLE_CHECKPOINT_STATE_KEYS:
                continue
            expected_tensor = expected_state.get(key)
            if expected_tensor is None or tuple(expected_tensor.shape) != tuple(submodule_state[key].shape):
                submodule_state.pop(key)
        allowed_missing_keys = set() if allowed_missing_keys is None else set(allowed_missing_keys)
        incompatible = module.load_state_dict(submodule_state, strict=False)
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)
        disallowed_missing_keys = missing_keys - allowed_missing_keys
        if disallowed_missing_keys:
            raise RuntimeError(
                f"Checkpoint is missing required weights for {prefix}: {sorted(disallowed_missing_keys)}."
            )
        if unexpected_keys:
            raise RuntimeError(
                f"Checkpoint has unexpected weights for {prefix}: {sorted(unexpected_keys)}."
            )

    def _allowed_dynamics_missing_keys(self, checkpoint: dict[str, Any]) -> set[str]:
        """Return optional missing dynamics keys for compatible depth-expansion warm starts."""

        checkpoint_dynamics = checkpoint.get("dynamics")
        if not isinstance(checkpoint_dynamics, dict):
            return set()
        checkpoint_config = checkpoint_dynamics.get("config")
        if not isinstance(checkpoint_config, dict):
            return set()
        checkpoint_num_blocks = checkpoint_config.get("num_blocks")
        if not isinstance(checkpoint_num_blocks, int):
            return set()
        current_num_blocks = int(self.model.dynamics.cfg.num_blocks)
        if checkpoint_num_blocks >= current_num_blocks:
            return set()
        allowed_missing_keys: set[str] = set()
        for key in self.model.dynamics.state_dict().keys():
            if not key.startswith("net.blocks."):
                continue
            block_index_text = key.split(".", maxsplit=3)[2]
            if not block_index_text.isdigit():
                continue
            if int(block_index_text) >= checkpoint_num_blocks:
                allowed_missing_keys.add(key)
        return allowed_missing_keys

    def _warm_start_extra_dynamics_blocks_from_checkpoint_tail(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """Seed extra tail blocks from the last compatible checkpoint block when depth increases."""

        checkpoint_dynamics = checkpoint.get("dynamics")
        if not isinstance(checkpoint_dynamics, dict):
            return
        checkpoint_config = checkpoint_dynamics.get("config")
        if not isinstance(checkpoint_config, dict):
            return
        checkpoint_num_blocks = checkpoint_config.get("num_blocks")
        if not isinstance(checkpoint_num_blocks, int):
            return
        current_blocks = self.model.dynamics.net.blocks
        current_num_blocks = len(current_blocks)
        if checkpoint_num_blocks < 1 or checkpoint_num_blocks >= current_num_blocks:
            return
        source_block = current_blocks[checkpoint_num_blocks - 1]
        source_state = source_block.state_dict()
        for block_index in range(checkpoint_num_blocks, current_num_blocks):
            current_blocks[block_index].load_state_dict(source_state)

    def _validation_setup_signature(self) -> dict[str, Any]:
        """Return the current validation-domain signature used for checkpoint selection."""

        return {
            "mode": self.cfg.mode,
            "dataset_format": self.cfg.dataset_format,
            "validation_split": self.cfg.resolved_validation_split(),
            "validation_episodes": self.cfg.resolved_validation_episodes(),
            "validation_max_frames": self.cfg.validation_max_frames,
            "task": self.cfg.task,
            "metaworld_task_index": self.cfg.metaworld_task_index,
            "metaworld_repo_id": self.cfg.metaworld_repo_id,
            "aloha_repo_id": self.cfg.aloha_repo_id,
            "maniskill_traj_h5": self.cfg.maniskill_traj_h5,
            "maniskill_traj_json": self.cfg.maniskill_traj_json,
            "camera": self.cfg.resolved_camera(),
            "frame_start": self.cfg.frame_start,
            "frame_end": self.cfg.frame_end,
            "validation_frame_end": self.cfg.resolved_validation_frame_end(),
            "resolution": self.cfg.resolution,
            "height": self.cfg.resolved_height(),
            "width": self.cfg.resolved_width(),
            "kl_beta": self.cfg.kl_beta,
            "recon_mse_weight": self.cfg.recon_mse_weight,
            "recon_l1_weight": self.cfg.recon_l1_weight,
            "recon_edge_weight": self.cfg.recon_edge_weight,
            "recon_motion_weight": self.cfg.recon_motion_weight,
            "recon_motion_edge_weight": self.cfg.recon_motion_edge_weight,
            "recon_motion_threshold": self.cfg.recon_motion_threshold,
            "recon_motion_dilation_kernel_size": self.cfg.recon_motion_dilation_kernel_size,
            "dynamics_context_frames": (
                int(self.model.dynamics.cfg.context_frames)
                if self.cfg.mode == "dynamics_only"
                else None
            ),
            "dynamics_target_frames": (
                int(self.model.dynamics.cfg.target_frames)
                if self.cfg.mode == "dynamics_only"
                else None
            ),
            "dynamics_validation_conditioning_frame_choices": (
                tuple(int(value) for value in self.model.dynamics.cfg.validation_conditioning_frame_choices)
                if self.cfg.mode == "dynamics_only"
                else None
            ),
            "dynamics_open_rollout_context_frames": (
                int(self.model.dynamics.cfg.open_rollout_context_frames)
                if self.cfg.mode == "dynamics_only"
                else None
            ),
            "dynamics_open_rollout_stride_frames": (
                None
                if self.cfg.mode != "dynamics_only"
                or self.model.dynamics.cfg.open_rollout_stride_frames is None
                else int(self.model.dynamics.cfg.open_rollout_stride_frames)
            ),
            "dynamics_action_representation": (
                self.cfg.resolved_dynamics_action_representation()
                if self.cfg.mode == "dynamics_only"
                else None
            ),
            "dynamics_action_scale": (
                self.cfg.resolved_dynamics_action_scale()
                if self.cfg.mode == "dynamics_only"
                else None
            ),
        }

    def _validation_setup_signature_from_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the saved validation-domain signature for one checkpoint."""

        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            checkpoint_config = {}
        dataset_format = str(checkpoint_config.get("dataset_format", "interactive_world_sim"))
        validation_split = str(checkpoint_config.get("validation_split", ""))
        split = str(checkpoint_config.get("split", "val"))
        resolved_validation_split = (
            "train"
            if dataset_format in {
                "lerobot_metaworld",
                "lerobot_aloha_sim_transfer_cube_scripted",
                "lerobot_so101_base_sim_pickplace",
                "maniskill_replay",
            }
            else validation_split or split
        )
        checkpoint_dynamics = checkpoint.get("dynamics")
        checkpoint_dynamics_config = (
            checkpoint_dynamics.get("config")
            if isinstance(checkpoint_dynamics, dict)
            and isinstance(checkpoint_dynamics.get("config"), dict)
            else {}
        )
        resolution = int(checkpoint_config.get("resolution", 128))
        raw_height = checkpoint_config.get("height")
        raw_width = checkpoint_config.get("width")
        raw_frame_start = checkpoint_config.get("frame_start")
        raw_frame_end = checkpoint_config.get("frame_end")
        raw_validation_max_frames = checkpoint_config.get("validation_max_frames")
        resolved_validation_frame_end = raw_frame_end
        if raw_validation_max_frames is not None:
            validation_start = 0 if raw_frame_start is None else int(raw_frame_start)
            capped_end = validation_start + int(raw_validation_max_frames) - 1
            resolved_validation_frame_end = (
                capped_end if raw_frame_end is None else min(int(raw_frame_end), capped_end)
            )
        return {
            "mode": str(checkpoint.get("mode", checkpoint_config.get("mode", self.cfg.mode))),
            "dataset_format": dataset_format,
            "validation_split": resolved_validation_split,
            "validation_episodes": (
                _normalized_optional_int_tuple(checkpoint_config.get("validation_episodes"))
                or (int(checkpoint_config.get("validation_episode", 0)),)
            ),
            "validation_max_frames": (
                None if raw_validation_max_frames is None else int(raw_validation_max_frames)
            ),
            "task": str(checkpoint_config.get("task", "single_grasp")),
            "metaworld_task_index": checkpoint_config.get("metaworld_task_index"),
            "metaworld_repo_id": str(
                checkpoint_config.get("metaworld_repo_id", METAWORLD_DATASET_ID)
            ),
            "aloha_repo_id": str(
                checkpoint_config.get(
                    "aloha_repo_id",
                    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
                )
            ),
            "maniskill_traj_h5": str(
                checkpoint_config.get("maniskill_traj_h5", MANISKILL_DEFAULT_TRAJ_H5)
            ),
            "maniskill_traj_json": str(
                checkpoint_config.get("maniskill_traj_json", MANISKILL_DEFAULT_TRAJ_JSON)
            ),
            "camera": (
                "observation.image"
                if dataset_format == "lerobot_metaworld"
                else (
                    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN
                    if dataset_format == "lerobot_aloha_sim_transfer_cube_scripted"
                    else (
                        SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN
                        if dataset_format == "lerobot_so101_base_sim_pickplace"
                        else (
                            str(checkpoint_config.get("maniskill_camera", MANISKILL_DEFAULT_CAMERA))
                            if dataset_format == "maniskill_replay"
                            else str(checkpoint_config.get("camera", "camera_1_color"))
                        )
                    )
                )
            ),
            "frame_start": raw_frame_start,
            "frame_end": raw_frame_end,
            "validation_frame_end": resolved_validation_frame_end,
            "resolution": resolution,
            "height": resolution if raw_height is None else int(raw_height),
            "width": resolution if raw_width is None else int(raw_width),
            "kl_beta": float(checkpoint_config.get("kl_beta", 1e-4)),
            "recon_mse_weight": float(checkpoint_config.get("recon_mse_weight", 1.0)),
            "recon_l1_weight": float(checkpoint_config.get("recon_l1_weight", 0.0)),
            "recon_edge_weight": float(checkpoint_config.get("recon_edge_weight", 0.0)),
            "recon_motion_weight": float(checkpoint_config.get("recon_motion_weight", 0.0)),
            "recon_motion_edge_weight": float(
                checkpoint_config.get("recon_motion_edge_weight", 0.0)
            ),
            "recon_motion_threshold": float(checkpoint_config.get("recon_motion_threshold", 0.02)),
            "recon_motion_dilation_kernel_size": int(
                checkpoint_config.get("recon_motion_dilation_kernel_size", 5)
            ),
            "dynamics_context_frames": (
                None
                if self.cfg.mode != "dynamics_only"
                else int(
                    checkpoint_dynamics_config.get(
                        "context_frames",
                        checkpoint_config.get(
                            "dynamics_context_frames",
                            DYNAMICS_FRAME_LAYOUT.context_frames,
                        ),
                    )
                )
            ),
            "dynamics_target_frames": (
                None
                if self.cfg.mode != "dynamics_only"
                else int(
                    checkpoint_dynamics_config.get(
                        "target_frames",
                        checkpoint_config.get(
                            "dynamics_target_frames",
                            DYNAMICS_FRAME_LAYOUT.target_frames,
                        ),
                    )
                )
            ),
            "dynamics_validation_conditioning_frame_choices": (
                None
                if self.cfg.mode != "dynamics_only"
                else _normalized_optional_int_tuple(
                    checkpoint_dynamics_config.get(
                        "validation_conditioning_frame_choices",
                        checkpoint_config.get("dynamics_validation_conditioning_frame_choices"),
                    )
                )
            ),
            "dynamics_open_rollout_context_frames": (
                None
                if self.cfg.mode != "dynamics_only"
                else int(
                    checkpoint_dynamics_config.get(
                        "open_rollout_context_frames",
                        checkpoint_config.get("dynamics_open_rollout_context_frames", 1),
                    )
                )
            ),
            "dynamics_open_rollout_stride_frames": (
                None
                if self.cfg.mode != "dynamics_only"
                else checkpoint_dynamics_config.get(
                    "open_rollout_stride_frames",
                    checkpoint_config.get("dynamics_open_rollout_stride_frames"),
                )
            ),
            "dynamics_action_representation": (
                None
                if self.cfg.mode != "dynamics_only"
                else resolved_dynamics_action_representation_from_config(
                    checkpoint_config,
                    mode_override=str(
                        checkpoint.get("mode", checkpoint_config.get("mode", self.cfg.mode))
                    ),
                )
            ),
            "dynamics_action_scale": (
                None
                if self.cfg.mode != "dynamics_only"
                else resolved_dynamics_action_scale_from_config(
                    checkpoint_config,
                    mode_override=str(
                        checkpoint.get("mode", checkpoint_config.get("mode", self.cfg.mode))
                    ),
                )
            ),
        }

    def _resume_uses_matching_validation_setup(self, checkpoint: dict[str, Any]) -> bool:
        """Return whether resumed validation metrics are comparable to the current run."""

        return self._validation_setup_signature() == self._validation_setup_signature_from_checkpoint(
            checkpoint
        )

    def _load_resume(self) -> int:
        """Restore a previous training checkpoint and return its step."""

        checkpoint = load_training_checkpoint(self.cfg.resume, self.device)
        self._assert_checkpoint_autoencoder_compatible(checkpoint, self.cfg.resume)
        checkpoint_step = int(checkpoint["step"])
        if self.cfg.mode == "ae_only":
            self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
            self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
        else:
            self._assert_checkpoint_dynamics_backend(checkpoint, self.cfg.resume)
            self._load_submodule_state("encoder", self.model.encoder, checkpoint["model_state"])
            self._load_submodule_state("decoder", self.model.decoder, checkpoint["model_state"])
            self._load_submodule_state(
                "dynamics",
                self.model.dynamics,
                checkpoint["model_state"],
                allowed_missing_keys=self._allowed_dynamics_missing_keys(checkpoint),
            )
            self._warm_start_extra_dynamics_blocks_from_checkpoint_tail(checkpoint)
        if checkpoint["optimizer_state"] is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                if "parameter group" not in str(exc):
                    raise
            self._apply_resume_optimizer_overrides()
        if checkpoint.get("scheduler_state") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        self._resume_validation_setup_matches = self._resume_uses_matching_validation_setup(
            checkpoint
        )
        if self._resume_validation_setup_matches:
            restored_best_metric, restored_best_step = self._restore_best_metric_record_from_metrics(
                up_to_step=checkpoint_step
            )
            checkpoint_best_metric = checkpoint.get("best_metric")
            active_best_metric = checkpoint_best_metric if restored_best_metric is None else restored_best_metric
            active_best_step = checkpoint_step if restored_best_metric is None else restored_best_step
            # Interrupted validations can leave a newer checkpoint on disk before the matching
            # validation summary lands in metrics.jsonl. Preserve that newer checkpoint-best value
            # when it clearly beats the replayed history.
            if (
                restored_best_metric is not None
                and checkpoint_best_metric is not None
                and (restored_best_step is None or checkpoint_step > restored_best_step)
                and float(checkpoint_best_metric) < float(restored_best_metric)
            ):
                active_best_metric = float(checkpoint_best_metric)
                active_best_step = checkpoint_step
            self.best_metric = active_best_metric
            if active_best_step == checkpoint_step:
                self._materialize_resumed_best_checkpoint(step=checkpoint_step)
        else:
            self.best_metric = None
        return checkpoint_step

    def _apply_resume_optimizer_overrides(self) -> None:
        """Apply user-requested optimizer overrides after loading resume state."""

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(self.cfg.lr)
            beta2 = float(param_group.get("betas", (self.cfg.optimizer_beta1, 0.999))[1])
            param_group["betas"] = (float(self.cfg.optimizer_beta1), beta2)
            if "initial_lr" in param_group:
                param_group["initial_lr"] = float(self.cfg.lr)

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

    def _materialize_resumed_best_checkpoint(self, step: int) -> None:
        """Write the inherited checkpoint into this run directory when it is still the active best."""

        if not self.cfg.resume:
            return
        best_path = self.checkpoints_dir / "best.pt"
        if best_path.exists():
            return
        self._save_checkpoint(best_path, step)

    def _restore_best_metric_record_from_metrics(
        self,
        up_to_step: int | None = None,
    ) -> tuple[float | None, int | None]:
        """Rebuild the active best metric and its step from saved validation records."""

        restore_path = self._restore_metrics_path()
        if restore_path is None:
            return None, None
        metric_name = self._validation_metric_name()
        best_metric: float | None = None
        best_step: int | None = None
        for line in restore_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record_step = int(record.get("step", 0))
            if up_to_step is not None and record_step > up_to_step:
                break
            validation = record.get("validation")
            if not isinstance(validation, dict):
                continue
            metric_value = validation_metric_value_from_stats(metric_name, validation)
            if metric_value is None:
                continue
            if best_metric is None or metric_value < best_metric:
                best_metric = metric_value
                best_step = record_step
        return best_metric, best_step

    def _restore_best_metric_from_metrics(self, up_to_step: int | None = None) -> float | None:
        """Rebuild the active best metric from saved validation records."""

        best_metric, _ = self._restore_best_metric_record_from_metrics(up_to_step=up_to_step)
        return best_metric

    def _extract_early_stop_metric_value(self, record: dict[str, Any]) -> float | None:
        """Extract the current mode's validation metric from one metrics record."""

        validation = record.get("validation")
        if not isinstance(validation, dict):
            return None
        metric_name = self._validation_metric_name()
        return validation_metric_value_from_stats(metric_name, validation)

    def _restore_early_stop_state(self, step: int) -> None:
        """Replay prior validation metrics to rebuild plateau state for resumes."""

        if not self._early_stop_enabled() or step <= 0:
            return
        if self.cfg.resume and not self._resume_validation_setup_matches:
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

        non_blocking = self.device.type == "cuda" and self._pin_memory_enabled()
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = (
                value.to(self.device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
            )
        return moved

    def _gradient_accumulation_steps(self) -> int:
        """Return the configured microbatch count per optimizer update."""

        return max(int(self.cfg.gradient_accumulation_steps), 1)

    def _effective_train_batch_size(self) -> int:
        """Return the effective batch size seen by each optimizer update."""

        return int(self.cfg.batch_size) * self._gradient_accumulation_steps()

    def _next_train_batch(self, train_iterator: Any) -> tuple[Any, dict[str, Any]]:
        """Return the next training batch, rewinding the iterator across epoch boundaries."""

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(self.train_loader)
            batch = next(train_iterator)
        return train_iterator, self._move_batch_to_device(batch)

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

        frames = batch["frames"]
        output = self.model.autoencode_video(frames, sample_posterior=True)
        flat_frames = rearrange(frames, "b t c h w -> (b t) c h w")
        flat_reconstructed = rearrange(output.reconstructed, "b t c h w -> (b t) c h w")
        frame_indices = torch.arange(frames.shape[1], device=frames.device)
        prev_frames = frames.index_select(1, torch.clamp(frame_indices - 1, min=0))
        next_frames = frames.index_select(
            1,
            torch.clamp(frame_indices + 1, max=int(frames.shape[1]) - 1),
        )
        recon_terms = reconstruction_loss_terms(
            flat_reconstructed,
            flat_frames,
            mse_weight=self.cfg.recon_mse_weight,
            l1_weight=self.cfg.recon_l1_weight,
            edge_weight=self.cfg.recon_edge_weight,
            motion_weight=self.cfg.recon_motion_weight,
            motion_edge_weight=self.cfg.recon_motion_edge_weight,
            motion_threshold=self.cfg.recon_motion_threshold,
            motion_dilation_kernel_size=self.cfg.recon_motion_dilation_kernel_size,
            prev_frame=rearrange(prev_frames, "b t c h w -> (b t) c h w"),
            next_frame=rearrange(next_frames, "b t c h w -> (b t) c h w"),
        )
        ae_loss = recon_terms["recon_loss"] + self.cfg.kl_beta * output.kl_loss
        return {
            "loss": ae_loss,
            "recon_loss": recon_terms["recon_loss"].detach(),
            "recon_mse": recon_terms["recon_mse"],
            "recon_l1": recon_terms["recon_l1"],
            "edge_l1": recon_terms["edge_l1"],
            "motion_l1": recon_terms["motion_l1"],
            "motion_edge_l1": recon_terms["motion_edge_l1"],
            "motion_mask_fraction": recon_terms["motion_mask_fraction"],
            "kl_loss": output.kl_loss.detach(),
            "ae_loss": ae_loss.detach(),
        }

    def _dynamics_only_training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run one frozen-autoencoder mixed-conditioning RF dynamics step."""

        context_frames = batch["context_frames"]
        target_frames = batch["target_frames"]
        actions = batch["actions"]
        future_target_frames = batch["future_target_frames"]
        future_actions = batch["future_actions"]
        with torch.no_grad():
            full_frame_chunk = torch.cat([context_frames, target_frames, future_target_frames], dim=1)
            extended_clean_latent_video = self.model.encode_frame_sequence(
                full_frame_chunk,
                deterministic=True,
            )
        clean_latent_video = extended_clean_latent_video[:, :, : self.model.dynamics.cfg.max_frames]
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
        active_self_forcing_loss_weight = self._active_dynamics_self_forcing_loss_weight()
        active_rollout_self_forcing_loss_weight = self._active_dynamics_rollout_self_forcing_loss_weight()
        latent_rf_self_forcing_mse, self_forcing_stats = self._dynamics_self_forcing_loss(
            clean_latent_video=clean_latent_video,
            extended_clean_latent_video=extended_clean_latent_video,
            predicted_velocity=predicted_velocity,
            dynamics_inputs=dynamics_inputs,
            future_actions=future_actions,
            loss_weight=active_self_forcing_loss_weight,
        )
        weighted_self_forcing_loss = active_self_forcing_loss_weight * latent_rf_self_forcing_mse
        latent_rf_rollout_self_forcing_mse, rollout_self_forcing_stats = (
            self._dynamics_rollout_self_forcing_loss(
                clean_latent_video=clean_latent_video,
                extended_clean_latent_video=extended_clean_latent_video,
                predicted_velocity=predicted_velocity,
                dynamics_inputs=dynamics_inputs,
                future_actions=future_actions,
            )
            if active_rollout_self_forcing_loss_weight > 0.0
            else (predicted_velocity.new_zeros(()), {})
        )
        weighted_rollout_self_forcing_loss = (
            active_rollout_self_forcing_loss_weight * latent_rf_rollout_self_forcing_mse
        )
        total_loss = latent_rf_mse + weighted_self_forcing_loss + weighted_rollout_self_forcing_loss
        metrics = {
            "loss": total_loss,
            "latent_rf_mse": latent_rf_mse.detach(),
            "target_sigma": dynamics_inputs.target_sigmas.mean().detach(),
            "active_self_forcing_loss_weight": torch.tensor(
                active_self_forcing_loss_weight,
                device=latent_rf_mse.device,
            ),
            "active_rollout_self_forcing_loss_weight": torch.tensor(
                active_rollout_self_forcing_loss_weight,
                device=latent_rf_mse.device,
            ),
        }
        if self.cfg.dynamics_self_forcing_loss_weight > 0.0:
            metrics.update(
                {
                    "latent_rf_self_forcing_mse": latent_rf_self_forcing_mse.detach(),
                    "latent_rf_self_forcing_weighted_loss": weighted_self_forcing_loss.detach(),
                    **self_forcing_stats,
                }
            )
        if self.cfg.dynamics_rollout_self_forcing_loss_weight > 0.0:
            metrics.update(
                {
                    "latent_rf_rollout_self_forcing_mse": latent_rf_rollout_self_forcing_mse.detach(),
                    "latent_rf_rollout_self_forcing_weighted_loss": weighted_rollout_self_forcing_loss.detach(),
                    **rollout_self_forcing_stats,
                }
            )
        if (
            self.cfg.dynamics_self_forcing_loss_weight > 0.0
            or self.cfg.dynamics_rollout_self_forcing_loss_weight > 0.0
        ):
            metrics["latent_rf_total_loss"] = total_loss.detach()
        return metrics

    def _dynamics_self_forcing_schedule_scale(self) -> float:
        """Return the warmup/ramp multiplier for the primary self-forcing objective."""

        return self._scheduled_loss_scale(
            warmup_steps=self.cfg.dynamics_self_forcing_warmup_steps,
            ramp_steps=self.cfg.dynamics_self_forcing_ramp_steps,
        )

    def _dynamics_rollout_self_forcing_schedule_scale(self) -> float:
        """Return the warmup/ramp multiplier for the rollout self-forcing auxiliary."""

        return self._scheduled_loss_scale(
            warmup_steps=self.cfg.dynamics_rollout_self_forcing_warmup_steps,
            ramp_steps=self.cfg.dynamics_rollout_self_forcing_ramp_steps,
        )

    def _scheduled_loss_scale(self, *, warmup_steps: int, ramp_steps: int) -> float:
        """Return the active multiplier for a warmup-then-ramp loss schedule."""

        if self.current_step < warmup_steps:
            return 0.0
        if ramp_steps > 0:
            transitioned_steps = self.current_step - warmup_steps + 1
            return min(max(float(transitioned_steps), 0.0) / float(ramp_steps), 1.0)
        return 1.0

    def _active_dynamics_self_forcing_loss_weight(self) -> float:
        """Return the active primary self-forcing weight after the configured warmup window."""

        return (
            float(self.cfg.dynamics_self_forcing_loss_weight)
            * self._dynamics_self_forcing_schedule_scale()
        )

    def _active_dynamics_rollout_self_forcing_loss_weight(self) -> float:
        """Return the active rollout self-forcing auxiliary weight after its own schedule."""

        return (
            float(self.cfg.dynamics_rollout_self_forcing_loss_weight)
            * self._dynamics_rollout_self_forcing_schedule_scale()
        )

    def _dynamics_self_forcing_loss(
        self,
        *,
        clean_latent_video: torch.Tensor,
        extended_clean_latent_video: torch.Tensor,
        predicted_velocity: torch.Tensor,
        dynamics_inputs: DynamicsTrainingInputs,
        future_actions: torch.Tensor,
        loss_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a DreamDojo-inspired causal self-forcing loss for later target frames."""

        zero = predicted_velocity.new_zeros(())
        if loss_weight <= 0.0:
            return zero, {}
        if self.cfg.dynamics_self_forcing_mode == "rollout":
            return self._dynamics_rollout_self_forcing_loss(
                clean_latent_video=clean_latent_video,
                extended_clean_latent_video=extended_clean_latent_video,
                predicted_velocity=predicted_velocity,
                dynamics_inputs=dynamics_inputs,
                future_actions=future_actions,
            )
        if self.model.dynamics.cfg.target_frames < 2:
            return zero, {}
        return self._dynamics_expanded_context_self_forcing_loss(
            clean_latent_video=clean_latent_video,
            predicted_velocity=predicted_velocity,
            dynamics_inputs=dynamics_inputs,
        )

    def _dynamics_expanded_context_self_forcing_loss(
        self,
        *,
        clean_latent_video: torch.Tensor,
        predicted_velocity: torch.Tensor,
        dynamics_inputs: DynamicsTrainingInputs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score the later within-chunk target after expanding the predicted prefix."""

        zero = predicted_velocity.new_zeros(())
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

    def _dynamics_rollout_self_forcing_loss(
        self,
        *,
        clean_latent_video: torch.Tensor,
        extended_clean_latent_video: torch.Tensor,
        predicted_velocity: torch.Tensor,
        dynamics_inputs: DynamicsTrainingInputs,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score future chunks with the same rollout context semantics used at inference."""

        zero = predicted_velocity.new_zeros(())
        rollout_chunks = self.cfg.dynamics_self_forcing_rollout_chunks
        if rollout_chunks < 1:
            return zero, {}
        rollout_context_frames = self.model.dynamics.cfg.open_rollout_context_frames
        rollout_target_frames = self.model.dynamics.cfg.max_frames - rollout_context_frames
        rollout_target_pixel_frames = self.model.temporal_downsample_factor * rollout_target_frames
        future_latent_video = extended_clean_latent_video[:, :, clean_latent_video.shape[2]:]
        expected_future_frames = rollout_chunks * rollout_target_frames
        if future_latent_video.shape[2] < expected_future_frames:
            raise ValueError(
                "extended_clean_latent_video is missing the future target frames required for rollout self-forcing."
            )
        if future_actions.shape[1] < rollout_chunks * rollout_target_pixel_frames:
            raise ValueError(
                "future_actions is missing the action horizon required for rollout self-forcing."
            )
        full_actions = torch.cat([dynamics_inputs.actions, future_actions], dim=1)
        predicted_history = self.model.dynamics.recover_clean_latent_video(
            noisy_latent_video=dynamics_inputs.noisy_latent_video,
            predicted_velocity=predicted_velocity,
            target_sigmas=dynamics_inputs.target_sigmas,
        ).detach()
        total_self_forcing_loss = zero
        stats: dict[str, torch.Tensor] = {}
        for chunk_index in range(rollout_chunks):
            future_start = chunk_index * rollout_target_frames
            future_stop = future_start + rollout_target_frames
            chunk_context = predicted_history[:, :, -rollout_context_frames:]
            chunk_targets = future_latent_video[:, :, future_start:future_stop]
            auxiliary_clean_latent_video = torch.cat([chunk_context, chunk_targets], dim=2)
            reference_noise = torch.randn_like(auxiliary_clean_latent_video)
            auxiliary_noisy_latent_video, auxiliary_target_velocity = self.model.dynamics.flow.interpolate(
                noise=reference_noise,
                clean=auxiliary_clean_latent_video,
                sigmas=dynamics_inputs.target_sigmas,
            )
            auxiliary_condition_mask = self.model.dynamics.make_condition_mask(
                auxiliary_clean_latent_video,
                num_conditional_frames=rollout_context_frames,
            )
            action_start = (chunk_index + 1) * rollout_target_pixel_frames
            action_stop = action_start + self.model.dynamics.cfg.num_action_per_chunk
            auxiliary_actions = full_actions[:, action_start:action_stop]
            auxiliary_predicted_velocity = self.model.dynamics(
                noisy_latent_video=auxiliary_noisy_latent_video,
                timesteps=dynamics_inputs.timesteps,
                condition_mask=auxiliary_condition_mask,
                actions=auxiliary_actions,
                conditioning_latent_video=auxiliary_clean_latent_video,
                target_velocity=auxiliary_target_velocity,
                use_video_condition=True,
            )
            step_loss = F.mse_loss(
                auxiliary_predicted_velocity[:, :, rollout_context_frames:],
                auxiliary_target_velocity[:, :, rollout_context_frames:],
            )
            total_self_forcing_loss = total_self_forcing_loss + step_loss
            stats[f"latent_rf_self_forcing_rollout_mse_chunk{chunk_index + 1}"] = step_loss.detach()
            predicted_chunk = self.model.dynamics.recover_clean_latent_video(
                noisy_latent_video=auxiliary_noisy_latent_video,
                predicted_velocity=auxiliary_predicted_velocity,
                target_sigmas=dynamics_inputs.target_sigmas,
            ).detach()
            predicted_history = torch.cat(
                [predicted_history, predicted_chunk[:, :, rollout_context_frames:]],
                dim=2,
            )
        return total_self_forcing_loss / rollout_chunks, stats

    @torch.no_grad()
    def _validate_ae_only_frames(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float, float, float, float, float, float]:
        """Validate AE reconstructions in chunks to avoid full-clip CUDA OOMs."""

        window_frames = self.model.latent_frames_to_pixel_frames(self.model.dynamics.cfg.max_frames)
        stride_frames = self.model.dynamics.cfg.num_action_per_chunk
        total_frames = int(frames.shape[0])
        reconstructed_chunks: list[torch.Tensor] = []
        kl_weighted_sum = 0.0
        total_unique_frames = 0
        for start in range(0, total_frames, stride_frames):
            actual_stop = min(start + window_frames, total_frames)
            frame_chunk = frames[start:actual_stop]
            if frame_chunk.shape[0] < window_frames:
                pad_frame = frame_chunk[-1:].expand(window_frames - frame_chunk.shape[0], -1, -1, -1)
                frame_chunk = torch.cat([frame_chunk, pad_frame], dim=0)
            output = self.model.autoencode_video(frame_chunk.unsqueeze(0), sample_posterior=False)
            reconstructed_chunk = output.reconstructed[0, : actual_stop - start]
            if start == 0:
                reconstructed_chunks.append(reconstructed_chunk.detach().cpu())
                unique_frames = int(reconstructed_chunk.shape[0])
            else:
                reconstructed_chunks.append(reconstructed_chunk[1:].detach().cpu())
                unique_frames = max(int(reconstructed_chunk.shape[0]) - 1, 0)
            kl_weighted_sum += float(output.kl_loss.item()) * unique_frames
            total_unique_frames += unique_frames
            if actual_stop == total_frames:
                break
        reconstructed = torch.cat(reconstructed_chunks, dim=0)
        prev_frames = frames.index_select(
            0,
            torch.clamp(torch.arange(total_frames, device=frames.device) - 1, min=0),
        )
        next_frames = frames.index_select(
            0,
            torch.clamp(torch.arange(total_frames, device=frames.device) + 1, max=total_frames - 1),
        )
        recon_terms = reconstruction_loss_terms(
            reconstructed.to(device=frames.device),
            frames,
            mse_weight=self.cfg.recon_mse_weight,
            l1_weight=self.cfg.recon_l1_weight,
            edge_weight=self.cfg.recon_edge_weight,
            motion_weight=self.cfg.recon_motion_weight,
            motion_edge_weight=self.cfg.recon_motion_edge_weight,
            motion_threshold=self.cfg.recon_motion_threshold,
            motion_dilation_kernel_size=self.cfg.recon_motion_dilation_kernel_size,
            prev_frame=prev_frames,
            next_frame=next_frames,
        )
        recon_loss = float(recon_terms["recon_loss"].item())
        recon_mse = float(recon_terms["recon_mse"].item())
        recon_l1 = float(recon_terms["recon_l1"].item())
        edge_l1 = float(recon_terms["edge_l1"].item())
        motion_l1 = float(recon_terms["motion_l1"].item())
        motion_edge_l1 = float(recon_terms["motion_edge_l1"].item())
        motion_mask_fraction = float(recon_terms["motion_mask_fraction"].item())
        kl_loss = kl_weighted_sum / max(total_unique_frames, 1)
        ae_loss = recon_loss + self.cfg.kl_beta * kl_loss
        return (
            reconstructed,
            recon_loss,
            recon_mse,
            recon_l1,
            edge_l1,
            motion_l1,
            motion_edge_l1,
            motion_mask_fraction,
            kl_loss,
            ae_loss,
        )

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
        context_latent_frames = (
            min(supported_counts)
            if num_conditional_frames is None
            else num_conditional_frames
        )
        if context_latent_frames not in supported_counts:
            raise ValueError(
                f"Expected num_conditional_frames from {supported_counts}, received {context_latent_frames}."
            )
        target_latent_frames = self.model.dynamics.cfg.max_frames - context_latent_frames
        context_pixel_frames = self.model.latent_frames_to_pixel_frames(context_latent_frames)
        target_pixel_frames = self.model.temporal_downsample_factor * target_latent_frames
        full_chunk_pixel_frames = self.model.latent_frames_to_pixel_frames(
            self.model.dynamics.cfg.max_frames
        )
        if frames.shape[0] < full_chunk_pixel_frames:
            raise ValueError(
                f"Dynamics validation requires at least {full_chunk_pixel_frames} pixel frames."
            )
        predicted_frames = [frames[:context_pixel_frames].detach().cpu()]
        total_frame_squared_error = 0.0
        total_frame_values = 0
        total_latent_squared_error = 0.0
        total_latent_values = 0
        per_target_frame_squared_error = [0.0 for _ in range(target_pixel_frames)]
        per_target_frame_values = [0 for _ in range(target_pixel_frames)]
        per_target_latent_squared_error = [0.0 for _ in range(target_latent_frames)]
        per_target_latent_values = [0 for _ in range(target_latent_frames)]
        predicted_motion_l1_total = 0.0
        ground_truth_motion_l1_total = 0.0
        motion_value_count = 0
        for target_start in range(context_pixel_frames, int(frames.shape[0]), target_pixel_frames):
            target_stop = min(target_start + target_pixel_frames, int(frames.shape[0]))
            target_chunk = frames[target_start:target_stop]
            padded_target_chunk = target_chunk
            if target_chunk.shape[0] < target_pixel_frames:
                pad_frame = target_chunk[-1:].expand(
                    target_pixel_frames - target_chunk.shape[0],
                    -1,
                    -1,
                    -1,
                )
                padded_target_chunk = torch.cat([target_chunk, pad_frame], dim=0)
            current_frames = frames[target_start - context_pixel_frames : target_start]
            clean_chunk_frames = torch.cat([current_frames, padded_target_chunk], dim=0).unsqueeze(0)
            clean_chunk_latent = self.model.encode_frame_sequence(clean_chunk_frames, deterministic=True)
            current_latent = clean_chunk_latent[:, :, :context_latent_frames]
            target_latent = clean_chunk_latent[:, :, context_latent_frames:]
            action_window = None
            if actions is not None:
                action_start = target_start - context_pixel_frames
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
            generator.manual_seed(target_start + context_latent_frames * 1000)
            predicted_latent = self.model.predict_next_latent(
                current_latent,
                actions=action_window,
                generator=generator,
            )
            predicted_frame = self.model.decode_target_latents(
                current_latent,
                predicted_latent,
                context_pixel_frames=context_pixel_frames,
                target_pixel_frames=int(target_chunk.shape[0]),
            )[0]
            predicted_frames.append(predicted_frame.detach().cpu())
            total_frame_squared_error += float(
                F.mse_loss(predicted_frame, target_chunk, reduction="sum").item()
            )
            total_frame_values += int(target_chunk.numel())
            if target_chunk.shape[0] == target_pixel_frames:
                total_latent_squared_error += float(
                    F.mse_loss(
                        predicted_latent,
                        target_latent,
                        reduction="sum",
                    ).item()
                )
                total_latent_values += int(target_latent.numel())
            for offset in range(int(target_chunk.shape[0])):
                per_target_frame_squared_error[offset] += float(
                    F.mse_loss(
                        predicted_frame[offset:offset + 1],
                        target_chunk[offset:offset + 1],
                        reduction="sum",
                    ).item()
                )
                per_target_frame_values[offset] += int(target_chunk[offset:offset + 1].numel())
            if target_chunk.shape[0] == target_pixel_frames:
                for offset in range(target_latent_frames):
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
            "seed_frames": int(context_pixel_frames),
            "loss_frames": int(frames.shape[0] - context_pixel_frames),
            "conditioning_latent_frames": int(context_latent_frames),
            "target_latent_frames": int(target_latent_frames),
            "target_pixel_frames": int(target_pixel_frames),
            "next_frame_mse": total_frame_squared_error / max(total_frame_values, 1),
            "next_latent_mse": total_latent_squared_error / max(total_latent_values, 1),
            "validation_style": (
                f"teacher_forced_{context_latent_frames}_context_{target_latent_frames}_target"
            ),
        }
        for offset in range(target_pixel_frames):
            if per_target_frame_values[offset] > 0:
                stats[f"next_frame_mse_target_{offset}"] = (
                    per_target_frame_squared_error[offset] / per_target_frame_values[offset]
                )
        for offset in range(target_latent_frames):
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

        context_latent_frames = self.model.dynamics.cfg.open_rollout_context_frames
        context_pixel_frames = self.model.latent_frames_to_pixel_frames(context_latent_frames)
        if frames.shape[0] <= context_pixel_frames:
            raise ValueError(
                f"Open-rollout validation requires more than {context_pixel_frames} pixel frames."
            )
        rollout_steps = int(frames.shape[0]) - context_pixel_frames
        seed_frames = frames[:context_pixel_frames].unsqueeze(0)
        rollout_actions = None if actions is None else actions.unsqueeze(0)
        initial_stride_latent_frames = self.model.resolved_rollout_stride_frames(
            context_latent_frames
        )
        predicted = self.model.rollout(
            seed_frames,
            steps=rollout_steps,
            actions=rollout_actions,
            stride_frames=self.model.dynamics.cfg.open_rollout_stride_frames,
        )[0]
        predicted_targets = predicted[context_pixel_frames:]
        target_frames = frames[context_pixel_frames:]
        predicted_motion_l1 = 0.0
        ground_truth_motion_l1 = 0.0
        frame_mse = float(F.mse_loss(predicted_targets, target_frames).item())
        frame_l1 = float(F.l1_loss(predicted_targets, target_frames).item())
        motion_ratio: float | None = None
        if predicted_targets.shape[0] > 1:
            predicted_motion_l1 = float(
                torch.abs(predicted_targets[1:] - predicted_targets[:-1]).mean().item()
            )
            ground_truth_motion_l1 = float(
                torch.abs(target_frames[1:] - target_frames[:-1]).mean().item()
            )
            motion_ratio = compute_motion_ratio(predicted_motion_l1, ground_truth_motion_l1)
        stats = {
            "open_rollout_seed_frames": int(context_pixel_frames),
            "open_rollout_loss_frames": int(rollout_steps),
            "open_rollout_stride_frames": (
                int(self.model.dynamics.cfg.open_rollout_stride_frames)
                if self.model.dynamics.cfg.open_rollout_stride_frames is not None
                else None
            ),
            "open_rollout_initial_stride_frames": int(
                self.model.temporal_downsample_factor * initial_stride_latent_frames
            ),
            "open_rollout_context_latent_frames": int(context_latent_frames),
            "open_rollout_decoded_frame_count": int(predicted.shape[0]),
            "open_rollout_predicted_frame_count": int(predicted.shape[0]),
            "open_rollout_frame_mse": frame_mse,
            "open_rollout_frame_l1": frame_l1,
            "open_rollout_consistency_score": open_rollout_consistency_score(
                frame_mse,
                motion_ratio,
            ),
            "open_rollout_validation_style": "open_rollout_autoregressive",
        }
        if predicted_targets.shape[0] > 1:
            stats["open_rollout_predicted_target_motion_l1"] = predicted_motion_l1
            stats["open_rollout_ground_truth_target_motion_l1"] = ground_truth_motion_l1
            stats["open_rollout_target_motion_ratio"] = motion_ratio
            stats["open_rollout_motion_log_error"] = motion_ratio_log_error(motion_ratio)
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

    def _validate_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        """Run validation for one batch and return frames, preview, context, and stats."""

        frames = batch["frames"][0]
        if self.cfg.mode == "ae_only":
            (
                reconstructed,
                recon_loss,
                recon_mse,
                recon_l1,
                edge_l1,
                motion_l1,
                motion_edge_l1,
                motion_mask_fraction,
                kl_loss,
                ae_loss,
            ) = self._validate_ae_only_frames(frames)
            stats = {
                "episode": int(batch["episode_idx"].reshape(-1)[0].item()),
                "input_frame_count": int(frames.shape[0]),
                "decoded_frame_count": int(reconstructed.shape[0]),
                "recon_loss": float(recon_loss),
                "recon_mse": float(recon_mse),
                "recon_l1": float(recon_l1),
                "edge_l1": float(edge_l1),
                "motion_l1": float(motion_l1),
                "motion_edge_l1": float(motion_edge_l1),
                "motion_mask_fraction": float(motion_mask_fraction),
                "kl_loss": kl_loss,
                "ae_loss": ae_loss,
                "mode": self.cfg.mode,
                "ae_backend": self.cfg.ae_backend,
                "dynamics_backend": self.model.dynamics_backend,
            }
            return frames, reconstructed, 0, stats

        clip_actions = batch.get("actions")
        validation_context_choices = self.model.dynamics.cfg.validation_conditioning_frame_choices
        primary_context_latent_frames = validation_context_choices[0]
        preview_frames, dynamics_stats = self._validate_dynamics_one_step(
            frames,
            actions=None if clip_actions is None else clip_actions[0],
            num_conditional_frames=primary_context_latent_frames,
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
        open_rollout_stats: dict[str, Any] = {}
        if self.cfg.resolved_run_open_rollout_validation():
            _, open_rollout_stats = self._validate_dynamics_open_rollout(
                frames,
                actions=None if clip_actions is None else clip_actions[0],
            )
        stats = {
            "episode": int(batch["episode_idx"].reshape(-1)[0].item()),
            **dynamics_stats,
            **self._suffix_validation_stats(
                dynamics_stats,
                (
                    f"{primary_context_latent_frames}to"
                    f"{self.model.dynamics.cfg.max_frames - primary_context_latent_frames}"
                ),
            ),
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
            "open_rollout_stride_frames": (
                None
                if self.model.dynamics.cfg.open_rollout_stride_frames is None
                else int(self.model.dynamics.cfg.open_rollout_stride_frames)
            ),
            "mode": self.cfg.mode,
            "ae_backend": self.cfg.ae_backend,
            "dynamics_backend": self.model.dynamics_backend,
        }
        teacher_forced_frame_metrics = teacher_forced_next_frame_mse_stats(stats)
        if teacher_forced_frame_metrics:
            stats["worst_case_next_frame_mse"] = max(teacher_forced_frame_metrics.values())
        return (
            frames,
            preview_frames,
            self.model.latent_frames_to_pixel_frames(primary_context_latent_frames),
            stats,
        )

    def _aggregate_validation_stats(
        self,
        per_episode_stats: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate validation stats across multiple clips by averaging numeric fields."""

        if not per_episode_stats:
            raise ValueError("Expected at least one validation stats record to aggregate.")
        if len(per_episode_stats) == 1:
            return dict(per_episode_stats[0])
        aggregated: dict[str, Any] = {
            "validation_episode_count": len(per_episode_stats),
            "validation_episodes": [
                int(stats["episode"])
                for stats in per_episode_stats
                if isinstance(stats.get("episode"), Integral)
            ],
        }
        all_keys = sorted({key for stats in per_episode_stats for key in stats})
        for key in all_keys:
            values = [stats[key] for stats in per_episode_stats if key in stats]
            if len(values) != len(per_episode_stats):
                continue
            if key == "episode":
                continue
            if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
                if all(isinstance(value, Integral) for value in values) and len(set(values)) == 1:
                    aggregated[key] = int(values[0])
                else:
                    aggregated[key] = float(
                        sum(float(value) for value in values) / len(values)
                    )
                continue
            first_value = values[0]
            if all(value == first_value for value in values):
                aggregated[key] = first_value
        return aggregated

    @torch.no_grad()
    def _validate(self, step: int) -> dict[str, Any]:
        """Run validation, export artifacts, and update the best checkpoint."""

        self.model.eval()
        validation_results: list[dict[str, Any]] = []
        for raw_batch in self.val_loader:
            batch = self._move_batch_to_device(raw_batch)
            frames, preview_frames, context_frames, stats = self._validate_batch(batch)
            validation_results.append(
                {
                    "frames": frames.detach().cpu(),
                    "preview_frames": preview_frames.detach().cpu(),
                    "context_frames": int(context_frames),
                    "stats": stats,
                }
            )
        stats = self._aggregate_validation_stats([result["stats"] for result in validation_results])
        metric_name = self._validation_metric_name()
        metric_value = float(stats[metric_name])
        is_best_checkpoint = self.best_metric is None or metric_value < self.best_metric
        if is_best_checkpoint:
            self.best_metric = metric_value
            self._save_checkpoint(self.checkpoints_dir / "best.pt", step)
        output_dir = self.run_dir / "samples" / f"step_{step:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_frame_counts: list[int] = []
        for result in validation_results:
            episode_value = result["stats"].get("episode", 0)
            episode_label = (
                int(episode_value)
                if isinstance(episode_value, Integral)
                else len(exported_frame_counts)
            )
            grid_path = output_dir / f"episode_{episode_label}_grid.png"
            video_path = output_dir / f"episode_{episode_label}.mp4"
            stats_path = output_dir / f"episode_{episode_label}_stats.json"
            build_side_by_side_grid(
                original=result["frames"],
                reconstructed=result["preview_frames"],
                max_frames=int(result["frames"].shape[0]),
                context_frames=int(result["context_frames"]),
            ).save(grid_path)
            exported_frame_count = write_side_by_side_mp4(
                original=result["frames"],
                reconstructed=result["preview_frames"],
                output_path=video_path,
                duration_ms=120,
                context_frames=int(result["context_frames"]),
            )
            exported_frame_counts.append(int(exported_frame_count))
            per_episode_stats = dict(result["stats"])
            per_episode_stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
            per_episode_stats["best_checkpoint"] = str(self.checkpoints_dir / "best.pt")
            per_episode_stats["is_best_checkpoint"] = bool(is_best_checkpoint)
            per_episode_stats["elapsed_run_seconds"] = self._elapsed_run_seconds()
            per_episode_stats["exported_video_frame_count"] = int(exported_frame_count)
            save_json(stats_path, per_episode_stats)
        stats["checkpoint"] = str(self.checkpoints_dir / "last.pt")
        stats["best_checkpoint"] = str(self.checkpoints_dir / "best.pt")
        stats["is_best_checkpoint"] = bool(is_best_checkpoint)
        stats["elapsed_run_seconds"] = self._elapsed_run_seconds()
        if len(exported_frame_counts) == 1:
            stats["exported_video_frame_count"] = int(exported_frame_counts[0])
        else:
            stats["exported_video_frame_count"] = float(
                sum(exported_frame_counts) / len(exported_frame_counts)
            )
            stats["validation_episode_count"] = len(validation_results)
            summary_path = output_dir / "validation_summary.json"
            save_json(summary_path, stats)
        append_jsonl(self.metrics_path, {"step": step, "validation": stats})
        if self.wandb_logger is not None:
            self.wandb_logger.log_validation_metrics(step, stats)
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
        if self.wandb_logger is not None:
            self.wandb_logger.log_early_stop_metrics(step, early_stop_record)
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

    def _should_run_validation(self, step: int) -> bool:
        """Return whether validation should run at the current training step."""

        return (
            self.cfg.validation_interval > 0
            and step >= self.cfg.validation_start_step
            and step % self.cfg.validation_interval == 0
        )

    def _start_wandb_run(self) -> WandbRunLogger | None:
        """Create one optional W&B logger for the current training run."""

        return WandbRunLogger.create(
            enabled=self.cfg.wandb_enabled,
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            group=self.cfg.wandb_group,
            name=self.cfg.wandb_name or self.run_name,
            tags=self.cfg.wandb_tags,
            mode=self.cfg.wandb_mode,
            config=self.cfg.to_dict(),
            run_dir=self.run_dir,
            resume_checkpoint=self.cfg.resume,
            run_id=self.cfg.wandb_run_id,
        )

    def _wandb_run_summary(self) -> dict[str, Any]:
        """Return flat run metadata worth mirroring into W&B summary fields."""

        return {
            "clip_dataset_format": self.cfg.dataset_format,
            "effective_train_batch_size": int(self._effective_train_batch_size()),
            "mode": self.cfg.mode,
            "resolved_train_loader_num_workers": int(self._resolved_train_loader_num_workers),
            "train_loader_num_workers_source": self._train_loader_worker_resolution_source,
        }

    def run(self) -> None:
        """Execute the configured training loop end to end."""

        self.run_started_at_monotonic = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.run_dir / "config.json", self.cfg.to_dict())
        self._restore_early_stop_state(self.current_step)
        self.wandb_logger = self._start_wandb_run()
        append_jsonl(
            self.metrics_path,
            {
                "run_start": {
                    "config": self.cfg.to_dict(),
                    "mode": self.cfg.mode,
                    "clip_metadata": self.cfg.clip_metadata(),
                    "effective_train_batch_size": int(self._effective_train_batch_size()),
                    "resolved_train_loader_num_workers": int(
                        self._resolved_train_loader_num_workers
                    ),
                    "train_loader_num_workers_source": self._train_loader_worker_resolution_source,
                }
            },
        )
        if self.wandb_logger is not None:
            self.wandb_logger.update_summary("run", self._wandb_run_summary())

        try:
            train_iterator = iter(self.train_loader)
            last_validation_step = -1
            while self.current_step < self.cfg.max_steps:
                self.model.train()
                loss_dict: dict[str, torch.Tensor] | None = None
                try:
                    loss_dict, train_iterator = self._execute_accumulated_training_step(train_iterator)
                except RuntimeError as error:
                    if not self._is_cuda_oom(error):
                        raise
                    recovered = self._reduce_batch_size_after_oom()
                    loss_dict = None
                    error = None
                    self._cleanup_after_cuda_oom()
                    if not recovered:
                        raise
                    train_iterator = iter(self.train_loader)
                    continue
                self.current_step += 1

                metric_record = {
                    "step": self.current_step,
                    "loss": float(loss_dict["loss"].detach().cpu()),
                    "elapsed_run_seconds": self._elapsed_run_seconds(),
                }
                for key, value in loss_dict.items():
                    if key == "loss":
                        continue
                    metric_record[key] = float(value.detach().cpu())
                append_jsonl(self.metrics_path, metric_record)
                if self.wandb_logger is not None:
                    self.wandb_logger.log_training_metrics(self.current_step, metric_record)
                if self.current_step == 1 or (
                    self.cfg.log_interval > 0 and self.current_step % self.cfg.log_interval == 0
                ):
                    print(json.dumps(metric_record, sort_keys=True))

                if self._should_run_validation(self.current_step):
                    validation_stats = self._validate(self.current_step)
                    last_validation_step = self.current_step
                    print(
                        json.dumps(
                            {"step": self.current_step, "validation": validation_stats},
                            sort_keys=True,
                        )
                    )
                    if self._handle_validation_early_stop(self.current_step, validation_stats):
                        return

                if (
                    self.cfg.checkpoint_interval > 0
                    and self.current_step % self.cfg.checkpoint_interval == 0
                ):
                    self._save_checkpoint(self.checkpoints_dir / "last.pt", self.current_step)

            if (
                last_validation_step != self.current_step
                and self.current_step >= self.cfg.validation_start_step
            ):
                validation_stats = self._validate(self.current_step)
                print(
                    json.dumps(
                        {"step": self.current_step, "validation": validation_stats},
                        sort_keys=True,
                    )
                )
                self._handle_validation_early_stop(self.current_step, validation_stats)
            self._save_checkpoint(self.checkpoints_dir / "last.pt", self.current_step)
        finally:
            if self.wandb_logger is not None:
                self.wandb_logger.finish()
                self.wandb_logger = None
