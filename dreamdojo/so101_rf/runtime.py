"""Shared runtime helpers for the standalone SO101 rectified-flow DreamDojo sandbox."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from dreamdojo.so101_rf.dynamics_transformer import (
    DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION,
    DYNAMICS_FRAME_LAYOUT,
)
from dreamdojo.so101_rf.model import WorldModel
from dreamdojo.so101_rf.wan_vae import (
    WanVAEConfig,
    dreamdojo_wan_state_dict,
    remap_dreamdojo_wan_state_dict,
)


DREAMDOJO_UPSTREAM_REPO_URL = "https://github.com/NVIDIA/DreamDojo.git"
DREAMDOJO_UPSTREAM_COMMIT = "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"
DREAMDOJO_UPSTREAM_COMMIT_DATE = "2026-03-21T18:59:22Z"
STANDALONE_CHECKPOINT_KIND = "dreamdojo_so101_rf_v1"
_TEACHER_FORCED_NEXT_FRAME_MSE_PATTERN = re.compile(r"^next_frame_mse(?:_\d+to\d+)?$")


def resolved_training_autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Return the preferred autocast dtype for one training device."""

    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def teacher_forced_next_frame_mse_stats(stats: dict[str, Any]) -> dict[str, float]:
    """Collect numeric teacher-forced next-frame MSE values from one stats dictionary."""

    collected: dict[str, float] = {}
    for key, value in stats.items():
        if not _TEACHER_FORCED_NEXT_FRAME_MSE_PATTERN.fullmatch(key):
            continue
        if isinstance(value, (int, float)):
            collected[key] = float(value)
    return collected


def compute_motion_ratio(predicted_motion_l1: float, ground_truth_motion_l1: float) -> float:
    """Return a stable predicted-to-ground-truth motion ratio."""

    if ground_truth_motion_l1 <= 1e-12:
        if predicted_motion_l1 <= 1e-12:
            return 1.0
        return math.inf
    return predicted_motion_l1 / ground_truth_motion_l1


def motion_ratio_log_error(motion_ratio: float) -> float:
    """Return a symmetric log-domain penalty for one motion ratio."""

    if not math.isfinite(motion_ratio) or motion_ratio <= 0.0:
        return math.inf
    return abs(math.log(motion_ratio))


def open_rollout_consistency_score(
    frame_mse: float,
    motion_ratio: float | None,
) -> float:
    """Combine open-rollout frame error with a motion-ratio penalty."""

    if not math.isfinite(frame_mse):
        return math.inf
    if motion_ratio is None:
        return float(frame_mse)
    log_error = motion_ratio_log_error(float(motion_ratio))
    if not math.isfinite(log_error):
        return math.inf
    return float(frame_mse) * (1.0 + log_error)


@torch.no_grad()
def run_teacher_forced_validation(
    model: WorldModel,
    frames: torch.Tensor,
    actions: torch.Tensor | None = None,
    *,
    num_conditional_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run teacher-forced validation over one unbatched frame clip."""

    supported_counts = model.dynamics.cfg.conditioning_frame_choices
    context_latent_frames = (
        min(supported_counts)
        if num_conditional_frames is None
        else int(num_conditional_frames)
    )
    if context_latent_frames not in supported_counts:
        raise ValueError(
            f"Expected num_conditional_frames from {supported_counts}, received {context_latent_frames}."
        )
    target_latent_frames = model.dynamics.cfg.max_frames - context_latent_frames
    context_pixel_frames = model.latent_frames_to_pixel_frames(context_latent_frames)
    target_pixel_frames = model.temporal_downsample_factor * target_latent_frames
    full_chunk_pixel_frames = model.latent_frames_to_pixel_frames(model.dynamics.cfg.max_frames)
    if frames.shape[0] < full_chunk_pixel_frames:
        raise ValueError(
            f"Teacher-forced validation requires at least {full_chunk_pixel_frames} pixel frames."
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
        clean_chunk_latent = model.encode_frame_sequence(clean_chunk_frames, deterministic=True)
        current_latent = clean_chunk_latent[:, :, :context_latent_frames]
        target_latent = clean_chunk_latent[:, :, context_latent_frames:]
        action_window = None
        if actions is not None:
            action_start = target_start - context_pixel_frames
            action_stop = action_start + model.dynamics.cfg.num_action_per_chunk
            action_window = actions[action_start:min(action_stop, int(actions.shape[0]))]
            if action_window.shape[0] < model.dynamics.cfg.num_action_per_chunk:
                pad_actions = torch.zeros(
                    model.dynamics.cfg.num_action_per_chunk - action_window.shape[0],
                    model.dynamics.cfg.action_dim,
                    device=actions.device,
                    dtype=actions.dtype,
                )
                action_window = torch.cat([action_window, pad_actions], dim=0)
            action_window = action_window.unsqueeze(0)
        generator = torch.Generator(device=current_latent.device.type)
        generator.manual_seed(target_start + context_latent_frames * 1000)
        predicted_latent = model.predict_next_latent(
            current_latent,
            actions=action_window,
            generator=generator,
        )
        predicted_frame = model.decode_target_latents(
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
                F.mse_loss(predicted_latent, target_latent, reduction="sum").item()
            )
            total_latent_values += int(target_latent.numel())
        for offset in range(int(target_chunk.shape[0])):
            per_target_frame_squared_error[offset] += float(
                F.mse_loss(
                    predicted_frame[offset : offset + 1],
                    target_chunk[offset : offset + 1],
                    reduction="sum",
                ).item()
            )
            per_target_frame_values[offset] += int(target_chunk[offset : offset + 1].numel())
        if target_chunk.shape[0] == target_pixel_frames:
            for offset in range(target_latent_frames):
                per_target_latent_squared_error[offset] += float(
                    F.mse_loss(
                        predicted_latent[:, :, offset : offset + 1],
                        target_latent[:, :, offset : offset + 1],
                        reduction="sum",
                    ).item()
                )
                per_target_latent_values[offset] += int(
                    target_latent[:, :, offset : offset + 1].numel()
                )
        if target_chunk.shape[0] > 1:
            predicted_motion_l1_total += float(
                torch.abs(predicted_frame[1:] - predicted_frame[:-1]).sum().item()
            )
            ground_truth_motion_l1_total += float(
                torch.abs(target_chunk[1:] - target_chunk[:-1]).sum().item()
            )
            motion_value_count += int((target_chunk.shape[0] - 1) * target_chunk[0].numel())
    preview_frames = torch.cat(predicted_frames, dim=0)
    stats: dict[str, Any] = {
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
        "validation_style": f"teacher_forced_{context_latent_frames}_context_{target_latent_frames}_target",
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
        stats["target_motion_ratio"] = compute_motion_ratio(predicted_motion_l1, ground_truth_motion_l1)
    frame_metrics = teacher_forced_next_frame_mse_stats(stats)
    if frame_metrics:
        stats["worst_case_next_frame_mse"] = max(frame_metrics.values())
    return preview_frames, stats


@torch.no_grad()
def run_open_rollout_validation(
    model: WorldModel,
    frames: torch.Tensor,
    actions: torch.Tensor | None = None,
    *,
    stride_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run autoregressive open-rollout validation over one unbatched frame clip."""

    context_latent_frames = model.dynamics.cfg.open_rollout_context_frames
    context_pixel_frames = model.latent_frames_to_pixel_frames(context_latent_frames)
    if frames.shape[0] <= context_pixel_frames:
        raise ValueError(
            f"Open-rollout validation requires more than {context_pixel_frames} pixel frames."
        )
    rollout_steps = int(frames.shape[0]) - context_pixel_frames
    seed_frames = frames[:context_pixel_frames].unsqueeze(0)
    rollout_actions = None if actions is None else actions.unsqueeze(0)
    initial_stride_latent_frames = model.resolved_rollout_stride_frames(
        context_latent_frames,
        stride_frames=stride_frames,
    )
    predicted = model.rollout(
        seed_frames,
        steps=rollout_steps,
        actions=rollout_actions,
        stride_frames=stride_frames,
    )[0]
    predicted_targets = predicted[context_pixel_frames:]
    target_frames = frames[context_pixel_frames:]
    frame_mse = float(F.mse_loss(predicted_targets, target_frames).item())
    frame_l1 = float(F.l1_loss(predicted_targets, target_frames).item())
    predicted_motion_l1 = 0.0
    ground_truth_motion_l1 = 0.0
    motion_ratio: float | None = None
    if predicted_targets.shape[0] > 1:
        predicted_motion_l1 = float(
            torch.abs(predicted_targets[1:] - predicted_targets[:-1]).mean().item()
        )
        ground_truth_motion_l1 = float(
            torch.abs(target_frames[1:] - target_frames[:-1]).mean().item()
        )
        motion_ratio = compute_motion_ratio(predicted_motion_l1, ground_truth_motion_l1)
    stats: dict[str, Any] = {
        "input_frame_count": int(frames.shape[0]),
        "decoded_frame_count": int(predicted.shape[0]),
        "predicted_frame_count": int(predicted.shape[0]),
        "exported_video_frame_count": int(predicted.shape[0]),
        "seed_frames": int(context_pixel_frames),
        "loss_frames": int(rollout_steps),
        "open_rollout_context_frames": int(context_latent_frames),
        "open_rollout_seed_frames": int(context_pixel_frames),
        "open_rollout_loss_frames": int(rollout_steps),
        "open_rollout_stride_frames": (
            model.dynamics.cfg.open_rollout_stride_frames
            if stride_frames is None
            else int(stride_frames)
        ),
        "open_rollout_initial_stride_frames": int(
            model.temporal_downsample_factor * initial_stride_latent_frames
        ),
        "open_rollout_context_latent_frames": int(context_latent_frames),
        "open_rollout_decoded_frame_count": int(predicted.shape[0]),
        "open_rollout_predicted_frame_count": int(predicted.shape[0]),
        "open_rollout_frame_mse": frame_mse,
        "open_rollout_frame_l1": frame_l1,
        "open_rollout_consistency_score": open_rollout_consistency_score(frame_mse, motion_ratio),
        "open_rollout_validation_style": "open_rollout_autoregressive",
    }
    if predicted_targets.shape[0] > 1:
        stats["open_rollout_predicted_target_motion_l1"] = predicted_motion_l1
        stats["open_rollout_ground_truth_target_motion_l1"] = ground_truth_motion_l1
        stats["open_rollout_target_motion_ratio"] = motion_ratio
        stats["open_rollout_motion_log_error"] = motion_ratio_log_error(float(motion_ratio))
    return predicted.detach().cpu(), stats


def resolved_dynamics_action_representation_from_config(
    config: Mapping[str, Any],
) -> str:
    """Resolve the serialized action representation for the standalone SO101 path."""

    return str(config.get("dynamics_action_representation", "relative_delta"))


def resolved_dynamics_action_scale_from_config(
    config: Mapping[str, Any],
) -> float:
    """Resolve the serialized action scale for the standalone SO101 path."""

    return float(config.get("dynamics_action_scale", 20.0))


def load_training_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load one standalone checkpoint payload onto the requested device."""

    return torch.load(Path(path), map_location=device, weights_only=False)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: WorldModel,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: dict[str, Any],
    best_metric: float | None,
) -> None:
    """Save one standalone checkpoint with DreamDojo provenance metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": STANDALONE_CHECKPOINT_KIND,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "best_metric": best_metric,
        "config": config,
        "ae_backend": model.ae_backend,
        "autoencoder": model.autoencoder_config(),
        "dynamics_backend": model.dynamics_backend,
        "dynamics": model.dynamics_config(),
        "dreamdojo_upstream": {
            "repo_url": DREAMDOJO_UPSTREAM_REPO_URL,
            "commit": DREAMDOJO_UPSTREAM_COMMIT,
            "commit_date": DREAMDOJO_UPSTREAM_COMMIT_DATE,
        },
    }
    torch.save(payload, output_path)


def load_wan_weights_into_model(model: WorldModel, checkpoint_path: str | Path) -> None:
    """Load raw DreamDojo Wan 2.2 tokenizer weights into one standalone model."""

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = dreamdojo_wan_state_dict(payload)
    if state_dict is None:
        raise ValueError(
            f"{checkpoint_path} is not a supported raw DreamDojo Wan 2.2 checkpoint."
        )
    remapped_state = remap_dreamdojo_wan_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    unexpected_keys = [key for key in unexpected if key]
    missing_keys = [
        key
        for key in missing
        if not key.startswith("dynamics.")
    ]
    if unexpected_keys:
        raise ValueError(
            f"Unexpected keys while loading Wan 2.2 weights: {unexpected_keys}."
        )
    if missing_keys:
        raise ValueError(
            "Failed to populate the standalone Wan encoder/decoder from the raw DreamDojo "
            f"checkpoint. Missing keys: {missing_keys}."
        )


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    infer_steps_override: int | None = None,
) -> WorldModel:
    """Instantiate one standalone world model from serialized checkpoint metadata."""

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing standalone config metadata.")
    dynamics_bundle = checkpoint.get("dynamics")
    dynamics_config = dynamics_bundle.get("config") if isinstance(dynamics_bundle, dict) else {}
    if not isinstance(dynamics_config, dict):
        dynamics_config = {}
    checkpoint_architecture_version = dynamics_config.get("architecture_version")
    if checkpoint_architecture_version != DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION:
        raise ValueError(
            "Checkpoint dynamics backbone is not compatible with this standalone DreamDojo "
            f"SO101 runtime. Expected architecture_version={DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION!r}, "
            f"received {checkpoint_architecture_version!r}."
        )
    model = WorldModel(
        latent_channels=int(config.get("latent_channels", 48)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        ae_backend=str(config.get("ae_backend", "wan")),
        resolution=int(config.get("resolution", 96)),
        height=config.get("height"),
        width=config.get("width"),
        dynamics_infer_steps=(
            int(infer_steps_override)
            if infer_steps_override is not None
            else int(dynamics_config.get("dynamics_infer_steps", config.get("dynamics_infer_steps", 35)))
        ),
        dynamics_train_timesteps=int(
            dynamics_config.get("dynamics_train_timesteps", config.get("dynamics_train_timesteps", 1000))
        ),
        dynamics_rf_shift=float(dynamics_config.get("dynamics_rf_shift", config.get("dynamics_rf_shift", 5.0))),
        conditional_frame_timestep=float(
            dynamics_config.get("conditional_frame_timestep", config.get("conditional_frame_timestep", -1.0))
        ),
        conditional_frame_sigma=float(
            dynamics_config.get("conditional_frame_sigma", config.get("conditional_frame_sigma", 0.0))
        ),
        dynamics_video_condition_dropout=float(
            dynamics_config.get("dynamics_video_condition_dropout", config.get("dynamics_video_condition_dropout", 0.0))
        ),
        dynamics_guidance_scale=float(
            dynamics_config.get("dynamics_guidance_scale", config.get("dynamics_guidance_scale", 0.0))
        ),
        dynamics_context_frames=int(
            dynamics_config.get("context_frames", config.get("dynamics_context_frames", DYNAMICS_FRAME_LAYOUT.context_frames))
        ),
        dynamics_target_frames=int(
            dynamics_config.get("target_frames", config.get("dynamics_target_frames", DYNAMICS_FRAME_LAYOUT.target_frames))
        ),
        dynamics_conditioning_frame_choices=dynamics_config.get(
            "conditioning_frame_choices",
            config.get("dynamics_conditioning_frame_choices"),
        ),
        dynamics_conditioning_frame_probabilities=dynamics_config.get(
            "conditioning_frame_probabilities",
            config.get("dynamics_conditioning_frame_probabilities"),
        ),
        dynamics_validation_conditioning_frame_choices=dynamics_config.get(
            "validation_conditioning_frame_choices",
            config.get("dynamics_validation_conditioning_frame_choices"),
        ),
        dynamics_open_rollout_context_frames=dynamics_config.get(
            "open_rollout_context_frames",
            config.get("dynamics_open_rollout_context_frames"),
        ),
        dynamics_open_rollout_stride_frames=dynamics_config.get(
            "open_rollout_stride_frames",
            config.get("dynamics_open_rollout_stride_frames"),
        ),
        dynamics_patch_spatial=int(
            dynamics_config.get("patch_spatial", config.get("dynamics_patch_spatial", 2))
        ),
        dynamics_model_channels=int(
            dynamics_config.get("model_channels", config.get("dynamics_model_channels", 1536))
        ),
        dynamics_num_blocks=int(
            dynamics_config.get("num_blocks", config.get("dynamics_num_blocks", 20))
        ),
        dynamics_num_heads=int(
            dynamics_config.get("num_heads", config.get("dynamics_num_heads", 12))
        ),
        dynamics_action_dim=int(
            dynamics_config.get("action_dim", config.get("dynamics_action_dim", 6))
        ),
        dynamics_action_conditioning_mode=str(
            dynamics_config.get(
                "action_conditioning_mode",
                config.get("dynamics_action_conditioning_mode", "chunk_per_frame"),
            )
        ),
        dynamics_zero_init_action_embedder=bool(
            dynamics_config.get(
                "zero_init_action_embedder",
                config.get("dynamics_zero_init_action_embedder", False),
            )
        ),
        dynamics_use_adaln_lora=bool(
            dynamics_config.get("use_adaln_lora", config.get("dynamics_use_adaln_lora", True))
        ),
        dynamics_adaln_lora_dim=int(
            dynamics_config.get("adaln_lora_dim", config.get("dynamics_adaln_lora_dim", 128))
        ),
        dynamics_rope_t_extrapolation_ratio=float(
            dynamics_config.get(
                "rope_t_extrapolation_ratio",
                config.get("dynamics_rope_t_extrapolation_ratio", 1.0),
            )
        ),
        dynamics_use_learned_temporal_embedding=bool(
            dynamics_config.get(
                "use_learned_temporal_embedding",
                config.get("dynamics_use_learned_temporal_embedding", False),
            )
        ),
        latent_normalization_stats=(
            checkpoint.get("autoencoder", {}).get("normalization_stats")
            if isinstance(checkpoint.get("autoencoder"), Mapping)
            else None
        ),
        wan_config=WanVAEConfig(
            dim=int(config.get("wan_dim", 160)),
            z_dim=int(config.get("latent_channels", 48)),
            num_res_blocks=int(config.get("wan_num_res_blocks", 2)),
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model
