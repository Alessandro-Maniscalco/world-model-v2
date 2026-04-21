"""Run VAE reconstructions on validation frames under several spatial transforms.

source .venv/bin/activate
python scripts/check/visualize_reconstruction_transforms.py \
  --checkpoint outputs/so101_base_pickplace_wan_ae_240x320_detail/checkpoints/last.pt \
  --output-dir outputs/vae_diagnostics/so101_ep0_transforms \
  --transform identity \
  --transform hflip \
  --transform hshift:16 \
  --transform hshift:-16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.dataset import ValidationClipDataset
from world_model_v2.experiment import (
    ExperimentConfig,
    checkpoint_ae_backend,
    load_training_checkpoint,
)
from world_model_v2.lerobot_video_dataset import (
    LeRobotVideoValidationClipDataset,
    SO101_BASE_SIM_PICKPLACE_ACTION_DIM,
    SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
)
from world_model_v2.maniskill_dataset import (
    MANISKILL_DEFAULT_ACTION_DIM,
    MANISKILL_DEFAULT_CAMERA,
    MANISKILL_DEFAULT_TRAJ_H5,
    MANISKILL_DEFAULT_TRAJ_JSON,
    ManiSkillValidationClipDataset,
)
from world_model_v2.metaworld_dataset import (
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM,
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN,
    METAWORLD_ACTION_DIM,
    METAWORLD_DATASET_ID,
    MetaWorldValidationClipDataset,
    AlohaValidationClipDataset,
)
from world_model_v2.model import WorldModel
from world_model_v2.reconstruction_diagnostics import (
    apply_spatial_transform,
    compute_motion_mask,
    compute_reconstruction_metrics,
    parse_transform_spec,
)
from world_model_v2.utils.checkpointing import save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4
from world_model_v2.wan_vae import WanVAEConfig


def parse_optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Normalize one optional integer sequence from checkpoint config JSON."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise ValueError(f"Expected an optional integer list, received {value!r}.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the transformed reconstruction check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--transform", action="append", default=[])
    parser.add_argument("--validation-episode", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--motion-threshold", type=float, default=0.03)
    parser.add_argument("--motion-dilation-radius", type=int, default=4)
    parser.add_argument("--max-grid-frames", type=int, default=24)
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    """Fail early when CUDA is requested but unavailable."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available to PyTorch. "
            "Rerun with --device cpu or install a compatible torch wheel."
        )


def resolved_action_dim(cfg: ExperimentConfig) -> int:
    """Return the action dimension that matches the checkpoint dataset family."""

    if cfg.dataset_format == "lerobot_metaworld":
        return METAWORLD_ACTION_DIM
    if cfg.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
        return ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_ACTION_DIM
    if cfg.dataset_format == "lerobot_so101_base_sim_pickplace":
        return SO101_BASE_SIM_PICKPLACE_ACTION_DIM
    if cfg.dataset_format == "maniskill_replay":
        return MANISKILL_DEFAULT_ACTION_DIM
    return 4


def build_checkpoint_config(checkpoint: dict[str, Any]) -> ExperimentConfig:
    """Rebuild the stored experiment config needed for clip loading and model init."""

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing the serialized config dictionary.")
    return ExperimentConfig(
        mode=str(checkpoint.get("mode", config.get("mode", "ae_only"))),
        dataset_format=str(config.get("dataset_format", "interactive_world_sim")),
        data_root=str(config.get("data_root", "data/full")),
        task=str(config.get("task", "single_grasp")),
        metaworld_task_index=(
            None if config.get("metaworld_task_index") is None else int(config["metaworld_task_index"])
        ),
        metaworld_repo_id=str(config.get("metaworld_repo_id", METAWORLD_DATASET_ID)),
        metaworld_cache_dir=str(config.get("metaworld_cache_dir", "")),
        aloha_repo_id=str(
            config.get("aloha_repo_id", ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID)
        ),
        aloha_cache_dir=str(config.get("aloha_cache_dir", "")),
        maniskill_traj_h5=str(config.get("maniskill_traj_h5", MANISKILL_DEFAULT_TRAJ_H5)),
        maniskill_traj_json=str(config.get("maniskill_traj_json", MANISKILL_DEFAULT_TRAJ_JSON)),
        maniskill_camera=str(config.get("maniskill_camera", MANISKILL_DEFAULT_CAMERA)),
        split=str(config.get("split", "val")),
        episode=int(config.get("episode", 0)),
        train_all_episodes=bool(config.get("train_all_episodes", False)),
        validation_split=str(config.get("validation_split", "")),
        validation_episode=int(config.get("validation_episode", 0)),
        validation_episodes=parse_optional_int_tuple(config.get("validation_episodes")),
        camera=str(config.get("camera", "camera_1_color")),
        frame_start=None if config.get("frame_start") is None else int(config["frame_start"]),
        frame_end=None if config.get("frame_end") is None else int(config["frame_end"]),
        resolution=int(config.get("resolution", 128)),
        height=None if config.get("height") is None else int(config["height"]),
        width=None if config.get("width") is None else int(config["width"]),
        latent_channels=int(config.get("latent_channels", 16)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        dynamics_infer_steps=int(config.get("dynamics_infer_steps", 16)),
        dynamics_train_timesteps=int(config.get("dynamics_train_timesteps", 1000)),
        dynamics_rf_shift=float(config.get("dynamics_rf_shift", 5.0)),
        conditional_frame_timestep=float(config.get("conditional_frame_timestep", -1.0)),
        conditional_frame_sigma=float(config.get("conditional_frame_sigma", 0.0)),
        dynamics_video_condition_dropout=float(config.get("dynamics_video_condition_dropout", 0.0)),
        dynamics_guidance_scale=float(config.get("dynamics_guidance_scale", 0.0)),
        dynamics_context_frames=int(config.get("dynamics_context_frames", 1)),
        dynamics_target_frames=int(config.get("dynamics_target_frames", 3)),
        dynamics_conditioning_frame_choices=parse_optional_int_tuple(
            config.get("dynamics_conditioning_frame_choices")
        ),
        dynamics_conditioning_frame_probabilities=(
            None
            if config.get("dynamics_conditioning_frame_probabilities") is None
            else tuple(float(value) for value in config["dynamics_conditioning_frame_probabilities"])
        ),
        dynamics_validation_conditioning_frame_choices=parse_optional_int_tuple(
            config.get("dynamics_validation_conditioning_frame_choices")
        ),
        dynamics_open_rollout_context_frames=(
            None
            if config.get("dynamics_open_rollout_context_frames") is None
            else int(config["dynamics_open_rollout_context_frames"])
        ),
        dynamics_open_rollout_stride_frames=(
            None
            if config.get("dynamics_open_rollout_stride_frames") is None
            else int(config["dynamics_open_rollout_stride_frames"])
        ),
        dynamics_model_channels=int(config.get("dynamics_model_channels", 256)),
        dynamics_num_blocks=int(config.get("dynamics_num_blocks", 4)),
        dynamics_num_heads=int(config.get("dynamics_num_heads", 4)),
        dynamics_action_conditioning_mode=str(
            config.get("dynamics_action_conditioning_mode", "chunk_per_frame")
        ),
        dynamics_zero_init_action_embedder=bool(
            config.get("dynamics_zero_init_action_embedder", False)
        ),
        dynamics_use_adaln_lora=bool(config.get("dynamics_use_adaln_lora", True)),
        dynamics_adaln_lora_dim=int(config.get("dynamics_adaln_lora_dim", 64)),
        dynamics_rope_t_extrapolation_ratio=float(
            config.get("dynamics_rope_t_extrapolation_ratio", 1.0)
        ),
        dynamics_use_learned_temporal_embedding=bool(
            config.get("dynamics_use_learned_temporal_embedding", False)
        ),
    )


def build_checkpoint_wan_config(checkpoint: dict[str, Any]) -> WanVAEConfig:
    """Rebuild one serialized Wan config, including legacy pre-patchify defaults."""

    autoencoder = checkpoint.get("autoencoder")
    if not isinstance(autoencoder, dict):
        raise ValueError("Checkpoint is missing autoencoder metadata.")
    raw_config = autoencoder.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint is missing the serialized autoencoder config.")
    payload = dict(raw_config)
    if "patch_size" not in payload:
        payload["patch_size"] = 1
    if "dec_dim" not in payload:
        payload["dec_dim"] = int(payload.get("dim", 64))
    if "temporal_window" not in payload:
        temporal_factor = 2 ** sum(bool(flag) for flag in payload.get("temperal_downsample", []))
        payload["temporal_window"] = max(1, int(temporal_factor))
    return WanVAEConfig.from_dict(payload)


def checkpoint_latent_normalization_stats(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Return serialized latent normalization stats when the checkpoint stored them."""

    autoencoder = checkpoint.get("autoencoder")
    if not isinstance(autoencoder, dict):
        return None
    raw_stats = autoencoder.get("normalization_stats")
    return raw_stats if isinstance(raw_stats, dict) else None


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    cfg: ExperimentConfig,
    device: torch.device,
) -> WorldModel:
    """Instantiate the stored model topology and load the saved state dict."""

    wan_config = build_checkpoint_wan_config(checkpoint)
    model = WorldModel(
        latent_channels=cfg.latent_channels,
        hidden_channels=cfg.hidden_channels,
        ae_backend=checkpoint_ae_backend(checkpoint),
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
        dynamics_model_channels=cfg.dynamics_model_channels,
        dynamics_num_blocks=cfg.dynamics_num_blocks,
        dynamics_num_heads=cfg.dynamics_num_heads,
        dynamics_action_dim=resolved_action_dim(cfg),
        dynamics_action_conditioning_mode=cfg.dynamics_action_conditioning_mode,
        dynamics_zero_init_action_embedder=cfg.dynamics_zero_init_action_embedder,
        dynamics_use_adaln_lora=cfg.dynamics_use_adaln_lora,
        dynamics_adaln_lora_dim=cfg.dynamics_adaln_lora_dim,
        dynamics_rope_t_extrapolation_ratio=cfg.dynamics_rope_t_extrapolation_ratio,
        dynamics_use_learned_temporal_embedding=cfg.dynamics_use_learned_temporal_embedding,
        wan_config=wan_config,
        latent_normalization_stats=checkpoint_latent_normalization_stats(checkpoint),
    ).to(device)
    model_state = checkpoint["model_state"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in model_state.items()
        if key.startswith("decoder.")
    }
    if not encoder_state or not decoder_state:
        raise KeyError("Checkpoint is missing encoder or decoder weights.")
    model.encoder.load_state_dict(encoder_state, strict=True)
    model.decoder.load_state_dict(decoder_state, strict=True)
    model.eval()
    return model


def load_validation_clip(
    cfg: ExperimentConfig,
    *,
    validation_episode: int | None,
    frame_start: int | None,
    frame_end: int | None,
) -> dict[str, torch.Tensor]:
    """Load the exact validation clip defined by the checkpoint config and overrides."""

    episode = cfg.validation_episode if validation_episode is None else int(validation_episode)
    start = cfg.frame_start if frame_start is None else frame_start
    end = cfg.frame_end if frame_end is None else frame_end
    if cfg.dataset_format == "lerobot_metaworld":
        dataset = MetaWorldValidationClipDataset(
            data_root=cfg.data_root,
            split=cfg.resolved_validation_split(),
            episode=episode,
            task_index=cfg.metaworld_task_index,
            frame_start=start,
            frame_end=end,
            resolution=cfg.resolution,
            height=cfg.height,
            width=cfg.width,
            repo_id=cfg.metaworld_repo_id,
            cache_dir=cfg.metaworld_cache_dir or None,
        )
        return dataset[0]
    if cfg.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
        dataset = AlohaValidationClipDataset(
            data_root=cfg.data_root,
            split=cfg.resolved_validation_split(),
            episode=episode,
            frame_start=start,
            frame_end=end,
            resolution=cfg.resolution,
            height=cfg.height,
            width=cfg.width,
            repo_id=cfg.aloha_repo_id,
            cache_dir=cfg.aloha_cache_dir or None,
        )
        return dataset[0]
    if cfg.dataset_format == "lerobot_so101_base_sim_pickplace":
        dataset = LeRobotVideoValidationClipDataset(
            data_root=cfg.data_root,
            split=cfg.resolved_validation_split(),
            episode=episode,
            frame_start=start,
            frame_end=end,
            resolution=cfg.resolution,
            height=cfg.height,
            width=cfg.width,
            repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
            image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        )
        return dataset[0]
    if cfg.dataset_format == "maniskill_replay":
        dataset = ManiSkillValidationClipDataset(
            data_root=cfg.data_root,
            split=cfg.resolved_validation_split(),
            episode=episode,
            frame_start=start,
            frame_end=end,
            resolution=cfg.resolution,
            height=cfg.height,
            width=cfg.width,
            traj_h5=cfg.maniskill_traj_h5,
            traj_json=cfg.maniskill_traj_json,
            camera=cfg.maniskill_camera,
        )
        return dataset[0]
    dataset = ValidationClipDataset(
        data_root=cfg.data_root,
        task=cfg.task,
        split=cfg.resolved_validation_split(),
        episode=episode,
        camera=cfg.resolved_camera(),
        frame_start=start,
        frame_end=end,
        resolution=cfg.resolution,
        height=cfg.height,
        width=cfg.width,
    )
    return dataset[0]


@torch.no_grad()
def reconstruct_in_chunks(
    model: WorldModel,
    frames: torch.Tensor,
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run deterministic VAE reconstruction in chunks to avoid transient OOMs."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    reconstructed_chunks: list[torch.Tensor] = []
    total_frames = int(frames.shape[0])
    for start in range(0, total_frames, int(chunk_size)):
        stop = min(start + int(chunk_size), total_frames)
        reconstructed_chunks.append(
            model.reconstruct(frames[start:stop].to(device), deterministic=True).cpu()
        )
    return torch.cat(reconstructed_chunks, dim=0)


def save_motion_mask(mask: torch.Tensor, output_path: Path) -> None:
    """Write one boolean motion mask as a grayscale PNG."""

    image = Image.fromarray(mask.to(torch.uint8).mul(255).cpu().numpy())
    image.save(output_path)


@torch.no_grad()
def run_transform_diagnostics(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device,
    chunk_size: int,
    transform_specs: list[str],
    validation_episode: int | None,
    frame_start: int | None,
    frame_end: int | None,
    motion_threshold: float,
    motion_dilation_radius: int,
    max_grid_frames: int,
) -> dict[str, Any]:
    """Reconstruct one validation clip under multiple transforms and export diagnostics."""

    device_obj = torch.device(device)
    validate_device(device_obj)
    checkpoint = load_training_checkpoint(checkpoint_path, device_obj)
    cfg = build_checkpoint_config(checkpoint)
    model = build_model_from_checkpoint(checkpoint, cfg, device_obj)
    clip = load_validation_clip(
        cfg,
        validation_episode=validation_episode,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    raw_frames = clip["frames"].detach().cpu()
    parsed_transforms = (
        [parse_transform_spec(spec) for spec in transform_specs]
        if transform_specs
        else [
            parse_transform_spec("identity"),
            parse_transform_spec("hflip"),
            parse_transform_spec("hshift:16"),
            parse_transform_spec("hshift:-16"),
        ]
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "mode": str(checkpoint["mode"]),
        "dataset_format": cfg.dataset_format,
        "episode": int(clip["episode_idx"].reshape(-1)[0].item()),
        "frame_start": int(clip["frame_idx"][0].item()),
        "frame_end": int(clip["frame_idx"][-1].item()),
        "input_frame_count": int(raw_frames.shape[0]),
        "chunk_size": int(chunk_size),
        "motion_threshold": float(motion_threshold),
        "motion_dilation_radius": int(motion_dilation_radius),
        "transforms": [],
    }

    for transform in parsed_transforms:
        transformed = apply_spatial_transform(raw_frames, transform)
        reconstructed = reconstruct_in_chunks(
            model,
            transformed,
            chunk_size=chunk_size,
            device=device_obj,
        )
        motion_mask = compute_motion_mask(
            transformed,
            threshold=motion_threshold,
            dilation_radius=motion_dilation_radius,
        )
        metrics = compute_reconstruction_metrics(
            transformed,
            reconstructed,
            motion_mask=motion_mask,
        )

        transform_dir = output_path / transform.name
        transform_dir.mkdir(parents=True, exist_ok=True)
        grid_path = transform_dir / "episode_0_grid.png"
        video_path = transform_dir / "episode_0.mp4"
        stats_path = transform_dir / "episode_0_stats.json"
        mask_path = transform_dir / "motion_mask.png"

        build_side_by_side_grid(
            original=transformed,
            reconstructed=reconstructed,
            max_frames=min(int(max_grid_frames), int(transformed.shape[0])),
        ).save(grid_path)
        exported_frame_count = write_side_by_side_mp4(
            original=transformed,
            reconstructed=reconstructed,
            output_path=video_path,
            duration_ms=120,
        )
        save_motion_mask(motion_mask, mask_path)

        transform_stats = {
            **metrics,
            "transform": transform.name,
            "horizontal_flip": bool(transform.horizontal_flip),
            "shift_x": int(transform.shift_x),
            "shift_y": int(transform.shift_y),
            "decoded_frame_count": int(reconstructed.shape[0]),
            "exported_video_frame_count": int(exported_frame_count),
            "output_grid": str(grid_path),
            "output_video": str(video_path),
            "output_motion_mask": str(mask_path),
        }
        if transform_stats["input_frame_count"] != transform_stats["decoded_frame_count"]:
            raise RuntimeError(f"Decoded frame count mismatch for {transform.name}: {transform_stats}")
        if transform_stats["decoded_frame_count"] != transform_stats["exported_video_frame_count"]:
            raise RuntimeError(f"Exported frame count mismatch for {transform.name}: {transform_stats}")
        save_json(stats_path, transform_stats)
        summary["transforms"].append(transform_stats)

    summary_path = output_path / "summary.json"
    save_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    """Run the transformed reconstruction CLI."""

    args = parse_args()
    result = run_transform_diagnostics(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        transform_specs=args.transform,
        validation_episode=args.validation_episode,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        motion_threshold=args.motion_threshold,
        motion_dilation_radius=args.motion_dilation_radius,
        max_grid_frames=args.max_grid_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
