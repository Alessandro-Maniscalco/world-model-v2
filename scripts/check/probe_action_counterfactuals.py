"""Probe how a trained action-conditioned world model responds to edited actions.

This smoke-check script loads one saved checkpoint, runs open-rollout inference on
one SO101 clip, and exports a comparison video for several action variants. The
loader mirrors the action representation serialized in the checkpoint config, so
new SO101 relative-action runs and older absolute-action runs are both probed in
their native action space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.dynamics_transformer import (
    DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION,
    DYNAMICS_FRAME_LAYOUT,
)
from world_model_v2.experiment import compute_motion_ratio, load_training_checkpoint
from world_model_v2.experiment import (
    resolved_dynamics_action_representation_from_config,
    resolved_dynamics_action_scale_from_config,
)
from world_model_v2.lerobot_video_dataset import load_lerobot_video_clip
from world_model_v2.model import WorldModel
from world_model_v2.utils.checkpointing import save_json
from world_model_v2.utils.visualization import (
    annotate_frame,
    frames_per_second_from_duration,
    tensor_to_uint8_images,
    write_mp4_frames,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the counterfactual action probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/so101_base_sim_pickplace_cache")
    parser.add_argument("--dataset-format", default="lerobot_so101_base_sim_pickplace")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=110)
    parser.add_argument("--frame-end", type=int, default=140)
    parser.add_argument("--resolution", type=int, default=208)
    parser.add_argument("--height", type=int, default=208)
    parser.add_argument("--width", type=int, default=272)
    parser.add_argument("--dynamics-infer-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--output-dir", default="outputs/action_counterfactuals")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    infer_steps_override: int | None,
) -> WorldModel:
    """Instantiate one world model from saved checkpoint metadata."""

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config metadata.")
    dynamics_bundle = checkpoint.get("dynamics")
    dynamics_config = dynamics_bundle.get("config") if isinstance(dynamics_bundle, dict) else {}
    if not isinstance(dynamics_config, dict):
        dynamics_config = {}
    checkpoint_architecture_version = dynamics_config.get("architecture_version")
    if checkpoint_architecture_version != DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION:
        raise ValueError(
            "Checkpoint dynamics backbone is not compatible with this DreamDojo-mechanics probe. "
            f"Expected architecture_version={DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION!r}, received "
            f"{checkpoint_architecture_version!r}."
        )
    model = WorldModel(
        latent_channels=int(config.get("latent_channels", 16)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        ae_backend=str(config.get("ae_backend", "wan")),
        resolution=int(config.get("resolution", 128)),
        height=config.get("height"),
        width=config.get("width"),
        dynamics_infer_steps=(
            int(infer_steps_override)
            if infer_steps_override is not None
            else int(dynamics_config.get("dynamics_infer_steps", config.get("dynamics_infer_steps", 16)))
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
            dynamics_config.get(
                "dynamics_video_condition_dropout",
                config.get("dynamics_video_condition_dropout", 0.0),
            )
        ),
        dynamics_guidance_scale=float(
            dynamics_config.get("dynamics_guidance_scale", config.get("dynamics_guidance_scale", 0.0))
        ),
        dynamics_context_frames=int(
            dynamics_config.get(
                "context_frames",
                config.get("dynamics_context_frames", DYNAMICS_FRAME_LAYOUT.context_frames),
            )
        ),
        dynamics_target_frames=int(
            dynamics_config.get(
                "target_frames",
                config.get("dynamics_target_frames", DYNAMICS_FRAME_LAYOUT.target_frames),
            )
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
            dynamics_config.get(
                "patch_spatial",
                config.get("dynamics_patch_spatial", 1),
            )
        ),
        dynamics_model_channels=int(dynamics_config.get("model_channels", config.get("dynamics_model_channels", 256))),
        dynamics_num_blocks=int(dynamics_config.get("num_blocks", config.get("dynamics_num_blocks", 4))),
        dynamics_num_heads=int(dynamics_config.get("num_heads", config.get("dynamics_num_heads", 4))),
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
            dynamics_config.get("adaln_lora_dim", config.get("dynamics_adaln_lora_dim", 64))
        ),
        dynamics_rope_t_extrapolation_ratio=float(
            dynamics_config.get(
                "rope_t_extrapolation_ratio",
                config.get("dynamics_rope_t_extrapolation_ratio", 1.0),
            )
        ),
        dynamics_action_dim=int(dynamics_config.get("action_dim", 4)),
        dynamics_use_learned_temporal_embedding=bool(
            dynamics_config.get(
                "use_learned_temporal_embedding",
                config.get("dynamics_use_learned_temporal_embedding", False),
            )
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory for this probe run."""

    if args.run_name:
        return Path(args.output_dir) / args.run_name
    checkpoint_name = Path(args.checkpoint).resolve().parent.parent.name
    return Path(args.output_dir) / (
        f"{checkpoint_name}_ep{args.episode}_f{args.frame_start}_{args.frame_end}"
    )


def load_clip(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any],
) -> dict[str, Any]:
    """Load one SO101 clip with action annotations."""

    if args.dataset_format != "lerobot_so101_base_sim_pickplace":
        raise ValueError(
            "This probe currently supports only dataset_format='lerobot_so101_base_sim_pickplace'."
        )
    action_representation = resolved_dynamics_action_representation_from_config(
        checkpoint_config,
        mode_override="dynamics_only",
    )
    action_scale = resolved_dynamics_action_scale_from_config(
        checkpoint_config,
        mode_override="dynamics_only",
    )
    clip = load_lerobot_video_clip(
        data_root=args.data_root,
        split=args.split,
        episode=args.episode,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        load_actions=True,
        action_representation=action_representation,
        action_scale=action_scale,
    )
    frames = clip["frames"]
    actions = clip["actions"]
    if actions.shape[0] != frames.shape[0] - 1:
        raise ValueError(
            "Expected one action per frame transition, but got "
            f"{actions.shape[0]} actions for {frames.shape[0]} frames."
        )
    clip["action_representation"] = action_representation
    clip["action_scale"] = action_scale
    return clip


def amplify_action_deltas(actions: torch.Tensor, factor: float) -> torch.Tensor:
    """Return one action sequence with per-step deltas scaled by `factor`."""

    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (T, A), received {tuple(actions.shape)}.")
    if actions.shape[0] <= 1:
        return actions.clone()
    scaled = actions.clone()
    deltas = actions[1:] - actions[:-1]
    scaled[1:] = scaled[:1] + torch.cumsum(deltas * float(factor), dim=0)
    return scaled


def build_action_variants(
    actions: torch.Tensor,
    *,
    action_representation: str,
) -> dict[str, torch.Tensor]:
    """Return the action variants to evaluate for one clip."""

    if action_representation == "relative_delta":
        halved = actions * 0.5
        reversed_actions = actions * -1.0
        doubled = actions * 2.0
    else:
        halved = amplify_action_deltas(actions, factor=0.5)
        reversed_actions = amplify_action_deltas(actions, factor=-1.0)
        doubled = amplify_action_deltas(actions, factor=2.0)
    return {
        "original": actions.clone(),
        "zero": torch.zeros_like(actions),
        "half_delta": halved,
        "reverse_delta": reversed_actions,
        "double_delta": doubled,
    }


def compute_variant_metrics(
    *,
    original_frames: torch.Tensor,
    predicted_frames: torch.Tensor,
    seed_frames: int,
    baseline_prediction: torch.Tensor | None,
) -> dict[str, float]:
    """Compute one compact metric bundle for a predicted rollout."""

    predicted_target = predicted_frames[seed_frames:]
    original_target = original_frames[seed_frames:]
    frame_mse = float(F.mse_loss(predicted_target, original_target).item())
    frame_l1 = float(F.l1_loss(predicted_target, original_target).item())
    predicted_motion_l1 = 0.0
    ground_truth_motion_l1 = 0.0
    target_motion_ratio = 0.0
    if predicted_target.shape[0] > 1:
        predicted_motion_l1 = float(
            torch.abs(predicted_target[1:] - predicted_target[:-1]).mean().item()
        )
        ground_truth_motion_l1 = float(
            torch.abs(original_target[1:] - original_target[:-1]).mean().item()
        )
        ratio = compute_motion_ratio(predicted_motion_l1, ground_truth_motion_l1)
        target_motion_ratio = 0.0 if ratio is None else float(ratio)
    stats = {
        "frame_mse": frame_mse,
        "frame_l1": frame_l1,
        "predicted_target_motion_l1": float(predicted_motion_l1),
        "ground_truth_target_motion_l1": float(ground_truth_motion_l1),
        "target_motion_ratio": float(target_motion_ratio),
    }
    if baseline_prediction is not None:
        baseline_target = baseline_prediction[seed_frames:]
        stats["prediction_delta_l1_vs_original"] = float(
            torch.abs(predicted_target - baseline_target).mean().item()
        )
        stats["prediction_delta_mse_vs_original"] = float(
            F.mse_loss(predicted_target, baseline_target).item()
        )
    return stats


def build_comparison_frames(
    *,
    original: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    context_frames: int,
) -> list[Image.Image]:
    """Render one annotated multi-column comparison video."""

    original_np = tensor_to_uint8_images(original)
    prediction_np = {label: tensor_to_uint8_images(frames) for label, frames in predictions.items()}
    ordered_labels = ["gt", *predictions.keys()]
    frames: list[Image.Image] = []
    for frame_index in range(original_np.shape[0]):
        border_color = (255, 0, 0) if frame_index < context_frames else (255, 255, 255)
        columns = [annotate_frame(original_np[frame_index], f"gt {frame_index}", border_color=border_color)]
        for label in predictions.keys():
            columns.append(
                annotate_frame(
                    prediction_np[label][frame_index],
                    f"{label} {frame_index}",
                    border_color=border_color,
                )
            )
        total_width = sum(image.width for image in columns)
        max_height = max(image.height for image in columns)
        canvas = Image.new("RGB", (total_width, max_height), "black")
        x_offset = 0
        for column in columns:
            canvas.paste(column, (x_offset, 0))
            x_offset += column.width
        frames.append(canvas)
    return frames


def build_contact_grid(rendered_frames: list[Image.Image], max_frames: int = 10) -> Image.Image:
    """Stack the first few rendered frames into one vertical contact sheet."""

    if not rendered_frames:
        raise ValueError("Need at least one rendered frame to build a grid.")
    selected = rendered_frames[:max_frames]
    width, height = selected[0].size
    canvas = Image.new("RGB", (width, height * len(selected)), "black")
    for index, image in enumerate(selected):
        canvas.paste(image, (0, index * height))
    return canvas


def main() -> None:
    """Run the counterfactual action probe and export artifacts."""

    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.checkpoint, device)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint is missing config metadata.")
    model = build_model_from_checkpoint(
        checkpoint,
        device=device,
        infer_steps_override=args.dynamics_infer_steps,
    )
    clip = load_clip(args, checkpoint_config)
    frames = clip["frames"].unsqueeze(0).to(device)
    actions = clip["actions"].to(device)
    action_representation = str(clip["action_representation"])
    action_scale = float(clip["action_scale"])
    seed_context_frames = int(model.dynamics.cfg.open_rollout_context_frames)
    rollout_steps = int(frames.shape[1]) - seed_context_frames
    if rollout_steps < 1:
        raise ValueError("Need more frames than the configured rollout seed length.")
    if seed_context_frames != 1:
        raise ValueError(
            "This probe expects the current 1-context-frame rollout setup, but checkpoint has "
            f"open_rollout_context_frames={seed_context_frames}."
        )
    variants = build_action_variants(
        actions,
        action_representation=action_representation,
    )
    predictions: dict[str, torch.Tensor] = {}
    stats: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode": int(args.episode),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "input_frame_count": int(frames.shape[1]),
        "seed_frames": seed_context_frames,
        "predicted_frame_count": int(frames.shape[1]),
        "dataset_format": args.dataset_format,
        "action_representation": action_representation,
        "action_scale": action_scale,
        "action_semantics_note": (
            "SO101 actions are relative next-step deltas in this checkpoint, so the "
            "'double_delta' variant multiplies those deltas directly."
            if action_representation == "relative_delta"
            else "SO101 actions are absolute next-step targets in this checkpoint, so the "
            "'double_delta' variant doubles per-step action changes instead of multiplying "
            "absolute actions by two."
        ),
        "variants": {},
    }
    seed_frames = frames[:, :seed_context_frames]
    baseline_prediction: torch.Tensor | None = None
    with torch.no_grad():
        for label, variant_actions in variants.items():
            rollout = model.rollout(
                seed_frames,
                steps=rollout_steps,
                actions=variant_actions.unsqueeze(0),
            )[0].detach().cpu()
            predictions[label] = rollout
            variant_stats = {
                "action_abs_mean": float(variant_actions.abs().mean().item()),
                "action_abs_max": float(variant_actions.abs().max().item()),
                "action_l1_vs_original": float(torch.abs(variant_actions - actions).mean().item()),
                **compute_variant_metrics(
                    original_frames=frames[0].detach().cpu(),
                    predicted_frames=rollout,
                    seed_frames=seed_context_frames,
                    baseline_prediction=baseline_prediction,
                ),
            }
            stats["variants"][label] = variant_stats
            if label == "original":
                baseline_prediction = rollout
    rendered_frames = build_comparison_frames(
        original=frames[0].detach().cpu(),
        predictions=predictions,
        context_frames=seed_context_frames,
    )
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "action_counterfactuals.mp4"
    grid_path = output_dir / "action_counterfactuals_grid.png"
    stats_path = output_dir / "action_counterfactuals_stats.json"
    exported_frames = write_mp4_frames(
        rendered_frames,
        video_path,
        fps=frames_per_second_from_duration(args.duration_ms),
    )
    build_contact_grid(rendered_frames).save(grid_path)
    stats["exported_video_frame_count"] = int(exported_frames)
    save_json(stats_path, stats)
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
