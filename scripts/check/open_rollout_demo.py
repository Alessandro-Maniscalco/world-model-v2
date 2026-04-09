"""Run an autoregressive open-rollout demo from a trained world-model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.experiment import load_training_checkpoint
from world_model_v2.metaworld_dataset import load_metaworld_clip
from world_model_v2.model import WorldModel
from world_model_v2.utils.checkpointing import save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the open-rollout demo script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--metaworld-task-index", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--dynamics-infer-steps", type=int, default=None)
    parser.add_argument("--dynamics-open-rollout-stride-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/open_rollout_demo")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    infer_steps_override: int | None,
) -> WorldModel:
    """Instantiate and load one world model from a saved checkpoint."""

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing config metadata.")
    dynamics_bundle = checkpoint.get("dynamics")
    dynamics_config = dynamics_bundle.get("config") if isinstance(dynamics_bundle, dict) else {}
    if not isinstance(dynamics_config, dict):
        dynamics_config = {}
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
        dynamics_rf_shift=float(
            dynamics_config.get("dynamics_rf_shift", config.get("dynamics_rf_shift", 5.0))
        ),
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
            dynamics_config.get("context_frames", config.get("dynamics_context_frames", 4))
        ),
        dynamics_target_frames=int(
            dynamics_config.get("target_frames", config.get("dynamics_target_frames", 1))
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
        dynamics_model_channels=int(
            dynamics_config.get("model_channels", config.get("dynamics_model_channels", 256))
        ),
        dynamics_num_blocks=int(
            dynamics_config.get("num_blocks", config.get("dynamics_num_blocks", 4))
        ),
        dynamics_num_heads=int(
            dynamics_config.get("num_heads", config.get("dynamics_num_heads", 4))
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
            dynamics_config.get("use_adaln_lora", config.get("dynamics_use_adaln_lora", False))
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
    """Return the output directory for the rollout demo artifacts."""

    if args.run_name:
        return Path(args.output_dir) / args.run_name
    return Path(args.output_dir) / (
        f"ep{args.episode}_task{args.metaworld_task_index}_f{args.frame_start}_{args.frame_end}"
    )


def main() -> None:
    """Load one checkpoint, run open rollout, and save preview artifacts."""

    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(
        checkpoint,
        device=device,
        infer_steps_override=args.dynamics_infer_steps,
    )
    clip = load_metaworld_clip(
        data_root=args.data_root,
        split=args.split,
        episode=args.episode,
        task_index=args.metaworld_task_index,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        load_actions=True,
    )
    frames = clip["frames"].unsqueeze(0).to(device)
    actions = clip["actions"].unsqueeze(0).to(device)
    seed_context_frames = model.dynamics.cfg.open_rollout_context_frames
    seed_frames = frames[:, :seed_context_frames]
    rollout_steps = int(frames.shape[1]) - seed_context_frames
    if rollout_steps < 1:
        raise ValueError(
            "Open rollout requires more frames than the configured context window."
        )
    with torch.no_grad():
        rollout = model.rollout(
            seed_frames,
            steps=rollout_steps,
            actions=actions,
            stride_frames=args.dynamics_open_rollout_stride_frames,
        )
    original = frames[0].detach().cpu()
    predicted = rollout[0].detach().cpu()
    predicted_only = predicted[seed_context_frames:]
    target_only = original[seed_context_frames:]
    stats = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "input_frame_count": int(original.shape[0]),
        "decoded_frame_count": int(predicted.shape[0]),
        "predicted_frame_count": int(predicted.shape[0]),
        "seed_frames": int(seed_context_frames),
        "loss_frames": int(rollout_steps),
        "open_rollout_stride_frames": (
            model.dynamics.cfg.open_rollout_stride_frames
            if args.dynamics_open_rollout_stride_frames is None
            else int(args.dynamics_open_rollout_stride_frames)
        ),
        "open_rollout_frame_mse": float(F.mse_loss(predicted_only, target_only).item()),
        "open_rollout_frame_l1": float(F.l1_loss(predicted_only, target_only).item()),
        "dynamics_infer_steps": int(model.dynamics.cfg.dynamics_infer_steps),
        "validation_style": "open_rollout_autoregressive",
    }
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_side_by_side_grid(
        original=original,
        reconstructed=predicted,
        max_frames=int(original.shape[0]),
        context_frames=seed_context_frames,
    ).save(output_dir / "rollout_grid.png")
    write_side_by_side_mp4(
        original=original,
        reconstructed=predicted,
        output_path=output_dir / "rollout.mp4",
        duration_ms=120,
        context_frames=seed_context_frames,
    )
    save_json(output_dir / "rollout_stats.json", stats)
    print(json.dumps({"output_dir": str(output_dir), **stats}, sort_keys=True))


if __name__ == "__main__":
    main()
