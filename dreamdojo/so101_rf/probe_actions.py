"""Probe standalone SO101 action counterfactuals from one trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dreamdojo.so101_rf.checkpointing import save_json
from dreamdojo.so101_rf.dataset import load_so101_video_clip
from dreamdojo.so101_rf.runtime import (
    build_model_from_checkpoint,
    compute_motion_ratio,
    load_training_checkpoint,
    resolved_dynamics_action_representation_from_config,
    resolved_dynamics_action_scale_from_config,
)
from dreamdojo.so101_rf.visualization import (
    annotate_frame,
    frames_per_second_from_duration,
    tensor_to_uint8_images,
    write_mp4_frames,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone SO101 action probe."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/so101_base_sim_pickplace_cache")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=110)
    parser.add_argument("--frame-end", type=int, default=140)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dynamics-infer-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--output-dir", default="dreamdojo/outputs/action_counterfactuals")
    parser.add_argument("--run-name", default="")
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory for one standalone action-probe run."""

    if args.run_name:
        return Path(args.output_dir) / args.run_name
    checkpoint_name = Path(args.checkpoint).resolve().parent.parent.name
    return Path(args.output_dir) / f"{checkpoint_name}_ep{args.episode}_f{args.frame_start}_{args.frame_end}"


def amplify_action_deltas(actions: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale cumulative action deltas for absolute-style action sequences."""

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
    """Return the standalone action variants to compare against the original rollout."""

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
    """Compute compact rollout metrics for one action-variant prediction."""

    predicted_target = predicted_frames[seed_frames:]
    original_target = original_frames[seed_frames:]
    frame_mse = float(F.mse_loss(predicted_target, original_target).item())
    frame_l1 = float(F.l1_loss(predicted_target, original_target).item())
    predicted_motion_l1 = 0.0
    ground_truth_motion_l1 = 0.0
    target_motion_ratio = 0.0
    if predicted_target.shape[0] > 1:
        predicted_motion_l1 = float(torch.abs(predicted_target[1:] - predicted_target[:-1]).mean().item())
        ground_truth_motion_l1 = float(torch.abs(original_target[1:] - original_target[:-1]).mean().item())
        target_motion_ratio = float(compute_motion_ratio(predicted_motion_l1, ground_truth_motion_l1))
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
    """Render one annotated multi-column comparison video for action variants."""

    original_images = tensor_to_uint8_images(original)
    prediction_images = {
        label: tensor_to_uint8_images(prediction)
        for label, prediction in predictions.items()
    }
    labels = ["ground_truth", *predictions.keys()]
    rows = [original_images, *prediction_images.values()]
    rendered_frames: list[Image.Image] = []
    for frame_idx in range(int(original.shape[0])):
        frame_images: list[Image.Image] = []
        for label, row in zip(labels, rows, strict=False):
            border_color = (255, 0, 0) if frame_idx < context_frames else (255, 255, 255)
            frame_images.append(annotate_frame(row[frame_idx], f"{label} {frame_idx}", border_color=border_color))
        total_width = sum(image.width for image in frame_images)
        max_height = max(image.height for image in frame_images)
        canvas = Image.new("RGB", (total_width, max_height), "black")
        left = 0
        for image in frame_images:
            canvas.paste(image, (left, 0))
            left += image.width
        rendered_frames.append(canvas)
    return rendered_frames


def main(argv: list[str] | None = None) -> None:
    """Load one checkpoint, run action counterfactual rollouts, and export comparisons."""

    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.checkpoint, device)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint is missing standalone config metadata.")
    model = build_model_from_checkpoint(
        checkpoint,
        device=device,
        infer_steps_override=args.dynamics_infer_steps,
    )
    action_representation = resolved_dynamics_action_representation_from_config(checkpoint_config)
    action_scale = resolved_dynamics_action_scale_from_config(checkpoint_config)
    clip = load_so101_video_clip(
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
    frames = clip["frames"].to(device)
    actions = clip["actions"].to(device)
    if actions.shape[0] != frames.shape[0] - 1:
        raise ValueError(
            "Expected one action per frame transition, but got "
            f"{actions.shape[0]} actions for {frames.shape[0]} frames."
        )
    variants = build_action_variants(
        actions,
        action_representation=action_representation,
    )
    seed_frames = model.latent_frames_to_pixel_frames(model.dynamics.cfg.open_rollout_context_frames)
    rollout_steps = int(frames.shape[0]) - seed_frames
    predictions: dict[str, torch.Tensor] = {}
    stats: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "action_representation": action_representation,
        "action_scale": float(action_scale),
        "seed_frames": int(seed_frames),
        "loss_frames": int(rollout_steps),
        "variants": {},
    }
    with torch.no_grad():
        baseline_prediction: torch.Tensor | None = None
        for label, variant_actions in variants.items():
            rollout = model.rollout(
                frames[:seed_frames].unsqueeze(0),
                steps=rollout_steps,
                actions=variant_actions.unsqueeze(0),
            )[0].detach().cpu()
            predictions[label] = rollout
            if label == "original":
                baseline_prediction = rollout
            stats["variants"][label] = {
                "action_abs_mean": float(variant_actions.abs().mean().item()),
                "action_abs_max": float(variant_actions.abs().max().item()),
                "action_l1_vs_original": float(torch.abs(variant_actions - actions).mean().item()),
                **compute_variant_metrics(
                    original_frames=frames.detach().cpu(),
                    predicted_frames=rollout,
                    seed_frames=seed_frames,
                    baseline_prediction=baseline_prediction if label != "original" else None,
                ),
            }
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "action_counterfactuals.mp4"
    grid_path = output_dir / "action_counterfactuals_grid.png"
    stats_path = output_dir / "action_counterfactuals_stats.json"
    comparison_frames = build_comparison_frames(
        original=frames.detach().cpu(),
        predictions=predictions,
        context_frames=seed_frames,
    )
    exported_frames = write_mp4_frames(
        comparison_frames,
        video_path,
        fps=frames_per_second_from_duration(args.duration_ms),
    )
    comparison_frames[0].save(grid_path)
    stats["exported_video_frame_count"] = int(exported_frames)
    save_json(stats_path, stats)
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
