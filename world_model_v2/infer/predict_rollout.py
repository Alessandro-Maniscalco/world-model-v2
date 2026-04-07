"""Predict an episode rollout with a saved Stage-2 or Stage-3 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from world_model_v2.config import DatasetConfig, RunConfig
from world_model_v2.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset
from world_model_v2.infer.reconstruct_episode import build_model_from_config, validate_device
from world_model_v2.utils.checkpointing import load_checkpoint, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


def parse_args() -> argparse.Namespace:
    """Parse rollout CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--camera", default="camera_1_color")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--context-size", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def predict_rollout(
    checkpoint_path: str | Path,
    data_root: str | Path,
    task: str,
    split: str,
    camera: str,
    episode: int,
    resolution: int,
    context_size: int,
    num_steps: int,
    max_frames: int,
    duration_ms: int,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Load a checkpoint, roll out one episode, and export artifacts."""

    device_obj = torch.device(device)
    validate_device(device_obj)
    checkpoint = load_checkpoint(checkpoint_path, device_obj)
    run_config = RunConfig.from_dict(checkpoint["config"])
    if run_config.algorithm.training_stage not in (2, 3):
        raise ValueError("Rollout prediction requires a Stage-2 or Stage-3 checkpoint.")

    dataset_cfg = DatasetConfig(
        data_root=str(data_root),
        task=task,
        split=split,
        obs_keys=(camera,),
        resolution=resolution,
        horizon=run_config.dataset.horizon,
        val_horizon=run_config.dataset.val_horizon,
        action_mode=run_config.dataset.action_mode,
    )
    model = build_model_from_config(
        RunConfig(
            dataset=dataset_cfg,
            algorithm=run_config.algorithm,
            experiment=run_config.experiment,
        ),
        device_obj,
    )
    model.set_normalization_stats(checkpoint.get("normalization_stats"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = RealAlohaDataset(dataset_cfg)
    episode_batch = dataset.load_episode_sequence(episode)
    device_batch = {
        "obs": {
            key: value.to(device_obj).unsqueeze(0)
            for key, value in episode_batch["obs"].items()
        },
        "action": episode_batch["action"].to(device_obj).unsqueeze(0),
        "episode_idx": episode_batch["episode_idx"].to(device_obj).unsqueeze(0),
    }
    preview = model.validation_step(
        device_batch,
        num_steps=num_steps,
        start_mode="noise",
        rollout_context_size=context_size if context_size > 0 else run_config.dataset.horizon,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    grid_path = output_path / f"episode_{episode}_grid.png"
    video_path = output_path / f"episode_{episode}.mp4"
    stats_path = output_path / f"episode_{episode}_stats.json"

    context_frames = int(preview["stats"].get("context_frames", 0))
    grid = build_side_by_side_grid(
        preview["original"],
        preview["reconstructed"],
        max_frames=max_frames,
        context_frames=context_frames,
    )
    grid.save(grid_path)
    exported_frame_count = write_side_by_side_mp4(
        preview["original"],
        preview["reconstructed"],
        video_path,
        duration_ms=duration_ms,
        context_frames=context_frames,
    )

    stats = dict(preview["stats"])
    stats["checkpoint"] = str(checkpoint_path)
    stats["exported_video_frame_count"] = int(exported_frame_count)
    if stats["input_frame_count"] != stats["predicted_frame_count"]:
        raise RuntimeError(f"Predicted frame count mismatch: {stats}")
    if stats["decoded_frame_count"] != stats["exported_video_frame_count"]:
        raise RuntimeError(f"Exported video frame count mismatch: {stats}")
    save_json(stats_path, stats)
    return {
        "grid_path": str(grid_path),
        "video_path": str(video_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main() -> None:
    """Run the rollout CLI."""

    args = parse_args()
    result = predict_rollout(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        task=args.task,
        split=args.split,
        camera=args.camera,
        episode=args.episode,
        resolution=args.resolution,
        context_size=args.context_size,
        num_steps=args.num_steps,
        max_frames=args.max_frames,
        duration_ms=args.duration_ms,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print(f"Wrote grid to {result['grid_path']}")
    print(f"Wrote mp4 to {result['video_path']}")


if __name__ == "__main__":
    main()
