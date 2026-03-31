"""Visualize a Stage-1 reconstruction for the first dataset frame after a horizontal flip.

source .venv/bin/activate
python scripts/check/visualize_stage1_first_frame_hflip.py \
  --checkpoint outputs/stage1/<run_name>/checkpoints/last.pt \
  --output-dir /tmp/stage1_first_frame_hflip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.config import DatasetConfig, RunConfig
from world_model_v2.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset
from world_model_v2.infer.reconstruct_episode import build_model_from_config, validate_device
from world_model_v2.utils.checkpointing import load_checkpoint, save_json
from world_model_v2.utils.visualization import build_side_by_side_grid


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the flipped first-frame reconstruction check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--camera", default="camera_1_color")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--start-mode", default="noisy-input", choices=["noisy-input", "noise"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_flipped_frame(
    dataset_cfg: DatasetConfig,
    camera: str,
    episode: int,
    frame_idx: int,
) -> tuple[torch.Tensor, int]:
    """Load one frame from the requested episode and flip it horizontally."""

    dataset = RealAlohaDataset(dataset_cfg)
    episode_batch = dataset.load_episode_sequence(episode)
    valid_length = int(episode_batch["valid_length"].item())
    if frame_idx < 0 or frame_idx >= valid_length:
        raise IndexError(
            f"frame_idx={frame_idx} is outside the valid range [0, {valid_length - 1}] "
            f"for episode {episode}."
        )
    frame = episode_batch["obs"][camera][frame_idx]
    return torch.flip(frame, dims=(-1,)), valid_length


@torch.no_grad()
def reconstruct_flipped_first_frame(
    checkpoint_path: str | Path,
    data_root: str | Path,
    task: str,
    split: str,
    camera: str,
    episode: int,
    frame_idx: int,
    resolution: int,
    num_steps: int,
    start_mode: str,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Reconstruct one horizontally flipped frame and save a GT-vs-recon image."""

    device_obj = torch.device(device)
    validate_device(device_obj)
    checkpoint = load_checkpoint(checkpoint_path, device_obj)
    run_config = RunConfig.from_dict(checkpoint["config"])
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

    flipped_frame, valid_length = load_flipped_frame(dataset_cfg, camera, episode, frame_idx)
    flipped_sequence = flipped_frame.unsqueeze(0)
    latent = model.encode(flipped_sequence.to(device_obj))
    reconstructed = model.reconstruct(
        {camera: flipped_sequence.to(device_obj)},
        num_steps=num_steps,
        start_mode=start_mode,
    ).cpu()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = output_path / f"episode_{episode}_frame_{frame_idx}_hflip_grid.png"
    stats_path = output_path / f"episode_{episode}_frame_{frame_idx}_hflip_stats.json"

    grid = build_side_by_side_grid(flipped_sequence, reconstructed, max_frames=1)
    grid.save(image_path)

    mse = torch.mean((reconstructed - flipped_sequence) ** 2).item()
    stats = {
        "checkpoint": str(checkpoint_path),
        "task": task,
        "split": split,
        "camera": camera,
        "episode": episode,
        "frame_idx": frame_idx,
        "episode_valid_length": valid_length,
        "frame_shape": list(flipped_sequence.shape),
        "latent_shape": list(latent.shape),
        "num_steps": num_steps,
        "start_mode": start_mode,
        "horizontal_flip": True,
        "mse": mse,
        "output_image": str(image_path),
    }
    save_json(stats_path, stats)
    return {
        "image_path": str(image_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main() -> None:
    """Run the flipped first-frame reconstruction CLI."""

    args = parse_args()
    result = reconstruct_flipped_first_frame(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        task=args.task,
        split=args.split,
        camera=args.camera,
        episode=args.episode,
        frame_idx=args.frame_idx,
        resolution=args.resolution,
        num_steps=args.num_steps,
        start_mode=args.start_mode,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print(f"Wrote grid to {result['image_path']}")


if __name__ == "__main__":
    main()
