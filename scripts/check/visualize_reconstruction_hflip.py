"""Run a checkpoint reconstruction on a horizontally flipped validation clip.

source .venv/bin/activate
python scripts/check/visualize_reconstruction_hflip.py \
  --checkpoint outputs/ae_only_single_grasp_ep0_f111_116/checkpoints/last.pt \
  --output-dir outputs/ae_only_single_grasp_ep0_f111_116/samples/step_001700_hflip
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

from world_model_v2.dataset import load_clip
from world_model_v2.experiment import checkpoint_ae_backend, load_training_checkpoint
from world_model_v2.model import WorldModel
from world_model_v2.utils.checkpointing import save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4


def resolve_image_size(config: dict[str, Any]) -> tuple[int, int]:
    """Resolve image height and width from saved checkpoint config values."""

    resolution = int(config["resolution"])
    height = config.get("height")
    width = config.get("width")
    resolved_height = resolution if height is None else int(height)
    resolved_width = resolution if width is None else int(width)
    return resolved_height, resolved_width


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the flipped reconstruction check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    """Fail early when CUDA is requested but unavailable."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available to PyTorch. "
            "Rerun with --device cpu or install a compatible torch wheel."
        )


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> WorldModel:
    """Instantiate the model using serialized checkpoint config."""

    config = checkpoint["config"]
    height, width = resolve_image_size(config)
    model = WorldModel(
        latent_channels=int(config["latent_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        ae_backend=checkpoint_ae_backend(checkpoint),
        resolution=int(config["resolution"]),
        height=height,
        width=width,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_flipped_clip(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Load the checkpoint clip slice and horizontally flip every frame."""

    clip_metadata = checkpoint["clip_metadata"]
    clip_height = clip_metadata.get("height")
    clip_width = clip_metadata.get("width")
    clip = load_clip(
        data_root=checkpoint["config"]["data_root"],
        task=clip_metadata["task"],
        split=clip_metadata["split"],
        episode=int(clip_metadata["episode"]),
        camera=clip_metadata["camera"],
        frame_start=clip_metadata["frame_start"],
        frame_end=clip_metadata["frame_end"],
        resolution=int(clip_metadata["resolution"]),
        height=None if clip_height is None else int(clip_height),
        width=None if clip_width is None else int(clip_width),
    )
    clip["frames"] = torch.flip(clip["frames"], dims=(-1,))
    return clip


@torch.no_grad()
def reconstruct_flipped_clip(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Reconstruct the flipped validation clip and export grid, MP4, and stats."""

    device_obj = torch.device(device)
    validate_device(device_obj)
    checkpoint = load_training_checkpoint(checkpoint_path, device_obj)
    model = build_model_from_checkpoint(checkpoint, device_obj)
    clip = load_flipped_clip(checkpoint)
    frames = clip["frames"]
    reconstructed = model.reconstruct(frames.to(device_obj), deterministic=True).cpu()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    grid_path = output_path / "episode_0_grid.png"
    video_path = output_path / "episode_0.mp4"
    stats_path = output_path / "episode_0_stats.json"

    build_side_by_side_grid(
        original=frames,
        reconstructed=reconstructed,
        max_frames=int(frames.shape[0]),
    ).save(grid_path)
    exported_frame_count = write_side_by_side_mp4(
        original=frames,
        reconstructed=reconstructed,
        output_path=video_path,
        duration_ms=120,
    )

    stats = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "mode": str(checkpoint["mode"]),
        "ae_backend": checkpoint_ae_backend(checkpoint),
        "episode": int(clip["episode_idx"].item()),
        "frame_start": int(clip["frame_idx"][0].item()),
        "frame_end": int(clip["frame_idx"][-1].item()),
        "input_frame_count": int(frames.shape[0]),
        "decoded_frame_count": int(reconstructed.shape[0]),
        "exported_video_frame_count": int(exported_frame_count),
        "horizontal_flip": True,
        "recon_mse": float(torch.mean((reconstructed - frames) ** 2).item()),
        "output_grid": str(grid_path),
        "output_video": str(video_path),
    }
    if stats["input_frame_count"] != stats["decoded_frame_count"]:
        raise RuntimeError(f"Decoded frame count mismatch: {stats}")
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
    """Run the flipped reconstruction CLI."""

    args = parse_args()
    result = reconstruct_flipped_clip(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print(f"Wrote grid to {result['grid_path']}")
    print(f"Wrote mp4 to {result['video_path']}")


if __name__ == "__main__":
    main()
