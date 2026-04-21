"""Run standalone SO101 autoregressive open-rollout validation from one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dreamdojo.so101_rf.checkpointing import save_json
from dreamdojo.so101_rf.dataset import load_so101_video_clip
from dreamdojo.so101_rf.runtime import (
    build_model_from_checkpoint,
    load_training_checkpoint,
    run_open_rollout_validation,
)
from dreamdojo.so101_rf.visualization import build_side_by_side_grid, write_side_by_side_mp4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone SO101 open-rollout validation."""

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
    parser.add_argument("--dynamics-open-rollout-stride-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="dreamdojo/outputs/open_rollout_demo")
    parser.add_argument("--run-name", default="")
    return parser.parse_args(argv)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory for one standalone open-rollout run."""

    if args.run_name:
        return Path(args.output_dir) / args.run_name
    return Path(args.output_dir) / (
        f"ep{args.episode}_f{args.frame_start}_{args.frame_end}"
    )


def main(argv: list[str] | None = None) -> None:
    """Load one checkpoint, run SO101 open rollout, and export preview artifacts."""

    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(
        checkpoint,
        device=device,
        infer_steps_override=args.dynamics_infer_steps,
    )
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
    )
    frames = clip["frames"].to(device)
    actions = clip["actions"].to(device)
    predicted, stats = run_open_rollout_validation(
        model,
        frames,
        actions,
        stride_frames=args.dynamics_open_rollout_stride_frames,
    )
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "open_rollout_grid.png"
    video_path = output_dir / "open_rollout.mp4"
    stats_path = output_dir / "open_rollout_stats.json"
    build_side_by_side_grid(
        original=frames.detach().cpu(),
        reconstructed=predicted,
        max_frames=int(frames.shape[0]),
        context_frames=int(stats["seed_frames"]),
    ).save(grid_path)
    exported_frame_count = write_side_by_side_mp4(
        original=frames.detach().cpu(),
        reconstructed=predicted,
        output_path=video_path,
        context_frames=int(stats["seed_frames"]),
    )
    stats.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "device": str(device),
            "dynamics_infer_steps": int(model.dynamics.cfg.dynamics_infer_steps),
            "exported_video_frame_count": int(exported_frame_count),
        }
    )
    save_json(stats_path, stats)
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
