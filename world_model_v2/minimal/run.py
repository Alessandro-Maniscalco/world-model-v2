"""CLI entrypoint for the minimal multi-mode world model experiment."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from world_model_v2.minimal.experiment import MinimalExperiment, MinimalExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the minimal training entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["joint", "ae_only", "dynamics_only"], default="joint")
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default="camera_1_color")
    parser.add_argument("--frame-start", type=int, default=111)
    parser.add_argument("--frame-end", type=int, default=116)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--early-stop-window-size", type=int, default=1)
    parser.add_argument("--early-stop-patience-windows", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-10)
    parser.add_argument("--early-stop-warmup-steps", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="outputs/minimal")
    parser.add_argument("--resume", default="")
    parser.add_argument("--load-encoder-decoder", default="")
    parser.add_argument("--load-dynamics", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def build_config(args: argparse.Namespace) -> MinimalExperimentConfig:
    """Convert parsed CLI arguments into the minimal experiment config."""

    return MinimalExperimentConfig(
        mode=args.mode,
        data_root=args.data_root,
        task=args.task,
        split=args.split,
        episode=args.episode,
        camera=args.camera,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        resolution=args.resolution,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        validation_interval=args.validation_interval,
        checkpoint_interval=args.checkpoint_interval,
        early_stop_window_size=args.early_stop_window_size,
        early_stop_patience_windows=args.early_stop_patience_windows,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup_steps=args.early_stop_warmup_steps,
        log_interval=args.log_interval,
        device=args.device,
        run_name=args.run_name,
        output_dir=args.output_dir,
        resume=args.resume,
        load_encoder_decoder=args.load_encoder_decoder,
        load_dynamics=args.load_dynamics,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the minimal training pipeline from the command line."""

    config = build_config(parse_args(argv))
    experiment = MinimalExperiment(config)
    experiment.run()
    print(
        json.dumps(
            {
                "run_dir": str(experiment.run_dir),
                "mode": config.mode,
                "step": experiment.current_step,
                "best_metric": experiment.best_metric,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
