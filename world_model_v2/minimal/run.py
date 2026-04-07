"""CLI entrypoint for the minimal VAE-plus-dynamics experiment."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from world_model_v2.minimal.experiment import MinimalExperiment, MinimalExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the minimal training entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ae_only", "dynamics_only"], default="ae_only")
    parser.add_argument(
        "--dataset-format",
        choices=["interactive_world_sim", "lerobot_metaworld"],
        default="interactive_world_sim",
    )
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--metaworld-task-index", type=int, default=None)
    parser.add_argument("--metaworld-repo-id", default="lerobot/metaworld_mt50")
    parser.add_argument("--metaworld-cache-dir", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--train-all-episodes", action="store_true")
    parser.add_argument("--validation-split", default="")
    parser.add_argument("--validation-episode", type=int, default=0)
    parser.add_argument("--camera", default="camera_1_color")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dynamics-infer-steps", type=int, default=16)
    parser.add_argument("--dynamics-train-timesteps", type=int, default=1000)
    parser.add_argument("--dynamics-rf-shift", type=float, default=5.0)
    parser.add_argument("--kl-beta", type=float, default=1e-4)
    parser.add_argument("--recon-mse-weight", type=float, default=1.0)
    parser.add_argument("--recon-l1-weight", type=float, default=0.0)
    parser.add_argument("--recon-edge-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
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
        dataset_format=args.dataset_format,
        data_root=args.data_root,
        task=args.task,
        metaworld_task_index=args.metaworld_task_index,
        metaworld_repo_id=args.metaworld_repo_id,
        metaworld_cache_dir=args.metaworld_cache_dir,
        split=args.split,
        episode=args.episode,
        train_all_episodes=args.train_all_episodes,
        validation_split=args.validation_split,
        validation_episode=args.validation_episode,
        camera=args.camera,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        ae_backend="wan",
        dynamics_infer_steps=args.dynamics_infer_steps,
        dynamics_train_timesteps=args.dynamics_train_timesteps,
        dynamics_rf_shift=args.dynamics_rf_shift,
        kl_beta=args.kl_beta,
        recon_mse_weight=args.recon_mse_weight,
        recon_l1_weight=args.recon_l1_weight,
        recon_edge_weight=args.recon_edge_weight,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
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
