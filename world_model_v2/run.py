"""Upstream-shaped plain-Python entrypoint for Stage-1 latent world model training."""

from __future__ import annotations

import argparse

import torch

from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig
from world_model_v2.experiments.latent_dynamics_experiment import LatentDynamicsExperiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args into dataset, algorithm, and experiment sections."""

    parser = argparse.ArgumentParser(description=__doc__)

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--data-root", default="data/full")
    dataset_group.add_argument("--task", default="single_grasp")
    dataset_group.add_argument("--split", default="train")
    dataset_group.add_argument("--camera", default="camera_1_color")
    dataset_group.add_argument("--obs-key", action="append", dest="obs_keys", default=None)
    dataset_group.add_argument("--resolution", type=int, default=128)
    dataset_group.add_argument("--horizon", type=int, default=1)
    dataset_group.add_argument("--val-horizon", type=int, default=200)
    dataset_group.add_argument("--action-mode", default="single_grasp")

    algorithm_group = parser.add_argument_group("algorithm")
    algorithm_group.add_argument(
        "--use-upstream-stage1-size",
        action="store_true",
        help=(
            "Use the upstream repo's published Stage-1 sizes for the overlapping "
            "config fields in this simplified implementation."
        ),
    )
    algorithm_group.add_argument("--training-stage", type=int, default=1)
    algorithm_group.add_argument("--latent-channels", type=int, default=4)
    algorithm_group.add_argument("--latent-dim", type=int, default=128)
    algorithm_group.add_argument("--hidden-channels", type=int, default=64)
    algorithm_group.add_argument("--timesteps", type=int, default=32)
    algorithm_group.add_argument("--sigma-min", type=float, default=0.01)
    algorithm_group.add_argument("--sigma-max", type=float, default=1.0)
    algorithm_group.add_argument("--infer-steps", type=int, default=2)

    experiment_group = parser.add_argument_group("experiment")
    experiment_group.add_argument("--run-name", default="")
    experiment_group.add_argument("--output-dir", default="outputs/stage1")
    experiment_group.add_argument("--batch-size", type=int, default=8)
    experiment_group.add_argument("--lr", type=float, default=2e-4)
    experiment_group.add_argument("--weight-decay", type=float, default=1e-4)
    experiment_group.add_argument("--max-steps", type=int, default=50)
    experiment_group.add_argument("--validation-interval", type=int, default=25)
    experiment_group.add_argument("--checkpoint-interval", type=int, default=25)
    experiment_group.add_argument("--save-preview-initial-minutes", type=float, default=0.0)
    experiment_group.add_argument("--save-preview-late-minutes", type=float, default=0.0)
    experiment_group.add_argument("--save-preview-switch-minutes", type=float, default=0.0)
    experiment_group.add_argument("--early-stop-window-size", type=int, default=0)
    experiment_group.add_argument("--early-stop-patience-windows", type=int, default=0)
    experiment_group.add_argument("--early-stop-min-delta", type=float, default=0.0)
    experiment_group.add_argument("--early-stop-warmup-steps", type=int, default=0)
    experiment_group.add_argument("--log-interval", type=int, default=5)
    experiment_group.add_argument("--num-workers", type=int, default=0)
    experiment_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    experiment_group.add_argument("--seed", type=int, default=7)
    experiment_group.add_argument("--resume", default="")

    return parser.parse_args(argv)


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Convert parsed args into the nested run configuration."""

    obs_keys = tuple(args.obs_keys or [args.camera])
    if args.use_upstream_stage1_size:
        upstream_sized = AlgorithmConfig.upstream_stage1(num_views=len(obs_keys))
        latent_channels = upstream_sized.latent_channels
        latent_dim = upstream_sized.latent_dim
        hidden_channels = upstream_sized.hidden_channels
        timesteps = upstream_sized.timesteps
        infer_steps = upstream_sized.infer_steps
    else:
        latent_channels = args.latent_channels
        latent_dim = args.latent_dim
        hidden_channels = args.hidden_channels
        timesteps = args.timesteps
        infer_steps = args.infer_steps

    return RunConfig(
        dataset=DatasetConfig(
            data_root=args.data_root,
            task=args.task,
            split=args.split,
            obs_keys=obs_keys,
            resolution=args.resolution,
            horizon=args.horizon,
            val_horizon=args.val_horizon,
            action_mode=args.action_mode,
        ),
        algorithm=AlgorithmConfig(
            training_stage=args.training_stage,
            latent_channels=latent_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            timesteps=timesteps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            infer_steps=infer_steps,
        ),
        experiment=ExperimentConfig(
            run_name=args.run_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval,
            save_preview_initial_minutes=args.save_preview_initial_minutes,
            save_preview_late_minutes=args.save_preview_late_minutes,
            save_preview_switch_minutes=args.save_preview_switch_minutes,
            early_stop_window_size=args.early_stop_window_size,
            early_stop_patience_windows=args.early_stop_patience_windows,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_warmup_steps=args.early_stop_warmup_steps,
            log_interval=args.log_interval,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """Build the experiment from CLI args and run Stage-1 training."""

    args = parse_args(argv)
    cfg = build_run_config(args)
    LatentDynamicsExperiment(cfg).run()


if __name__ == "__main__":
    main()
