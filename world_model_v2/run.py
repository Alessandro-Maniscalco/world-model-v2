"""Upstream-shaped plain-Python entrypoint for three-stage latent world model training."""

from __future__ import annotations

import argparse
import sys

import torch

from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig
from world_model_v2.experiments.latent_dynamics_experiment import LatentDynamicsExperiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args into dataset, algorithm, and experiment sections."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
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
    algorithm_group.add_argument("--dyn-infer-steps", type=int, default=1)
    algorithm_group.add_argument("--load-ae", default="")
    algorithm_group.add_argument("--action-dim", type=int, default=4)
    algorithm_group.add_argument("--dynamics-hidden-channels", type=int, default=64)
    algorithm_group.add_argument("--action-emb-dim", type=int, default=128)
    algorithm_group.add_argument("--dynamics-attention-heads", type=int, default=4)
    algorithm_group.add_argument("--mask-prev-action", action="store_true")
    algorithm_group.add_argument("--sampling-strategy", default="uniform")
    algorithm_group.add_argument("--prev-frame-noise-scale", type=float, default=0.1)
    algorithm_group.add_argument("--last-frame-loss-only", action="store_true")
    algorithm_group.add_argument("--loss-weighting", default="auto")
    algorithm_group.add_argument("--stage3-latent-noise-std", type=float, default=0.02)

    experiment_group = parser.add_argument_group("experiment")
    experiment_group.add_argument("--run-name", default="")
    experiment_group.add_argument("--output-dir", default="outputs/stage1")
    experiment_group.add_argument("--batch-size", type=int, default=8)
    experiment_group.add_argument("--lr", type=float, default=2e-4)
    experiment_group.add_argument("--weight-decay", type=float, default=1e-4)
    experiment_group.add_argument("--grad-clip-norm", type=float, default=0.0)
    experiment_group.add_argument("--lr-scheduler", default="none")
    experiment_group.add_argument("--warmup-steps", type=int, default=0)
    experiment_group.add_argument("--max-steps", type=int, default=50)
    experiment_group.add_argument("--validation-interval", type=int, default=25)
    experiment_group.add_argument("--checkpoint-interval", type=int, default=25)
    experiment_group.add_argument("--save-preview-initial-minutes", type=float, default=0.0)
    experiment_group.add_argument("--save-preview-late-minutes", type=float, default=0.0)
    experiment_group.add_argument("--save-preview-switch-minutes", type=float, default=0.0)
    experiment_group.add_argument("--early-stop-metric", default="training_loss")
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

    args = parser.parse_args(raw_argv)
    provided_dests: set[str] = set()
    for token in raw_argv:
        option = token.split("=", maxsplit=1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            provided_dests.add(action.dest)
    setattr(args, "_provided_dests", provided_dests)
    return args


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Convert parsed args into the nested run configuration."""

    obs_keys = tuple(args.obs_keys or [args.camera])
    provided_dests = set(getattr(args, "_provided_dests", set()))
    stage2_dataset_defaults = DatasetConfig.upstream_stage2() if args.training_stage == 2 else None
    stage2_algorithm_defaults = (
        AlgorithmConfig.upstream_stage2(num_views=len(obs_keys)) if args.training_stage == 2 else None
    )
    stage2_experiment_defaults = ExperimentConfig.upstream_stage2() if args.training_stage == 2 else None

    def pick(dest: str, current: object, stage2_default: object | None = None) -> object:
        """Select the explicit CLI value or a Stage-2-specific default."""

        if dest in provided_dests or stage2_default is None:
            return current
        return stage2_default

    if args.use_upstream_stage1_size:
        upstream_sized = AlgorithmConfig.upstream_stage1(num_views=len(obs_keys))
        latent_channels = upstream_sized.latent_channels
        latent_dim = upstream_sized.latent_dim
        hidden_channels = upstream_sized.hidden_channels
        timesteps = upstream_sized.timesteps
        infer_steps = upstream_sized.infer_steps
    elif stage2_algorithm_defaults is not None:
        latent_channels = int(
            pick("latent_channels", args.latent_channels, stage2_algorithm_defaults.latent_channels)
        )
        latent_dim = int(pick("latent_dim", args.latent_dim, stage2_algorithm_defaults.latent_dim))
        hidden_channels = int(
            pick("hidden_channels", args.hidden_channels, stage2_algorithm_defaults.hidden_channels)
        )
        timesteps = int(pick("timesteps", args.timesteps, stage2_algorithm_defaults.timesteps))
        infer_steps = int(pick("infer_steps", args.infer_steps, stage2_algorithm_defaults.infer_steps))
    else:
        latent_channels = args.latent_channels
        latent_dim = args.latent_dim
        hidden_channels = args.hidden_channels
        timesteps = args.timesteps
        infer_steps = args.infer_steps

    if args.training_stage in (2, 3) and not args.load_ae:
        raise ValueError("--load-ae is required for training-stage 2 and 3")

    return RunConfig(
        dataset=DatasetConfig(
            data_root=str(pick("data_root", args.data_root)),
            task=str(pick("task", args.task)),
            split=str(pick("split", args.split)),
            obs_keys=obs_keys,
            resolution=int(pick("resolution", args.resolution)),
            horizon=int(
                pick(
                    "horizon",
                    args.horizon,
                    stage2_dataset_defaults.horizon if stage2_dataset_defaults is not None else None,
                )
            ),
            val_horizon=int(
                pick(
                    "val_horizon",
                    args.val_horizon,
                    stage2_dataset_defaults.val_horizon if stage2_dataset_defaults is not None else None,
                )
            ),
            action_mode=str(pick("action_mode", args.action_mode)),
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
            dyn_infer_steps=int(
                pick(
                    "dyn_infer_steps",
                    args.dyn_infer_steps,
                    stage2_algorithm_defaults.dyn_infer_steps if stage2_algorithm_defaults is not None else None,
                )
            ),
            load_ae=args.load_ae,
            action_dim=args.action_dim,
            dynamics_hidden_channels=int(
                pick(
                    "dynamics_hidden_channels",
                    args.dynamics_hidden_channels,
                    stage2_algorithm_defaults.dynamics_hidden_channels
                    if stage2_algorithm_defaults is not None
                    else None,
                )
            ),
            action_emb_dim=int(
                pick(
                    "action_emb_dim",
                    args.action_emb_dim,
                    stage2_algorithm_defaults.action_emb_dim if stage2_algorithm_defaults is not None else None,
                )
            ),
            dynamics_attention_heads=int(
                pick(
                    "dynamics_attention_heads",
                    args.dynamics_attention_heads,
                    stage2_algorithm_defaults.dynamics_attention_heads
                    if stage2_algorithm_defaults is not None
                    else None,
                )
            ),
            mask_prev_action=bool(
                pick(
                    "mask_prev_action",
                    args.mask_prev_action,
                    stage2_algorithm_defaults.mask_prev_action if stage2_algorithm_defaults is not None else None,
                )
            ),
            sampling_strategy=str(
                pick(
                    "sampling_strategy",
                    args.sampling_strategy,
                    stage2_algorithm_defaults.sampling_strategy if stage2_algorithm_defaults is not None else None,
                )
            ),
            prev_frame_noise_scale=float(
                pick(
                    "prev_frame_noise_scale",
                    args.prev_frame_noise_scale,
                    stage2_algorithm_defaults.prev_frame_noise_scale
                    if stage2_algorithm_defaults is not None
                    else None,
                )
            ),
            last_frame_loss_only=bool(
                pick(
                    "last_frame_loss_only",
                    args.last_frame_loss_only,
                    stage2_algorithm_defaults.last_frame_loss_only
                    if stage2_algorithm_defaults is not None
                    else None,
                )
            ),
            loss_weighting=str(
                pick(
                    "loss_weighting",
                    args.loss_weighting,
                    stage2_algorithm_defaults.loss_weighting if stage2_algorithm_defaults is not None else None,
                )
            ),
            stage3_latent_noise_std=args.stage3_latent_noise_std,
        ),
        experiment=ExperimentConfig(
            run_name=args.run_name,
            output_dir=str(
                pick(
                    "output_dir",
                    args.output_dir,
                    stage2_experiment_defaults.output_dir if stage2_experiment_defaults is not None else None,
                )
            ),
            batch_size=int(
                pick(
                    "batch_size",
                    args.batch_size,
                    stage2_experiment_defaults.batch_size if stage2_experiment_defaults is not None else None,
                )
            ),
            lr=float(
                pick("lr", args.lr, stage2_experiment_defaults.lr if stage2_experiment_defaults is not None else None)
            ),
            weight_decay=float(
                pick(
                    "weight_decay",
                    args.weight_decay,
                    stage2_experiment_defaults.weight_decay if stage2_experiment_defaults is not None else None,
                )
            ),
            grad_clip_norm=float(
                pick(
                    "grad_clip_norm",
                    args.grad_clip_norm,
                    stage2_experiment_defaults.grad_clip_norm
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            lr_scheduler=str(
                pick(
                    "lr_scheduler",
                    args.lr_scheduler,
                    stage2_experiment_defaults.lr_scheduler
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            warmup_steps=int(
                pick(
                    "warmup_steps",
                    args.warmup_steps,
                    stage2_experiment_defaults.warmup_steps
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            max_steps=int(
                pick(
                    "max_steps",
                    args.max_steps,
                    stage2_experiment_defaults.max_steps if stage2_experiment_defaults is not None else None,
                )
            ),
            validation_interval=int(
                pick(
                    "validation_interval",
                    args.validation_interval,
                    stage2_experiment_defaults.validation_interval
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            checkpoint_interval=int(
                pick(
                    "checkpoint_interval",
                    args.checkpoint_interval,
                    stage2_experiment_defaults.checkpoint_interval
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            save_preview_initial_minutes=args.save_preview_initial_minutes,
            save_preview_late_minutes=args.save_preview_late_minutes,
            save_preview_switch_minutes=args.save_preview_switch_minutes,
            early_stop_metric=str(
                pick(
                    "early_stop_metric",
                    args.early_stop_metric,
                    stage2_experiment_defaults.early_stop_metric
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            early_stop_window_size=int(
                pick(
                    "early_stop_window_size",
                    args.early_stop_window_size,
                    stage2_experiment_defaults.early_stop_window_size
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            early_stop_patience_windows=int(
                pick(
                    "early_stop_patience_windows",
                    args.early_stop_patience_windows,
                    stage2_experiment_defaults.early_stop_patience_windows
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            early_stop_min_delta=float(
                pick(
                    "early_stop_min_delta",
                    args.early_stop_min_delta,
                    stage2_experiment_defaults.early_stop_min_delta
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            early_stop_warmup_steps=int(
                pick(
                    "early_stop_warmup_steps",
                    args.early_stop_warmup_steps,
                    stage2_experiment_defaults.early_stop_warmup_steps
                    if stage2_experiment_defaults is not None
                    else None,
                )
            ),
            log_interval=int(
                pick(
                    "log_interval",
                    args.log_interval,
                    stage2_experiment_defaults.log_interval if stage2_experiment_defaults is not None else None,
                )
            ),
            num_workers=int(
                pick(
                    "num_workers",
                    args.num_workers,
                    stage2_experiment_defaults.num_workers if stage2_experiment_defaults is not None else None,
                )
            ),
            device=args.device,
            seed=args.seed,
            resume=args.resume,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """Build the experiment from CLI args and run the requested training stage."""

    args = parse_args(argv)
    cfg = build_run_config(args)
    LatentDynamicsExperiment(cfg).run()


if __name__ == "__main__":
    main()
