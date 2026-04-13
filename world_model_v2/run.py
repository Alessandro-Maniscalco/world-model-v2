"""CLI entrypoint for the root Wan-VAE plus RF-DiT experiment."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from world_model_v2.experiment import Experiment, ExperimentConfig


def parse_int_csv(value: str | None) -> tuple[int, ...] | None:
    """Parse one optional comma-separated integer list from the CLI."""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(f"Expected a comma-separated integer list, received {value!r}.")
    return tuple(int(part) for part in parts)


def parse_float_csv(value: str | None) -> tuple[float, ...] | None:
    """Parse one optional comma-separated float list from the CLI."""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if any(part == "" for part in parts):
        raise ValueError(f"Expected a comma-separated float list, received {value!r}.")
    return tuple(float(part) for part in parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the world-model training entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ae_only", "dynamics_only"], default="ae_only")
    parser.add_argument(
        "--dataset-format",
        choices=[
            "interactive_world_sim",
            "lerobot_metaworld",
            "lerobot_aloha_sim_transfer_cube_scripted",
            "lerobot_so101_base_sim_pickplace",
            "maniskill_replay",
        ],
        default="interactive_world_sim",
    )
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--metaworld-task-index", type=int, default=None)
    parser.add_argument("--metaworld-repo-id", default="lerobot/metaworld_mt50")
    parser.add_argument("--metaworld-cache-dir", default="")
    parser.add_argument(
        "--aloha-repo-id",
        default="lerobot/aloha_sim_transfer_cube_scripted",
    )
    parser.add_argument("--aloha-cache-dir", default="")
    parser.add_argument("--maniskill-traj-h5", default="trajectory.rgb.pd_joint_pos.physx_cpu.h5")
    parser.add_argument("--maniskill-traj-json", default="trajectory.rgb.pd_joint_pos.physx_cpu.json")
    parser.add_argument("--maniskill-camera", default="base_camera")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--train-all-episodes", action="store_true")
    parser.add_argument("--validation-split", default="")
    parser.add_argument("--validation-episode", type=int, default=0)
    parser.add_argument("--validation-episodes", default=None)
    parser.add_argument("--camera", default="camera_1_color")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dynamics-infer-steps", type=int, default=35)
    parser.add_argument("--dynamics-train-timesteps", type=int, default=1000)
    parser.add_argument("--dynamics-rf-shift", type=float, default=5.0)
    parser.add_argument("--conditional-frame-timestep", type=float, default=-1.0)
    parser.add_argument("--conditional-frame-sigma", type=float, default=0.0)
    parser.add_argument("--dynamics-video-condition-dropout", type=float, default=0.0)
    parser.add_argument("--dynamics-guidance-scale", type=float, default=0.0)
    parser.add_argument("--dynamics-self-forcing-loss-weight", type=float, default=0.0)
    parser.add_argument("--dynamics-rollout-self-forcing-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--dynamics-self-forcing-mode",
        choices=["expanded_context", "rollout"],
        default="expanded_context",
    )
    parser.add_argument("--dynamics-self-forcing-warmup-steps", type=int, default=0)
    parser.add_argument("--dynamics-self-forcing-ramp-steps", type=int, default=0)
    parser.add_argument("--dynamics-rollout-self-forcing-warmup-steps", type=int, default=0)
    parser.add_argument("--dynamics-rollout-self-forcing-ramp-steps", type=int, default=0)
    parser.add_argument("--dynamics-self-forcing-rollout-chunks", type=int, default=0)
    parser.add_argument(
        "--dynamics-context-frames",
        type=int,
        default=DYNAMICS_FRAME_LAYOUT.context_frames,
    )
    parser.add_argument(
        "--dynamics-target-frames",
        type=int,
        default=DYNAMICS_FRAME_LAYOUT.target_frames,
    )
    parser.add_argument("--dynamics-conditioning-frame-choices", default=None)
    parser.add_argument("--dynamics-conditioning-frame-probabilities", default=None)
    parser.add_argument("--dynamics-validation-conditioning-frame-choices", default=None)
    parser.add_argument("--dynamics-open-rollout-context-frames", type=int, default=None)
    parser.add_argument("--dynamics-open-rollout-stride-frames", type=int, default=None)
    parser.add_argument("--dynamics-model-channels", type=int, default=256)
    parser.add_argument("--dynamics-num-blocks", type=int, default=4)
    parser.add_argument("--dynamics-num-heads", type=int, default=4)
    parser.add_argument(
        "--dynamics-action-conditioning-mode",
        choices=["chunk_per_frame"],
        default="chunk_per_frame",
    )
    parser.add_argument("--dynamics-zero-init-action-embedder", action="store_true")
    parser.add_argument(
        "--dynamics-use-adaln-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dynamics-adaln-lora-dim", type=int, default=64)
    parser.add_argument("--dynamics-rope-t-extrapolation-ratio", type=float, default=1.0)
    parser.add_argument(
        "--dynamics-use-learned-temporal-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dynamics-validation-metric",
        choices=[
            "next_frame_mse",
            "open_rollout_frame_mse",
            "open_rollout_consistency_score",
        ],
        default="next_frame_mse",
    )
    parser.add_argument(
        "--dynamics-run-open-rollout-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--kl-beta", type=float, default=1e-4)
    parser.add_argument("--recon-mse-weight", type=float, default=1.0)
    parser.add_argument("--recon-l1-weight", type=float, default=0.0)
    parser.add_argument("--recon-edge-weight", type=float, default=0.0)
    parser.add_argument("--recon-motion-weight", type=float, default=0.0)
    parser.add_argument("--recon-motion-threshold", type=float, default=0.02)
    parser.add_argument("--recon-motion-dilation-kernel-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--early-stop-window-size", type=int, default=1)
    parser.add_argument("--early-stop-patience-windows", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-10)
    parser.add_argument("--early-stop-warmup-steps", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--resume", default="")
    parser.add_argument("--load-encoder-decoder", default="")
    parser.add_argument("--load-dynamics", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Convert parsed CLI arguments into the experiment config."""

    return ExperimentConfig(
        mode=args.mode,
        dataset_format=args.dataset_format,
        data_root=args.data_root,
        task=args.task,
        metaworld_task_index=args.metaworld_task_index,
        metaworld_repo_id=args.metaworld_repo_id,
        metaworld_cache_dir=args.metaworld_cache_dir,
        aloha_repo_id=args.aloha_repo_id,
        aloha_cache_dir=args.aloha_cache_dir,
        maniskill_traj_h5=args.maniskill_traj_h5,
        maniskill_traj_json=args.maniskill_traj_json,
        maniskill_camera=args.maniskill_camera,
        split=args.split,
        episode=args.episode,
        train_all_episodes=args.train_all_episodes,
        validation_split=args.validation_split,
        validation_episode=args.validation_episode,
        validation_episodes=parse_int_csv(args.validation_episodes),
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
        conditional_frame_timestep=args.conditional_frame_timestep,
        conditional_frame_sigma=args.conditional_frame_sigma,
        dynamics_video_condition_dropout=args.dynamics_video_condition_dropout,
        dynamics_guidance_scale=args.dynamics_guidance_scale,
        dynamics_self_forcing_loss_weight=args.dynamics_self_forcing_loss_weight,
        dynamics_rollout_self_forcing_loss_weight=args.dynamics_rollout_self_forcing_loss_weight,
        dynamics_self_forcing_mode=args.dynamics_self_forcing_mode,
        dynamics_self_forcing_warmup_steps=args.dynamics_self_forcing_warmup_steps,
        dynamics_self_forcing_ramp_steps=args.dynamics_self_forcing_ramp_steps,
        dynamics_rollout_self_forcing_warmup_steps=args.dynamics_rollout_self_forcing_warmup_steps,
        dynamics_rollout_self_forcing_ramp_steps=args.dynamics_rollout_self_forcing_ramp_steps,
        dynamics_self_forcing_rollout_chunks=args.dynamics_self_forcing_rollout_chunks,
        dynamics_context_frames=args.dynamics_context_frames,
        dynamics_target_frames=args.dynamics_target_frames,
        dynamics_conditioning_frame_choices=parse_int_csv(
            args.dynamics_conditioning_frame_choices
        ),
        dynamics_conditioning_frame_probabilities=parse_float_csv(
            args.dynamics_conditioning_frame_probabilities
        ),
        dynamics_validation_conditioning_frame_choices=parse_int_csv(
            args.dynamics_validation_conditioning_frame_choices
        ),
        dynamics_open_rollout_context_frames=args.dynamics_open_rollout_context_frames,
        dynamics_open_rollout_stride_frames=args.dynamics_open_rollout_stride_frames,
        dynamics_model_channels=args.dynamics_model_channels,
        dynamics_num_blocks=args.dynamics_num_blocks,
        dynamics_num_heads=args.dynamics_num_heads,
        dynamics_action_conditioning_mode=args.dynamics_action_conditioning_mode,
        dynamics_zero_init_action_embedder=args.dynamics_zero_init_action_embedder,
        dynamics_use_adaln_lora=args.dynamics_use_adaln_lora,
        dynamics_adaln_lora_dim=args.dynamics_adaln_lora_dim,
        dynamics_rope_t_extrapolation_ratio=args.dynamics_rope_t_extrapolation_ratio,
        dynamics_use_learned_temporal_embedding=args.dynamics_use_learned_temporal_embedding,
        dynamics_validation_metric=args.dynamics_validation_metric,
        dynamics_run_open_rollout_validation=args.dynamics_run_open_rollout_validation,
        kl_beta=args.kl_beta,
        recon_mse_weight=args.recon_mse_weight,
        recon_l1_weight=args.recon_l1_weight,
        recon_edge_weight=args.recon_edge_weight,
        recon_motion_weight=args.recon_motion_weight,
        recon_motion_threshold=args.recon_motion_threshold,
        recon_motion_dilation_kernel_size=args.recon_motion_dilation_kernel_size,
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
    """Run the world-model training pipeline from the command line."""

    config = build_config(parse_args(argv))
    experiment = Experiment(config)
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
