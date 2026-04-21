"""Tests for the root CLI configuration helpers."""

from __future__ import annotations

import pytest

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from world_model_v2.run import build_config, parse_args
from world_model_v2.wan_vae import DEFAULT_WAN_DIM, DEFAULT_WAN_NUM_RES_BLOCKS, DEFAULT_WAN_Z_DIM


def test_run_parse_args_uses_expected_defaults() -> None:
    """The CLI parser should default to full-episode training bounds."""

    args = parse_args([])
    config = build_config(args)
    assert config.mode == "ae_only"
    assert config.dataset_format == "interactive_world_sim"
    assert config.split == "val"
    assert config.train_all_episodes is False
    assert config.validation_split == ""
    assert config.validation_episode == 0
    assert config.validation_episodes is None
    assert config.validation_max_frames is None
    assert config.camera == "camera_1_color"
    assert config.frame_start is None
    assert config.frame_end is None
    assert config.resolution == 128
    assert config.height is None
    assert config.width is None
    assert config.wan_dim == DEFAULT_WAN_DIM
    assert config.latent_channels == DEFAULT_WAN_Z_DIM
    assert config.wan_num_res_blocks == DEFAULT_WAN_NUM_RES_BLOCKS
    assert config.hidden_channels == 64
    assert config.ae_backend == "wan"
    assert config.dynamics_infer_steps == 35
    assert config.dynamics_train_timesteps == 1000
    assert config.dynamics_rf_shift == 5.0
    assert config.conditional_frame_timestep == -1.0
    assert config.conditional_frame_sigma == 0.0
    assert config.dynamics_video_condition_dropout == 0.0
    assert config.dynamics_guidance_scale == 0.0
    assert config.dynamics_self_forcing_loss_weight == 0.0
    assert config.dynamics_rollout_self_forcing_loss_weight == 0.0
    assert config.dynamics_self_forcing_mode == "expanded_context"
    assert config.dynamics_self_forcing_warmup_steps == 0
    assert config.dynamics_self_forcing_ramp_steps == 0
    assert config.dynamics_rollout_self_forcing_warmup_steps == 0
    assert config.dynamics_rollout_self_forcing_ramp_steps == 0
    assert config.dynamics_self_forcing_rollout_chunks == 0
    assert config.dynamics_context_frames == DYNAMICS_FRAME_LAYOUT.context_frames
    assert config.dynamics_target_frames == DYNAMICS_FRAME_LAYOUT.target_frames
    assert config.dynamics_conditioning_frame_choices is None
    assert config.dynamics_conditioning_frame_probabilities is None
    assert config.dynamics_validation_conditioning_frame_choices is None
    assert config.dynamics_open_rollout_context_frames is None
    assert config.dynamics_open_rollout_stride_frames is None
    assert config.dynamics_patch_spatial == 1
    assert config.dynamics_model_channels == 256
    assert config.dynamics_num_blocks == 4
    assert config.dynamics_num_heads == 4
    assert config.dynamics_action_conditioning_mode == "chunk_per_frame"
    assert config.dynamics_zero_init_action_embedder is False
    assert config.dynamics_use_adaln_lora is True
    assert config.dynamics_adaln_lora_dim == 64
    assert config.dynamics_rope_t_extrapolation_ratio == 1.0
    assert config.dynamics_use_learned_temporal_embedding is False
    assert config.dynamics_validation_metric == "next_frame_mse"
    assert config.dynamics_run_open_rollout_validation is None
    assert config.kl_beta == 1e-4
    assert config.recon_mse_weight == 1.0
    assert config.recon_l1_weight == 0.0
    assert config.recon_edge_weight == 0.0
    assert config.recon_motion_weight == 0.0
    assert config.recon_motion_edge_weight == 0.0
    assert config.recon_motion_threshold == 0.02
    assert config.recon_motion_dilation_kernel_size == 5
    assert config.batch_size == 64
    assert config.gradient_accumulation_steps == 1
    assert config.dataloader_num_workers is None
    assert config.dataloader_prefetch_factor == 2
    assert config.dataloader_pin_memory is None
    assert config.auto_batch_size is False
    assert config.lr == 1e-4
    assert config.lr_warmup_steps == 200
    assert config.optimizer_beta1 == 0.95
    assert config.validation_interval == 250
    assert config.validation_start_step == 0
    assert config.checkpoint_interval == 250
    assert config.early_stop_window_size == 1
    assert config.early_stop_patience_windows == 5
    assert config.early_stop_min_delta == 1e-10
    assert config.early_stop_warmup_steps == 300
    assert config.wandb_enabled is False
    assert config.wandb_project == "world-model-v2"
    assert config.wandb_entity == ""
    assert config.wandb_group == ""
    assert config.wandb_name == ""
    assert config.wandb_tags is None
    assert config.wandb_mode == "online"
    assert config.wandb_run_id == ""


def test_run_build_config_preserves_load_flags() -> None:
    """The config builder should keep the requested partial-load checkpoint paths."""

    args = parse_args(
        [
            "--mode",
            "dynamics_only",
            "--load-encoder-decoder",
            "encoder_decoder.pt",
            "--load-dynamics",
            "dynamics.pt",
        ]
    )
    config = build_config(args)
    assert config.mode == "dynamics_only"
    assert config.load_encoder_decoder == "encoder_decoder.pt"
    assert config.load_dynamics == "dynamics.pt"


def test_run_build_config_preserves_wandb_flags() -> None:
    """The config builder should keep the requested W&B logging controls."""

    args = parse_args(
        [
            "--wandb",
            "--wandb-project",
            "wm-v2",
            "--wandb-entity",
            "openai",
            "--wandb-group",
            "so101",
            "--wandb-name",
            "trial-17",
            "--wandb-tags",
            "ae,debug",
            "--wandb-mode",
            "offline",
            "--wandb-run-id",
            "existing-run-id",
        ]
    )
    config = build_config(args)
    assert config.wandb_enabled is True
    assert config.wandb_project == "wm-v2"
    assert config.wandb_entity == "openai"
    assert config.wandb_group == "so101"
    assert config.wandb_name == "trial-17"
    assert config.wandb_tags == ("ae", "debug")
    assert config.wandb_mode == "offline"
    assert config.wandb_run_id == "existing-run-id"


def test_run_build_config_preserves_validation_start_step() -> None:
    """The config builder should keep the requested validation delay step."""

    args = parse_args(
        [
            "--validation-interval",
            "250",
            "--validation-start-step",
            "30000",
        ]
    )
    config = build_config(args)
    assert config.validation_interval == 250
    assert config.validation_start_step == 30000


def test_run_build_config_preserves_validation_max_frames() -> None:
    """The config builder should keep the requested validation frame cap."""

    args = parse_args(["--validation-max-frames", "49"])
    config = build_config(args)
    assert config.validation_max_frames == 49


def test_run_build_config_preserves_wan_autoencoder_shape() -> None:
    """The config builder should keep the requested Wan autoencoder shape knobs."""

    args = parse_args(
        [
            "--wan-dim",
            "96",
            "--latent-channels",
            "64",
            "--wan-num-res-blocks",
            "1",
        ]
    )
    config = build_config(args)
    assert config.wan_dim == 96
    assert config.latent_channels == 64
    assert config.wan_num_res_blocks == 1


def test_run_build_config_preserves_dataloader_flags() -> None:
    """The config builder should keep the requested dataloader performance knobs."""

    args = parse_args(
        [
            "--dataloader-num-workers",
            "4",
            "--dataloader-prefetch-factor",
            "3",
            "--dataloader-pin-memory",
        ]
    )
    config = build_config(args)
    assert config.dataloader_num_workers == 4
    assert config.dataloader_prefetch_factor == 3
    assert config.dataloader_pin_memory is True


def test_run_build_config_preserves_gradient_accumulation_steps() -> None:
    """The config builder should keep the requested gradient accumulation factor."""

    args = parse_args(
        [
            "--batch-size",
            "1",
            "--grad-accum-steps",
            "2",
        ]
    )
    config = build_config(args)
    assert config.batch_size == 1
    assert config.gradient_accumulation_steps == 2


def test_run_build_config_preserves_rf_dynamics_flags() -> None:
    """The config builder should keep the requested RF dynamics settings."""

    args = parse_args(
        [
            "--dynamics-infer-steps",
            "8",
            "--dynamics-train-timesteps",
            "256",
            "--dynamics-rf-shift",
            "3.5",
            "--conditional-frame-timestep",
            "0.75",
            "--conditional-frame-sigma",
            "0.125",
            "--dynamics-video-condition-dropout",
            "0.35",
            "--dynamics-guidance-scale",
            "2.0",
            "--dynamics-self-forcing-loss-weight",
            "0.4",
            "--dynamics-rollout-self-forcing-loss-weight",
            "0.15",
            "--dynamics-self-forcing-mode",
            "rollout",
            "--dynamics-self-forcing-warmup-steps",
            "75",
            "--dynamics-self-forcing-ramp-steps",
            "125",
            "--dynamics-rollout-self-forcing-warmup-steps",
            "25",
            "--dynamics-rollout-self-forcing-ramp-steps",
            "50",
            "--dynamics-self-forcing-rollout-chunks",
            "2",
            "--dynamics-model-channels",
            "384",
            "--dynamics-num-blocks",
            "6",
            "--dynamics-num-heads",
            "8",
            "--dynamics-zero-init-action-embedder",
            "--no-dynamics-use-adaln-lora",
            "--dynamics-adaln-lora-dim",
            "96",
            "--dynamics-rope-t-extrapolation-ratio",
            "1.5",
            "--dynamics-use-learned-temporal-embedding",
            "--dynamics-validation-metric",
            "open_rollout_frame_mse",
        ]
    )
    config = build_config(args)
    assert config.dynamics_infer_steps == 8
    assert config.dynamics_train_timesteps == 256
    assert config.dynamics_rf_shift == 3.5
    assert config.conditional_frame_timestep == 0.75
    assert config.conditional_frame_sigma == 0.125
    assert config.dynamics_video_condition_dropout == 0.35
    assert config.dynamics_guidance_scale == 2.0
    assert config.dynamics_self_forcing_loss_weight == 0.4
    assert config.dynamics_rollout_self_forcing_loss_weight == 0.15
    assert config.dynamics_self_forcing_mode == "rollout"
    assert config.dynamics_self_forcing_warmup_steps == 75
    assert config.dynamics_self_forcing_ramp_steps == 125
    assert config.dynamics_rollout_self_forcing_warmup_steps == 25
    assert config.dynamics_rollout_self_forcing_ramp_steps == 50
    assert config.dynamics_self_forcing_rollout_chunks == 2
    assert config.dynamics_model_channels == 384
    assert config.dynamics_num_blocks == 6
    assert config.dynamics_num_heads == 8
    assert config.dynamics_action_conditioning_mode == "chunk_per_frame"
    assert config.dynamics_zero_init_action_embedder is True
    assert config.dynamics_use_adaln_lora is False
    assert config.dynamics_adaln_lora_dim == 96
    assert config.dynamics_rope_t_extrapolation_ratio == 1.5
    assert config.dynamics_use_learned_temporal_embedding is True
    assert config.dynamics_validation_metric == "open_rollout_frame_mse"


def test_run_rejects_removed_global_chunk_action_mode() -> None:
    """The parser should reject the removed global-chunk action mode."""

    with pytest.raises(SystemExit):
        parse_args(["--dynamics-action-conditioning-mode", "global_chunk"])


def test_run_build_config_preserves_custom_dynamics_layout_flags() -> None:
    """The config builder should keep custom layout, sampling, and validation controls."""

    args = parse_args(
        [
            "--dynamics-context-frames",
            "1",
            "--dynamics-target-frames",
            "3",
            "--dynamics-conditioning-frame-choices",
            "1,2",
            "--dynamics-conditioning-frame-probabilities",
            "0.25,0.75",
            "--dynamics-validation-conditioning-frame-choices",
            "1",
            "--dynamics-open-rollout-context-frames",
            "1",
            "--dynamics-open-rollout-stride-frames",
            "1",
            "--dynamics-patch-spatial",
            "2",
        ]
    )
    config = build_config(args)
    assert config.dynamics_context_frames == 1
    assert config.dynamics_target_frames == 3
    assert config.dynamics_conditioning_frame_choices == (1, 2)
    assert config.dynamics_conditioning_frame_probabilities == (0.25, 0.75)
    assert config.dynamics_validation_conditioning_frame_choices == (1,)
    assert config.dynamics_open_rollout_context_frames == 1
    assert config.dynamics_open_rollout_stride_frames == 1
    assert config.dynamics_patch_spatial == 2


def test_run_build_config_preserves_rollout_consistency_validation_metric() -> None:
    """The config builder should accept the motion-aware open-rollout metric."""

    args = parse_args(["--dynamics-validation-metric", "open_rollout_consistency_score"])
    config = build_config(args)
    assert config.dynamics_validation_metric == "open_rollout_consistency_score"


def test_run_build_config_preserves_explicit_open_rollout_validation_toggle() -> None:
    """The config builder should keep explicit open-rollout validation toggles."""

    enabled_args = parse_args(["--dynamics-run-open-rollout-validation"])
    enabled_config = build_config(enabled_args)
    assert enabled_config.dynamics_run_open_rollout_validation is True

    disabled_args = parse_args(["--no-dynamics-run-open-rollout-validation"])
    disabled_config = build_config(disabled_args)
    assert disabled_config.dynamics_run_open_rollout_validation is False


def test_run_build_config_preserves_kl_flag() -> None:
    """The config builder should keep the requested KL setting."""

    args = parse_args(["--kl-beta", "0.002"])
    config = build_config(args)
    assert config.ae_backend == "wan"
    assert config.kl_beta == 0.002


def test_run_build_config_preserves_reconstruction_loss_flags() -> None:
    """The config builder should keep the requested reconstruction-loss weights."""

    args = parse_args(
        [
            "--recon-mse-weight",
            "0.25",
            "--recon-l1-weight",
            "1.0",
            "--recon-edge-weight",
            "0.5",
            "--recon-motion-weight",
            "0.75",
            "--recon-motion-edge-weight",
            "0.25",
            "--recon-motion-threshold",
            "0.03",
            "--recon-motion-dilation-kernel-size",
            "7",
        ]
    )
    config = build_config(args)
    assert config.recon_mse_weight == 0.25
    assert config.recon_l1_weight == 1.0
    assert config.recon_edge_weight == 0.5
    assert config.recon_motion_weight == 0.75
    assert config.recon_motion_edge_weight == 0.25
    assert config.recon_motion_threshold == 0.03
    assert config.recon_motion_dilation_kernel_size == 7


def test_run_build_config_preserves_explicit_frame_bounds() -> None:
    """The config builder should keep explicit frame-window overrides."""

    args = parse_args(["--frame-start", "111", "--frame-end", "116"])
    config = build_config(args)
    assert config.frame_start == 111
    assert config.frame_end == 116


def test_run_build_config_preserves_rectangular_resize_bounds() -> None:
    """The config builder should keep explicit rectangular resize overrides."""

    args = parse_args(["--height", "240", "--width", "320"])
    config = build_config(args)
    assert config.height == 240
    assert config.width == 320


def test_run_build_config_preserves_auto_batch_flag() -> None:
    """The config builder should keep the requested auto-batch setting."""

    args = parse_args(["--auto-batch-size"])
    config = build_config(args)
    assert config.auto_batch_size is True


def test_run_build_config_preserves_optimizer_beta1() -> None:
    """The config builder should keep the requested AdamW beta1 override."""

    args = parse_args(["--optimizer-beta1", "0.9"])
    config = build_config(args)
    assert config.optimizer_beta1 == 0.9


def test_run_build_config_preserves_learning_rate_warmup_steps() -> None:
    """The config builder should keep the requested LR warmup duration."""

    args = parse_args(["--lr-warmup-steps", "125"])
    config = build_config(args)
    assert config.lr_warmup_steps == 125


def test_run_build_config_preserves_all_episode_training_flags() -> None:
    """The config builder should keep the all-episode AE training settings."""

    args = parse_args(
        [
            "--split",
            "train",
            "--train-all-episodes",
            "--validation-split",
            "val",
            "--validation-episode",
            "3",
        ]
    )
    config = build_config(args)
    assert config.split == "train"
    assert config.train_all_episodes is True
    assert config.validation_split == "val"
    assert config.validation_episode == 3


def test_run_build_config_preserves_multiple_validation_episodes() -> None:
    """The config builder should keep explicit multi-episode validation selection."""

    args = parse_args(["--validation-episodes", "0,2,4"])
    config = build_config(args)

    assert config.validation_episode == 0
    assert config.validation_episodes == (0, 2, 4)


def test_run_build_config_preserves_metaworld_flags() -> None:
    """The config builder should keep the requested MetaWorld dataset settings."""

    args = parse_args(
        [
            "--dataset-format",
            "lerobot_metaworld",
            "--metaworld-task-index",
            "24",
            "--metaworld-cache-dir",
            "data/metaworld_cache",
        ]
    )
    config = build_config(args)
    assert config.dataset_format == "lerobot_metaworld"
    assert config.metaworld_task_index == 24
    assert config.metaworld_cache_dir == "data/metaworld_cache"


def test_run_build_config_preserves_aloha_flags() -> None:
    """The config builder should keep the requested ALOHA dataset settings."""

    args = parse_args(
        [
            "--dataset-format",
            "lerobot_aloha_sim_transfer_cube_scripted",
            "--aloha-cache-dir",
            "data/aloha_cache",
        ]
    )
    config = build_config(args)
    assert config.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted"
    assert config.aloha_cache_dir == "data/aloha_cache"


def test_run_build_config_preserves_maniskill_flags() -> None:
    """The config builder should keep the requested ManiSkill dataset settings."""

    args = parse_args(
        [
            "--dataset-format",
            "maniskill_replay",
            "--data-root",
            "data/maniskill_raw/PickCube-v1/motionplanning",
            "--maniskill-traj-h5",
            "trajectory.rgb.pd_joint_pos.physx_cpu.h5",
            "--maniskill-camera",
            "base_camera",
        ]
    )
    config = build_config(args)
    assert config.dataset_format == "maniskill_replay"
    assert config.data_root == "data/maniskill_raw/PickCube-v1/motionplanning"
    assert config.maniskill_traj_h5 == "trajectory.rgb.pd_joint_pos.physx_cpu.h5"
    assert config.maniskill_camera == "base_camera"


def test_run_build_config_preserves_lerobot_so101_base_sim_pickplace_format() -> None:
    """The config builder should keep the requested SO-101 sim dataset format."""

    args = parse_args(
        [
            "--mode",
            "dynamics_only",
            "--dataset-format",
            "lerobot_so101_base_sim_pickplace",
            "--data-root",
            "data/so101_base_sim_pickplace_cache",
        ]
    )
    config = build_config(args)
    assert config.dataset_format == "lerobot_so101_base_sim_pickplace"
    assert config.data_root == "data/so101_base_sim_pickplace_cache"
    assert config.resolved_dynamics_action_representation() == "relative_delta"
    assert config.resolved_dynamics_action_scale() == 20.0


def test_run_build_config_allows_explicit_absolute_so101_actions() -> None:
    """The config builder should preserve explicit action-representation overrides."""

    args = parse_args(
        [
            "--dataset-format",
            "lerobot_so101_base_sim_pickplace",
            "--dynamics-action-representation",
            "absolute",
            "--dynamics-action-scale",
            "7.5",
        ]
    )

    config = build_config(args)

    assert config.dynamics_action_representation == "absolute"
    assert config.dynamics_action_scale == 7.5
    assert config.resolved_dynamics_action_representation() == "absolute"
    assert config.resolved_dynamics_action_scale() == 1.0


def test_run_rejects_removed_joint_mode() -> None:
    """The parser should reject the removed joint training mode."""

    with pytest.raises(SystemExit):
        parse_args(["--mode", "joint"])


def test_run_rejects_removed_backend_flag() -> None:
    """The parser should reject the removed backend selector."""

    with pytest.raises(SystemExit):
        parse_args(["--ae-backend", "conv"])
