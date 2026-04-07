"""Tests for the minimal CLI configuration helpers."""

from __future__ import annotations

import pytest

from world_model_v2.minimal.run import build_config, parse_args


def test_minimal_run_parse_args_uses_expected_defaults() -> None:
    """The CLI parser should default to full-episode training bounds."""

    args = parse_args([])
    config = build_config(args)
    assert config.mode == "ae_only"
    assert config.dataset_format == "interactive_world_sim"
    assert config.split == "val"
    assert config.train_all_episodes is False
    assert config.validation_split == ""
    assert config.validation_episode == 0
    assert config.camera == "camera_1_color"
    assert config.frame_start is None
    assert config.frame_end is None
    assert config.resolution == 128
    assert config.height is None
    assert config.width is None
    assert config.latent_channels == 16
    assert config.hidden_channels == 64
    assert config.ae_backend == "wan"
    assert config.kl_beta == 1e-4
    assert config.recon_mse_weight == 1.0
    assert config.recon_l1_weight == 0.0
    assert config.recon_edge_weight == 0.0
    assert config.batch_size == 32
    assert config.auto_batch_size is False
    assert config.lr == 1e-4
    assert config.early_stop_window_size == 1
    assert config.early_stop_patience_windows == 5
    assert config.early_stop_min_delta == 1e-10
    assert config.early_stop_warmup_steps == 300


def test_minimal_run_build_config_preserves_load_flags() -> None:
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


def test_minimal_run_build_config_preserves_kl_flag() -> None:
    """The config builder should keep the requested KL setting."""

    args = parse_args(["--kl-beta", "0.002"])
    config = build_config(args)
    assert config.ae_backend == "wan"
    assert config.kl_beta == 0.002


def test_minimal_run_build_config_preserves_reconstruction_loss_flags() -> None:
    """The config builder should keep the requested reconstruction-loss weights."""

    args = parse_args(
        [
            "--recon-mse-weight",
            "0.25",
            "--recon-l1-weight",
            "1.0",
            "--recon-edge-weight",
            "0.5",
        ]
    )
    config = build_config(args)
    assert config.recon_mse_weight == 0.25
    assert config.recon_l1_weight == 1.0
    assert config.recon_edge_weight == 0.5


def test_minimal_run_build_config_preserves_explicit_frame_bounds() -> None:
    """The config builder should keep explicit frame-window overrides."""

    args = parse_args(["--frame-start", "111", "--frame-end", "116"])
    config = build_config(args)
    assert config.frame_start == 111
    assert config.frame_end == 116


def test_minimal_run_build_config_preserves_rectangular_resize_bounds() -> None:
    """The config builder should keep explicit rectangular resize overrides."""

    args = parse_args(["--height", "240", "--width", "320"])
    config = build_config(args)
    assert config.height == 240
    assert config.width == 320


def test_minimal_run_build_config_preserves_auto_batch_flag() -> None:
    """The config builder should keep the requested auto-batch setting."""

    args = parse_args(["--auto-batch-size"])
    config = build_config(args)
    assert config.auto_batch_size is True


def test_minimal_run_build_config_preserves_all_episode_training_flags() -> None:
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


def test_minimal_run_build_config_preserves_metaworld_flags() -> None:
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


def test_minimal_run_rejects_removed_joint_mode() -> None:
    """The parser should reject the removed joint training mode."""

    with pytest.raises(SystemExit):
        parse_args(["--mode", "joint"])


def test_minimal_run_rejects_removed_backend_flag() -> None:
    """The parser should reject the removed backend selector."""

    with pytest.raises(SystemExit):
        parse_args(["--ae-backend", "conv"])
