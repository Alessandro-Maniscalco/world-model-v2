"""Tests for the minimal CLI configuration helpers."""

from __future__ import annotations

from world_model_v2.minimal.run import build_config, parse_args


def test_minimal_run_parse_args_uses_expected_defaults() -> None:
    """The CLI parser should expose the planned default debug settings."""

    args = parse_args([])
    config = build_config(args)
    assert config.mode == "joint"
    assert config.split == "val"
    assert config.camera == "camera_1_color"
    assert config.frame_start == 111
    assert config.frame_end == 116
    assert config.resolution == 128
    assert config.latent_channels == 4
    assert config.hidden_channels == 64
    assert config.batch_size == 32
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


def test_minimal_run_build_config_preserves_early_stop_flags() -> None:
    """The config builder should keep the requested early-stop settings."""

    args = parse_args(
        [
            "--early-stop-window-size",
            "3",
            "--early-stop-patience-windows",
            "2",
            "--early-stop-min-delta",
            "0.01",
            "--early-stop-warmup-steps",
            "7",
        ]
    )
    config = build_config(args)
    assert config.early_stop_window_size == 3
    assert config.early_stop_patience_windows == 2
    assert config.early_stop_min_delta == 0.01
    assert config.early_stop_warmup_steps == 7
