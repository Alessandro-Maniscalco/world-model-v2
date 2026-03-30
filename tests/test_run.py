"""Tests for the top-level run entrypoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from world_model_v2.run import build_run_config, main, parse_args


def test_run_main_executes_tiny_cpu_training(fake_dataset_root: Path, tmp_path: Path) -> None:
    """The top-level runner should complete a tiny Stage-1 training run."""

    output_dir = tmp_path / "run_outputs"
    main(
        [
            "--data-root",
            str(fake_dataset_root),
            "--resolution",
            "32",
            "--val-horizon",
            "6",
            "--latent-dim",
            "64",
            "--hidden-channels",
            "32",
            "--timesteps",
            "8",
            "--infer-steps",
            "2",
            "--run-name",
            "tiny_main",
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--max-steps",
            "1",
            "--validation-interval",
            "1",
            "--checkpoint-interval",
            "1",
            "--log-interval",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert (output_dir / "tiny_main" / "checkpoints" / "last.pt").exists()


def test_build_run_config_supports_upstream_sized_flag() -> None:
    """The runner should expose the upstream-sized Stage-1 preset."""

    args = parse_args(
        [
            "--obs-key",
            "camera_1_color",
            "--obs-key",
            "camera_0_color",
            "--use-upstream-stage1-size",
        ]
    )
    cfg = build_run_config(args)
    assert cfg.algorithm.latent_channels == 8
    assert cfg.algorithm.latent_dim == 512
    assert cfg.algorithm.hidden_channels == 64
    assert cfg.algorithm.timesteps == 1000
    assert cfg.algorithm.infer_steps == 3


def test_build_run_config_supports_early_stop_flags() -> None:
    """The runner should parse the plateau-stop CLI arguments into the experiment config."""

    args = parse_args(
        [
            "--early-stop-window-size",
            "200",
            "--early-stop-patience-windows",
            "4",
            "--early-stop-min-delta",
            "0.005",
            "--early-stop-warmup-steps",
            "1000",
        ]
    )
    cfg = build_run_config(args)
    assert cfg.experiment.early_stop_window_size == 200
    assert cfg.experiment.early_stop_patience_windows == 4
    assert cfg.experiment.early_stop_min_delta == 0.005
    assert cfg.experiment.early_stop_warmup_steps == 1000


def test_build_run_config_supports_stage2_flags() -> None:
    """The runner should expose the Stage-2 and Stage-3 algorithm options."""

    args = parse_args(
        [
            "--training-stage",
            "2",
            "--load-ae",
            "stage1.pt",
            "--dyn-infer-steps",
            "3",
            "--action-dim",
            "4",
            "--dynamics-hidden-channels",
            "32",
            "--action-emb-dim",
            "96",
            "--dynamics-attention-heads",
            "2",
            "--mask-prev-action",
            "--sampling-strategy",
            "terminal_only",
            "--prev-frame-noise-scale",
            "0.2",
            "--last-frame-loss-only",
            "--loss-weighting",
            "uniform",
            "--stage3-latent-noise-std",
            "0.05",
        ]
    )
    cfg = build_run_config(args)
    assert cfg.algorithm.training_stage == 2
    assert cfg.algorithm.load_ae == "stage1.pt"
    assert cfg.algorithm.dyn_infer_steps == 3
    assert cfg.algorithm.action_dim == 4
    assert cfg.algorithm.dynamics_hidden_channels == 32
    assert cfg.algorithm.action_emb_dim == 96
    assert cfg.algorithm.dynamics_attention_heads == 2
    assert cfg.algorithm.mask_prev_action is True
    assert cfg.algorithm.sampling_strategy == "terminal_only"
    assert cfg.algorithm.prev_frame_noise_scale == 0.2
    assert cfg.algorithm.last_frame_loss_only is True
    assert cfg.algorithm.loss_weighting == "uniform"
    assert cfg.algorithm.stage3_latent_noise_std == 0.05


def test_build_run_config_requires_load_ae_for_stage2_and_stage3() -> None:
    """Later training stages should fail fast when no bootstrap checkpoint is given."""

    with pytest.raises(ValueError):
        build_run_config(parse_args(["--training-stage", "2"]))
    with pytest.raises(ValueError):
        build_run_config(parse_args(["--training-stage", "3"]))
