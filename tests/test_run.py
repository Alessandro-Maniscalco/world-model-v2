"""Tests for the top-level run entrypoint."""

from __future__ import annotations

from pathlib import Path

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
