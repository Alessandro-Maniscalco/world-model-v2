"""Tests for the compact metrics plotting helper."""

from __future__ import annotations

from pathlib import Path

from scripts.check.plot_last_run_metrics import (
    build_plot,
    collect_training_series,
    collect_validation_series,
    filter_records_min_step,
    select_training_metric_names,
    select_validation_metric_names,
)


def test_collect_series_keeps_numeric_metrics_and_skips_validation_metadata() -> None:
    """Series collection should keep numeric metrics while skipping validation metadata."""

    records = [
        {
            "step": 1,
            "loss": 1.5,
            "ae_loss": 1.5,
            "recon_mse": 1.2,
            "kl_loss": 0.3,
        },
        {
            "step": 2,
            "loss": 0.5,
            "ae_loss": 0.5,
            "recon_mse": 0.4,
            "kl_loss": 0.1,
        },
        {
            "step": 2,
            "validation": {
                "ae_loss": 0.6,
                "next_frame_mse": 0.3,
                "recon_mse": 0.55,
                "kl_loss": 0.05,
                "episode": 0,
                "decoded_frame_count": 20,
                "elapsed_run_seconds": 1.5,
            },
        },
    ]

    assert collect_training_series(records) == {
        "loss": [(1, 1.5), (2, 0.5)],
        "ae_loss": [(1, 1.5), (2, 0.5)],
        "recon_mse": [(1, 1.2), (2, 0.4)],
        "kl_loss": [(1, 0.3), (2, 0.1)],
    }
    assert collect_validation_series(records) == {
        "ae_loss": [(2, 0.6)],
        "next_frame_mse": [(2, 0.3)],
        "recon_mse": [(2, 0.55)],
        "kl_loss": [(2, 0.05)],
    }
    assert select_training_metric_names(collect_training_series(records)) == ("loss",)
    assert select_validation_metric_names(records, collect_validation_series(records)) == ("ae_loss",)


def test_build_plot_summary_reports_only_selected_metrics(tmp_path: Path) -> None:
    """Plot summaries should only expose the two requested loss curves."""

    run_dir = tmp_path / "run"
    output_path = run_dir / "metrics_validation_plot.png"
    records = [
        {"run_start": {"config": {"mode": "ae_only"}}},
        {
            "step": 1,
            "loss": 1.0,
            "ae_loss": 1.0,
            "recon_mse": 0.9,
            "kl_loss": 0.1,
        },
        {
            "step": 2,
            "loss": 0.25,
            "ae_loss": 0.25,
            "recon_mse": 0.2,
            "kl_loss": 0.05,
        },
        {
            "step": 2,
            "validation": {
                "ae_loss": 0.4,
                "recon_mse": 0.35,
                "kl_loss": 0.05,
            },
        },
    ]

    result = build_plot(run_dir, records, output_path)

    assert output_path.exists()
    assert result["latest"] == {"loss": 0.25, "validation.ae_loss": 0.4}


def test_filter_records_min_step_keeps_metadata_and_drops_earlier_steps() -> None:
    """Minimum-step filtering should preserve metadata while trimming old metrics."""

    records = [
        {"run_start": {"config": {"mode": "ae_only"}}},
        {"step": 1, "loss": 1.0},
        {"step": 2, "loss": 0.5},
        {"step": 2, "validation": {"ae_loss": 0.4}},
    ]

    assert filter_records_min_step(records, 2) == [
        {"run_start": {"config": {"mode": "ae_only"}}},
        {"step": 2, "loss": 0.5},
        {"step": 2, "validation": {"ae_loss": 0.4}},
    ]


def test_build_plot_supports_dynamics_validation_metrics(tmp_path: Path) -> None:
    """Dynamics validation runs should plot their preferred metrics when AE loss is absent."""

    run_dir = tmp_path / "run"
    output_path = run_dir / "metrics_validation_plot.png"
    records = [
        {
            "run_start": {
                "config": {
                    "mode": "dynamics_only",
                    "dynamics_validation_metric": "open_rollout_consistency_score",
                }
            }
        },
        {
            "step": 355,
            "validation": {
                "open_rollout_consistency_score": 0.01,
                "next_frame_mse": 0.02,
                "open_rollout_frame_mse": 0.03,
                "next_latent_mse": 0.04,
                "target_motion_ratio": 0.8,
                "decoded_frame_count": 20,
                "elapsed_run_seconds": 10.0,
            },
        },
    ]

    result = build_plot(run_dir, records, output_path)

    assert output_path.exists()
    assert result["latest"] == {
        "validation.open_rollout_consistency_score": 0.01,
        "validation.next_frame_mse": 0.02,
        "validation.open_rollout_frame_mse": 0.03,
        "validation.next_latent_mse": 0.04,
    }
