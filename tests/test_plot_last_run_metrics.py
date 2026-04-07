"""Tests for the compact metrics plotting helper."""

from __future__ import annotations

from pathlib import Path

from scripts.check.plot_last_run_metrics import (
    build_plot,
    collect_training_series,
    collect_validation_series,
)


def test_collect_series_keeps_only_total_train_and_validation_loss() -> None:
    """Series collection should ignore every metric except train loss and validation AE loss."""

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
                "recon_mse": 0.55,
                "kl_loss": 0.05,
                "episode": 0,
            },
        },
    ]

    assert collect_training_series(records) == {"loss": [(1, 1.5), (2, 0.5)]}
    assert collect_validation_series(records) == {"ae_loss": [(2, 0.6)]}


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
