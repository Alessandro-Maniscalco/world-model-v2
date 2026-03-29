"""Tests for the latent-dynamics experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig
from world_model_v2.experiments.latent_dynamics_experiment import LatentDynamicsExperiment
from world_model_v2.utils.checkpointing import save_checkpoint


def test_experiment_run_writes_checkpoint_and_preview(fake_dataset_root: Path, tmp_path: Path) -> None:
    """A tiny CPU run should save checkpoints and validation artifacts."""

    cfg = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
            resolution=32,
            horizon=1,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=1,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
        ),
        experiment=ExperimentConfig(
            run_name="tiny_experiment",
            output_dir=str(tmp_path / "outputs"),
            batch_size=2,
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            log_interval=1,
            device="cpu",
        ),
    )
    LatentDynamicsExperiment(cfg).run()
    run_dir = tmp_path / "outputs" / "tiny_experiment"
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0_stats.json").exists()


def test_experiment_uses_two_phase_wall_clock_preview_schedule(
    fake_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The experiment should switch from the initial save cadence to the later cadence."""

    cfg = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
            resolution=32,
            horizon=1,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=1,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
        ),
        experiment=ExperimentConfig(
            run_name="schedule_experiment",
            output_dir=str(tmp_path / "outputs"),
            batch_size=2,
            max_steps=1,
            validation_interval=0,
            checkpoint_interval=0,
            save_preview_initial_minutes=10.0,
            save_preview_late_minutes=30.0,
            save_preview_switch_minutes=60.0,
            log_interval=1,
            device="cpu",
        ),
    )
    experiment = LatentDynamicsExperiment(cfg)
    assert experiment._current_save_preview_interval_seconds(5.0 * 60.0) == 10.0 * 60.0
    assert experiment._current_save_preview_interval_seconds(65.0 * 60.0) == 30.0 * 60.0
    assert experiment._is_time_save_preview_due(10.0 * 60.0, 0.0)
    assert not experiment._is_time_save_preview_due(5.0 * 60.0, 0.0)


def test_experiment_can_stop_early_on_plateau(fake_dataset_root: Path, tmp_path: Path) -> None:
    """A plateau-configured run should stop before the max step count when it stops improving."""

    cfg = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
            resolution=32,
            horizon=1,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=1,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
        ),
        experiment=ExperimentConfig(
            run_name="early_stop_experiment",
            output_dir=str(tmp_path / "outputs"),
            batch_size=2,
            max_steps=20,
            validation_interval=0,
            checkpoint_interval=0,
            early_stop_window_size=1,
            early_stop_patience_windows=1,
            early_stop_min_delta=10.0,
            early_stop_warmup_steps=1,
            log_interval=1,
            device="cpu",
        ),
    )
    LatentDynamicsExperiment(cfg).run()
    metrics_path = tmp_path / "outputs" / "early_stop_experiment" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    assert stopped_records[-1]["stopped"]["reason"] == "plateau"
    assert stopped_records[-1]["step"] < 20


def test_experiment_restores_plateau_state_from_resume_run_metrics(
    fake_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """A clean resumed run should seed early-stop state from the original run's metrics."""

    source_run_dir = tmp_path / "source_run"
    checkpoints_dir = source_run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    metrics_path = source_run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "".join(
            [
                json.dumps({"step": 1, "loss": 0.5}) + "\n",
                json.dumps({"step": 2, "loss": 0.4}) + "\n",
                json.dumps({"step": 3, "loss": 0.45}) + "\n",
            ]
        ),
        encoding="utf-8",
    )

    cfg = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
            resolution=32,
            horizon=1,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=1,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
        ),
        experiment=ExperimentConfig(run_name="source", device="cpu"),
    )
    model = LatentDynamicsExperiment(cfg).model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = checkpoints_dir / "last.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        3,
        cfg.to_dict(),
        {"image_range": [0.0, 1.0], "action_min": [0, 0, 0, 0], "action_max": [1, 1, 1, 1]},
    )

    resume_cfg = RunConfig(
        dataset=DatasetConfig(
            data_root=str(fake_dataset_root),
            resolution=32,
            horizon=1,
            val_horizon=6,
        ),
        algorithm=AlgorithmConfig(
            training_stage=1,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
        ),
        experiment=ExperimentConfig(
            run_name="resume_target",
            output_dir=str(tmp_path / "new_outputs"),
            device="cpu",
            resume=str(checkpoint_path),
            early_stop_window_size=1,
            early_stop_patience_windows=2,
            early_stop_min_delta=0.0,
            early_stop_warmup_steps=0,
        ),
    )
    current_metrics_path = tmp_path / "new_outputs" / "resume_target" / "metrics.jsonl"
    current_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    current_metrics_path.write_text(json.dumps({"run_start": {"note": "fresh"}}) + "\n", encoding="utf-8")
    experiment = LatentDynamicsExperiment(resume_cfg)
    experiment._restore_early_stop_state(step=3)
    assert experiment.best_window_loss == 0.4
    assert experiment.non_improving_windows == 1
