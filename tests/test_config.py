"""Tests for nested run configuration serialization."""

from __future__ import annotations

from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig


def test_run_config_round_trip_preserves_nested_sections() -> None:
    """Nested configs should serialize and deserialize without losing structure."""

    cfg = RunConfig(
        dataset=DatasetConfig(obs_keys=("camera_1_color",), horizon=1, val_horizon=6),
        algorithm=AlgorithmConfig(training_stage=1, latent_dim=64),
        experiment=ExperimentConfig(
            run_name="demo",
            device="cpu",
            save_preview_initial_minutes=10.0,
            save_preview_late_minutes=30.0,
            save_preview_switch_minutes=60.0,
            early_stop_window_size=100,
            early_stop_patience_windows=3,
            early_stop_min_delta=0.01,
            early_stop_warmup_steps=500,
        ),
    )
    restored = RunConfig.from_dict(cfg.to_dict())
    assert restored.dataset.obs_keys == ("camera_1_color",)
    assert restored.algorithm.latent_dim == 64
    assert restored.experiment.run_name == "demo"
    assert restored.experiment.save_preview_initial_minutes == 10.0
    assert restored.experiment.save_preview_late_minutes == 30.0
    assert restored.experiment.save_preview_switch_minutes == 60.0
    assert restored.experiment.early_stop_window_size == 100
    assert restored.experiment.early_stop_patience_windows == 3
    assert restored.experiment.early_stop_min_delta == 0.01
    assert restored.experiment.early_stop_warmup_steps == 500


def test_upstream_stage1_config_matches_published_overlapping_sizes() -> None:
    """The upstream-sized helper should mirror the overlapping Stage-1 config values."""

    cfg = AlgorithmConfig.upstream_stage1(num_views=2)
    assert cfg.training_stage == 1
    assert cfg.latent_channels == 8
    assert cfg.latent_dim == 512
    assert cfg.hidden_channels == 64
    assert cfg.timesteps == 1000
    assert cfg.infer_steps == 3
