"""Tests for nested run configuration serialization."""

from __future__ import annotations

from world_model_v2.config import AlgorithmConfig, DatasetConfig, ExperimentConfig, RunConfig


def test_run_config_round_trip_preserves_nested_sections() -> None:
    """Nested configs should serialize and deserialize without losing structure."""

    cfg = RunConfig(
        dataset=DatasetConfig(obs_keys=("camera_1_color",), horizon=1, val_horizon=6),
        algorithm=AlgorithmConfig(
            training_stage=2,
            latent_dim=64,
            dyn_infer_steps=2,
            load_ae="stage1.pt",
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=96,
            dynamics_attention_heads=2,
            mask_prev_action=True,
            sampling_strategy="terminal_only",
            prev_frame_noise_scale=0.2,
            last_frame_loss_only=True,
            loss_weighting="uniform",
            stage3_latent_noise_std=0.05,
        ),
        experiment=ExperimentConfig(
            run_name="demo",
            device="cpu",
            grad_clip_norm=1.0,
            lr_scheduler="linear",
            warmup_steps=1000,
            save_preview_initial_minutes=10.0,
            save_preview_late_minutes=30.0,
            save_preview_switch_minutes=60.0,
            early_stop_metric="validation_dyn_loss",
            early_stop_window_size=100,
            early_stop_patience_windows=3,
            early_stop_min_delta=0.01,
            early_stop_warmup_steps=500,
        ),
    )
    restored = RunConfig.from_dict(cfg.to_dict())
    assert restored.dataset.obs_keys == ("camera_1_color",)
    assert restored.algorithm.latent_dim == 64
    assert restored.algorithm.training_stage == 2
    assert restored.algorithm.dyn_infer_steps == 2
    assert restored.algorithm.load_ae == "stage1.pt"
    assert restored.algorithm.action_dim == 4
    assert restored.algorithm.dynamics_hidden_channels == 32
    assert restored.algorithm.action_emb_dim == 96
    assert restored.algorithm.dynamics_attention_heads == 2
    assert restored.algorithm.mask_prev_action is True
    assert restored.algorithm.sampling_strategy == "terminal_only"
    assert restored.algorithm.prev_frame_noise_scale == 0.2
    assert restored.algorithm.last_frame_loss_only is True
    assert restored.algorithm.loss_weighting == "uniform"
    assert restored.algorithm.stage3_latent_noise_std == 0.05
    assert restored.experiment.run_name == "demo"
    assert restored.experiment.grad_clip_norm == 1.0
    assert restored.experiment.lr_scheduler == "linear"
    assert restored.experiment.warmup_steps == 1000
    assert restored.experiment.save_preview_initial_minutes == 10.0
    assert restored.experiment.save_preview_late_minutes == 30.0
    assert restored.experiment.save_preview_switch_minutes == 60.0
    assert restored.experiment.early_stop_metric == "validation_dyn_loss"
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


def test_upstream_stage2_config_matches_published_overlapping_sizes() -> None:
    """The upstream Stage-2 helper should mirror the local overlap with the upstream recipe."""

    cfg = AlgorithmConfig.upstream_stage2(num_views=2)
    assert cfg.training_stage == 2
    assert cfg.latent_channels == 8
    assert cfg.latent_dim == 512
    assert cfg.hidden_channels == 64
    assert cfg.timesteps == 1000
    assert cfg.infer_steps == 3
    assert cfg.dyn_infer_steps == 1
    assert cfg.action_emb_dim == 512
    assert cfg.sampling_strategy == "terminal_only"
    assert cfg.loss_weighting == "uniform"


def test_upstream_stage2_experiment_config_matches_published_runtime_defaults() -> None:
    """The upstream Stage-2 runtime helper should mirror the local published defaults."""

    cfg = ExperimentConfig.upstream_stage2()
    assert cfg.output_dir == "outputs/stage2"
    assert cfg.batch_size == 4
    assert cfg.lr == 8e-5
    assert cfg.grad_clip_norm == 1.0
    assert cfg.lr_scheduler == "linear"
    assert cfg.warmup_steps == 10000
    assert cfg.max_steps == 200005
    assert cfg.validation_interval == 30000
    assert cfg.checkpoint_interval == 10000
    assert cfg.early_stop_metric == "validation_dyn_loss"
    assert cfg.early_stop_window_size == 1
    assert cfg.early_stop_patience_windows == 3
    assert cfg.early_stop_min_delta == 5e-4
    assert cfg.early_stop_warmup_steps == 60000
    assert cfg.log_interval == 10
    assert cfg.num_workers == 4
