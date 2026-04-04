"""Tests for the minimal multi-mode experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from world_model_v2.minimal.experiment import (
    MinimalExperiment,
    MinimalExperimentConfig,
    load_minimal_checkpoint,
    save_minimal_checkpoint,
)
from world_model_v2.minimal.model import MinimalWorldModel


def _all_trainable(module: torch.nn.Module) -> bool:
    """Return whether every parameter in a module is trainable."""

    return all(parameter.requires_grad for parameter in module.parameters())


def _all_frozen(module: torch.nn.Module) -> bool:
    """Return whether every parameter in a module is frozen."""

    return all(not parameter.requires_grad for parameter in module.parameters())


def test_minimal_experiment_joint_trains_all_modules(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Joint mode should keep every submodule trainable."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="joint",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="joint_modes",
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    assert _all_trainable(experiment.model.encoder)
    assert _all_trainable(experiment.model.decoder)
    assert _all_trainable(experiment.model.dynamics)


def test_minimal_experiment_ae_only_freezes_dynamics(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should freeze dynamics while training the autoencoder."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="ae_only_mode",
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    assert _all_trainable(experiment.model.encoder)
    assert _all_trainable(experiment.model.decoder)
    assert _all_frozen(experiment.model.dynamics)


def test_minimal_experiment_dynamics_only_requires_encoder_decoder_checkpoint(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should fail fast without frozen encoder/decoder weights."""

    with pytest.raises(ValueError, match="load-encoder-decoder"):
        MinimalExperiment(
            MinimalExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="missing_checkpoint",
                device="cpu",
            )
        )


def test_minimal_experiment_rejects_validation_plateau_without_validation_interval(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Validation plateau stopping should fail fast when periodic validation is disabled."""

    with pytest.raises(ValueError, match="validation_interval > 0"):
        MinimalExperiment(
            MinimalExperimentConfig(
                mode="joint",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="invalid_early_stop",
                validation_interval=0,
                early_stop_window_size=1,
                early_stop_patience_windows=1,
                device="cpu",
            )
        )


def test_minimal_checkpoint_round_trip(tmp_path: Path, fake_long_dataset_root: Path) -> None:
    """Minimal checkpoints should save and load the expected metadata."""

    checkpoint_path = tmp_path / "round_trip.pt"
    model = MinimalWorldModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = MinimalExperimentConfig(
        mode="joint",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        device="cpu",
    )
    save_minimal_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        step=3,
        config=config.to_dict(),
        mode=config.mode,
        clip_metadata=config.clip_metadata(),
        best_metric=0.5,
    )
    checkpoint = load_minimal_checkpoint(checkpoint_path, device="cpu")
    assert checkpoint["step"] == 3
    assert checkpoint["mode"] == "joint"
    assert checkpoint["clip_metadata"]["frame_start"] == 111


def test_minimal_experiment_can_partial_load_encoder_decoder(
    fake_long_dataset_root: Path,
    saved_minimal_joint_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading encoder/decoder weights should copy those submodules from a minimal checkpoint."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_encoder_decoder",
            load_encoder_decoder=str(saved_minimal_joint_checkpoint),
            device="cpu",
        )
    )
    encoder_weight = next(experiment.model.encoder.parameters())
    decoder_weight = next(experiment.model.decoder.parameters())
    dynamics_weight = next(experiment.model.dynamics.parameters())
    assert torch.allclose(encoder_weight, torch.full_like(encoder_weight, 0.25))
    assert torch.allclose(decoder_weight, torch.full_like(decoder_weight, 0.5))
    assert not torch.allclose(dynamics_weight, torch.full_like(dynamics_weight, 0.75))


def test_minimal_experiment_can_partial_load_dynamics(
    fake_long_dataset_root: Path,
    saved_minimal_joint_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading dynamics weights should copy only the dynamics submodule."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="joint",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_dynamics",
            load_dynamics=str(saved_minimal_joint_checkpoint),
            device="cpu",
        )
    )
    dynamics_weight = next(experiment.model.dynamics.parameters())
    encoder_weight = next(experiment.model.encoder.parameters())
    assert torch.allclose(dynamics_weight, torch.full_like(dynamics_weight, 0.75))
    assert not torch.allclose(encoder_weight, torch.full_like(encoder_weight, 0.25))


@pytest.mark.parametrize(
    ("mode", "expected_stat_key"),
    [
        ("ae_only", "recon_mse"),
        ("joint", "rollout_mse"),
        ("dynamics_only", "rollout_mse"),
    ],
)
def test_minimal_experiment_run_writes_artifacts_for_each_mode(
    mode: str,
    expected_stat_key: str,
    fake_long_dataset_root: Path,
    saved_minimal_joint_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """A one-step run should write checkpoints and validation artifacts in every mode."""

    config = MinimalExperimentConfig(
        mode=mode,
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name=f"smoke_{mode}",
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        log_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_minimal_joint_checkpoint) if mode == "dynamics_only" else "",
    )
    experiment = MinimalExperiment(config)
    experiment.run()
    run_dir = tmp_path / "outputs" / f"smoke_{mode}"
    stats = load_minimal_checkpoint(run_dir / "checkpoints" / "best.pt", device="cpu")
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0_grid.png").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0.gif").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0_stats.json").exists()
    assert stats["best_metric"] is not None
    if mode == "ae_only":
        payload = (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
        assert expected_stat_key in payload
    else:
        payload = (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
        assert '"predicted_frame_count": 6' in payload
        assert expected_stat_key in payload


def test_minimal_experiment_can_stop_early_on_ae_validation_plateau(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should stop early when reconstruction validation plateaus."""

    config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="ae_plateau",
        lr=0.0,
        max_steps=5,
        validation_interval=1,
        checkpoint_interval=0,
        early_stop_window_size=1,
        early_stop_patience_windows=1,
        early_stop_min_delta=1e-6,
        early_stop_warmup_steps=0,
        device="cpu",
    )
    experiment = MinimalExperiment(config)
    experiment.run()
    assert experiment.current_step == 2
    metrics_path = tmp_path / "outputs" / "ae_plateau" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    early_stop_records = [record for record in records if "early_stop" in record]
    assert early_stop_records
    assert stopped_records
    assert stopped_records[-1]["stopped"]["reason"] == "plateau"


def test_minimal_experiment_can_stop_early_on_joint_validation_plateau(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Joint mode should stop early when rollout validation plateaus."""

    config = MinimalExperimentConfig(
        mode="joint",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="joint_plateau",
        lr=0.0,
        max_steps=5,
        validation_interval=1,
        checkpoint_interval=0,
        early_stop_window_size=1,
        early_stop_patience_windows=1,
        early_stop_min_delta=1e-6,
        early_stop_warmup_steps=0,
        device="cpu",
    )
    experiment = MinimalExperiment(config)
    experiment.run()
    assert experiment.current_step == 2
    metrics_path = tmp_path / "outputs" / "joint_plateau" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    validation_records = [record for record in records if "validation" in record]
    assert validation_records[-1]["validation"]["rollout_mse"] >= 0.0


def test_minimal_experiment_can_disable_plateau_stopping(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Setting the early-stop window size to zero should leave training running to max_steps."""

    config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="no_plateau_stop",
        lr=0.0,
        max_steps=3,
        validation_interval=1,
        checkpoint_interval=0,
        early_stop_window_size=0,
        early_stop_patience_windows=1,
        early_stop_warmup_steps=0,
        device="cpu",
    )
    experiment = MinimalExperiment(config)
    experiment.run()
    assert experiment.current_step == 3
    metrics_path = tmp_path / "outputs" / "no_plateau_stop" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert not stopped_records


def test_minimal_experiment_restores_plateau_state_from_resume_metrics(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """A resumed run should reuse prior validation history when evaluating plateau stops."""

    source_config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_source",
        lr=0.0,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        early_stop_window_size=1,
        early_stop_patience_windows=1,
        early_stop_min_delta=1e-6,
        early_stop_warmup_steps=0,
        device="cpu",
    )
    source_experiment = MinimalExperiment(source_config)
    source_experiment.run()
    resume_path = tmp_path / "outputs" / "resume_source" / "checkpoints" / "last.pt"

    resumed_config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_target",
        resume=str(resume_path),
        lr=0.0,
        max_steps=2,
        validation_interval=1,
        checkpoint_interval=0,
        early_stop_window_size=1,
        early_stop_patience_windows=1,
        early_stop_min_delta=1e-6,
        early_stop_warmup_steps=0,
        device="cpu",
    )
    resumed_experiment = MinimalExperiment(resumed_config)
    resumed_experiment.run()
    assert resumed_experiment.current_step == 2
    metrics_path = tmp_path / "outputs" / "resume_target" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    assert stopped_records[-1]["step"] == 2
