"""Tests for the minimal multi-mode experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from world_model_v2.minimal.experiment import (
    MinimalExperiment,
    MinimalExperimentConfig,
    checkpoint_ae_backend,
    load_minimal_checkpoint,
    reconstruction_loss_terms,
    save_minimal_checkpoint,
)
from world_model_v2.minimal.model import MinimalWorldModel
from world_model_v2.utils.checkpointing import append_jsonl


DEBUG_FRAME_KWARGS = {"frame_start": 111, "frame_end": 116}


def _all_trainable(module: torch.nn.Module) -> bool:
    """Return whether every parameter in a module is trainable."""

    return all(parameter.requires_grad for parameter in module.parameters())


def _all_frozen(module: torch.nn.Module) -> bool:
    """Return whether every parameter in a module is frozen."""

    return all(not parameter.requires_grad for parameter in module.parameters())


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
            **DEBUG_FRAME_KWARGS,
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    assert _all_trainable(experiment.model.encoder)
    assert _all_trainable(experiment.model.decoder)
    assert _all_frozen(experiment.model.dynamics)


def test_minimal_experiment_dynamics_only_freezes_autoencoder(
    fake_long_dataset_root: Path,
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should freeze the autoencoder while training dynamics."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="dynamics_only_mode",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint),
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    assert _all_frozen(experiment.model.encoder)
    assert _all_frozen(experiment.model.decoder)
    assert _all_trainable(experiment.model.dynamics)


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
                **DEBUG_FRAME_KWARGS,
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
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="invalid_early_stop",
                **DEBUG_FRAME_KWARGS,
                validation_interval=0,
                early_stop_window_size=1,
                early_stop_patience_windows=1,
                device="cpu",
            )
        )


def test_minimal_experiment_ae_only_can_train_all_episodes_with_separate_validation_clip(
    fake_multi_episode_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should flatten full train episodes while keeping one validation clip."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="ae_only",
            data_root=str(fake_multi_episode_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=0,
            output_dir=str(tmp_path / "outputs"),
            run_name="all_episodes_ae",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 245
    validation_batch = next(iter(experiment.val_loader))
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["frames"].shape[1] == 130


def test_minimal_experiment_supports_metaworld_ae_training(
    fake_metaworld_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should build the MT50 lazy frame dataset and validation clip loader."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="ae_only",
            dataset_format="lerobot_metaworld",
            data_root=str(fake_metaworld_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=0,
            metaworld_task_index=0,
            batch_size=2,
            resolution=8,
            output_dir=str(tmp_path / "outputs"),
            run_name="metaworld_ae",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 7
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["frame"].shape == (2, 3, 8, 8)
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["frames"].shape[1:] == (4, 3, 8, 8)


def test_minimal_experiment_auto_batch_size_uses_full_clip_on_cpu(
    fake_long_dataset_root: Path,
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Auto batch sizing should resolve to the full episode size on CPU."""

    ae_experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="auto_batch_ae",
            auto_batch_size=True,
            device="cpu",
        )
    )
    dyn_experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="auto_batch_dyn",
            auto_batch_size=True,
            load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint),
            device="cpu",
        )
    )
    assert ae_experiment.cfg.batch_size == 130
    assert ae_experiment.train_loader.batch_size == 130
    assert dyn_experiment.cfg.batch_size == 129
    assert dyn_experiment.train_loader.batch_size == 129


def test_minimal_checkpoint_round_trip(tmp_path: Path, fake_long_dataset_root: Path) -> None:
    """Minimal checkpoints should save and load the expected metadata."""

    checkpoint_path = tmp_path / "round_trip.pt"
    model = MinimalWorldModel(ae_backend="wan")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        ae_backend="wan",
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
    assert checkpoint["mode"] == "ae_only"
    assert checkpoint["clip_metadata"]["frame_start"] is None
    assert checkpoint_ae_backend(checkpoint) == "wan"


def test_minimal_checkpoint_backend_falls_back_to_autoencoder_metadata() -> None:
    """Checkpoint backend detection should read the serialized autoencoder metadata."""

    assert checkpoint_ae_backend({"autoencoder": {"backend": "wan"}}) == "wan"


def test_reconstruction_loss_terms_capture_edge_changes() -> None:
    """Edge loss should increase when a sharp structure shifts between frames."""

    target = torch.zeros(1, 3, 8, 8)
    target[:, :, 2:6, 2:4] = 1.0
    predicted = torch.zeros(1, 3, 8, 8)
    predicted[:, :, 2:6, 3:5] = 1.0
    terms = reconstruction_loss_terms(
        predicted,
        target,
        mse_weight=1.0,
        l1_weight=1.0,
        edge_weight=1.0,
    )
    assert float(terms["recon_mse"].item()) > 0.0
    assert float(terms["recon_l1"].item()) > 0.0
    assert float(terms["edge_l1"].item()) > 0.0
    assert float(terms["recon_loss"].item()) > 0.0


def test_minimal_experiment_can_partial_load_encoder_decoder(
    fake_long_dataset_root: Path,
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading encoder/decoder weights should copy those submodules from a minimal checkpoint."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_encoder_decoder",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint),
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
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading dynamics weights should copy only the dynamics submodule."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_dynamics",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint),
            load_dynamics=str(saved_minimal_wan_ae_checkpoint),
            device="cpu",
        )
    )
    dynamics_weight = next(experiment.model.dynamics.parameters())
    encoder_weight = next(experiment.model.encoder.parameters())
    assert torch.allclose(dynamics_weight, torch.full_like(dynamics_weight, 0.75))
    assert torch.allclose(encoder_weight, torch.full_like(encoder_weight, 0.25))


def test_minimal_experiment_rejects_removed_backend_config(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The minimal experiment should reject the removed conv backend configuration."""

    with pytest.raises(ValueError, match="only supports the Wan VAE"):
        MinimalExperiment(
            MinimalExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="removed_backend",
                **DEBUG_FRAME_KWARGS,
                ae_backend="conv",
                device="cpu",
            )
        )


def test_minimal_experiment_can_stop_early_on_ae_validation_plateau(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should stop early once validation AE loss stops improving."""

    config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="ae_plateau",
        **DEBUG_FRAME_KWARGS,
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
    assert not experiment._handle_validation_early_stop(1, {"ae_loss": 1.0})
    assert experiment._handle_validation_early_stop(2, {"ae_loss": 1.0})
    metrics_path = tmp_path / "outputs" / "ae_plateau" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    early_stop_records = [record for record in records if "early_stop" in record]
    assert early_stop_records
    assert stopped_records
    assert stopped_records[-1]["stopped"]["reason"] == "plateau"


def test_minimal_experiment_can_stop_early_on_dynamics_validation_plateau(
    fake_long_dataset_root: Path,
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should stop early once rollout validation stops improving."""

    config = MinimalExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="dynamics_plateau",
        **DEBUG_FRAME_KWARGS,
        load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint),
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
    assert not experiment._handle_validation_early_stop(1, {"rollout_mse": 0.5})
    assert experiment._handle_validation_early_stop(2, {"rollout_mse": 0.5})
    metrics_path = tmp_path / "outputs" / "dynamics_plateau" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    assert stopped_records[-1]["step"] == 2


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
        **DEBUG_FRAME_KWARGS,
        lr=0.0,
        max_steps=0,
        validation_interval=1,
        checkpoint_interval=1,
        early_stop_window_size=1,
        early_stop_patience_windows=1,
        early_stop_min_delta=1e-6,
        early_stop_warmup_steps=0,
        device="cpu",
    )
    source_experiment = MinimalExperiment(source_config)
    source_experiment._save_checkpoint(
        tmp_path / "outputs" / "resume_source" / "checkpoints" / "last.pt",
        step=1,
    )
    append_jsonl(
        tmp_path / "outputs" / "resume_source" / "metrics.jsonl",
        {"step": 1, "validation": {"ae_loss": 1.0}},
    )
    resume_path = tmp_path / "outputs" / "resume_source" / "checkpoints" / "last.pt"

    resumed_config = MinimalExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_target",
        **DEBUG_FRAME_KWARGS,
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
    resumed_experiment._restore_early_stop_state(resumed_experiment.current_step)
    assert resumed_experiment.current_step == 1
    assert resumed_experiment.best_window_loss == pytest.approx(1.0)
    assert resumed_experiment._handle_validation_early_stop(2, {"ae_loss": 1.0})
    metrics_path = tmp_path / "outputs" / "resume_target" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    assert stopped_records[-1]["step"] == 2
