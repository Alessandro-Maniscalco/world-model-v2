"""Tests for the root multi-mode experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from world_model_v2.experiment import (
    Experiment,
    ExperimentConfig,
    checkpoint_ae_backend,
    checkpoint_dynamics_backend,
    load_training_checkpoint,
    reconstruction_loss_terms,
    save_training_checkpoint,
)
from world_model_v2.dynamics_transformer import DynamicsTrainingInputs
from world_model_v2.model import WorldModel
from world_model_v2.utils.checkpointing import append_jsonl


DEBUG_FRAME_KWARGS = {"frame_start": 111, "frame_end": 116}


def _all_trainable(module: torch.nn.Module) -> bool:
    """Return whether every parameter in a module is trainable."""

    return all(parameter.requires_grad for parameter in module.parameters())


def _all_frozen(module: torch.nn.Module) -> bool:
    """Return whether every parameter in a module is frozen."""

    return all(not parameter.requires_grad for parameter in module.parameters())


def test_experiment_ae_only_freezes_dynamics(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should freeze dynamics while training the autoencoder."""

    experiment = Experiment(
        ExperimentConfig(
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


def test_experiment_dynamics_only_freezes_autoencoder(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should freeze the autoencoder while training dynamics."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="dynamics_only_mode",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            device="cpu",
        )
    )
    assert _all_frozen(experiment.model.encoder)
    assert _all_frozen(experiment.model.decoder)
    assert _all_trainable(experiment.model.dynamics)


def test_experiment_dynamics_only_requires_encoder_decoder_checkpoint(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should fail fast without frozen encoder/decoder weights."""

    with pytest.raises(ValueError, match="load-encoder-decoder"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="missing_checkpoint",
                **DEBUG_FRAME_KWARGS,
                device="cpu",
            )
        )


def test_experiment_dynamics_only_requires_at_least_five_frames(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should reject a four-frame clip with no valid 5-frame window."""

    with pytest.raises(ValueError, match="at least 5 frames"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="single_frame_dyn",
                frame_start=111,
                frame_end=114,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                device="cpu",
            )
        )


def test_experiment_supports_custom_dynamics_layout_controls(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """The experiment should wire custom layout, validation, and rollout settings through."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="custom_layout",
            frame_start=111,
            frame_end=112,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            conditional_frame_sigma=1e-4,
            dynamics_context_frames=1,
            dynamics_target_frames=1,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_open_rollout_stride_frames=1,
            device="cpu",
        )
    )

    assert experiment.model.dynamics.cfg.context_frames == 1
    assert experiment.model.dynamics.cfg.target_frames == 1
    assert experiment.model.dynamics.cfg.conditional_frame_sigma == pytest.approx(1e-4)
    assert experiment.model.dynamics.cfg.conditioning_frame_choices == (1,)
    assert experiment.model.dynamics.cfg.conditioning_frame_probabilities == (1.0,)
    assert experiment.model.dynamics.cfg.validation_conditioning_frame_choices == (1,)
    assert experiment.model.dynamics.cfg.open_rollout_context_frames == 1
    assert experiment.model.dynamics.cfg.open_rollout_stride_frames == 1
    assert len(experiment.train_dataset) == 1


def test_experiment_rejects_negative_self_forcing_weight(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative self-forcing weights should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_self_forcing_loss_weight"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_self_forcing",
                dynamics_self_forcing_loss_weight=-0.1,
                device="cpu",
            )
        )


def test_experiment_rejects_negative_rollout_self_forcing_weight(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative rollout self-forcing weights should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_rollout_self_forcing_loss_weight"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_rollout_self_forcing",
                dynamics_rollout_self_forcing_loss_weight=-0.1,
                device="cpu",
            )
        )


def test_experiment_rejects_negative_self_forcing_warmup_steps(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative self-forcing warmup steps should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_self_forcing_warmup_steps"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_self_forcing_warmup",
                dynamics_self_forcing_warmup_steps=-1,
                device="cpu",
            )
        )


def test_experiment_rejects_negative_self_forcing_ramp_steps(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative self-forcing ramp steps should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_self_forcing_ramp_steps"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_self_forcing_ramp",
                dynamics_self_forcing_ramp_steps=-1,
                device="cpu",
            )
        )


def test_experiment_rejects_negative_rollout_self_forcing_warmup_steps(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative rollout self-forcing warmup steps should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_rollout_self_forcing_warmup_steps"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_rollout_self_forcing_warmup",
                dynamics_rollout_self_forcing_warmup_steps=-1,
                device="cpu",
            )
        )


def test_experiment_rejects_negative_rollout_self_forcing_ramp_steps(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative rollout self-forcing ramp steps should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_rollout_self_forcing_ramp_steps"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_rollout_self_forcing_ramp",
                dynamics_rollout_self_forcing_ramp_steps=-1,
                device="cpu",
            )
        )


def test_experiment_rejects_unknown_self_forcing_mode(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Unknown self-forcing modes should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_self_forcing_mode"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="invalid_self_forcing_mode",
                dynamics_self_forcing_mode="bad_mode",
                device="cpu",
            )
        )


def test_experiment_rejects_negative_self_forcing_rollout_chunks(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Negative rollout chunk counts should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_self_forcing_rollout_chunks"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="negative_self_forcing_rollout_chunks",
                dynamics_self_forcing_rollout_chunks=-1,
                device="cpu",
            )
        )


def test_experiment_rejects_rollout_self_forcing_without_future_chunks(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Rollout self-forcing should require at least one extra future chunk."""

    with pytest.raises(ValueError, match="dynamics_self_forcing_rollout_chunks"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="missing_rollout_chunks",
                dynamics_self_forcing_loss_weight=0.5,
                dynamics_self_forcing_mode="rollout",
                dynamics_self_forcing_rollout_chunks=0,
                load_encoder_decoder="unused.pt",
                device="cpu",
            )
        )


def test_experiment_rejects_duplicate_rollout_self_forcing_configuration(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Rollout mode should not also receive the additive rollout auxiliary weight."""

    with pytest.raises(ValueError, match="dynamics_rollout_self_forcing_loss_weight"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="duplicate_rollout_self_forcing",
                dynamics_self_forcing_loss_weight=0.5,
                dynamics_rollout_self_forcing_loss_weight=0.25,
                dynamics_self_forcing_mode="rollout",
                dynamics_self_forcing_rollout_chunks=1,
                load_encoder_decoder="unused.pt",
                device="cpu",
            )
        )


def test_experiment_rejects_out_of_range_conditional_frame_sigma(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Conditioning sigma outside `[0, 1]` should fail fast during config validation."""

    with pytest.raises(ValueError, match="conditional_frame_sigma"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="invalid_conditional_sigma",
                conditional_frame_sigma=1.5,
                device="cpu",
            )
        )


def test_experiment_rejects_non_positive_open_rollout_stride(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Non-positive rollout stride should fail fast during config validation."""

    with pytest.raises(ValueError, match="dynamics_open_rollout_stride_frames"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="invalid_rollout_stride",
                dynamics_open_rollout_stride_frames=0,
                device="cpu",
            )
        )


def test_dynamics_training_step_adds_weighted_self_forcing_loss(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamics training should add the configured weighted self-forcing auxiliary loss."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="self_forcing_loss",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=0.5,
            device="cpu",
        )
    )
    batch = {
        "context_frames": torch.randn(2, 1, 3, 8, 8),
        "target_frames": torch.randn(2, 2, 3, 8, 8),
        "future_target_frames": torch.empty(2, 0, 3, 8, 8),
        "actions": torch.zeros(2, 2, 4),
        "future_actions": torch.empty(2, 0, 4),
    }
    clean_latent_video = torch.randn(2, 4, 3, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.ones_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=torch.zeros_like(clean_latent_video),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(2, 3),
        condition_mask=torch.zeros(2, 1, 3, 8, 8),
        actions=batch["actions"],
        target_sigmas=torch.ones(2),
        num_conditional_frames=torch.ones(2, dtype=torch.long),
        use_video_condition=torch.ones(2, dtype=torch.bool),
    )

    monkeypatch.setattr(
        experiment.model,
        "encode_context_frames",
        lambda images, deterministic=True: clean_latent_video[:, :, :1],
    )
    monkeypatch.setattr(
        experiment.model,
        "encode_frame_sequence",
        lambda images, deterministic=True: clean_latent_video[:, :, 1:],
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "prepare_training_inputs",
        lambda clean_latent_video, actions=None: dynamics_inputs,
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "forward",
        lambda **kwargs: predicted_velocity,
    )
    monkeypatch.setattr(
        experiment,
        "_dynamics_self_forcing_loss",
        lambda **kwargs: (
            torch.tensor(2.0),
            {"latent_rf_self_forcing_mse_ctx2": torch.tensor(2.0)},
        ),
    )

    loss_dict = experiment._dynamics_only_training_step(batch)

    assert loss_dict["latent_rf_mse"] == pytest.approx(1.0)
    assert loss_dict["latent_rf_self_forcing_mse"] == pytest.approx(2.0)
    assert loss_dict["latent_rf_self_forcing_weighted_loss"] == pytest.approx(1.0)
    assert loss_dict["latent_rf_total_loss"] == pytest.approx(2.0)
    assert loss_dict["loss"] == pytest.approx(2.0)
    assert loss_dict["latent_rf_self_forcing_mse_ctx2"] == pytest.approx(2.0)
    assert loss_dict["active_rollout_self_forcing_loss_weight"] == pytest.approx(0.0)


def test_dynamics_training_step_adds_rollout_self_forcing_auxiliary_loss(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamics training should support a rollout auxiliary on top of the primary self-forcing loss."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="hybrid_self_forcing_loss",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=0.5,
            dynamics_rollout_self_forcing_loss_weight=0.25,
            dynamics_self_forcing_rollout_chunks=1,
            device="cpu",
        )
    )
    batch = {
        "context_frames": torch.randn(2, 1, 3, 8, 8),
        "target_frames": torch.randn(2, 2, 3, 8, 8),
        "future_target_frames": torch.randn(2, 2, 3, 8, 8),
        "actions": torch.zeros(2, 2, 4),
        "future_actions": torch.zeros(2, 2, 4),
    }
    clean_latent_video = torch.randn(2, 4, 3, 8, 8)
    future_target_latent_video = torch.randn(2, 4, 2, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.ones_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=torch.zeros_like(clean_latent_video),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(2, 3),
        condition_mask=torch.zeros(2, 1, 3, 8, 8),
        actions=batch["actions"],
        target_sigmas=torch.ones(2),
        num_conditional_frames=torch.ones(2, dtype=torch.long),
        use_video_condition=torch.ones(2, dtype=torch.bool),
    )

    monkeypatch.setattr(
        experiment.model,
        "encode_context_frames",
        lambda images, deterministic=True: clean_latent_video[:, :, :1],
    )

    def fake_encode_frame_sequence(images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return the right latent slice for target versus future-target encoding."""

        del deterministic
        if images.shape[1] == 2 and torch.equal(images, batch["target_frames"]):
            return clean_latent_video[:, :, 1:]
        if images.shape[1] == 2 and torch.equal(images, batch["future_target_frames"]):
            return future_target_latent_video
        raise AssertionError("Unexpected image batch passed to encode_frame_sequence.")

    monkeypatch.setattr(experiment.model, "encode_frame_sequence", fake_encode_frame_sequence)
    monkeypatch.setattr(
        experiment.model.dynamics,
        "prepare_training_inputs",
        lambda clean_latent_video, actions=None: dynamics_inputs,
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "forward",
        lambda **kwargs: predicted_velocity,
    )
    monkeypatch.setattr(
        experiment,
        "_dynamics_self_forcing_loss",
        lambda **kwargs: (
            torch.tensor(2.0),
            {"latent_rf_self_forcing_mse_ctx2": torch.tensor(2.0)},
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_dynamics_rollout_self_forcing_loss",
        lambda **kwargs: (
            torch.tensor(4.0),
            {"latent_rf_self_forcing_rollout_mse_chunk1": torch.tensor(4.0)},
        ),
    )

    loss_dict = experiment._dynamics_only_training_step(batch)

    assert loss_dict["latent_rf_mse"] == pytest.approx(1.0)
    assert loss_dict["latent_rf_self_forcing_mse"] == pytest.approx(2.0)
    assert loss_dict["latent_rf_self_forcing_weighted_loss"] == pytest.approx(1.0)
    assert loss_dict["latent_rf_rollout_self_forcing_mse"] == pytest.approx(4.0)
    assert loss_dict["latent_rf_rollout_self_forcing_weighted_loss"] == pytest.approx(1.0)
    assert loss_dict["active_rollout_self_forcing_loss_weight"] == pytest.approx(0.25)
    assert loss_dict["latent_rf_total_loss"] == pytest.approx(3.0)
    assert loss_dict["loss"] == pytest.approx(3.0)


def test_dynamics_training_step_disables_self_forcing_during_warmup(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamics training should keep self-forcing inactive until the warmup window ends."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="self_forcing_warmup",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=0.5,
            dynamics_self_forcing_warmup_steps=10,
            device="cpu",
        )
    )
    batch = {
        "context_frames": torch.randn(2, 1, 3, 8, 8),
        "target_frames": torch.randn(2, 2, 3, 8, 8),
        "future_target_frames": torch.empty(2, 0, 3, 8, 8),
        "actions": torch.zeros(2, 2, 4),
        "future_actions": torch.empty(2, 0, 4),
    }
    clean_latent_video = torch.randn(2, 4, 3, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.ones_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=torch.zeros_like(clean_latent_video),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(2, 3),
        condition_mask=torch.zeros(2, 1, 3, 8, 8),
        actions=batch["actions"],
        target_sigmas=torch.ones(2),
        num_conditional_frames=torch.ones(2, dtype=torch.long),
        use_video_condition=torch.ones(2, dtype=torch.bool),
    )
    captured_loss_weight: list[float] = []

    monkeypatch.setattr(
        experiment.model,
        "encode_context_frames",
        lambda images, deterministic=True: clean_latent_video[:, :, :1],
    )
    monkeypatch.setattr(
        experiment.model,
        "encode_frame_sequence",
        lambda images, deterministic=True: clean_latent_video[:, :, 1:],
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "prepare_training_inputs",
        lambda clean_latent_video, actions=None: dynamics_inputs,
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "forward",
        lambda **kwargs: predicted_velocity,
    )

    def fake_self_forcing_loss(**kwargs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Record the active warmup-controlled loss weight before returning a dummy loss."""

        captured_loss_weight.append(float(kwargs["loss_weight"]))
        return torch.tensor(2.0), {"latent_rf_self_forcing_mse_ctx2": torch.tensor(2.0)}

    monkeypatch.setattr(experiment, "_dynamics_self_forcing_loss", fake_self_forcing_loss)

    loss_dict = experiment._dynamics_only_training_step(batch)

    assert captured_loss_weight == [0.0]
    assert loss_dict["active_self_forcing_loss_weight"] == pytest.approx(0.0)
    assert loss_dict["latent_rf_self_forcing_weighted_loss"] == pytest.approx(0.0)
    assert loss_dict["loss"] == pytest.approx(1.0)


def test_dynamics_training_step_ramps_self_forcing_weight_after_warmup(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamics training should linearly ramp self-forcing weight after warmup."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="self_forcing_ramp",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=0.5,
            dynamics_self_forcing_warmup_steps=10,
            dynamics_self_forcing_ramp_steps=20,
            device="cpu",
        )
    )
    batch = {
        "context_frames": torch.randn(2, 1, 3, 8, 8),
        "target_frames": torch.randn(2, 2, 3, 8, 8),
        "future_target_frames": torch.empty(2, 0, 3, 8, 8),
        "actions": torch.zeros(2, 2, 4),
        "future_actions": torch.empty(2, 0, 4),
    }
    clean_latent_video = torch.randn(2, 4, 3, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.ones_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=torch.zeros_like(clean_latent_video),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(2, 3),
        condition_mask=torch.zeros(2, 1, 3, 8, 8),
        actions=batch["actions"],
        target_sigmas=torch.ones(2),
        num_conditional_frames=torch.ones(2, dtype=torch.long),
        use_video_condition=torch.ones(2, dtype=torch.bool),
    )
    captured_loss_weight: list[float] = []

    monkeypatch.setattr(
        experiment.model,
        "encode_context_frames",
        lambda images, deterministic=True: clean_latent_video[:, :, :1],
    )
    monkeypatch.setattr(
        experiment.model,
        "encode_frame_sequence",
        lambda images, deterministic=True: clean_latent_video[:, :, 1:],
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "prepare_training_inputs",
        lambda clean_latent_video, actions=None: dynamics_inputs,
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "forward",
        lambda **kwargs: predicted_velocity,
    )

    def fake_self_forcing_loss(**kwargs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Record the active ramp-controlled loss weight before returning a dummy loss."""

        captured_loss_weight.append(float(kwargs["loss_weight"]))
        return torch.tensor(2.0), {"latent_rf_self_forcing_mse_ctx2": torch.tensor(2.0)}

    monkeypatch.setattr(experiment, "_dynamics_self_forcing_loss", fake_self_forcing_loss)
    experiment.current_step = 19

    loss_dict = experiment._dynamics_only_training_step(batch)

    assert captured_loss_weight == [pytest.approx(0.25)]
    assert loss_dict["active_self_forcing_loss_weight"] == pytest.approx(0.25)
    assert loss_dict["latent_rf_self_forcing_weighted_loss"] == pytest.approx(0.5)
    assert loss_dict["loss"] == pytest.approx(1.5)


def test_dynamics_training_step_delays_rollout_self_forcing_auxiliary_during_own_warmup(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollout auxiliary should stay inactive until its own warmup window ends."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="rollout_aux_warmup",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=0.5,
            dynamics_rollout_self_forcing_loss_weight=0.25,
            dynamics_rollout_self_forcing_warmup_steps=10,
            dynamics_self_forcing_rollout_chunks=1,
            device="cpu",
        )
    )
    batch = {
        "context_frames": torch.randn(2, 1, 3, 8, 8),
        "target_frames": torch.randn(2, 2, 3, 8, 8),
        "future_target_frames": torch.randn(2, 2, 3, 8, 8),
        "actions": torch.zeros(2, 2, 4),
        "future_actions": torch.zeros(2, 2, 4),
    }
    clean_latent_video = torch.randn(2, 4, 3, 8, 8)
    future_target_latent_video = torch.randn(2, 4, 2, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.ones_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=torch.zeros_like(clean_latent_video),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(2, 3),
        condition_mask=torch.zeros(2, 1, 3, 8, 8),
        actions=batch["actions"],
        target_sigmas=torch.ones(2),
        num_conditional_frames=torch.ones(2, dtype=torch.long),
        use_video_condition=torch.ones(2, dtype=torch.bool),
    )

    monkeypatch.setattr(
        experiment.model,
        "encode_context_frames",
        lambda images, deterministic=True: clean_latent_video[:, :, :1],
    )

    def fake_encode_frame_sequence(images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return the right latent slice for target versus future-target encoding."""

        del deterministic
        if images.shape[1] == 2 and torch.equal(images, batch["target_frames"]):
            return clean_latent_video[:, :, 1:]
        if images.shape[1] == 2 and torch.equal(images, batch["future_target_frames"]):
            return future_target_latent_video
        raise AssertionError("Unexpected image batch passed to encode_frame_sequence.")

    monkeypatch.setattr(experiment.model, "encode_frame_sequence", fake_encode_frame_sequence)
    monkeypatch.setattr(
        experiment.model.dynamics,
        "prepare_training_inputs",
        lambda clean_latent_video, actions=None: dynamics_inputs,
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "forward",
        lambda **kwargs: predicted_velocity,
    )
    monkeypatch.setattr(
        experiment,
        "_dynamics_self_forcing_loss",
        lambda **kwargs: (
            torch.tensor(2.0),
            {"latent_rf_self_forcing_mse_ctx2": torch.tensor(2.0)},
        ),
    )
    rollout_calls = {"count": 0}

    def fake_rollout_self_forcing_loss(**kwargs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Record whether the rollout helper ran while the auxiliary was warming up."""

        rollout_calls["count"] += 1
        return torch.tensor(4.0), {"latent_rf_self_forcing_rollout_mse_chunk1": torch.tensor(4.0)}

    monkeypatch.setattr(experiment, "_dynamics_rollout_self_forcing_loss", fake_rollout_self_forcing_loss)

    loss_dict = experiment._dynamics_only_training_step(batch)

    assert loss_dict["active_self_forcing_loss_weight"] == pytest.approx(0.5)
    assert loss_dict["active_rollout_self_forcing_loss_weight"] == pytest.approx(0.0)
    assert rollout_calls["count"] == 0
    assert loss_dict["latent_rf_rollout_self_forcing_weighted_loss"] == pytest.approx(0.0)
    assert loss_dict["loss"] == pytest.approx(2.0)


def test_dynamics_training_step_ramps_rollout_self_forcing_auxiliary_after_own_warmup(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollout auxiliary should use its own warmup/ramp schedule instead of sharing the primary one."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="rollout_aux_ramp",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=0.5,
            dynamics_rollout_self_forcing_loss_weight=0.25,
            dynamics_rollout_self_forcing_warmup_steps=10,
            dynamics_rollout_self_forcing_ramp_steps=20,
            dynamics_self_forcing_rollout_chunks=1,
            device="cpu",
        )
    )
    batch = {
        "context_frames": torch.randn(2, 1, 3, 8, 8),
        "target_frames": torch.randn(2, 2, 3, 8, 8),
        "future_target_frames": torch.randn(2, 2, 3, 8, 8),
        "actions": torch.zeros(2, 2, 4),
        "future_actions": torch.zeros(2, 2, 4),
    }
    clean_latent_video = torch.randn(2, 4, 3, 8, 8)
    future_target_latent_video = torch.randn(2, 4, 2, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.ones_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=torch.zeros_like(clean_latent_video),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(2, 3),
        condition_mask=torch.zeros(2, 1, 3, 8, 8),
        actions=batch["actions"],
        target_sigmas=torch.ones(2),
        num_conditional_frames=torch.ones(2, dtype=torch.long),
        use_video_condition=torch.ones(2, dtype=torch.bool),
    )

    monkeypatch.setattr(
        experiment.model,
        "encode_context_frames",
        lambda images, deterministic=True: clean_latent_video[:, :, :1],
    )

    def fake_encode_frame_sequence(images: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return the right latent slice for target versus future-target encoding."""

        del deterministic
        if images.shape[1] == 2 and torch.equal(images, batch["target_frames"]):
            return clean_latent_video[:, :, 1:]
        if images.shape[1] == 2 and torch.equal(images, batch["future_target_frames"]):
            return future_target_latent_video
        raise AssertionError("Unexpected image batch passed to encode_frame_sequence.")

    monkeypatch.setattr(experiment.model, "encode_frame_sequence", fake_encode_frame_sequence)
    monkeypatch.setattr(
        experiment.model.dynamics,
        "prepare_training_inputs",
        lambda clean_latent_video, actions=None: dynamics_inputs,
    )
    monkeypatch.setattr(
        experiment.model.dynamics,
        "forward",
        lambda **kwargs: predicted_velocity,
    )
    monkeypatch.setattr(
        experiment,
        "_dynamics_self_forcing_loss",
        lambda **kwargs: (
            torch.tensor(2.0),
            {"latent_rf_self_forcing_mse_ctx2": torch.tensor(2.0)},
        ),
    )
    monkeypatch.setattr(
        experiment,
        "_dynamics_rollout_self_forcing_loss",
        lambda **kwargs: (
            torch.tensor(4.0),
            {"latent_rf_self_forcing_rollout_mse_chunk1": torch.tensor(4.0)},
        ),
    )
    experiment.current_step = 19

    loss_dict = experiment._dynamics_only_training_step(batch)

    assert loss_dict["active_self_forcing_loss_weight"] == pytest.approx(0.5)
    assert loss_dict["active_rollout_self_forcing_loss_weight"] == pytest.approx(0.125)
    assert loss_dict["latent_rf_self_forcing_weighted_loss"] == pytest.approx(1.0)
    assert loss_dict["latent_rf_rollout_self_forcing_weighted_loss"] == pytest.approx(0.5)
    assert loss_dict["loss"] == pytest.approx(2.5)


def test_dynamics_self_forcing_loss_uses_predicted_prefix_for_later_targets(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-forcing should create a longer causal prefix and score only the later target."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="self_forcing_helper",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=1.0,
            device="cpu",
        )
    )
    clean_latent_video = torch.randn(1, 4, 3, 8, 8)
    target_velocity = torch.zeros_like(clean_latent_video)
    predicted_velocity = torch.zeros_like(clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=clean_latent_video.clone(),
        conditioning_latent_video=clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(1, 3),
        condition_mask=experiment.model.dynamics.make_condition_mask(
            clean_latent_video,
            num_conditional_frames=1,
        ),
        actions=torch.zeros(1, 2, 4),
        target_sigmas=torch.ones(1),
        num_conditional_frames=torch.ones(1, dtype=torch.long),
        use_video_condition=torch.ones(1, dtype=torch.bool),
    )
    captured_conditioning_prefixes: list[torch.Tensor] = []

    def fake_forward(**kwargs: torch.Tensor) -> torch.Tensor:
        """Return a one-unit later-target error while recording the self-forced prefix."""

        conditioning_latent_video = kwargs["conditioning_latent_video"]
        target_velocity = kwargs["target_velocity"]
        captured_conditioning_prefixes.append(conditioning_latent_video[:, :, :2].detach().clone())
        output = target_velocity.clone()
        output[:, :, 2:] = output[:, :, 2:] + 1.0
        return output

    monkeypatch.setattr(experiment.model.dynamics, "forward", fake_forward)

    self_forcing_loss, stats = experiment._dynamics_self_forcing_loss(
        clean_latent_video=clean_latent_video,
        extended_clean_latent_video=clean_latent_video,
        predicted_velocity=predicted_velocity,
        dynamics_inputs=dynamics_inputs,
        future_actions=torch.empty(1, 0, 4),
        loss_weight=1.0,
    )

    assert self_forcing_loss == pytest.approx(1.0)
    assert stats["latent_rf_self_forcing_mse_ctx2"] == pytest.approx(1.0)
    assert len(captured_conditioning_prefixes) == 1
    assert torch.allclose(captured_conditioning_prefixes[0], clean_latent_video[:, :, :2])


def test_dynamics_rollout_self_forcing_loss_uses_same_context_rollout_semantics(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollout self-forcing should score future chunks using the rollout context length, not an expanded prefix."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="rollout_self_forcing_helper",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            dynamics_context_frames=1,
            dynamics_target_frames=2,
            dynamics_conditioning_frame_choices=(1,),
            dynamics_conditioning_frame_probabilities=(1.0,),
            dynamics_validation_conditioning_frame_choices=(1,),
            dynamics_open_rollout_context_frames=1,
            dynamics_self_forcing_loss_weight=1.0,
            dynamics_self_forcing_mode="rollout",
            dynamics_self_forcing_rollout_chunks=1,
            device="cpu",
        )
    )
    primary_clean_latent_video = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 2, 2)
    future_latent_video = torch.arange(12, 20, dtype=torch.float32).view(1, 1, 2, 2, 2)
    extended_clean_latent_video = torch.cat([primary_clean_latent_video, future_latent_video], dim=2)
    target_velocity = torch.zeros_like(primary_clean_latent_video)
    predicted_velocity = torch.zeros_like(primary_clean_latent_video)
    dynamics_inputs = DynamicsTrainingInputs(
        noisy_latent_video=primary_clean_latent_video.clone(),
        conditioning_latent_video=primary_clean_latent_video,
        target_velocity=target_velocity,
        timesteps=torch.ones(1, 3),
        condition_mask=experiment.model.dynamics.make_condition_mask(
            primary_clean_latent_video,
            num_conditional_frames=1,
        ),
        actions=torch.tensor([[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]]]),
        target_sigmas=torch.ones(1),
        num_conditional_frames=torch.ones(1, dtype=torch.long),
        use_video_condition=torch.ones(1, dtype=torch.bool),
    )
    future_actions = torch.tensor([[[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]])
    captured_contexts: list[torch.Tensor] = []
    captured_actions: list[torch.Tensor] = []

    monkeypatch.setattr(
        experiment.model.dynamics.flow,
        "interpolate",
        lambda noise, clean, sigmas: (clean.clone(), torch.zeros_like(clean)),
    )

    def fake_forward(**kwargs: torch.Tensor) -> torch.Tensor:
        """Return a unit error on future targets while recording rollout contexts and actions."""

        captured_contexts.append(kwargs["conditioning_latent_video"][:, :, :1].detach().clone())
        captured_actions.append(kwargs["actions"].detach().clone())
        output = kwargs["target_velocity"].clone()
        output[:, :, 1:] = output[:, :, 1:] + 1.0
        return output

    monkeypatch.setattr(experiment.model.dynamics, "forward", fake_forward)

    self_forcing_loss, stats = experiment._dynamics_self_forcing_loss(
        clean_latent_video=primary_clean_latent_video,
        extended_clean_latent_video=extended_clean_latent_video,
        predicted_velocity=predicted_velocity,
        dynamics_inputs=dynamics_inputs,
        future_actions=future_actions,
        loss_weight=1.0,
    )

    assert self_forcing_loss == pytest.approx(1.0)
    assert stats["latent_rf_self_forcing_rollout_mse_chunk1"] == pytest.approx(1.0)
    assert len(captured_contexts) == 1
    assert torch.equal(captured_contexts[0], primary_clean_latent_video[:, :, 2:3])
    assert torch.equal(
        captured_actions[0],
        torch.tensor([[[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]]),
    )


def test_experiment_rejects_validation_plateau_without_validation_interval(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Validation plateau stopping should fail fast when periodic validation is disabled."""

    with pytest.raises(ValueError, match="validation_interval > 0"):
        Experiment(
            ExperimentConfig(
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


def test_experiment_ae_only_can_train_all_episodes_with_separate_validation_clip(
    fake_multi_episode_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should flatten full train episodes while keeping one validation clip."""

    experiment = Experiment(
        ExperimentConfig(
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


def test_experiment_supports_metaworld_ae_training(
    fake_metaworld_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should build the MT50 lazy frame dataset and validation clip loader."""

    experiment = Experiment(
        ExperimentConfig(
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
    assert len(experiment.train_dataset) == 8
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["frame"].shape == (2, 3, 8, 8)
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["frames"].shape[1:] == (5, 3, 8, 8)


def test_experiment_auto_batch_size_uses_full_clip_on_cpu(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Auto batch sizing should resolve to the full episode size on CPU."""

    ae_experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="auto_batch_ae",
            auto_batch_size=True,
            device="cpu",
        )
    )
    dyn_experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="auto_batch_dyn",
            auto_batch_size=True,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            device="cpu",
        )
    )
    assert ae_experiment.cfg.batch_size == 130
    assert ae_experiment.train_loader.batch_size == 130
    assert dyn_experiment.cfg.batch_size == 126
    assert dyn_experiment.train_loader.batch_size == 126


def test_checkpoint_round_trip(tmp_path: Path, fake_long_dataset_root: Path) -> None:
    """World-model checkpoints should save and load the expected metadata."""

    checkpoint_path = tmp_path / "round_trip.pt"
    model = WorldModel(ae_backend="wan")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = ExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        ae_backend="wan",
        device="cpu",
    )
    save_training_checkpoint(
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
    checkpoint = load_training_checkpoint(checkpoint_path, device="cpu")
    assert checkpoint["step"] == 3
    assert checkpoint["mode"] == "ae_only"
    assert checkpoint["clip_metadata"]["frame_start"] is None
    assert checkpoint_ae_backend(checkpoint) == "wan"
    assert checkpoint_dynamics_backend(checkpoint) == "rf_dit"


def test_checkpoint_backend_falls_back_to_autoencoder_metadata() -> None:
    """Checkpoint backend detection should read the serialized autoencoder metadata."""

    assert checkpoint_ae_backend({"autoencoder": {"backend": "wan"}}) == "wan"


def test_checkpoint_dynamics_backend_falls_back_to_legacy_conv_label() -> None:
    """Missing dynamics metadata should be treated as the removed legacy backend."""

    assert checkpoint_dynamics_backend({}) == "legacy_conv"


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


def test_experiment_can_partial_load_encoder_decoder(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading encoder and decoder weights should copy those submodules from a checkpoint."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_encoder_decoder",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            device="cpu",
        )
    )
    encoder_weight = next(experiment.model.encoder.parameters())
    decoder_weight = next(experiment.model.decoder.parameters())
    dynamics_weight = next(experiment.model.dynamics.parameters())
    assert torch.allclose(encoder_weight, torch.full_like(encoder_weight, 0.25))
    assert torch.allclose(decoder_weight, torch.full_like(decoder_weight, 0.5))
    assert not torch.allclose(dynamics_weight, torch.full_like(dynamics_weight, 0.75))


def test_experiment_can_partial_load_dynamics(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading dynamics weights should copy only the dynamics submodule."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_dynamics",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            load_dynamics=str(saved_world_model_ae_checkpoint),
            device="cpu",
        )
    )
    dynamics_weight = next(experiment.model.dynamics.parameters())
    encoder_weight = next(experiment.model.encoder.parameters())
    assert torch.allclose(dynamics_weight, torch.full_like(dynamics_weight, 0.75))
    assert torch.allclose(encoder_weight, torch.full_like(encoder_weight, 0.25))


def test_experiment_accepts_compatible_dynamics_checkpoint_without_action_conditioning_metadata(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Missing action-conditioning metadata should not block compatible RF-DiT warm starts."""

    compatible_path = tmp_path / "legacy_action_metadata_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["action_conditioning_mode"] = None
    checkpoint["dynamics"]["config"]["conditioning_frame_choices"] = None
    torch.save(checkpoint, compatible_path)

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="legacy_action_metadata",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            load_dynamics=str(compatible_path),
            device="cpu",
        )
    )

    dynamics_weight = next(experiment.model.dynamics.parameters())
    assert torch.allclose(dynamics_weight, torch.full_like(dynamics_weight, 0.75))


def test_experiment_accepts_older_dynamics_checkpoint_with_new_temporal_embedding_enabled(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Older RF-DiT checkpoints should warm start when the learned temporal embedding is newly enabled."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="temporal_embedding_warm_start",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            load_dynamics=str(saved_world_model_ae_checkpoint),
            dynamics_use_learned_temporal_embedding=True,
            device="cpu",
        )
    )

    assert experiment.model.dynamics.cfg.use_learned_temporal_embedding is True
    assert experiment.model.dynamics.net.temporal_pos_embed is not None
    assert torch.count_nonzero(experiment.model.dynamics.net.temporal_pos_embed) == 0


def test_experiment_rejects_explicitly_mismatched_action_conditioning_mode(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Explicit action-conditioning mismatches should still fail fast."""

    incompatible_path = tmp_path / "global_chunk_action_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["action_conditioning_mode"] = "global_chunk"
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(ValueError, match="action_conditioning_mode=global_chunk"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="global_chunk_action_dynamics",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(incompatible_path),
                device="cpu",
            )
        )


def test_experiment_accepts_global_chunk_action_checkpoint_when_requested(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """DreamDojo-style global-chunk checkpoints should load when the run requests them."""

    compatible_path = tmp_path / "global_chunk_action_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["action_conditioning_mode"] = "global_chunk"
    checkpoint["model_state"]["dynamics.net.action_embedder_B_D.fc1.weight"] = torch.zeros(1024, 16)
    checkpoint["model_state"]["dynamics.net.action_embedder_B_3D.fc1.weight"] = torch.zeros(3072, 16)
    torch.save(checkpoint, compatible_path)

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="global_chunk_action_dynamics",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            load_dynamics=str(compatible_path),
            dynamics_action_conditioning_mode="global_chunk",
            device="cpu",
        )
    )

    assert experiment.model.dynamics.cfg.action_conditioning_mode == "global_chunk"


def test_experiment_rejects_legacy_global_chunk_action_checkpoint_without_metadata(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Old flattened-action checkpoints should fail clearly even when metadata is missing."""

    incompatible_path = tmp_path / "legacy_global_chunk_action_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["action_conditioning_mode"] = None
    checkpoint["model_state"]["dynamics.net.action_embedder_B_D.fc1.weight"] = torch.zeros(1024, 16)
    checkpoint["model_state"]["dynamics.net.action_embedder_B_3D.fc1.weight"] = torch.zeros(3072, 16)
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(ValueError, match="action_conditioning_mode=global_chunk"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="legacy_global_chunk_action_dynamics",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(incompatible_path),
                device="cpu",
            )
        )


def test_experiment_rejects_legacy_conv_dynamics_checkpoint(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading old conv-dynamics checkpoints should fail with a clear backend error."""

    legacy_path = tmp_path / "legacy_conv_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint.pop("dynamics_backend", None)
    checkpoint.pop("dynamics", None)
    torch.save(checkpoint, legacy_path)

    with pytest.raises(ValueError, match="legacy"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="legacy_dynamics",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(legacy_path),
                device="cpu",
            )
        )


def test_experiment_rejects_stub_action_dynamics_checkpoint(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading a pre-action dynamics checkpoint should fail with a clear compatibility error."""

    incompatible_path = tmp_path / "stub_action_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["num_action_per_chunk"] = 5
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(ValueError, match="stub-action|Older stub-action checkpoints"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="stub_action_dynamics",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(incompatible_path),
                device="cpu",
            )
        )


def test_experiment_rejects_old_three_to_two_dynamics_checkpoint(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading an older 3->2 dynamics checkpoint should fail clearly."""

    incompatible_path = tmp_path / "three_to_two_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["context_frames"] = 3
    checkpoint["dynamics"]["config"]["target_frames"] = 2
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(ValueError, match="context_frames=3|target_frames=2"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="three_to_two_dynamics",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(incompatible_path),
                device="cpu",
            )
        )


def test_experiment_rejects_removed_backend_config(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The experiment should reject the removed conv backend configuration."""

    with pytest.raises(ValueError, match="only supports the Wan VAE"):
        Experiment(
            ExperimentConfig(
                mode="ae_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="removed_backend",
                **DEBUG_FRAME_KWARGS,
                ae_backend="conv",
                device="cpu",
            )
        )


def test_experiment_can_stop_early_on_ae_validation_plateau(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should stop early once validation AE loss stops improving."""

    config = ExperimentConfig(
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
    experiment = Experiment(config)
    assert not experiment._handle_validation_early_stop(1, {"ae_loss": 1.0})
    assert experiment._handle_validation_early_stop(2, {"ae_loss": 1.0})
    metrics_path = tmp_path / "outputs" / "ae_plateau" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    early_stop_records = [record for record in records if "early_stop" in record]
    assert early_stop_records
    assert stopped_records
    assert stopped_records[-1]["stopped"]["reason"] == "plateau"


def test_experiment_can_stop_early_on_dynamics_validation_plateau(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should stop early once one-step validation stops improving."""

    config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="dynamics_plateau",
        **DEBUG_FRAME_KWARGS,
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
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
    experiment = Experiment(config)
    assert not experiment._handle_validation_early_stop(1, {"next_frame_mse": 0.5})
    assert experiment._handle_validation_early_stop(2, {"next_frame_mse": 0.5})
    metrics_path = tmp_path / "outputs" / "dynamics_plateau" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    assert stopped_records[-1]["step"] == 2


def test_experiment_restores_plateau_state_from_resume_metrics(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """A resumed run should reuse prior validation history when evaluating plateau stops."""

    source_config = ExperimentConfig(
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
    source_experiment = Experiment(source_config)
    source_experiment._save_checkpoint(
        tmp_path / "outputs" / "resume_source" / "checkpoints" / "last.pt",
        step=1,
    )
    append_jsonl(
        tmp_path / "outputs" / "resume_source" / "metrics.jsonl",
        {"step": 1, "validation": {"ae_loss": 1.0}},
    )
    resume_path = tmp_path / "outputs" / "resume_source" / "checkpoints" / "last.pt"

    resumed_config = ExperimentConfig(
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
    resumed_experiment = Experiment(resumed_config)
    resumed_experiment._restore_early_stop_state(resumed_experiment.current_step)
    assert resumed_experiment.current_step == 1
    assert resumed_experiment.best_window_loss == pytest.approx(1.0)
    assert resumed_experiment._handle_validation_early_stop(2, {"ae_loss": 1.0})
    metrics_path = tmp_path / "outputs" / "resume_target" / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    stopped_records = [record for record in records if "stopped" in record]
    assert stopped_records
    assert stopped_records[-1]["step"] == 2
