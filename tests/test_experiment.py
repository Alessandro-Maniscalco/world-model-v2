"""Tests for the root multi-mode experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from world_model_v2.experiment import (
    Experiment,
    ExperimentConfig,
    _normalized_rng_state_tensor,
    checkpoint_ae_backend,
    checkpoint_dynamics_backend,
    compute_motion_ratio,
    load_training_checkpoint,
    motion_ratio_log_error,
    open_rollout_consistency_score,
    reconstruction_loss_terms,
    restore_rng_state,
    save_training_checkpoint,
    validation_metric_value_from_stats,
)
from world_model_v2.dynamics_transformer import DynamicsTrainingInputs
from world_model_v2.model import WorldModel
from world_model_v2.utils.checkpointing import append_jsonl


DEBUG_FRAME_KWARGS = {"frame_start": 111, "frame_end": 123}


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


def test_experiment_dynamics_only_requires_at_least_one_temporal_window(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should reject clips shorter than one 13-frame temporal window."""

    with pytest.raises(ValueError, match="at least 13 pixel frames"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="single_frame_dyn",
                frame_start=111,
                frame_end=113,
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
            frame_end=115,
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


def test_open_rollout_consistency_score_penalizes_motion_ratio_mismatch() -> None:
    """The rollout-consistency metric should worsen symmetrically away from ratio 1."""

    base_score = open_rollout_consistency_score(0.01, 1.0)
    under_motion_score = open_rollout_consistency_score(0.01, 0.5)
    over_motion_score = open_rollout_consistency_score(0.01, 2.0)

    assert base_score == pytest.approx(0.01)
    assert under_motion_score == pytest.approx(over_motion_score)
    assert under_motion_score > base_score
    assert motion_ratio_log_error(1.0) == pytest.approx(0.0)
    assert compute_motion_ratio(0.0, 0.0) == pytest.approx(1.0)


def test_validation_metric_value_from_stats_can_derive_rollout_consistency() -> None:
    """Compatibility metric restoration should derive consistency from older rollout stats."""

    stats = {
        "open_rollout_frame_mse": 0.01,
        "open_rollout_target_motion_ratio": 2.0,
    }

    derived = validation_metric_value_from_stats("open_rollout_consistency_score", stats)

    assert derived == pytest.approx(open_rollout_consistency_score(0.01, 2.0))


def test_normalized_rng_state_tensor_returns_cpu_uint8_tensor() -> None:
    """RNG-state normalization should coerce tensors into PyTorch's expected format."""

    raw_state = torch.arange(16, dtype=torch.int64)[::2]

    normalized_state = _normalized_rng_state_tensor(raw_state)

    assert normalized_state.device.type == "cpu"
    assert normalized_state.dtype == torch.uint8
    assert normalized_state.is_contiguous()
    assert normalized_state.tolist() == [0, 2, 4, 6, 8, 10, 12, 14]


def test_restore_rng_state_normalizes_torch_and_cuda_rng_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RNG restore should feed CPU uint8 tensors into the PyTorch RNG setters."""

    observed: dict[str, object] = {}

    def fake_set_rng_state(state: torch.Tensor) -> None:
        """Capture the normalized CPU RNG state."""

        observed["torch"] = state

    def fake_cuda_is_available() -> bool:
        """Pretend CUDA is available so the CUDA restore path runs."""

        return True

    def fake_set_rng_state_all(states: list[torch.Tensor]) -> None:
        """Capture the normalized CUDA RNG states."""

        observed["torch_cuda"] = states

    monkeypatch.setattr(torch, "set_rng_state", fake_set_rng_state)
    monkeypatch.setattr(torch.cuda, "is_available", fake_cuda_is_available)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", fake_set_rng_state_all)

    restore_rng_state(
        {
            "torch": torch.arange(8, dtype=torch.int64),
            "torch_cuda": [torch.arange(4, dtype=torch.int16)],
        }
    )

    torch_state = observed["torch"]
    assert isinstance(torch_state, torch.Tensor)
    assert torch_state.device.type == "cpu"
    assert torch_state.dtype == torch.uint8
    assert torch_state.tolist() == list(range(8))

    cuda_states = observed["torch_cuda"]
    assert isinstance(cuda_states, list)
    assert len(cuda_states) == 1
    assert cuda_states[0].device.type == "cpu"
    assert cuda_states[0].dtype == torch.uint8
    assert cuda_states[0].tolist() == list(range(4))


def test_experiment_auto_batch_size_uses_requested_batch_as_cuda_probe_start() -> None:
    """CUDA auto-batch probing should start from the requested batch size."""

    class DummyDataset:
        """Expose a fixed dataset length for batch-size resolution tests."""

        def __len__(self) -> int:
            """Return the synthetic dataset length."""

            return 100

    experiment = object.__new__(Experiment)
    experiment.cfg = ExperimentConfig(
        batch_size=64,
        auto_batch_size=True,
        dataset_format="lerobot_metaworld",
        device="cuda",
    )
    experiment.device = torch.device("cuda")
    captured: dict[str, int] = {}

    def fake_probe(
        dataset: DummyDataset,
        requested_batch_size: int,
        max_batch_size: int,
    ) -> int:
        """Capture the probe inputs and return a synthetic fit result."""

        captured["requested_batch_size"] = requested_batch_size
        captured["max_batch_size"] = max_batch_size
        return 17

    experiment._probe_cuda_batch_size = fake_probe  # type: ignore[method-assign]

    resolved_batch_size = experiment._resolve_train_batch_size(DummyDataset())

    assert resolved_batch_size == 17
    assert captured["requested_batch_size"] == 64
    assert captured["max_batch_size"] == 100


def test_experiment_probe_cuda_batch_size_grows_beyond_requested_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA auto-batch probing should keep growing when larger batches fit."""

    class DummyDataset:
        """Expose a fixed dataset length for auto-batch probing tests."""

        def __len__(self) -> int:
            """Return the synthetic dataset length."""

            return 100

    experiment = object.__new__(Experiment)
    observed_batch_sizes: list[int] = []

    def fake_batch_size_fits(dataset: DummyDataset, batch_size: int) -> bool:
        """Pretend every batch up to forty-eight samples fits in memory."""

        observed_batch_sizes.append(batch_size)
        return batch_size <= 48

    experiment._batch_size_fits = fake_batch_size_fits  # type: ignore[method-assign]
    monkeypatch.setattr(torch, "get_rng_state", lambda: torch.arange(4, dtype=torch.uint8))
    monkeypatch.setattr(torch, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    resolved_batch_size = experiment._probe_cuda_batch_size(
        DummyDataset(),
        requested_batch_size=8,
        max_batch_size=100,
    )

    assert resolved_batch_size == 48
    assert 16 in observed_batch_sizes
    assert 32 in observed_batch_sizes
    assert 48 in observed_batch_sizes
    assert 64 in observed_batch_sizes


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
        """Return one combined latent sequence for the concatenated temporal training chunk."""

        del deterministic
        expected_images = torch.cat(
            [batch["context_frames"], batch["target_frames"], batch["future_target_frames"]],
            dim=1,
        )
        if torch.equal(images, expected_images):
            return torch.cat([clean_latent_video, future_target_latent_video], dim=2)
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
        """Return one combined latent sequence for the concatenated temporal training chunk."""

        del deterministic
        expected_images = torch.cat(
            [batch["context_frames"], batch["target_frames"], batch["future_target_frames"]],
            dim=1,
        )
        if torch.equal(images, expected_images):
            return torch.cat([clean_latent_video, future_target_latent_video], dim=2)
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
        """Return one combined latent sequence for the concatenated temporal training chunk."""

        del deterministic
        expected_images = torch.cat(
            [batch["context_frames"], batch["target_frames"], batch["future_target_frames"]],
            dim=1,
        )
        if torch.equal(images, expected_images):
            return torch.cat([clean_latent_video, future_target_latent_video], dim=2)
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
        actions=torch.arange(32, dtype=torch.float32).view(1, 8, 4),
        target_sigmas=torch.ones(1),
        num_conditional_frames=torch.ones(1, dtype=torch.long),
        use_video_condition=torch.ones(1, dtype=torch.bool),
    )
    future_actions = torch.arange(32, 64, dtype=torch.float32).view(1, 8, 4)
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
    assert torch.equal(captured_actions[0], future_actions)


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


def test_experiment_rejects_disabling_open_rollout_for_open_rollout_metric(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Open-rollout metrics should fail fast when rollout validation is explicitly disabled."""

    with pytest.raises(ValueError, match="requires open-rollout stats"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="invalid_open_rollout_validation_toggle",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                dynamics_validation_metric="open_rollout_frame_mse",
                dynamics_run_open_rollout_validation=False,
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
    assert len(experiment.train_dataset) == 221
    validation_batch = next(iter(experiment.val_loader))
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["frames"].shape[1] == 130


def test_experiment_supports_metaworld_ae_training(
    fake_metaworld_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should exclude validation episodes from MT50 all-episode training."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            dataset_format="lerobot_metaworld",
            data_root=str(fake_metaworld_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=1,
            metaworld_task_index=0,
            batch_size=2,
            resolution=8,
            dynamics_target_frames=1,
            output_dir=str(tmp_path / "outputs"),
            run_name="metaworld_ae",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 1
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["frames"].shape == (1, 5, 3, 8, 8)
    assert train_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 1
    assert validation_batch["frames"].shape[1:] == (3, 3, 8, 8)


def test_experiment_supports_aloha_ae_training(
    fake_aloha_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should exclude validation episodes from ALOHA all-episode training."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            dataset_format="lerobot_aloha_sim_transfer_cube_scripted",
            data_root=str(fake_aloha_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=1,
            batch_size=2,
            resolution=8,
            dynamics_target_frames=1,
            output_dir=str(tmp_path / "outputs"),
            run_name="aloha_ae",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 1
    assert experiment.model.dynamics.cfg.action_dim == 14
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["frames"].shape == (1, 5, 3, 8, 8)
    assert train_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 1
    assert validation_batch["frames"].shape[1:] == (3, 3, 8, 8)


def test_experiment_supports_maniskill_ae_training(
    fake_maniskill_replay_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should exclude validation episodes from ManiSkill all-episode training."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            dataset_format="maniskill_replay",
            data_root=str(fake_maniskill_replay_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=1,
            batch_size=2,
            resolution=8,
            dynamics_target_frames=1,
            output_dir=str(tmp_path / "outputs"),
            run_name="maniskill_ae",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 1
    assert experiment.model.dynamics.cfg.action_dim == 8
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["frames"].shape == (1, 5, 3, 8, 8)
    assert train_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 1
    assert validation_batch["frames"].shape[1:] == (3, 3, 8, 8)


def test_experiment_supports_lerobot_so101_base_sim_pickplace_ae_training(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only mode should exclude validation episodes from SO-101 all-episode training."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            dataset_format="lerobot_so101_base_sim_pickplace",
            data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=1,
            batch_size=2,
            resolution=8,
            dynamics_target_frames=1,
            output_dir=str(tmp_path / "outputs"),
            run_name="lerobot_so101_base_sim_pickplace_ae",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 1
    assert experiment.model.dynamics.cfg.action_dim == 6
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["frames"].shape == (1, 5, 3, 8, 8)
    assert train_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 1
    assert validation_batch["frames"].shape[1:] == (3, 3, 8, 8)


def test_experiment_dynamics_only_can_train_all_episodes_with_separate_validation_clip(
    fake_multi_episode_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should flatten valid windows across the full train split."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_multi_episode_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=0,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            output_dir=str(tmp_path / "outputs"),
            run_name="all_episodes_dynamics",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 221
    assert experiment.train_dataset[118]["episode_idx"].item() == 1
    validation_batch = next(iter(experiment.val_loader))
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["frames"].shape[1] == 130


def test_experiment_ae_motion_loss_uses_clip_batches(
    fake_lerobot_so101_base_sim_pickplace_root: Path,
    tmp_path: Path,
) -> None:
    """Motion-weighted AE runs should still receive temporal clip batches for reconstruction."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            dataset_format="lerobot_so101_base_sim_pickplace",
            data_root=str(fake_lerobot_so101_base_sim_pickplace_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            validation_episode=1,
            batch_size=2,
            resolution=8,
            dynamics_target_frames=1,
            recon_motion_weight=0.5,
            output_dir=str(tmp_path / "outputs"),
            run_name="lerobot_so101_motion_ae",
            device="cpu",
        )
    )
    train_batch = next(iter(experiment.train_loader))
    assert train_batch["frames"].shape == (1, 5, 3, 8, 8)
    assert "prev_frame" not in train_batch
    assert "next_frame" not in train_batch


def test_experiment_supports_metaworld_dynamics_training_across_all_episodes(
    fake_metaworld_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only mode should exclude validation episodes from MT50 training."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            dataset_format="lerobot_metaworld",
            data_root=str(fake_metaworld_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="val",
            metaworld_task_index=0,
            batch_size=2,
            resolution=8,
            dynamics_context_frames=1,
            dynamics_target_frames=1,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            output_dir=str(tmp_path / "outputs"),
            run_name="metaworld_dynamics_all_episodes",
            validation_episode=1,
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 1
    assert experiment.train_dataset[0]["episode_idx"].item() == 0
    train_batch = next(iter(experiment.train_loader))
    validation_batch = next(iter(experiment.val_loader))
    assert train_batch["context_frames"].shape[1:] == (1, 3, 8, 8)
    assert train_batch["target_frames"].shape[1:] == (4, 3, 8, 8)
    assert train_batch["episode_idx"].reshape(-1)[0].item() == 0
    assert validation_batch["episode_idx"].reshape(-1)[0].item() == 1
    assert validation_batch["frames"].shape[1:] == (3, 3, 8, 8)


def test_experiment_excludes_validation_episodes_from_same_split_all_episode_training(
    fake_multi_episode_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics all-episode training should exclude validation episodes when sharing a split."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_multi_episode_dataset_root),
            split="train",
            train_all_episodes=True,
            validation_split="train",
            validation_episodes=(0,),
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            output_dir=str(tmp_path / "outputs"),
            run_name="all_episodes_excluding_validation",
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 103
    assert experiment.train_dataset[0]["episode_idx"].item() == 1


def test_experiment_excludes_metaworld_validation_episodes_from_all_episode_training(
    fake_metaworld_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """MetaWorld all-episode dynamics training should exclude selected validation episodes."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            dataset_format="lerobot_metaworld",
            data_root=str(fake_metaworld_dataset_root),
            split="train",
            train_all_episodes=True,
            metaworld_task_index=0,
            batch_size=2,
            resolution=8,
            dynamics_context_frames=1,
            dynamics_target_frames=1,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            output_dir=str(tmp_path / "outputs"),
            run_name="metaworld_excluding_validation",
            validation_episodes=(1,),
            device="cpu",
        )
    )
    assert len(experiment.train_dataset) == 1
    assert experiment.train_dataset[0]["episode_idx"].item() == 0


def test_experiment_auto_batch_size_uses_requested_ceiling_on_cpu(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """CPU auto batch sizing should respect the requested batch-size ceiling."""

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
    assert ae_experiment.cfg.batch_size == 32
    assert ae_experiment.train_loader.batch_size == 32
    assert dyn_experiment.cfg.batch_size == 32
    assert dyn_experiment.train_loader.batch_size == 32


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
    assert set(checkpoint["rng_state"]) >= {"python", "numpy", "torch"}
    assert checkpoint_ae_backend(checkpoint) == "wan"
    assert checkpoint_dynamics_backend(checkpoint) == "rf_dit"


def test_experiment_resume_restores_rng_state(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Resumes should restore saved RNG state instead of resetting to the CLI seed."""

    checkpoint_path = tmp_path / "resume_rng.pt"
    config = ExperimentConfig(
        mode="ae_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_rng_source",
        ae_backend="wan",
        device="cpu",
    )
    source_experiment = Experiment(config)

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    save_training_checkpoint(
        path=checkpoint_path,
        model=source_experiment.model,
        optimizer=source_experiment.optimizer,
        scheduler=None,
        step=3,
        config=config.to_dict(),
        mode=config.mode,
        clip_metadata=config.clip_metadata(),
        best_metric=0.5,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    Experiment(
        ExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_rng_target",
            device="cpu",
            resume=str(checkpoint_path),
        )
    )

    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.rand()) == pytest.approx(expected_numpy)
    assert float(torch.rand(1).item()) == pytest.approx(expected_torch)


def test_validate_records_non_best_checkpoint_metadata(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation artifacts should mark non-improving steps without overwriting best-path metadata."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="non_best_validation_metadata",
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            device="cpu",
        )
    )

    def fake_validate_dynamics_one_step(
        frames: torch.Tensor,
        actions: torch.Tensor | None = None,
        num_conditional_frames: int = 1,
    ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
        """Return stable teacher-forced validation outputs for metadata checks."""

        del actions, num_conditional_frames
        return frames.clone(), {
            "input_frame_count": int(frames.shape[0]),
            "decoded_frame_count": int(frames.shape[0]),
            "predicted_frame_count": int(frames.shape[0]),
            "seed_frames": 1,
            "loss_frames": int(frames.shape[0]) - 1,
            "next_frame_mse": 0.02,
            "next_latent_mse": 0.03,
            "validation_style": "teacher_forced_1_context_3_target",
        }

    def fake_validate_dynamics_open_rollout(
        frames: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int | str | None]]:
        """Return stable open-rollout validation outputs for metadata checks."""

        del actions
        return frames.clone(), {
            "open_rollout_seed_frames": 1,
            "open_rollout_loss_frames": int(frames.shape[0]) - 1,
            "open_rollout_stride_frames": None,
            "open_rollout_initial_stride_frames": 3,
            "open_rollout_decoded_frame_count": int(frames.shape[0]),
            "open_rollout_predicted_frame_count": int(frames.shape[0]),
            "open_rollout_frame_mse": 0.02,
            "open_rollout_frame_l1": 0.04,
            "open_rollout_validation_style": "open_rollout_autoregressive",
        }

    monkeypatch.setattr(experiment, "_validate_dynamics_one_step", fake_validate_dynamics_one_step)
    monkeypatch.setattr(experiment, "_validate_dynamics_open_rollout", fake_validate_dynamics_open_rollout)

    experiment.best_metric = 0.01
    experiment._save_checkpoint(experiment.checkpoints_dir / "best.pt", step=0)

    stats = experiment._validate(step=1)
    saved_stats = json.loads(
        (experiment.run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(
            encoding="utf-8"
        )
    )

    assert stats["is_best_checkpoint"] is False
    assert stats["checkpoint"] == str(experiment.checkpoints_dir / "last.pt")
    assert stats["best_checkpoint"] == str(experiment.checkpoints_dir / "best.pt")
    assert saved_stats["is_best_checkpoint"] is False
    assert (experiment.checkpoints_dir / "best.pt").exists()


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


def test_reconstruction_loss_terms_upweight_motion_regions() -> None:
    """Motion loss should focus reconstruction pressure on pixels that move across neighbors."""

    target = torch.zeros(1, 3, 8, 8)
    target[:, :, 3:5, 3:5] = 1.0
    prev_frame = torch.zeros_like(target)
    prev_frame[:, :, 3:5, 1:3] = 1.0
    next_frame = torch.zeros_like(target)
    next_frame[:, :, 3:5, 5:7] = 1.0
    predicted = torch.zeros_like(target)

    base_terms = reconstruction_loss_terms(
        predicted,
        target,
        mse_weight=1.0,
        l1_weight=1.0,
        edge_weight=0.0,
    )
    motion_terms = reconstruction_loss_terms(
        predicted,
        target,
        mse_weight=1.0,
        l1_weight=1.0,
        edge_weight=0.0,
        motion_weight=1.0,
        motion_threshold=0.05,
        motion_dilation_kernel_size=3,
        prev_frame=prev_frame,
        next_frame=next_frame,
    )

    assert float(motion_terms["motion_l1"].item()) > 0.0
    assert float(motion_terms["motion_mask_fraction"].item()) > 0.0
    assert float(motion_terms["recon_loss"].item()) > float(base_terms["recon_loss"].item())


def test_reconstruction_loss_terms_require_motion_context_when_enabled() -> None:
    """Motion-weighted loss should fail fast if neighboring GT frames are unavailable."""

    target = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError, match="prev_frame and next_frame"):
        reconstruction_loss_terms(
            target,
            target,
            mse_weight=1.0,
            l1_weight=0.0,
            edge_weight=0.0,
            motion_weight=1.0,
        )


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


def test_experiment_can_warm_start_deeper_dynamics_with_tail_blocks(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Loading dynamics into a deeper same-width model should seed tail blocks from the checkpoint tail."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="load_deeper_dynamics",
            **DEBUG_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            load_dynamics=str(saved_world_model_ae_checkpoint),
            dynamics_num_blocks=6,
            device="cpu",
        )
    )

    loaded_block_weight = experiment.model.dynamics.net.blocks[0].self_attn.q_proj.weight
    source_tail_weight = experiment.model.dynamics.net.blocks[3].self_attn.q_proj.weight
    tail_block_weight = experiment.model.dynamics.net.blocks[5].self_attn.q_proj.weight

    assert torch.allclose(loaded_block_weight, torch.full_like(loaded_block_weight, 0.75))
    assert torch.allclose(source_tail_weight, torch.full_like(source_tail_weight, 0.75))
    assert torch.allclose(tail_block_weight, source_tail_weight)


def test_experiment_can_resume_deeper_dynamics_with_tail_blocks(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Resuming into a deeper same-width model should also seed tail blocks from the checkpoint tail."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_deeper_dynamics",
            **DEBUG_FRAME_KWARGS,
            resume=str(saved_world_model_ae_checkpoint),
            dynamics_num_blocks=6,
            lr=1e-6,
            device="cpu",
        )
    )

    assert experiment.current_step == 5
    loaded_block_weight = experiment.model.dynamics.net.blocks[0].self_attn.q_proj.weight
    source_tail_weight = experiment.model.dynamics.net.blocks[3].self_attn.q_proj.weight
    tail_block_weight = experiment.model.dynamics.net.blocks[5].self_attn.q_proj.weight

    assert torch.allclose(loaded_block_weight, torch.full_like(loaded_block_weight, 0.75))
    assert torch.allclose(source_tail_weight, torch.full_like(source_tail_weight, 0.75))
    assert torch.allclose(tail_block_weight, source_tail_weight)


def test_experiment_rejects_missing_action_conditioning_metadata(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Missing action-conditioning metadata should fail fast for the new backbone."""

    compatible_path = tmp_path / "legacy_action_metadata_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["action_conditioning_mode"] = None
    checkpoint["dynamics"]["config"]["conditioning_frame_choices"] = None
    torch.save(checkpoint, compatible_path)

    with pytest.raises(ValueError, match="action_conditioning_mode=None"):
        Experiment(
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


def test_experiment_rejects_unsupported_learned_temporal_embedding_flag(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """The experiment should fail fast on the removed learned temporal embedding flag."""

    with pytest.raises(ValueError, match="dynamics_use_learned_temporal_embedding"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="temporal_embedding_removed",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(saved_world_model_ae_checkpoint),
                dynamics_use_learned_temporal_embedding=True,
                device="cpu",
            )
        )


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


def test_experiment_rejects_missing_dynamics_architecture_version(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Old checkpoints missing the DreamDojo architecture tag should fail clearly."""

    compatible_path = tmp_path / "missing_architecture_version_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"].pop("architecture_version", None)
    torch.save(checkpoint, compatible_path)

    with pytest.raises(ValueError, match="architecture_version"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="missing_architecture_version",
                **DEBUG_FRAME_KWARGS,
                load_encoder_decoder=str(saved_world_model_ae_checkpoint),
                load_dynamics=str(compatible_path),
                device="cpu",
            )
        )


def test_experiment_rejects_incompatible_dynamics_architecture_version(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Checkpoints from older DiT backbones should fail with a clear architecture error."""

    incompatible_path = tmp_path / "old_architecture_checkpoint.pt"
    checkpoint = load_training_checkpoint(saved_world_model_ae_checkpoint, device="cpu")
    checkpoint["dynamics"]["config"]["architecture_version"] = "dreamdojo_torch_small_v0"
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(ValueError, match="architecture_version"):
        Experiment(
            ExperimentConfig(
                mode="dynamics_only",
                data_root=str(fake_long_dataset_root),
                output_dir=str(tmp_path / "outputs"),
                run_name="old_architecture_version",
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
