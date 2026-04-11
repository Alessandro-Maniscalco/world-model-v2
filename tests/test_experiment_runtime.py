"""Runtime-heavy Wan experiment tests kept out of the fast root suite."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import pytest
import torch

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from world_model_v2.experiment import (
    Experiment,
    ExperimentConfig,
    load_training_checkpoint,
    validation_metric_value_from_stats,
)


DEBUG_FRAME_KWARGS = {"frame_start": 111, "frame_end": 116}
DYNAMICS_FIVE_FRAME_KWARGS = {"frame_start": 111, "frame_end": 115}
DYNAMICS_TWO_FRAME_KWARGS = {"frame_start": 111, "frame_end": 112}


def test_wan_ae_only_training_step_reports_kl_terms(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only steps should expose reconstruction, KL, and total loss metrics."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="ae_metrics",
            **DEBUG_FRAME_KWARGS,
            device="cpu",
        )
    )
    batch = experiment._move_batch_to_device(next(iter(experiment.train_loader)))
    metrics = experiment._ae_only_training_step(batch)
    assert set(metrics) == {"loss", "recon_loss", "recon_mse", "recon_l1", "edge_l1", "kl_loss", "ae_loss"}
    assert float(metrics["kl_loss"]) >= 0.0
    assert torch.isclose(metrics["loss"], metrics["ae_loss"])


def test_wan_dynamics_only_training_step_reports_rf_terms(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only steps should expose RF loss terms for the target latent frame."""

    experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="dynamics_metrics",
            **DYNAMICS_FIVE_FRAME_KWARGS,
            load_encoder_decoder=str(saved_world_model_ae_checkpoint),
            device="cpu",
        )
    )
    batch = experiment._move_batch_to_device(next(iter(experiment.train_loader)))
    metrics = experiment._dynamics_only_training_step(batch)
    assert set(metrics) == {
        "loss",
        "latent_rf_mse",
        "target_sigma",
        "conditioning_frames_mean",
        "use_video_condition_mean",
        "active_self_forcing_loss_weight",
        "active_rollout_self_forcing_loss_weight",
    }
    assert float(metrics["active_self_forcing_loss_weight"]) == 0.0
    assert float(metrics["active_rollout_self_forcing_loss_weight"]) == 0.0
    assert torch.isclose(metrics["loss"], metrics["latent_rf_mse"])
    assert float(metrics["target_sigma"]) > 0.0
    assert (
        DYNAMICS_FRAME_LAYOUT.conditioning_frame_choices[0]
        <= float(metrics["conditioning_frames_mean"])
        <= DYNAMICS_FRAME_LAYOUT.conditioning_frame_choices[-1]
    )
    assert 0.0 <= float(metrics["use_video_condition_mean"]) <= 1.0


@pytest.mark.parametrize(
    ("mode", "expected_stat_key", "frame_kwargs", "expected_predicted_count"),
    [
        ("ae_only", "ae_loss", DEBUG_FRAME_KWARGS, 6),
        ("dynamics_only", "next_frame_mse", DYNAMICS_FIVE_FRAME_KWARGS, 5),
    ],
)
def test_wan_experiment_run_writes_artifacts_for_each_mode(
    mode: str,
    expected_stat_key: str,
    frame_kwargs: dict[str, int],
    expected_predicted_count: int,
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A one-step run should write checkpoints and validation artifacts in every mode."""

    config = ExperimentConfig(
        mode=mode,
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name=f"smoke_{mode}",
        **frame_kwargs,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        log_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint) if mode == "dynamics_only" else "",
    )
    experiment = Experiment(config)
    experiment.run()
    captured = capsys.readouterr()
    run_dir = tmp_path / "outputs" / f"smoke_{mode}"
    stats = load_training_checkpoint(run_dir / "checkpoints" / "best.pt", device="cpu")
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0_grid.png").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0.mp4").exists()
    assert (run_dir / "samples" / "step_000001" / "episode_0_stats.json").exists()
    assert stats["best_metric"] is not None
    payload = (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
    saved_stats = json.loads(payload)
    assert expected_stat_key in payload
    assert stats["best_metric"] == pytest.approx(saved_stats[expected_stat_key])
    assert saved_stats["elapsed_run_seconds"] >= 0.0
    metrics_records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_record = next(record for record in metrics_records if "validation" in record)
    assert validation_record["validation"]["elapsed_run_seconds"] >= 0.0
    assert saved_stats["checkpoint"] == str(run_dir / "checkpoints" / "last.pt")
    assert saved_stats["best_checkpoint"] == str(run_dir / "checkpoints" / "best.pt")
    assert saved_stats["is_best_checkpoint"] is True
    assert validation_record["validation"]["best_checkpoint"] == str(
        run_dir / "checkpoints" / "best.pt"
    )
    assert '"elapsed_run_seconds"' in captured.out
    if mode == "ae_only":
        assert '"kl_loss"' in payload
    else:
        assert saved_stats["predicted_frame_count"] == expected_predicted_count
        assert '"predicted_frame_count": 5' in payload
        assert '"next_latent_mse"' in payload
        assert '"open_rollout_frame_mse"' in payload
        assert saved_stats["seed_frames"] == DYNAMICS_FRAME_LAYOUT.conditioning_frame_choices[0]
        assert saved_stats["loss_frames"] == (
            saved_stats["input_frame_count"] - DYNAMICS_FRAME_LAYOUT.conditioning_frame_choices[0]
        )
        assert saved_stats["open_rollout_seed_frames"] == DYNAMICS_FRAME_LAYOUT.context_frames
        assert saved_stats["open_rollout_loss_frames"] == (
            saved_stats["input_frame_count"] - DYNAMICS_FRAME_LAYOUT.context_frames
        )
        assert saved_stats["open_rollout_validation_style"] == "open_rollout_autoregressive"
        assert saved_stats["validation_style"] == "teacher_forced_1_context_3_target"
        assert "next_frame_mse_1to3" in saved_stats
        assert saved_stats["validation_style_1to3"] == "teacher_forced_1_context_3_target"
        assert saved_stats["worst_case_next_frame_mse"] == pytest.approx(
            saved_stats["next_frame_mse_1to3"]
        )
        assert stats["dynamics_backend"] == "rf_dit"


def test_dynamics_run_can_select_best_checkpoint_by_open_rollout_metric(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics runs should optionally choose the best checkpoint by open-rollout MSE."""

    config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="smoke_dynamics_open_rollout_metric",
        **DYNAMICS_FIVE_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
        dynamics_validation_metric="open_rollout_frame_mse",
    )
    experiment = Experiment(config)
    experiment.run()
    run_dir = tmp_path / "outputs" / "smoke_dynamics_open_rollout_metric"
    checkpoint = load_training_checkpoint(run_dir / "checkpoints" / "best.pt", device="cpu")
    saved_stats = json.loads(
        (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
    )
    assert checkpoint["best_metric"] == pytest.approx(saved_stats["open_rollout_frame_mse"])


def test_dynamics_run_can_average_validation_across_multiple_episodes(
    fake_multi_episode_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics validation should aggregate checkpoint selection across multiple episodes."""

    config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_multi_episode_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="smoke_dynamics_multi_validation",
        split="train",
        validation_split="train",
        validation_episodes=(0, 1),
        frame_start=100,
        frame_end=104,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        log_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
    )
    experiment = Experiment(config)
    experiment.run()

    run_dir = tmp_path / "outputs" / "smoke_dynamics_multi_validation"
    checkpoint = load_training_checkpoint(run_dir / "checkpoints" / "best.pt", device="cpu")
    episode_0_stats = json.loads(
        (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
    )
    episode_1_stats = json.loads(
        (run_dir / "samples" / "step_000001" / "episode_1_stats.json").read_text(encoding="utf-8")
    )
    summary_stats = json.loads(
        (run_dir / "samples" / "step_000001" / "validation_summary.json").read_text(encoding="utf-8")
    )
    metrics_records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_record = next(record for record in metrics_records if "validation" in record)

    assert summary_stats["validation_episode_count"] == 2
    assert summary_stats["validation_episodes"] == [0, 1]
    assert validation_record["validation"]["validation_episode_count"] == 2
    assert validation_record["validation"]["validation_episodes"] == [0, 1]
    assert summary_stats["next_frame_mse"] == pytest.approx(
        (episode_0_stats["next_frame_mse"] + episode_1_stats["next_frame_mse"]) / 2.0
    )
    assert summary_stats["open_rollout_frame_mse"] == pytest.approx(
        (episode_0_stats["open_rollout_frame_mse"] + episode_1_stats["open_rollout_frame_mse"])
        / 2.0
    )
    assert checkpoint["best_metric"] == pytest.approx(summary_stats["next_frame_mse"])


@pytest.mark.parametrize(
    ("run_name", "target_frames", "expected_validation_style"),
    [
        ("smoke_dynamics_1to1", 1, "teacher_forced_1_context_1_target"),
        ("smoke_dynamics_1to2", 2, "teacher_forced_1_context_2_target"),
        ("smoke_dynamics_1to3", 3, "teacher_forced_1_context_3_target"),
    ],
)
def test_dynamics_run_supports_one_context_layouts(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
    run_name: str,
    target_frames: int,
    expected_validation_style: str,
) -> None:
    """A one-step run should work for 1-context dynamics layouts with 1-3 targets."""

    config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name=run_name,
        **DEBUG_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        batch_size=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
        dynamics_context_frames=1,
        dynamics_target_frames=target_frames,
        dynamics_conditioning_frame_choices=(1,),
        dynamics_conditioning_frame_probabilities=(1.0,),
        dynamics_validation_conditioning_frame_choices=(1,),
        dynamics_open_rollout_context_frames=1,
        dynamics_open_rollout_stride_frames=1,
    )
    experiment = Experiment(config)
    experiment.run()

    run_dir = tmp_path / "outputs" / run_name
    saved_stats = json.loads(
        (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
    )
    checkpoint = load_training_checkpoint(run_dir / "checkpoints" / "best.pt", device="cpu")

    assert checkpoint["best_metric"] == pytest.approx(saved_stats["next_frame_mse"])
    assert saved_stats["seed_frames"] == 1
    assert saved_stats["loss_frames"] == 5
    assert saved_stats["open_rollout_seed_frames"] == 1
    assert saved_stats["open_rollout_loss_frames"] == 5
    assert saved_stats["predicted_frame_count"] == 6
    assert saved_stats["validation_style"] == expected_validation_style
    assert saved_stats[f"next_frame_mse_1to{target_frames}"] == pytest.approx(
        saved_stats["next_frame_mse"]
    )
    assert saved_stats["conditioning_frame_choices"] == [1]
    assert saved_stats["validation_conditioning_frame_choices"] == [1]
    assert saved_stats["open_rollout_context_frames"] == 1
    assert saved_stats["open_rollout_stride_frames"] == 1
    assert saved_stats["open_rollout_initial_stride_frames"] == 1


def test_runtime_records_initial_default_open_rollout_stride(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Validation stats should expose the implicit first rollout stride for default chunking."""

    config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="default_rollout_stride",
        **DEBUG_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        batch_size=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
        dynamics_context_frames=1,
        dynamics_target_frames=3,
        dynamics_conditioning_frame_choices=(1,),
        dynamics_conditioning_frame_probabilities=(1.0,),
        dynamics_validation_conditioning_frame_choices=(1,),
        dynamics_open_rollout_context_frames=1,
    )
    experiment = Experiment(config)
    experiment.run()

    run_dir = tmp_path / "outputs" / "default_rollout_stride"
    saved_stats = json.loads(
        (run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
    )

    assert saved_stats["open_rollout_stride_frames"] is None
    assert saved_stats["open_rollout_initial_stride_frames"] == 3


def test_resume_rebuilds_best_metric_for_changed_dynamics_validation_metric(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Resumes should restore the best value for the currently selected validation metric."""

    initial_config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_source",
        **DYNAMICS_FIVE_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
    )
    initial_experiment = Experiment(initial_config)
    initial_experiment.run()
    source_run_dir = tmp_path / "outputs" / "resume_source"
    saved_stats = json.loads(
        (source_run_dir / "samples" / "step_000001" / "episode_0_stats.json").read_text(encoding="utf-8")
    )

    resumed_experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_target",
            **DYNAMICS_FIVE_FRAME_KWARGS,
            device="cpu",
            resume=str(source_run_dir / "checkpoints" / "best.pt"),
            dynamics_validation_metric="open_rollout_frame_mse",
        )
    )

    assert resumed_experiment.best_metric == pytest.approx(saved_stats["open_rollout_frame_mse"])


def test_resume_rebuilds_derived_rollout_consistency_metric_from_older_logs(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Resumes should derive rollout consistency from older logs that predate the new metric."""

    initial_config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_consistency_source",
        **DYNAMICS_FIVE_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
    )
    initial_experiment = Experiment(initial_config)
    initial_experiment.run()
    source_run_dir = tmp_path / "outputs" / "resume_consistency_source"
    stats_path = source_run_dir / "samples" / "step_000001" / "episode_0_stats.json"
    saved_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_consistency = validation_metric_value_from_stats(
        "open_rollout_consistency_score",
        saved_stats,
    )
    assert expected_consistency is not None

    metrics_path = source_run_dir / "metrics.jsonl"
    patched_records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        validation = record.get("validation")
        if isinstance(validation, dict):
            validation.pop("open_rollout_consistency_score", None)
            validation.pop("open_rollout_motion_log_error", None)
        patched_records.append(record)
    metrics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in patched_records),
        encoding="utf-8",
    )

    resumed_experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_consistency_target",
            **DYNAMICS_FIVE_FRAME_KWARGS,
            device="cpu",
            resume=str(source_run_dir / "checkpoints" / "best.pt"),
            dynamics_validation_metric="open_rollout_consistency_score",
        )
    )

    assert resumed_experiment.best_metric == pytest.approx(expected_consistency)


def test_resume_applies_requested_learning_rate_override(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Resumes should honor the current config learning rate instead of the checkpoint's one."""

    initial_config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_lr_source",
        **DYNAMICS_FIVE_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        device="cpu",
        lr=2e-5,
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
    )
    initial_experiment = Experiment(initial_config)
    initial_experiment.run()
    source_run_dir = tmp_path / "outputs" / "resume_lr_source"

    resumed_experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_lr_target",
            **DYNAMICS_FIVE_FRAME_KWARGS,
            device="cpu",
            lr=1e-5,
            resume=str(source_run_dir / "checkpoints" / "best.pt"),
        )
    )

    assert all(group["lr"] == pytest.approx(1e-5) for group in resumed_experiment.optimizer.param_groups)


def test_resume_best_metric_ignores_future_source_validation_records(
    fake_long_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Resume state should only replay validation history up to the checkpoint step."""

    initial_config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_long_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_cutoff_source",
        **DYNAMICS_FIVE_FRAME_KWARGS,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
    )
    initial_experiment = Experiment(initial_config)
    initial_experiment.run()
    source_run_dir = tmp_path / "outputs" / "resume_cutoff_source"
    stats_path = source_run_dir / "samples" / "step_000001" / "episode_0_stats.json"
    saved_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_consistency = validation_metric_value_from_stats(
        "open_rollout_consistency_score",
        saved_stats,
    )
    assert expected_consistency is not None

    metrics_path = source_run_dir / "metrics.jsonl"
    patched_records = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        patched_records.append(json.loads(line))
    synthetic_validation = dict(saved_stats)
    synthetic_validation.pop("open_rollout_consistency_score", None)
    synthetic_validation.pop("open_rollout_motion_log_error", None)
    synthetic_validation["open_rollout_frame_mse"] = float(saved_stats["open_rollout_frame_mse"]) * 0.5
    synthetic_validation["open_rollout_target_motion_ratio"] = 1.0
    derived_future_consistency = validation_metric_value_from_stats(
        "open_rollout_consistency_score",
        synthetic_validation,
    )
    assert derived_future_consistency is not None
    assert derived_future_consistency < expected_consistency
    synthetic_validation["open_rollout_consistency_score"] = derived_future_consistency
    patched_records.append({"step": 2, "validation": synthetic_validation})
    metrics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in patched_records),
        encoding="utf-8",
    )

    resumed_run_dir = tmp_path / "outputs" / "resume_cutoff_target"
    resumed_experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_cutoff_target",
            **DYNAMICS_FIVE_FRAME_KWARGS,
            device="cpu",
            resume=str(source_run_dir / "checkpoints" / "best.pt"),
            dynamics_validation_metric="open_rollout_consistency_score",
        )
    )

    assert resumed_experiment.best_metric == pytest.approx(expected_consistency)
    resumed_best_checkpoint = load_training_checkpoint(
        resumed_run_dir / "checkpoints" / "best.pt",
        device="cpu",
    )
    assert int(resumed_best_checkpoint["step"]) == 1
    assert float(resumed_best_checkpoint["best_metric"]) == pytest.approx(expected_consistency)


def test_resume_resets_best_metric_when_validation_episodes_change(
    fake_multi_episode_dataset_root: Path,
    saved_world_model_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Resume should not inherit best-checkpoint state across a changed validation domain."""

    initial_config = ExperimentConfig(
        mode="dynamics_only",
        data_root=str(fake_multi_episode_dataset_root),
        output_dir=str(tmp_path / "outputs"),
        run_name="resume_multi_source",
        split="train",
        validation_split="train",
        validation_episode=0,
        frame_start=100,
        frame_end=104,
        max_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        device="cpu",
        load_encoder_decoder=str(saved_world_model_ae_checkpoint),
    )
    initial_experiment = Experiment(initial_config)
    initial_experiment.run()
    source_run_dir = tmp_path / "outputs" / "resume_multi_source"

    resumed_experiment = Experiment(
        ExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_multi_episode_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="resume_multi_target",
            split="train",
            validation_split="train",
            validation_episodes=(0, 1),
            frame_start=100,
            frame_end=104,
            max_steps=1,
            validation_interval=1,
            checkpoint_interval=1,
            device="cpu",
            resume=str(source_run_dir / "checkpoints" / "best.pt"),
        )
    )

    assert resumed_experiment.best_metric is None

    resumed_experiment.run()
    resumed_run_dir = tmp_path / "outputs" / "resume_multi_target"
    summary_stats = json.loads(
        (resumed_run_dir / "samples" / "step_000001" / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    resumed_best_checkpoint = load_training_checkpoint(
        resumed_run_dir / "checkpoints" / "best.pt",
        device="cpu",
    )

    assert summary_stats["validation_episode_count"] == 2
    assert float(resumed_best_checkpoint["best_metric"]) == pytest.approx(
        summary_stats["next_frame_mse"]
    )


def test_wan_experiment_auto_batch_size_cleans_up_and_retries_after_oom(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The trainer should clean up and continue after an auto-batch CUDA OOM."""

    experiment = Experiment(
        ExperimentConfig(
            mode="ae_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="oom_retry",
            **DEBUG_FRAME_KWARGS,
            auto_batch_size=True,
            batch_size=8,
            max_steps=1,
            validation_interval=0,
            checkpoint_interval=0,
            early_stop_window_size=0,
            device="cpu",
        )
    )
    state = {"raised": False, "cleaned": False}
    original_execute_training_step = experiment._execute_training_step

    def flaky_execute(self: Experiment, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Raise one synthetic OOM before delegating to the real training step."""

        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("CUDA out of memory")
        return original_execute_training_step(batch)

    def fake_reduce(self: Experiment) -> bool:
        """Shrink the batch size so the retried step can continue."""

        self.cfg.batch_size = 3
        self.train_loader = self._build_train_loader(self.train_dataset)
        return True

    def fake_cleanup(self: Experiment) -> None:
        """Record that the cleanup hook ran after the synthetic OOM."""

        state["cleaned"] = True

    experiment._execute_training_step = MethodType(flaky_execute, experiment)
    experiment._reduce_batch_size_after_oom = MethodType(fake_reduce, experiment)
    experiment._cleanup_after_cuda_oom = MethodType(fake_cleanup, experiment)
    experiment.run()
    assert experiment.current_step == 1
    assert experiment.cfg.batch_size == 3
    assert state["cleaned"] is True
