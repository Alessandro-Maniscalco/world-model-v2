"""Runtime-heavy Wan experiment tests kept out of the fast minimal suite."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import pytest
import torch

from world_model_v2.minimal.experiment import (
    MinimalExperiment,
    MinimalExperimentConfig,
    load_minimal_checkpoint,
)


DEBUG_FRAME_KWARGS = {"frame_start": 111, "frame_end": 116}
DYNAMICS_FIVE_FRAME_KWARGS = {"frame_start": 111, "frame_end": 115}


def test_wan_ae_only_training_step_reports_kl_terms(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """AE-only steps should expose reconstruction, KL, and total loss metrics."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
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
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Dynamics-only steps should expose RF loss terms for the target latent frame."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
            mode="dynamics_only",
            data_root=str(fake_long_dataset_root),
            output_dir=str(tmp_path / "outputs"),
            run_name="dynamics_metrics",
            **DYNAMICS_FIVE_FRAME_KWARGS,
            load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint),
            device="cpu",
        )
    )
    batch = experiment._move_batch_to_device(next(iter(experiment.train_loader)))
    metrics = experiment._dynamics_only_training_step(batch)
    assert set(metrics) == {"loss", "latent_rf_mse", "target_sigma"}
    assert torch.isclose(metrics["loss"], metrics["latent_rf_mse"])
    assert float(metrics["target_sigma"]) > 0.0


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
    saved_minimal_wan_ae_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """A one-step run should write checkpoints and validation artifacts in every mode."""

    config = MinimalExperimentConfig(
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
        load_encoder_decoder=str(saved_minimal_wan_ae_checkpoint) if mode == "dynamics_only" else "",
    )
    experiment = MinimalExperiment(config)
    experiment.run()
    run_dir = tmp_path / "outputs" / f"smoke_{mode}"
    stats = load_minimal_checkpoint(run_dir / "checkpoints" / "best.pt", device="cpu")
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
    if mode == "ae_only":
        assert '"kl_loss"' in payload
    else:
        assert saved_stats["predicted_frame_count"] == expected_predicted_count
        assert '"predicted_frame_count": 5' in payload
        assert '"next_latent_mse"' in payload
        assert saved_stats["seed_frames"] == 3
        assert saved_stats["loss_frames"] == 2
        assert saved_stats["validation_style"] == "teacher_forced_three_context_two_target"
        assert stats["dynamics_backend"] == "rf_dit"


def test_wan_experiment_auto_batch_size_cleans_up_and_retries_after_oom(
    fake_long_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The trainer should clean up and continue after an auto-batch CUDA OOM."""

    experiment = MinimalExperiment(
        MinimalExperimentConfig(
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

    def flaky_execute(self: MinimalExperiment, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Raise one synthetic OOM before delegating to the real training step."""

        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("CUDA out of memory")
        return original_execute_training_step(batch)

    def fake_reduce(self: MinimalExperiment) -> bool:
        """Shrink the batch size so the retried step can continue."""

        self.cfg.batch_size = 3
        self.train_loader = self._build_train_loader(self.train_dataset)
        return True

    def fake_cleanup(self: MinimalExperiment) -> None:
        """Record that the cleanup hook ran after the synthetic OOM."""

        state["cleaned"] = True

    experiment._execute_training_step = MethodType(flaky_execute, experiment)
    experiment._reduce_batch_size_after_oom = MethodType(fake_reduce, experiment)
    experiment._cleanup_after_cuda_oom = MethodType(fake_cleanup, experiment)
    experiment.run()
    assert experiment.current_step == 1
    assert experiment.cfg.batch_size == 3
    assert state["cleaned"] is True
