"""Tests for the standalone SO101 DreamDojo-style rectified-flow package."""

from __future__ import annotations

from pathlib import Path

import torch

from dreamdojo.so101_rf.dataset import SO101_RELATIVE_ACTION_SCALE, so101_relative_actions_from_absolute_targets
from dreamdojo.so101_rf.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from dreamdojo.so101_rf.model import WorldModel
from dreamdojo.so101_rf.runtime import (
    DREAMDOJO_UPSTREAM_COMMIT,
    build_model_from_checkpoint,
    load_training_checkpoint,
    run_teacher_forced_validation,
    save_training_checkpoint,
)
from dreamdojo.so101_rf.train import (
    DEFAULT_VALIDATION_METRIC,
    TrainConfig,
    dynamics_training_step,
    resolve_max_steps,
)


def build_small_world_model() -> WorldModel:
    """Return a compact standalone world model for fast CPU tests."""

    return WorldModel(
        latent_channels=48,
        hidden_channels=64,
        ae_backend="wan",
        resolution=32,
        height=32,
        width=32,
        dynamics_context_frames=1,
        dynamics_target_frames=3,
        dynamics_patch_spatial=2,
        dynamics_model_channels=64,
        dynamics_num_blocks=1,
        dynamics_num_heads=4,
        dynamics_action_dim=6,
        dynamics_action_conditioning_mode="chunk_per_frame",
        dynamics_use_adaln_lora=True,
        dynamics_adaln_lora_dim=16,
    )


def build_small_checkpoint_config() -> dict[str, object]:
    """Return checkpoint config metadata for the compact standalone test model."""

    return {
        "latent_channels": 48,
        "hidden_channels": 64,
        "ae_backend": "wan",
        "resolution": 32,
        "height": 32,
        "width": 32,
        "dynamics_infer_steps": 35,
        "dynamics_train_timesteps": 1000,
        "dynamics_rf_shift": 5.0,
        "dynamics_context_frames": 1,
        "dynamics_target_frames": 3,
        "dynamics_patch_spatial": 2,
        "dynamics_model_channels": 64,
        "dynamics_num_blocks": 1,
        "dynamics_num_heads": 4,
        "dynamics_action_dim": 6,
        "dynamics_action_conditioning_mode": "chunk_per_frame",
        "dynamics_action_representation": "relative_delta",
        "dynamics_action_scale": SO101_RELATIVE_ACTION_SCALE,
        "dynamics_adaln_lora_dim": 16,
    }


def test_standalone_frame_layout_keeps_expected_temporal_boundaries() -> None:
    """The standalone frame layout should keep the requested SO101 `1 -> 3` boundaries."""

    assert DYNAMICS_FRAME_LAYOUT.context_frames == 1
    assert DYNAMICS_FRAME_LAYOUT.target_frames == 3
    assert DYNAMICS_FRAME_LAYOUT.max_frames == 4
    assert DYNAMICS_FRAME_LAYOUT.pixel_frames_for_latent_frames(4) == 13
    assert DYNAMICS_FRAME_LAYOUT.latent_frames_for_pixel_frames(13) == 4
    assert DYNAMICS_FRAME_LAYOUT.num_action_per_chunk == 12


def test_standalone_so101_relative_actions_match_expected_formula() -> None:
    """The standalone SO101 action helper should convert absolute targets to scaled deltas."""

    actions = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.6, 0.9]], dtype=torch.float32)
    states = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.3, 0.4]], dtype=torch.float32)
    expected = (actions - states[:, : actions.shape[1]]) * SO101_RELATIVE_ACTION_SCALE
    converted = so101_relative_actions_from_absolute_targets(
        actions,
        states,
        scale=SO101_RELATIVE_ACTION_SCALE,
    )
    torch.testing.assert_close(converted, expected)


def test_standalone_world_model_builds_requested_dit_shape() -> None:
    """The standalone world model should preserve the requested `1536/20/12` DiT wiring."""

    model = WorldModel(
        latent_channels=48,
        hidden_channels=64,
        ae_backend="wan",
        resolution=96,
        height=96,
        width=128,
        dynamics_context_frames=1,
        dynamics_target_frames=3,
        dynamics_patch_spatial=2,
        dynamics_model_channels=1536,
        dynamics_num_blocks=20,
        dynamics_num_heads=12,
        dynamics_action_dim=6,
        dynamics_action_conditioning_mode="chunk_per_frame",
        dynamics_use_adaln_lora=True,
        dynamics_adaln_lora_dim=128,
    )

    assert model.dynamics.cfg.patch_spatial == 2
    assert model.dynamics.cfg.context_frames == 1
    assert model.dynamics.cfg.target_frames == 3
    assert model.dynamics.cfg.model_channels == 1536
    assert model.dynamics.cfg.num_blocks == 20
    assert model.dynamics.cfg.num_heads == 12
    assert model.dynamics.cfg.adaln_lora_dim == 128
    assert DEFAULT_VALIDATION_METRIC == "worst_case_next_frame_mse"


def test_standalone_checkpoint_saves_upstream_commit_and_roundtrips(tmp_path: Path) -> None:
    """Standalone checkpoints should retain DreamDojo provenance and remain loadable."""

    model = build_small_world_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=3,
        config=build_small_checkpoint_config(),
        best_metric=0.25,
    )

    checkpoint = load_training_checkpoint(checkpoint_path, "cpu")
    reloaded = build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))

    assert checkpoint["dreamdojo_upstream"]["commit"] == DREAMDOJO_UPSTREAM_COMMIT
    assert checkpoint["best_metric"] == 0.25
    assert reloaded.dynamics.cfg.model_channels == 64
    assert reloaded.dynamics.cfg.target_frames == 3


def test_standalone_training_step_runs_on_fake_batch() -> None:
    """A fake SO101 batch should run through Wan encode, RF prep, and DiT loss computation."""

    model = build_small_world_model()
    model.configure_trainability("dynamics_only")
    batch = {
        "context_frames": torch.rand(1, 1, 3, 32, 32),
        "target_frames": torch.rand(1, 12, 3, 32, 32),
        "future_target_frames": torch.zeros(1, 0, 3, 32, 32),
        "actions": torch.rand(1, 12, 6),
        "future_actions": torch.zeros(1, 0, 6),
    }

    metrics = dynamics_training_step(model, batch)

    assert set(metrics) == {"loss", "latent_rf_mse", "target_sigma"}
    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(metrics["latent_rf_mse"])
    assert torch.isfinite(metrics["target_sigma"])


def test_standalone_teacher_forced_validation_reports_expected_boundaries() -> None:
    """Teacher-forced validation should preserve the standalone temporal accounting."""

    model = build_small_world_model()
    frames = torch.rand(13, 3, 32, 32)
    actions = torch.rand(12, 6)

    preview_frames, stats = run_teacher_forced_validation(model, frames, actions)

    assert tuple(preview_frames.shape) == tuple(frames.shape)
    assert stats["input_frame_count"] == 13
    assert stats["decoded_frame_count"] == 13
    assert stats["predicted_frame_count"] == 13
    assert stats["seed_frames"] == 1
    assert stats["target_latent_frames"] == 3
    assert stats["target_pixel_frames"] == 12
    assert "worst_case_next_frame_mse" in stats


def test_standalone_resolve_max_steps_uses_exact_two_epoch_formula() -> None:
    """The standalone trainer should default to `ceil(2 * windows / batch)` steps."""

    cfg = TrainConfig(
        batch_size=16,
        num_epochs=2.0,
        max_steps=None,
        wan_vae_path="dummy.pt",
    )

    assert resolve_max_steps(28112, 16, cfg) == 3514


def test_standalone_package_does_not_runtime_import_world_model_v2() -> None:
    """The standalone DreamDojo lane should not import the legacy package at runtime."""

    package_root = Path("dreamdojo/so101_rf")
    source_paths = sorted(path for path in package_root.glob("*.py") if path.is_file())

    for source_path in source_paths:
        assert "world_model_v2" not in source_path.read_text(encoding="utf-8")
