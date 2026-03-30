"""Tests for the upstream-shaped latent world model."""

from __future__ import annotations

import torch

from world_model_v2.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from world_model_v2.config import AlgorithmConfig


def make_batch(batch_size: int = 2, horizon: int = 1, resolution: int = 128) -> dict[str, object]:
    """Create a small synthetic batch with upstream-shaped observation keys."""

    return {
        "obs": {"camera_1_color": torch.rand(batch_size, horizon, 3, resolution, resolution)},
        "action": torch.rand(batch_size, horizon, 4),
        "episode_idx": torch.zeros(batch_size, dtype=torch.long),
    }


def test_training_step_returns_loss_dict() -> None:
    """Stage-1 training should return the expected loss payload."""

    model = LatentWorldModel(
        AlgorithmConfig(training_stage=1, latent_channels=4, latent_dim=64, hidden_channels=32, timesteps=8),
        obs_keys=("camera_1_color",),
    )
    outputs = model.training_step(make_batch())
    assert set(outputs) == {"loss", "recon_loss", "clean_loss", "latents"}
    assert outputs["loss"].ndim == 0
    assert outputs["latents"].shape == (2, 4, 32, 32)


def test_reconstruct_preserves_sequence_shape() -> None:
    """Reconstruction should preserve batch and horizon axes."""

    model = LatentWorldModel(
        AlgorithmConfig(training_stage=1, latent_channels=4, latent_dim=64, hidden_channels=32, timesteps=8, infer_steps=2),
        obs_keys=("camera_1_color",),
    )
    reconstructed = model.reconstruct(make_batch()["obs"], num_steps=2)
    assert reconstructed.shape == (2, 1, 3, 128, 128)


def test_stage2_training_step_returns_full_window_latent_predictions(saved_stage1_checkpoint: str) -> None:
    """Stage 2 should produce a scalar loss and one latent prediction per input frame."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.bootstrap_from_checkpoint(saved_stage1_checkpoint, device="cpu")
    outputs = model.training_step(make_batch(batch_size=1, horizon=4, resolution=32))
    assert set(outputs) == {"loss", "dyn_loss_teacher_forced", "pred_latents"}
    assert outputs["loss"].ndim == 0
    assert outputs["pred_latents"].shape == (1, 4, 4, 8, 8)


def test_stage2_terminal_only_noise_levels_match_upstream_layout(saved_stage1_checkpoint: str) -> None:
    """Stage 2 should match the upstream terminal-only timestep layout."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=20,
            dyn_infer_steps=1,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
            sampling_strategy="terminal_only",
            prev_frame_noise_scale=0.1,
        ),
        obs_keys=("camera_1_color",),
    )
    t, s = model._sample_stage2_noise_levels(batch_size=2, horizon=5, device="cpu")
    assert torch.equal(t[:, :-1], s[:, :-1])
    assert torch.all(t[:, :-1] >= 1)
    assert torch.all(t[:, :-1] < max(2, int(20 * 0.1)))
    assert torch.all(t[:, -1] == 19)
    assert torch.all(s[:, -1] == 0)


def test_stage2_action_masking_matches_upstream_behavior(saved_stage1_checkpoint: str) -> None:
    """Stage-2 action masking should preserve all slots by default and only keep the last slot when enabled."""

    actions = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    default_model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    masked_model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
            mask_prev_action=True,
        ),
        obs_keys=("camera_1_color",),
    )
    assert torch.equal(default_model._prepare_dynamics_actions(actions), actions)
    masked_actions = masked_model._prepare_dynamics_actions(actions)
    assert torch.all(masked_actions[:, :-1] == 0)
    assert torch.equal(masked_actions[:, -1], actions[:, -1])


def test_stage2_teacher_forced_predictions_do_not_depend_on_later_future_frames(
    saved_stage1_checkpoint: str,
) -> None:
    """Earlier Stage-2 predictions should ignore later future frames in the batch."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.bootstrap_from_checkpoint(saved_stage1_checkpoint, device="cpu")
    batch = make_batch(batch_size=1, horizon=4, resolution=32)
    changed_batch = make_batch(batch_size=1, horizon=4, resolution=32)
    changed_batch["obs"]["camera_1_color"] = batch["obs"]["camera_1_color"].clone()
    changed_batch["action"] = batch["action"].clone()
    changed_batch["episode_idx"] = batch["episode_idx"].clone()
    changed_batch["obs"]["camera_1_color"][:, 3] = torch.rand_like(changed_batch["obs"]["camera_1_color"][:, 3])
    torch.manual_seed(1234)
    base_outputs = model.training_step(batch)
    torch.manual_seed(1234)
    changed_outputs = model.training_step(changed_batch)
    assert torch.allclose(
        base_outputs["pred_latents"][:, :3],
        changed_outputs["pred_latents"][:, :3],
        atol=1e-5,
        rtol=1e-4,
    )


def test_stage2_validation_rollout_preserves_frame_count(saved_stage1_checkpoint: str) -> None:
    """Stage 2 validation should roll out one image prediction per input frame."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
            dyn_infer_steps=1,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.bootstrap_from_checkpoint(saved_stage1_checkpoint, device="cpu")
    preview = model.validation_step(
        make_batch(batch_size=1, horizon=4, resolution=32),
        num_steps=1,
        rollout_context_size=2,
    )
    assert preview["reconstructed"].shape == (4, 3, 32, 32)
    assert preview["stats"]["predicted_frame_count"] == 4
    assert preview["stats"]["decoded_frame_count"] == 4
    assert "dyn_loss_rollout" in preview["stats"]


def test_stage3_freezes_encoder_and_dynamics(saved_stage2_checkpoint: str) -> None:
    """Stage 3 should only fine-tune the decoder after bootstrapping from Stage 2."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=3,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            infer_steps=2,
            dyn_infer_steps=1,
            load_ae=saved_stage2_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
            stage3_latent_noise_std=0.01,
        ),
        obs_keys=("camera_1_color",),
    )
    model.bootstrap_from_checkpoint(saved_stage2_checkpoint, device="cpu")
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.decoder.parameters())
    assert model.dynamics is not None
    assert not any(parameter.requires_grad for parameter in model.dynamics.parameters())
    outputs = model.training_step(make_batch(batch_size=1, horizon=1, resolution=32))
    assert outputs["loss"].ndim == 0
