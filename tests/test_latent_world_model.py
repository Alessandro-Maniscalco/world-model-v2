"""Tests for the upstream-shaped latent world model."""

from __future__ import annotations

from unittest.mock import patch

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
    assert set(outputs) == {"loss", "dyn_loss", "pred_latents"}
    assert outputs["loss"].ndim == 0
    assert outputs["pred_latents"].shape == (1, 4, 4, 8, 8)


def test_stage2_training_step_uses_second_hop_when_dyn_infer_steps_exceeds_one(
    saved_stage1_checkpoint: str,
) -> None:
    """Stage 2 should run the extra clean-target hop when more than one dynamics step is requested."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=2,
            load_ae=saved_stage1_checkpoint,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.bootstrap_from_checkpoint(saved_stage1_checkpoint, device="cpu")
    with patch.object(model, "_stage2_forward", wraps=model._stage2_forward) as wrapped:
        outputs = model.training_step(make_batch(batch_size=1, horizon=4, resolution=32))
    assert wrapped.call_count == 2
    assert outputs["loss"].ndim == 0


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


def test_stage2_prepare_actions_normalizes_to_minus_one_one() -> None:
    """Stage-2 action preparation should map checkpoint min/max stats into `[-1, 1]`."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.set_normalization_stats(
        {
            "action_min": [0.0, 10.0, -2.0, 3.0],
            "action_max": [10.0, 30.0, 2.0, 3.0],
        }
    )
    actions = torch.tensor(
        [
            [
                [0.0, 20.0, 0.0, 3.0],
                [10.0, 10.0, 2.0, 3.0],
            ]
        ]
    )
    prepared = model._prepare_stage2_actions(actions, expected_steps=2)
    expected = torch.tensor(
        [
            [
                [-1.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 1.0, 0.0],
            ]
        ]
    )
    assert torch.allclose(prepared, expected)


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
    assert "dyn_loss" in preview["stats"]
    assert preview["stats"]["prediction_mode"] == "open_loop_rollout"
    assert preview["stats"]["context_frames"] == 0
    assert preview["stats"]["seed_frames"] == 1
    assert preview["stats"]["rollout_window"] == 2


def test_stage2_validation_rollout_keeps_the_first_ground_truth_latent(
    saved_stage1_checkpoint: str,
) -> None:
    """Validation rollout should seed from the first encoded latent exactly once."""

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
    batch = make_batch(batch_size=1, horizon=5, resolution=32)
    obs, _ = model.concatenate_observations(batch["obs"])
    latents = model._encode_sequence(obs)
    actions = model._validate_actions(torch.as_tensor(batch["action"]), obs.shape[1])
    rolled = model.rollout_validation_episode(latents, actions, context_size=2)
    assert torch.allclose(rolled[:, 0], latents[:, 0])


def test_stage2_validation_rollout_uses_a_continuous_sliding_window(
    saved_stage1_checkpoint: str,
) -> None:
    """Validation rollout should keep a single sliding history window across the full episode."""

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
    batch = make_batch(batch_size=1, horizon=6, resolution=32)
    obs, _ = model.concatenate_observations(batch["obs"])
    latents = model._encode_sequence(obs)
    actions = model._validate_actions(torch.as_tensor(batch["action"]), obs.shape[1])
    captured_window_lengths: list[int] = []
    captured_action_lengths: list[int] = []

    def capture_window(
        current: torch.Tensor,
        action_window: torch.Tensor,
        schedule: list[tuple[torch.Tensor, torch.Tensor]],
        stabilization_timestep: int,
    ) -> torch.Tensor:
        captured_window_lengths.append(int(current.shape[1]))
        captured_action_lengths.append(int(action_window.shape[1]))
        assert stabilization_timestep == 1
        return current

    with (
        patch.object(model, "_denoise_rollout_window", side_effect=capture_window),
        patch.object(model, "_postprocess_rollout_latents", side_effect=lambda latents: latents),
    ):
        rolled = model.rollout_validation_episode(latents, actions, context_size=4)
    assert rolled.shape[1] == 6
    assert captured_window_lengths == [2, 3, 4, 4, 4]
    assert captured_action_lengths == [2, 3, 4, 4, 4]


def test_stage2_rollout_updates_the_active_window_not_only_the_newest_slot() -> None:
    """Open-loop rollout should refresh the active prediction window instead of only appending one slot."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.set_normalization_stats({"action_min": [0.0, 0.0, 0.0, 0.0], "action_max": [1.0, 1.0, 1.0, 1.0]})
    latents = torch.zeros(1, 5, 4, 8, 8)
    actions = torch.zeros(1, 5, 4)

    def overwrite_window(
        current: torch.Tensor,
        action_window: torch.Tensor,
        schedule: list[tuple[torch.Tensor, torch.Tensor]],
        stabilization_timestep: int,
    ) -> torch.Tensor:
        del action_window, schedule, stabilization_timestep
        value = float(current.shape[1] + 1)
        return torch.full_like(current, value)

    with (
        patch.object(model, "_denoise_rollout_window", side_effect=overwrite_window),
        patch.object(model, "_postprocess_rollout_latents", side_effect=lambda sequence: sequence),
    ):
        rolled = model.rollout_validation_episode(latents, actions, context_size=4)

    assert rolled.shape[1] == 5
    assert torch.all(rolled[:, 0] == 5.0)
    assert torch.all(rolled[:, 1:] == 5.0)


def test_stage2_rollout_normalizes_once_after_generation() -> None:
    """Open-loop rollout should normalize the finished latent sequence only once after generation."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.set_normalization_stats({"action_min": [0.0, 0.0, 0.0, 0.0], "action_max": [1.0, 1.0, 1.0, 1.0]})
    initial_latents = torch.rand(1, 1, 4, 8, 8)
    prepared_actions = model._prepare_stage2_actions(torch.zeros(1, 4, 4), expected_steps=4)

    with (
        patch.object(model, "_denoise_rollout_window", side_effect=lambda current, *_args: current),
        patch.object(model, "_normalize_latents", side_effect=lambda latents: latents) as wrapped_normalize,
    ):
        rolled = model.rollout_latents(initial_latents, prepared_actions, context_size=3)

    assert rolled.shape[1] == 4
    assert wrapped_normalize.call_count == 1


def test_stage2_teacher_forced_rollout_stays_closer_than_open_loop_under_drift() -> None:
    """Teacher-forced rollout should stay closer to the ground truth than open loop under compounding drift."""

    model = LatentWorldModel(
        AlgorithmConfig(
            training_stage=2,
            latent_channels=4,
            latent_dim=64,
            hidden_channels=32,
            timesteps=8,
            dyn_infer_steps=1,
            action_dim=4,
            dynamics_hidden_channels=32,
            action_emb_dim=64,
            dynamics_attention_heads=4,
        ),
        obs_keys=("camera_1_color",),
    )
    model.set_normalization_stats({"action_min": [0.0, 0.0, 0.0, 0.0], "action_max": [1.0, 1.0, 1.0, 1.0]})
    latents = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1, 1).repeat(1, 1, 4, 8, 8)
    actions = torch.zeros(1, 5, 4)

    def drift_to_history_mean(
        current: torch.Tensor,
        action_window: torch.Tensor,
        schedule: list[tuple[torch.Tensor, torch.Tensor]],
        stabilization_timestep: int,
    ) -> torch.Tensor:
        del action_window, schedule, stabilization_timestep
        denoised = current.clone()
        history_mean = current[:, :-1].mean(dim=1, keepdim=True)
        denoised[:, -1:] = history_mean
        return denoised

    with (
        patch.object(model, "_denoise_rollout_window", side_effect=drift_to_history_mean),
        patch.object(model, "_postprocess_rollout_latents", side_effect=lambda sequence: sequence),
    ):
        teacher_forced = model._teacher_forced_rollout_latents(latents, actions, context_size=4)
        open_loop = model.rollout_validation_episode(latents, actions, context_size=4)

    teacher_loss = torch.mean((teacher_forced[:, 1:] - latents[:, 1:]) ** 2)
    open_loop_loss = torch.mean((open_loop[:, 1:] - latents[:, 1:]) ** 2)
    assert teacher_loss < open_loop_loss


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
