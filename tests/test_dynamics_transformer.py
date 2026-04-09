"""Tests for the rectified-flow transformer dynamics module."""

from __future__ import annotations

import pytest
import torch

from world_model_v2.dynamics_transformer import (
    DYNAMICS_FRAME_LAYOUT,
    DynamicsTransformerConfig,
    RectifiedFlowDynamics,
)


def build_dynamics(
    *,
    context_frames: int = DYNAMICS_FRAME_LAYOUT.context_frames,
    target_frames: int = DYNAMICS_FRAME_LAYOUT.target_frames,
    conditional_frame_sigma: float = 0.0,
    conditioning_frame_choices: tuple[int, ...] | None = None,
    conditioning_frame_probabilities: tuple[float, ...] | None = None,
    validation_conditioning_frame_choices: tuple[int, ...] | None = None,
    open_rollout_context_frames: int | None = None,
    open_rollout_stride_frames: int | None = None,
    action_conditioning_mode: str = "chunk_per_frame",
    zero_init_action_embedder: bool = False,
    use_learned_temporal_embedding: bool = False,
) -> RectifiedFlowDynamics:
    """Create a small RF DiT dynamics module for unit tests."""

    return RectifiedFlowDynamics(
        DynamicsTransformerConfig(
            max_img_h=8,
            max_img_w=8,
            context_frames=context_frames,
            target_frames=target_frames,
            max_frames=context_frames + target_frames,
            conditioning_frame_choices=conditioning_frame_choices,
            conditioning_frame_probabilities=conditioning_frame_probabilities,
            validation_conditioning_frame_choices=validation_conditioning_frame_choices,
            open_rollout_context_frames=open_rollout_context_frames,
            open_rollout_stride_frames=open_rollout_stride_frames,
            in_channels=4,
            out_channels=4,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=64,
            num_blocks=2,
            num_heads=4,
            action_dim=4,
            action_conditioning_mode=action_conditioning_mode,
            zero_init_action_embedder=zero_init_action_embedder,
            use_learned_temporal_embedding=use_learned_temporal_embedding,
            conditional_frame_sigma=conditional_frame_sigma,
            dynamics_infer_steps=4,
            dynamics_train_timesteps=32,
            dynamics_rf_shift=5.0,
        )
    )


def test_rf_dit_forward_preserves_video_shape() -> None:
    """The tiny DiT should return one velocity tensor per latent input."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(2, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((2, 5), 10.0)
    condition_mask = dynamics.make_condition_mask(
        conditioning_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    actions = dynamics.make_zero_actions(batch_size=2, device=noisy_latent_video.device, dtype=noisy_latent_video.dtype)
    output = dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=actions,
        conditioning_latent_video=conditioning_latent_video,
    )
    assert output.shape == noisy_latent_video.shape
    assert torch.isfinite(output).all()


def test_rf_dit_can_add_learned_temporal_embedding() -> None:
    """The tiny DiT should optionally carry one learned temporal token bias."""

    dynamics = build_dynamics(use_learned_temporal_embedding=True)

    assert dynamics.cfg.use_learned_temporal_embedding is True
    assert dynamics.net.temporal_pos_embed is not None
    assert dynamics.net.temporal_pos_embed.shape == (1, dynamics.cfg.max_frames, 1, 1, 64)


def test_prepare_training_inputs_uses_full_clip_rf_interpolation() -> None:
    """RF training inputs should noise the full clip and share one timestep per item."""

    dynamics = build_dynamics()
    clean_latent_video = torch.randn(2, 4, 5, 8, 8)
    prepared = dynamics.prepare_training_inputs(clean_latent_video)
    assert prepared.noisy_latent_video.shape == clean_latent_video.shape
    assert prepared.conditioning_latent_video.shape == clean_latent_video.shape
    assert prepared.target_velocity.shape == clean_latent_video.shape
    assert prepared.timesteps.shape == (2, 5)
    assert prepared.actions.shape == (2, 4, 4)
    sigma = prepared.target_sigmas.view(-1, 1, 1, 1, 1)
    assert torch.allclose(prepared.conditioning_latent_video, clean_latent_video)
    assert torch.allclose(
        prepared.noisy_latent_video,
        clean_latent_video + sigma * prepared.target_velocity,
        atol=1e-6,
    )
    assert not torch.allclose(
        prepared.target_velocity[:, :, : dynamics.cfg.context_frames],
        torch.zeros_like(prepared.target_velocity[:, :, : dynamics.cfg.context_frames]),
    )
    assert prepared.num_conditional_frames.shape == (2,)
    assert set(prepared.num_conditional_frames.tolist()).issubset(
        set(DYNAMICS_FRAME_LAYOUT.conditioning_frame_choices)
    )
    for batch_index, conditioning_frames in enumerate(prepared.num_conditional_frames.tolist()):
        assert torch.all(prepared.condition_mask[batch_index, :, :conditioning_frames] == 1.0)
        assert torch.all(prepared.condition_mask[batch_index, :, conditioning_frames:] == 0.0)
    assert torch.allclose(prepared.timesteps, prepared.timesteps[:, :1].expand_as(prepared.timesteps))
    assert torch.all(prepared.target_sigmas > 0.0)
    assert prepared.use_video_condition.shape == (2,)
    assert prepared.use_video_condition.dtype == torch.bool


def test_make_condition_mask_supports_per_item_conditioning_counts() -> None:
    """Condition masks should support different conditioning counts within one batch."""

    dynamics = build_dynamics()
    latents = torch.randn(2, 4, 5, 8, 8)
    condition_mask = dynamics.make_condition_mask(latents, num_conditional_frames=torch.tensor([3, 4]))
    assert torch.all(condition_mask[0, :, :3] == 1.0)
    assert torch.all(condition_mask[0, :, 3:] == 0.0)
    assert torch.all(condition_mask[1, :, :4] == 1.0)
    assert torch.all(condition_mask[1, :, 4:] == 0.0)


def test_make_condition_mask_can_allow_unregistered_conditioning_counts() -> None:
    """Explicit masks should optionally allow causal self-forcing counts outside the train set."""

    dynamics = build_dynamics(
        context_frames=1,
        target_frames=2,
        conditioning_frame_choices=(1,),
        conditioning_frame_probabilities=(1.0,),
        validation_conditioning_frame_choices=(1,),
        open_rollout_context_frames=1,
    )
    latents = torch.randn(1, 4, 3, 8, 8)

    condition_mask = dynamics.make_condition_mask(
        latents,
        num_conditional_frames=2,
        allow_unregistered=True,
    )

    assert torch.all(condition_mask[:, :, :2] == 1.0)
    assert torch.all(condition_mask[:, :, 2:] == 0.0)


def test_recover_clean_latent_video_inverts_rf_parameterization() -> None:
    """Clean-latent recovery should invert the RF `x_t = x_0 + sigma * v` parameterization."""

    dynamics = build_dynamics()
    clean = torch.randn(2, 4, 5, 8, 8)
    velocity = torch.randn_like(clean)
    sigmas = torch.tensor([0.25, 0.75], dtype=clean.dtype)
    noisy = clean + sigmas.view(-1, 1, 1, 1, 1) * velocity

    recovered = dynamics.recover_clean_latent_video(
        noisy_latent_video=noisy,
        predicted_velocity=velocity,
        target_sigmas=sigmas,
    )

    assert torch.allclose(recovered, clean, atol=1e-6, rtol=1e-6)


def test_make_zero_actions_matches_transition_chunk_shape() -> None:
    """Zero-action fallback should follow DreamDojo's four-action chunk semantics."""

    dynamics = build_dynamics()
    actions = dynamics.make_zero_actions(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)
    assert actions.shape == (3, DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    assert torch.count_nonzero(actions) == 0


def test_prepare_training_inputs_respects_conditioning_frame_probabilities() -> None:
    """Sampling conditioning lengths should honor explicit per-choice probabilities."""

    dynamics = build_dynamics(
        context_frames=2,
        target_frames=2,
        conditioning_frame_choices=(1, 2),
        conditioning_frame_probabilities=(1.0, 0.0),
        validation_conditioning_frame_choices=(1, 2),
        open_rollout_context_frames=1,
    )
    clean_latent_video = torch.randn(6, 4, 4, 8, 8)

    prepared = dynamics.prepare_training_inputs(clean_latent_video)

    assert prepared.num_conditional_frames.tolist() == [1, 1, 1, 1, 1, 1]


def test_config_preserves_open_rollout_stride_frames() -> None:
    """The RF dynamics config should retain an explicit rollout overlap stride."""

    dynamics = build_dynamics(
        context_frames=1,
        target_frames=2,
        conditioning_frame_choices=(1,),
        conditioning_frame_probabilities=(1.0,),
        validation_conditioning_frame_choices=(1,),
        open_rollout_context_frames=1,
        open_rollout_stride_frames=1,
    )

    assert dynamics.cfg.open_rollout_stride_frames == 1


def test_forward_rejects_wrong_action_horizon() -> None:
    """The DiT should fail clearly when the action chunk length is wrong."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 10.0)
    condition_mask = dynamics.make_condition_mask(
        conditioning_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    with pytest.raises(ValueError, match="action steps per chunk"):
        dynamics(
            noisy_latent_video=noisy_latent_video,
            timesteps=timesteps,
            condition_mask=condition_mask,
            actions=torch.zeros(1, DYNAMICS_FRAME_LAYOUT.max_frames, 4),
            conditioning_latent_video=conditioning_latent_video,
        )


def test_nonzero_actions_change_timestep_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Active action conditioning should change the timestep embedding passed into the blocks."""

    dynamics = build_dynamics()
    clean_latent_video = torch.randn(1, 4, 5, 8, 8)
    timesteps = torch.full((1, 5), 9.0)
    condition_mask = dynamics.make_condition_mask(
        clean_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    captured: list[torch.Tensor] = []
    original_forward = dynamics.net.blocks[0].forward

    def capture_forward(
        self: object,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture the first block's timestep embedding before delegating."""

        captured.append(timestep_embedding.detach().clone())
        return original_forward(x, timestep_embedding, cos, sin, adaln_lora=adaln_lora)

    monkeypatch.setattr(
        dynamics.net.blocks[0],
        "forward",
        capture_forward.__get__(dynamics.net.blocks[0], type(dynamics.net.blocks[0])),
    )
    zero_actions = dynamics.make_zero_actions(batch_size=1, device=clean_latent_video.device, dtype=clean_latent_video.dtype)
    nonzero_actions = torch.arange(
        DYNAMICS_FRAME_LAYOUT.num_action_per_chunk * 4,
        dtype=clean_latent_video.dtype,
    ).view(1, DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=zero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=nonzero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    assert len(captured) == 2
    assert not torch.allclose(captured[0], captured[1])


def test_action_chunk_conditioning_aligns_actions_to_future_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DreamDojo action-chunk conditioning should leave frame 0 unmodified and affect frame 1+."""

    dynamics = build_dynamics()
    clean_latent_video = torch.randn(1, 4, 5, 8, 8)
    timesteps = torch.full((1, 5), 9.0)
    condition_mask = dynamics.make_condition_mask(
        clean_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    captured: list[torch.Tensor] = []
    original_forward = dynamics.net.blocks[0].forward

    def capture_forward(
        self: object,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture the timestep embedding so frame alignment can be asserted."""

        captured.append(timestep_embedding.detach().clone())
        return original_forward(x, timestep_embedding, cos, sin, adaln_lora=adaln_lora)

    monkeypatch.setattr(
        dynamics.net.blocks[0],
        "forward",
        capture_forward.__get__(dynamics.net.blocks[0], type(dynamics.net.blocks[0])),
    )
    zero_actions = dynamics.make_zero_actions(batch_size=1, device=clean_latent_video.device, dtype=clean_latent_video.dtype)
    first_transition_only = zero_actions.clone()
    first_transition_only[:, 0, :] = 1.0
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=zero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=first_transition_only,
        conditioning_latent_video=clean_latent_video,
    )
    assert len(captured) == 2
    zero_embedding, action_embedding = captured
    assert torch.allclose(zero_embedding[:, 0], action_embedding[:, 0])
    assert not torch.allclose(zero_embedding[:, 1], action_embedding[:, 1])


def test_global_chunk_action_conditioning_broadcasts_one_embedding_to_all_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DreamDojo's global chunk mode should affect every frame with one shared action embedding."""

    dynamics = build_dynamics(action_conditioning_mode="global_chunk")
    clean_latent_video = torch.randn(1, 4, 5, 8, 8)
    timesteps = torch.full((1, 5), 9.0)
    condition_mask = dynamics.make_condition_mask(
        clean_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    captured: list[torch.Tensor] = []
    original_forward = dynamics.net.blocks[0].forward

    def capture_forward(
        self: object,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture the timestep embedding so global broadcasting can be asserted."""

        captured.append(timestep_embedding.detach().clone())
        return original_forward(x, timestep_embedding, cos, sin, adaln_lora=adaln_lora)

    monkeypatch.setattr(
        dynamics.net.blocks[0],
        "forward",
        capture_forward.__get__(dynamics.net.blocks[0], type(dynamics.net.blocks[0])),
    )
    zero_actions = dynamics.make_zero_actions(
        batch_size=1,
        device=clean_latent_video.device,
        dtype=clean_latent_video.dtype,
    )
    nonzero_actions = torch.arange(
        DYNAMICS_FRAME_LAYOUT.num_action_per_chunk * 4,
        dtype=clean_latent_video.dtype,
    ).view(1, DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=zero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=nonzero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    assert len(captured) == 2
    zero_embedding, action_embedding = captured
    assert not torch.allclose(zero_embedding[:, 0], action_embedding[:, 0])
    assert torch.allclose(action_embedding[:, 0], action_embedding[:, -1])


def test_zero_init_action_embedder_starts_as_noop() -> None:
    """Zero-init action embedders should ignore actions until training updates them."""

    dynamics = build_dynamics(
        action_conditioning_mode="global_chunk",
        zero_init_action_embedder=True,
    )
    clean_latent_video = torch.randn(1, 4, 5, 8, 8)
    timesteps = torch.full((1, 5), 9.0)
    condition_mask = dynamics.make_condition_mask(
        clean_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    zero_actions = dynamics.make_zero_actions(
        batch_size=1,
        device=clean_latent_video.device,
        dtype=clean_latent_video.dtype,
    )
    nonzero_actions = torch.arange(
        DYNAMICS_FRAME_LAYOUT.num_action_per_chunk * 4,
        dtype=clean_latent_video.dtype,
    ).view(1, DYNAMICS_FRAME_LAYOUT.num_action_per_chunk, 4)
    zero_output = dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=zero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    nonzero_output = dynamics(
        noisy_latent_video=clean_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=nonzero_actions,
        conditioning_latent_video=clean_latent_video,
    )
    assert torch.allclose(zero_output, nonzero_output)


def test_forward_repins_conditioned_frames_and_overwrites_conditioned_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper should repin context frames before the net and overwrite their velocity."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 12.0)
    condition_mask = dynamics.make_condition_mask(
        conditioning_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    actions = dynamics.make_zero_actions(batch_size=1, device=noisy_latent_video.device, dtype=noisy_latent_video.dtype)
    target_velocity = torch.randn_like(conditioning_latent_video)
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the repinned model input and return a constant target-frame velocity."""

        del timesteps_B_T, condition_video_input_mask_B_C_T_H_W, action
        captured["x"] = x_B_C_T_H_W.detach().clone()
        return torch.full_like(x_B_C_T_H_W, 7.0)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    output = dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=actions,
        conditioning_latent_video=conditioning_latent_video,
        target_velocity=target_velocity,
    )
    assert torch.allclose(
        captured["x"][:, :, : dynamics.cfg.context_frames],
        conditioning_latent_video[:, :, : dynamics.cfg.context_frames],
    )
    assert torch.allclose(
        output[:, :, : dynamics.cfg.context_frames],
        target_velocity[:, :, : dynamics.cfg.context_frames],
    )
    assert torch.allclose(
        output[:, :, dynamics.cfg.context_frames :],
        torch.full_like(output[:, :, dynamics.cfg.context_frames :], 7.0),
    )


def test_forward_zeroes_conditioned_input_when_video_condition_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DreamDojo-style dropout should zero masked conditioning inputs before the net."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 12.0)
    condition_mask = dynamics.make_condition_mask(
        conditioning_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the DiT input after the conditioning dropout gate."""

        del timesteps_B_T, condition_video_input_mask_B_C_T_H_W, action
        captured["x"] = x_B_C_T_H_W.detach().clone()
        return torch.zeros_like(x_B_C_T_H_W)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=None,
        conditioning_latent_video=conditioning_latent_video,
        use_video_condition=False,
    )
    assert torch.count_nonzero(captured["x"][:, :, : dynamics.cfg.context_frames]) == 0


def test_forward_applies_conditional_frame_timestep_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conditioned frames should receive a dedicated timestep before the DiT forward."""

    dynamics = RectifiedFlowDynamics(
        DynamicsTransformerConfig(
            max_img_h=8,
            max_img_w=8,
            max_frames=5,
            in_channels=4,
            out_channels=4,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=64,
            num_blocks=2,
            num_heads=4,
            action_dim=4,
            conditional_frame_timestep=0.5,
            dynamics_infer_steps=4,
            dynamics_train_timesteps=32,
            dynamics_rf_shift=5.0,
        )
    )
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    noisy_latent_video = torch.randn_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 12.0)
    condition_mask = dynamics.make_condition_mask(
        conditioning_latent_video,
        num_conditional_frames=3,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the effective timesteps passed into the DiT."""

        del x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W, action
        captured["timesteps"] = timesteps_B_T.detach().clone()
        return torch.zeros(1, 4, 5, 8, 8)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=None,
        conditioning_latent_video=conditioning_latent_video,
    )
    assert torch.allclose(captured["timesteps"][:, :3], torch.full((1, 3), 0.5))
    assert torch.allclose(captured["timesteps"][:, 3:], torch.full((1, 2), 12.0))


def test_forward_applies_conditional_frame_sigma_to_repinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conditioned frames should be repinned with the configured tiny sigma during training."""

    dynamics = build_dynamics(conditional_frame_sigma=0.25)
    conditioning_latent_video = torch.zeros(1, 4, 5, 8, 8)
    noisy_latent_video = torch.full_like(conditioning_latent_video, 9.0)
    target_velocity = torch.ones_like(conditioning_latent_video)
    timesteps = torch.full((1, 5), 12.0)
    condition_mask = dynamics.make_condition_mask(
        conditioning_latent_video,
        num_conditional_frames=3,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the repinned DiT input after conditional-sigma preprocessing."""

        del timesteps_B_T, condition_video_input_mask_B_C_T_H_W, action
        captured["x"] = x_B_C_T_H_W.detach().clone()
        return torch.zeros_like(x_B_C_T_H_W)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    dynamics(
        noisy_latent_video=noisy_latent_video,
        timesteps=timesteps,
        condition_mask=condition_mask,
        actions=None,
        conditioning_latent_video=conditioning_latent_video,
        target_velocity=target_velocity,
    )
    assert torch.allclose(captured["x"][:, :, :3], torch.full((1, 4, 3, 8, 8), 0.25))
    assert torch.allclose(captured["x"][:, :, 3:], torch.full((1, 4, 2, 8, 8), 9.0))


def test_sample_conditioned_latent_video_repins_context_with_conditional_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling should pin context frames at the configured tiny conditioning sigma."""

    dynamics = build_dynamics(
        context_frames=2,
        target_frames=3,
        conditional_frame_sigma=0.1,
        conditioning_frame_choices=(2,),
        conditioning_frame_probabilities=(1.0,),
        validation_conditioning_frame_choices=(2,),
        open_rollout_context_frames=2,
    )
    conditioning_latent_video = torch.zeros(1, 4, 5, 8, 8)
    generator = torch.Generator(device="cpu").manual_seed(123)
    expected_noise = torch.randn(
        1,
        4,
        5,
        8,
        8,
        generator=torch.Generator(device="cpu").manual_seed(123),
    )
    captured: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(
        dynamics.flow,
        "make_inference_schedule",
        lambda num_steps, device, dtype: (
            torch.tensor([1.0], device=device, dtype=dtype),
            torch.tensor([1.0, 0.0], device=device, dtype=dtype),
        ),
    )

    def fake_forward(**kwargs: torch.Tensor) -> torch.Tensor:
        """Capture the denoising input passed into the first sampling step."""

        captured["x"] = kwargs["noisy_latent_video"].detach().clone()
        return torch.zeros_like(kwargs["noisy_latent_video"])

    monkeypatch.setattr(dynamics, "forward", fake_forward)
    dynamics.sample_conditioned_latent_video(
        conditioning_latent_video=conditioning_latent_video,
        num_conditional_frames=2,
        generator=generator,
        infer_steps=1,
    )
    assert torch.allclose(captured["x"][:, :, :2], expected_noise[:, :, :2] * 0.1)
    assert torch.allclose(captured["x"][:, :, 2:], expected_noise[:, :, 2:])


def test_sample_conditioned_latent_video_uses_cfg_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling should combine conditioned and unconditioned velocities with CFG."""

    dynamics = build_dynamics()
    conditioning_latent_video = torch.randn(1, 4, 5, 8, 8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    recorded_flags: list[bool] = []

    def fake_forward(
        self: RectifiedFlowDynamics,
        noisy_latent_video: torch.Tensor,
        timesteps: torch.Tensor,
        condition_mask: torch.Tensor,
        actions: torch.Tensor | None,
        conditioning_latent_video: torch.Tensor,
        target_velocity: torch.Tensor | None = None,
        reference_noise: torch.Tensor | None = None,
        use_video_condition: bool | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return different target-frame velocities for cond and uncond branches."""

        del noisy_latent_video, timesteps, condition_mask, actions, conditioning_latent_video, target_velocity, reference_noise
        flag = bool(use_video_condition)
        recorded_flags.append(flag)
        output = torch.zeros(1, 4, 5, 8, 8)
        output[:, :, dynamics.cfg.context_frames :] = 2.0 if flag else 1.0
        return output

    def fake_make_inference_schedule(
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one deterministic single-step RF schedule."""

        del num_steps
        return (
            torch.tensor([1.0], device=device, dtype=dtype),
            torch.tensor([1.0, 0.0], device=device, dtype=dtype),
        )

    def fake_step(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma_from: torch.Tensor,
        sigma_to: torch.Tensor,
    ) -> torch.Tensor:
        """Expose the guided velocity directly as the next sample."""

        del sample, sigma_from, sigma_to
        return velocity

    monkeypatch.setattr(RectifiedFlowDynamics, "forward", fake_forward)
    monkeypatch.setattr(dynamics.flow, "make_inference_schedule", fake_make_inference_schedule)
    monkeypatch.setattr(dynamics.flow, "step", fake_step)
    sampled = dynamics.sample_conditioned_latent_video(
        conditioning_latent_video=conditioning_latent_video,
        num_conditional_frames=dynamics.cfg.context_frames,
        generator=generator,
        guidance_scale=1.5,
        infer_steps=1,
    )
    assert sampled.shape == conditioning_latent_video.shape
    assert recorded_flags == [True, False]
    expected_target = torch.full_like(
        sampled[:, :, dynamics.cfg.context_frames :],
        2.0 + 1.5 * (2.0 - 1.0),
    )
    assert torch.allclose(sampled[:, :, dynamics.cfg.context_frames :], expected_target)


def test_sample_next_latent_keeps_context_pinned_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default RF sampling should repin context once per step when CFG is disabled."""

    dynamics = build_dynamics()
    context = torch.randn(2, 4, DYNAMICS_FRAME_LAYOUT.context_frames, 8, 8)
    captured_inputs: list[torch.Tensor] = []

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Capture every full latent clip passed into the DiT during sampling."""

        del timesteps_B_T, condition_video_input_mask_B_C_T_H_W, action
        captured_inputs.append(x_B_C_T_H_W.detach().clone())
        return torch.zeros_like(x_B_C_T_H_W)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    generator = torch.Generator(device=context.device.type)
    generator.manual_seed(7)
    next_latent = dynamics.sample_next_latent(context, generator=generator)
    assert next_latent.shape == (2, 4, DYNAMICS_FRAME_LAYOUT.target_frames, 8, 8)
    assert torch.isfinite(next_latent).all()
    assert len(captured_inputs) == dynamics.cfg.dynamics_infer_steps
    for latent_video in captured_inputs:
        assert torch.allclose(latent_video[:, :, : dynamics.cfg.context_frames], context)


def test_sample_next_latent_supports_three_frame_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling should also work for the shorter mixed-training three-frame context."""

    dynamics = build_dynamics()
    context = torch.randn(1, 4, 3, 8, 8)

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero velocity while asserting cond and uncond prefixes are both valid."""

        del timesteps_B_T, action
        assert torch.all(condition_video_input_mask_B_C_T_H_W[:, :, :3] == 1.0)
        assert torch.all(condition_video_input_mask_B_C_T_H_W[:, :, 3:] == 0.0)
        conditioned_prefix = x_B_C_T_H_W[:, :, :3]
        assert torch.allclose(conditioned_prefix, context) or torch.count_nonzero(conditioned_prefix) == 0
        return torch.zeros_like(x_B_C_T_H_W)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    next_latent = dynamics.sample_next_latent(context)
    assert next_latent.shape == (1, 4, 2, 8, 8)


def test_sample_next_latent_supports_one_context_three_target_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling should support a DreamDojo-style one-context three-target chunk."""

    dynamics = build_dynamics(
        context_frames=1,
        target_frames=3,
        conditioning_frame_choices=(1,),
        conditioning_frame_probabilities=(1.0,),
        validation_conditioning_frame_choices=(1,),
        open_rollout_context_frames=1,
    )
    context = torch.randn(1, 4, 1, 8, 8)

    def fake_forward(
        *,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero velocity while asserting the one-frame conditioning mask."""

        del timesteps_B_T, action
        assert torch.all(condition_video_input_mask_B_C_T_H_W[:, :, :1] == 1.0)
        assert torch.all(condition_video_input_mask_B_C_T_H_W[:, :, 1:] == 0.0)
        assert torch.allclose(x_B_C_T_H_W[:, :, :1], context) or torch.count_nonzero(
            x_B_C_T_H_W[:, :, :1]
        ) == 0
        return torch.zeros_like(x_B_C_T_H_W)

    monkeypatch.setattr(dynamics.net, "forward", fake_forward)
    next_latent = dynamics.sample_next_latent(context)
    assert next_latent.shape == (1, 4, 3, 8, 8)
