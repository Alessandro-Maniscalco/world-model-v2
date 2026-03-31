"""Three-stage latent world model with Stage 1 reconstruction and Stage 2/3 rollout."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_v2.algorithms.latent_dynamics.noise_scheduler import Stage1NoiseScheduler
from world_model_v2.algorithms.models.cm_decoder import CMDecoder
from world_model_v2.algorithms.models.cm_latent_dynamics import CMLatentDynamics
from world_model_v2.algorithms.models.cnn_encoder import CNNEncoder
from world_model_v2.config import AlgorithmConfig
from world_model_v2.utils.checkpointing import load_checkpoint


class LatentWorldModel(nn.Module):
    """Own stage-aware encoding, decoding, dynamics, training loss, and validation behavior."""

    def __init__(self, cfg: AlgorithmConfig, obs_keys: tuple[str, ...]) -> None:
        """Create the requested Stage 1, 2, or 3 latent world model."""

        super().__init__()
        if cfg.training_stage not in (1, 2, 3):
            raise NotImplementedError(f"Unsupported training_stage={cfg.training_stage}")
        self.cfg = cfg
        self.obs_keys = tuple(obs_keys)
        self.image_channels = 3 * len(self.obs_keys)
        self.encoder = CNNEncoder(
            image_channels=self.image_channels,
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
        )
        self.decoder = CMDecoder(
            image_channels=self.image_channels,
            latent_channels=cfg.latent_channels,
            hidden_channels=cfg.hidden_channels,
            latent_dim=cfg.latent_dim,
        )
        self.dynamics = (
            CMLatentDynamics(
                latent_channels=cfg.latent_channels,
                hidden_channels=cfg.dynamics_hidden_channels,
                cond_dim=cfg.action_emb_dim,
                action_dim=cfg.action_dim,
                action_emb_dim=cfg.action_emb_dim,
                attention_heads=cfg.dynamics_attention_heads,
            )
            if cfg.training_stage in (2, 3)
            else None
        )
        self.noise_scheduler = Stage1NoiseScheduler(
            timesteps=cfg.timesteps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
        )
        self.normalization_stats: dict[str, Any] | None = None
        self._action_norm_scale: torch.Tensor | None = None
        self._action_norm_offset: torch.Tensor | None = None
        self.configure_stage_trainability()

    def to(self, *args: object, **kwargs: object) -> "LatentWorldModel":
        """Move the model and its noise scheduler to the requested device."""

        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.noise_scheduler.to(device)
        return self

    def configure_stage_trainability(self) -> None:
        """Freeze or unfreeze modules according to the current training stage."""

        self._set_module_trainable(self.encoder, self.cfg.training_stage == 1)
        self._set_module_trainable(self.decoder, self.cfg.training_stage in (1, 3))
        if self.dynamics is not None:
            self._set_module_trainable(self.dynamics, self.cfg.training_stage == 2)

    def bootstrap_from_checkpoint(
        self,
        checkpoint_path: str,
        device: torch.device | str,
    ) -> None:
        """Load encoder, decoder, and optional dynamics weights from a prior stage."""

        checkpoint = load_checkpoint(checkpoint_path, device)
        self.set_normalization_stats(checkpoint.get("normalization_stats"))
        model_state = checkpoint["model_state"]
        self._load_submodule_state("encoder", self.encoder, model_state)
        self._load_submodule_state("decoder", self.decoder, model_state)
        if self.dynamics is not None:
            dynamics_keys = [key for key in model_state if key.startswith("dynamics.")]
            if self.cfg.training_stage == 3 and not dynamics_keys:
                raise KeyError("Stage 3 bootstrap checkpoint must include dynamics weights.")
            if dynamics_keys:
                self._load_submodule_state("dynamics", self.dynamics, model_state)
        self.configure_stage_trainability()

    def set_normalization_stats(self, normalization_stats: Mapping[str, object] | None) -> None:
        """Cache checkpoint or dataset normalization stats for Stage-2 action conditioning."""

        if normalization_stats is None:
            self.normalization_stats = None
            self._action_norm_scale = None
            self._action_norm_offset = None
            return

        self.normalization_stats = dict(normalization_stats)
        if "action_min" not in normalization_stats or "action_max" not in normalization_stats:
            self._action_norm_scale = None
            self._action_norm_offset = None
            return

        action_min = torch.as_tensor(normalization_stats["action_min"], dtype=torch.float32)
        action_max = torch.as_tensor(normalization_stats["action_max"], dtype=torch.float32)
        if action_min.shape != action_max.shape:
            raise ValueError("Action normalization stats must have matching action_min/action_max shapes.")

        action_range = action_max - action_min
        constant_dims = action_range.abs() < 1e-6
        safe_range = action_range.clone()
        safe_range[constant_dims] = 2.0

        scale = torch.full_like(action_min, 2.0) / safe_range
        offset = -1.0 - scale * action_min
        scale[constant_dims] = 1.0
        offset[constant_dims] = -action_min[constant_dims]

        self._action_norm_scale = scale
        self._action_norm_offset = offset

    def _load_submodule_state(
        self,
        prefix: str,
        module: nn.Module,
        model_state: Mapping[str, torch.Tensor],
    ) -> None:
        """Load one named submodule from a saved checkpoint state dict."""

        prefix_with_dot = f"{prefix}."
        submodule_state = {
            key.removeprefix(prefix_with_dot): value
            for key, value in model_state.items()
            if key.startswith(prefix_with_dot)
        }
        if not submodule_state:
            raise KeyError(f"Checkpoint is missing weights for {prefix}.")
        module.load_state_dict(submodule_state, strict=True)

    def _set_module_trainable(self, module: nn.Module, trainable: bool) -> None:
        """Toggle gradients for every parameter in a module."""

        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise latent normalization."""

        norms = torch.linalg.vector_norm(latents, dim=1, keepdim=True).clamp_min(1e-6)
        return latents / norms

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and apply channel-wise latent normalization."""

        return self._normalize_latents(self.encoder(images))

    def decode(
        self,
        noisy_images: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode noisy images toward their lower-noise targets."""

        return self.decoder(noisy_images, t, s, latents)

    def concatenate_observations(
        self,
        obs_batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, bool]:
        """Concatenate configured observations across the channel dimension."""

        views = [obs_batch[key] for key in self.obs_keys]
        if not views:
            raise ValueError("No observations were provided.")
        had_batch = views[0].ndim == 5
        if not had_batch:
            views = [view.unsqueeze(0) for view in views]
        combined = torch.cat(views, dim=2)
        return combined.float(), had_batch

    def _flatten_images(self, obs: torch.Tensor) -> torch.Tensor:
        """Flatten batch and time axes into an image batch."""

        return obs.reshape(-1, obs.shape[2], obs.shape[3], obs.shape[4])

    def _encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a `(B, T, C, H, W)` observation tensor into latent grids."""

        batch_size, horizon = obs.shape[:2]
        flat_images = self._flatten_images(obs)
        latents = self.encode(flat_images)
        return latents.reshape(batch_size, horizon, latents.shape[1], latents.shape[2], latents.shape[3])

    def _validate_actions(self, actions: torch.Tensor, expected_steps: int) -> torch.Tensor:
        """Validate the shape of one raw action sequence tensor."""

        if actions.shape[-1] != self.cfg.action_dim:
            raise ValueError(
                f"Expected action dim {self.cfg.action_dim}, got {actions.shape[-1]}"
            )
        if actions.shape[1] != expected_steps:
            raise ValueError(
                f"Expected action horizon {expected_steps}, got {actions.shape[1]}"
            )
        return actions.float()

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to the upstream-style `[-1, 1]` range using cached stats."""

        if self.cfg.training_stage == 1:
            return actions
        if self._action_norm_scale is None or self._action_norm_offset is None:
            raise RuntimeError(
                "Stage 2 and 3 require action normalization stats. "
                "Call set_normalization_stats with checkpoint or dataset stats first."
            )
        scale = self._action_norm_scale.to(device=actions.device, dtype=actions.dtype)
        offset = self._action_norm_offset.to(device=actions.device, dtype=actions.dtype)
        return actions * scale + offset

    def _prepare_stage2_actions(
        self,
        actions: torch.Tensor,
        expected_steps: int,
    ) -> torch.Tensor:
        """Validate, normalize, and mask one action sequence for Stage-2 dynamics."""

        validated = self._validate_actions(actions, expected_steps)
        normalized = self._normalize_actions(validated)
        return self._prepare_dynamics_actions(normalized)

    def _prepare_dynamics_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Prepare action conditioning with the configured masking strategy."""

        conditioned = actions.clone()
        if self.cfg.mask_prev_action:
            conditioned[:, :-1] = 0.0
        return conditioned

    def _slice_dynamics_action_window(
        self,
        actions: torch.Tensor,
        frame_start: int,
        frame_end_exclusive: int,
    ) -> torch.Tensor:
        """Slice one already-prepared rollout action window."""

        return actions[:, frame_start:frame_end_exclusive].clone()

    def _make_dynamics_noise_slot(
        self,
        reference: torch.Tensor,
        start_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize a rollout target slot with scheduler-scaled Gaussian noise."""

        sigma = self.noise_scheduler.sigmas[start_timestep].to(reference.device).view(1, 1, 1, 1, 1)
        return torch.randn_like(reference) * sigma

    def _stage2_forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        stop_timesteps: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Stage-2 dynamics model through the scheduler-style consistency merge."""

        if self.dynamics is None:
            raise RuntimeError("Stage 2 requires an initialized dynamics model.")
        predicted_clean = self.dynamics(sample, timesteps, stop_timesteps, actions)
        return self.noise_scheduler.ctm_calc_out(
            sample,
            predicted_clean,
            timesteps,
            stop_timesteps,
        )

    def _expand_loss_weights(
        self,
        weights: torch.Tensor,
        target_ndim: int,
    ) -> torch.Tensor:
        """Expand per-frame loss weights to match a latent or image tensor."""

        expanded = weights
        while expanded.ndim < target_ndim:
            expanded = expanded.unsqueeze(-1)
        return expanded

    def _resolve_stage2_loss_weighting(self) -> str:
        """Resolve the effective Stage-2 loss weighting strategy."""

        if self.cfg.loss_weighting == "auto":
            return "uniform"
        return self.cfg.loss_weighting

    def _stage2_rollout_stabilization_timestep(self) -> int:
        """Return the smallest nonzero timestep used for stabilized rollout context frames."""

        if self.noise_scheduler.timesteps <= 1:
            return 0
        return 1

    def _postprocess_rollout_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize a completed predicted latent sequence once after rollout finishes."""

        flat_latents = latents.reshape(-1, latents.shape[2], latents.shape[3], latents.shape[4])
        normalized = self._normalize_latents(flat_latents)
        return normalized.reshape_as(latents)

    def _denoise_rollout_window(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        schedule: list[tuple[torch.Tensor, torch.Tensor]],
        stabilization_timestep: int,
    ) -> torch.Tensor:
        """Run the Stage-2 denoising schedule over one active rollout window."""

        for start, end in schedule:
            t = torch.full(
                current.shape[:2],
                stabilization_timestep,
                device=current.device,
                dtype=torch.long,
            )
            s = torch.full_like(t, stabilization_timestep)
            t[:, -1] = start
            s[:, -1] = end
            predicted_sequence = self._stage2_forward(current, t, s, actions)
            if self.cfg.last_frame_loss_only:
                current[:, -1:] = predicted_sequence[:, -1:]
            else:
                current = predicted_sequence
        return current

    def _sample_stage2_noise_levels(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample full-window Stage-2 timestep pairs using the configured strategy."""

        return self.noise_scheduler.sample_stage2_noise_levels(
            batch_size=batch_size,
            horizon=horizon,
            device=device,
            sampling_strategy=self.cfg.sampling_strategy,
            prev_frame_noise_scale=self.cfg.prev_frame_noise_scale,
            dyn_infer_steps=self.cfg.dyn_infer_steps,
        )

    def _decoder_training_step_from_latents(
        self,
        flat_images: torch.Tensor,
        latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the Stage-1/Stage-3 decoder denoising objective."""

        t, s = self.noise_scheduler.sample_training_pair(
            flat_images.shape[0],
            flat_images.device,
        )
        noisy_t = self.noise_scheduler.add_noise(flat_images, t)
        target_s = self.noise_scheduler.add_noise(flat_images, s)
        pred_s = self.decode(noisy_t, t, s, latents)
        weights = self.noise_scheduler.get_weights(t).view(-1, 1, 1, 1)
        recon_loss = ((pred_s - target_s) ** 2 * weights).mean()
        clean_pred = self.decode(
            flat_images,
            torch.zeros_like(t),
            torch.zeros_like(s),
            latents,
        )
        clean_loss = F.mse_loss(clean_pred, flat_images)
        loss = recon_loss + 0.1 * clean_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "clean_loss": clean_loss.detach(),
            "latents": latents.detach(),
        }

    def _stage1_training_step(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        """Run the Stage-1 autoencoder objective for one batch."""

        obs, _ = self.concatenate_observations(batch["obs"])  # type: ignore[arg-type]
        flat_images = self._flatten_images(obs)
        latents = self.encode(flat_images)
        return self._decoder_training_step_from_latents(flat_images, latents)

    def _stage2_training_step(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        """Run the Stage-2 latent dynamics objective for one batch."""

        if self.dynamics is None:
            raise RuntimeError("Stage 2 requires an initialized dynamics model.")
        obs, _ = self.concatenate_observations(batch["obs"])  # type: ignore[arg-type]
        if obs.shape[1] < 2:
            raise ValueError("Stage 2 training requires horizon >= 2.")
        actions = self._prepare_stage2_actions(
            torch.as_tensor(batch["action"], device=obs.device),
            obs.shape[1],
        )
        with torch.no_grad():
            latents = self._encode_sequence(obs)
        t, s = self._sample_stage2_noise_levels(
            batch_size=latents.shape[0],
            horizon=latents.shape[1],
            device=obs.device,
        )
        noisy_t, noisy_s = self.noise_scheduler.add_noise_to_t_s(latents, t, s)
        pred_s = self._stage2_forward(noisy_t, t, s, actions)
        weights_t = self.noise_scheduler.get_weights(
            t,
            weighting=self._resolve_stage2_loss_weighting(),
        )
        loss_s = F.mse_loss(pred_s, noisy_s.detach(), reduction="none")
        loss_s = loss_s * self._expand_loss_weights(weights_t, loss_s.ndim)

        loss = loss_s
        pred_latents = pred_s
        if self.cfg.dyn_infer_steps > 1:
            u = torch.zeros_like(s)
            pred_u = self._stage2_forward(noisy_s, s, u, actions)
            weights_s = self.noise_scheduler.get_weights(
                s,
                weighting=self._resolve_stage2_loss_weighting(),
            )
            loss_u = F.mse_loss(pred_u, latents.detach(), reduction="none")
            loss_u = loss_u * self._expand_loss_weights(weights_s, loss_u.ndim)
            loss = loss + loss_u
            pred_latents = pred_u
        if self.cfg.last_frame_loss_only:
            loss = loss[:, -1:]
        dyn_loss = loss.mean()
        return {
            "loss": dyn_loss,
            "dyn_loss": dyn_loss.detach(),
            "pred_latents": pred_latents.detach(),
        }

    def _stage3_training_step(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        """Run the Stage-3 decoder finetuning objective for one batch."""

        obs, _ = self.concatenate_observations(batch["obs"])  # type: ignore[arg-type]
        flat_images = self._flatten_images(obs)
        with torch.no_grad():
            latents = self.encode(flat_images)
        jitter = torch.randn_like(latents) * self.cfg.stage3_latent_noise_std
        noisy_latents = self._normalize_latents(latents + jitter)
        return self._decoder_training_step_from_latents(flat_images, noisy_latents)

    def training_step(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        """Dispatch to the stage-specific training objective."""

        if self.cfg.training_stage == 1:
            return self._stage1_training_step(batch)
        if self.cfg.training_stage == 2:
            return self._stage2_training_step(batch)
        return self._stage3_training_step(batch)

    @torch.no_grad()
    def render_from_latents(
        self,
        latents: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Decode a latent sequence into images by denoising from pure noise."""

        batch_size, horizon, _, height, width = latents.shape
        flat_latents = latents.reshape(-1, latents.shape[2], latents.shape[3], latents.shape[4])
        current = torch.rand(
            batch_size * horizon,
            self.image_channels,
            height * 4,
            width * 4,
            device=latents.device,
            dtype=latents.dtype,
        )
        schedule = self.noise_scheduler.make_sampling_schedule(max(1, num_steps), latents.device)
        for start, end in schedule:
            t = start.expand(current.shape[0])
            s = end.expand(current.shape[0])
            current = self.decode(current, t, s, flat_latents)
        return current.reshape(batch_size, horizon, self.image_channels, current.shape[-2], current.shape[-1]).clamp(0.0, 1.0)

    @torch.no_grad()
    def reconstruct(
        self,
        obs_batch: Mapping[str, torch.Tensor],
        num_steps: int = 4,
        start_mode: str = "noisy-input",
    ) -> torch.Tensor:
        """Reconstruct a batch of observation sequences by iterative denoising."""

        obs, had_batch = self.concatenate_observations(obs_batch)
        batch_size, horizon, channels, height, width = obs.shape
        flat_images = obs.reshape(-1, channels, height, width)
        latents = self.encode(flat_images)
        if start_mode == "noise":
            current = torch.rand_like(flat_images)
        elif start_mode == "noisy-input":
            max_t = torch.full(
                (flat_images.shape[0],),
                self.noise_scheduler.timesteps - 1,
                device=flat_images.device,
                dtype=torch.long,
            )
            current = self.noise_scheduler.add_noise(flat_images, max_t)
        else:
            raise ValueError(f"Unsupported start_mode: {start_mode}")

        schedule = self.noise_scheduler.make_sampling_schedule(num_steps, flat_images.device)
        for start, end in schedule:
            t = start.expand(flat_images.shape[0])
            s = end.expand(flat_images.shape[0])
            current = self.decode(current, t, s, latents)

        reconstructed = current.reshape(batch_size, horizon, channels, height, width)
        if had_batch:
            return reconstructed.clamp(0.0, 1.0)
        return reconstructed.squeeze(0).clamp(0.0, 1.0)

    @torch.no_grad()
    def rollout_latents(
        self,
        initial_latents: torch.Tensor,
        actions: torch.Tensor,
        context_size: int,
    ) -> torch.Tensor:
        """Autoregress future latents with a continuous sliding latent-history window."""

        if self.dynamics is None:
            raise RuntimeError("Latent rollout requires a dynamics model.")
        if actions.shape[1] < initial_latents.shape[1]:
            raise ValueError(
                f"Expected at least {initial_latents.shape[1]} action steps, got {actions.shape[1]}"
            )

        predictions = initial_latents.clone()
        max_context = max(2, context_size)
        future_steps = actions.shape[1] - predictions.shape[1]
        schedule = self.noise_scheduler.make_sampling_schedule(
            max(1, self.cfg.dyn_infer_steps),
            initial_latents.device,
        )
        first_start = schedule[0][0]
        stabilization_timestep = self._stage2_rollout_stabilization_timestep()
        for future_idx in range(future_steps):
            current_end = predictions.shape[1] + 1
            current_start = max(0, current_end - max_context)
            active_predictions = predictions[:, current_start:]
            current = torch.cat(
                [
                    active_predictions,
                    self._make_dynamics_noise_slot(active_predictions[:, :1], first_start),
                ],
                dim=1,
            )
            action_window = self._slice_dynamics_action_window(actions, current_start, current_end)
            current = self._denoise_rollout_window(
                current,
                action_window,
                schedule,
                stabilization_timestep,
            )
            if not self.cfg.last_frame_loss_only:
                predictions[:, current_start:] = current[:, :-1]
            predictions = torch.cat([predictions, current[:, -1:]], dim=1)
        return self._postprocess_rollout_latents(predictions)

    @torch.no_grad()
    def _teacher_forced_rollout_latents(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        context_size: int,
    ) -> torch.Tensor:
        """Predict each next latent from ground-truth context for Stage-2 debugging and tests."""

        prepared_actions = self._prepare_stage2_actions(actions, latents.shape[1])
        schedule = self.noise_scheduler.make_sampling_schedule(
            max(1, self.cfg.dyn_infer_steps),
            latents.device,
        )
        first_start = schedule[0][0]
        stabilization_timestep = self._stage2_rollout_stabilization_timestep()
        predictions = [latents[:, :1]]
        max_context = max(2, context_size)

        for target_idx in range(1, latents.shape[1]):
            current_end = target_idx + 1
            current_start = max(0, current_end - max_context)
            active_context = latents[:, current_start:target_idx]
            current = torch.cat(
                [
                    active_context,
                    self._make_dynamics_noise_slot(active_context[:, :1], first_start),
                ],
                dim=1,
            )
            action_window = self._slice_dynamics_action_window(
                prepared_actions,
                current_start,
                current_end,
            )
            current = self._denoise_rollout_window(
                current,
                action_window,
                schedule,
                stabilization_timestep,
            )
            predictions.append(current[:, -1:])
        return self._postprocess_rollout_latents(torch.cat(predictions, dim=1))

    @torch.no_grad()
    def rollout_validation_episode(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        context_size: int,
    ) -> torch.Tensor:
        """Roll out a full validation episode from the first latent with a sliding context window."""

        if latents.shape[1] != actions.shape[1]:
            raise ValueError(
                f"Expected matching latent/action lengths, got {latents.shape[1]} and {actions.shape[1]}"
            )
        prepared_actions = self._prepare_stage2_actions(actions, latents.shape[1])
        return self.rollout_latents(latents[:, :1], prepared_actions, context_size)

    @torch.no_grad()
    def validation_step(
        self,
        batch: dict[str, object],
        num_steps: int = 4,
        start_mode: str = "noisy-input",
        rollout_context_size: int = 1,
    ) -> dict[str, object]:
        """Produce a Stage-appropriate validation preview and summary stats."""

        obs, _ = self.concatenate_observations(batch["obs"])  # type: ignore[arg-type]
        if self.cfg.training_stage == 1:
            reconstructed = self.reconstruct(
                batch["obs"],  # type: ignore[arg-type]
                num_steps=num_steps,
                start_mode=start_mode,
            )
            if reconstructed.ndim == 4:
                reconstructed = reconstructed.unsqueeze(0)
            latent_shape = list(
                self.encode(obs[:, :1].reshape(-1, obs.shape[2], obs.shape[3], obs.shape[4])).shape
            )
            stats = {
                "episode": int(torch.as_tensor(batch["episode_idx"]).reshape(-1)[0].item()),
                "input_frame_count": int(obs.shape[1]),
                "decoded_frame_count": int(reconstructed.shape[1]),
                "latent_shape": latent_shape,
                "action_shape": list(torch.as_tensor(batch["action"]).shape[-2:]),
                "start_mode": start_mode,
                "num_steps": num_steps,
                "training_stage": self.cfg.training_stage,
            }
            return {
                "original": obs[0].detach().cpu(),
                "reconstructed": reconstructed[0].detach().cpu(),
                "stats": stats,
            }

        actions = self._validate_actions(torch.as_tensor(batch["action"], device=obs.device), obs.shape[1])
        latents = self._encode_sequence(obs)
        rollout_window = max(2, int(rollout_context_size))
        rolled_latents = self.rollout_validation_episode(latents, actions, rollout_window)
        rendered = self.render_from_latents(rolled_latents, num_steps=max(1, num_steps))
        dyn_loss = F.mse_loss(rolled_latents[:, 1:], latents[:, 1:]).item()
        latent_shape = list(rolled_latents[:, :1].reshape(-1, *rolled_latents.shape[2:]).shape)
        stats = {
            "episode": int(torch.as_tensor(batch["episode_idx"]).reshape(-1)[0].item()),
            "input_frame_count": int(obs.shape[1]),
            "predicted_frame_count": int(rolled_latents.shape[1]),
            "decoded_frame_count": int(rendered.shape[1]),
            "dyn_loss": dyn_loss,
            "latent_shape": latent_shape,
            "action_shape": list(actions.shape[-2:]),
            "prediction_mode": "open_loop_rollout",
            "context_frames": 0,
            "seed_frames": 1,
            "num_steps": num_steps,
            "rollout_window": rollout_window,
            "training_stage": self.cfg.training_stage,
        }
        return {
            "original": obs[0].detach().cpu(),
            "reconstructed": rendered[0].detach().cpu(),
            "stats": stats,
        }
