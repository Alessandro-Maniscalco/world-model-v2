"""Rectified-flow transformer dynamics for configurable Wan-latent video clips."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DynamicsFrameLayout:
    """Describe the shared context and target frame layout for dynamics training."""

    context_frames: int = 1
    target_frames: int = 3
    temporal_compression_ratio: int = 4

    @property
    def max_frames(self) -> int:
        """Return the total number of latent frames in one dynamics chunk."""

        return self.context_frames + self.target_frames

    @property
    def num_action_per_chunk(self) -> int:
        """Return the number of transition actions aligned to one frame chunk."""

        return self.temporal_compression_ratio * (self.max_frames - 1)

    def pixel_frames_for_latent_frames(self, latent_frames: int) -> int:
        """Return the Wan-style pixel-frame count for one latent-frame count."""

        if latent_frames < 1:
            raise ValueError("latent_frames must be positive.")
        return 1 + (int(latent_frames) - 1) * self.temporal_compression_ratio

    def latent_frames_for_pixel_frames(self, pixel_frames: int) -> int:
        """Return the Wan-style latent-frame count for one pixel-frame count."""

        if pixel_frames < 1:
            raise ValueError("pixel_frames must be positive.")
        return 1 + (int(pixel_frames) - 1) // self.temporal_compression_ratio

    @property
    def context_pixel_frames(self) -> int:
        """Return the pixel-frame count represented by the context latent frames."""

        return self.pixel_frames_for_latent_frames(self.context_frames)

    @property
    def target_pixel_frames(self) -> int:
        """Return the pixel-frame count represented by the target latent frames."""

        return self.temporal_compression_ratio * self.target_frames

    @property
    def max_pixel_frames(self) -> int:
        """Return the total pixel-frame count represented by one latent chunk."""

        return self.pixel_frames_for_latent_frames(self.max_frames)

    @property
    def conditioning_frame_choices(self) -> tuple[int, ...]:
        """Return the default mixed teacher-conditioning counts for one frame layout."""

        shorter_context = max(1, self.context_frames - self.target_frames)
        if shorter_context == self.context_frames:
            return (self.context_frames,)
        return (shorter_context, self.context_frames)


DYNAMICS_FRAME_LAYOUT = DynamicsFrameLayout()
DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION = "dreamdojo_torch_small_v2_temporal4"


def _normalize_conditioning_frame_choices(
    context_frames: int,
    target_frames: int,
    conditioning_frame_choices: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return validated conditioning counts for the configured total clip length."""

    max_frames = context_frames + target_frames
    if conditioning_frame_choices is None:
        return DynamicsFrameLayout(
            context_frames=context_frames,
            target_frames=target_frames,
        ).conditioning_frame_choices
    normalized = tuple(int(value) for value in conditioning_frame_choices)
    if not normalized:
        raise ValueError("conditioning_frame_choices must contain at least one frame count.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("conditioning_frame_choices must not contain duplicates.")
    for conditioning_frames in normalized:
        if conditioning_frames < 1 or conditioning_frames >= max_frames:
            raise ValueError(
                "conditioning_frame_choices must stay within [1, max_frames - 1]."
            )
    return normalized


def _normalize_conditioning_frame_probabilities(
    conditioning_frame_probabilities: Sequence[float] | None,
    conditioning_frame_choices: tuple[int, ...],
) -> tuple[float, ...] | None:
    """Return validated sampling probabilities for the configured conditioning counts."""

    if conditioning_frame_probabilities is None:
        return None
    normalized = tuple(float(value) for value in conditioning_frame_probabilities)
    if len(normalized) != len(conditioning_frame_choices):
        raise ValueError(
            "conditioning_frame_probabilities must match conditioning_frame_choices in length."
        )
    if any(probability < 0.0 for probability in normalized):
        raise ValueError("conditioning_frame_probabilities must be non-negative.")
    probability_sum = math.fsum(normalized)
    if probability_sum <= 0.0:
        raise ValueError("conditioning_frame_probabilities must sum to a positive value.")
    return tuple(probability / probability_sum for probability in normalized)


def _normalize_validation_conditioning_frame_choices(
    validation_conditioning_frame_choices: Sequence[int] | None,
    conditioning_frame_choices: tuple[int, ...],
) -> tuple[int, ...]:
    """Return validated conditioning counts to report during validation."""

    if validation_conditioning_frame_choices is None:
        return conditioning_frame_choices
    normalized = tuple(int(value) for value in validation_conditioning_frame_choices)
    if not normalized:
        raise ValueError(
            "validation_conditioning_frame_choices must contain at least one frame count."
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("validation_conditioning_frame_choices must not contain duplicates.")
    invalid = [
        conditioning_frames
        for conditioning_frames in normalized
        if conditioning_frames not in conditioning_frame_choices
    ]
    if invalid:
        raise ValueError(
            "validation_conditioning_frame_choices must be drawn from "
            f"{conditioning_frame_choices}, received {invalid}."
        )
    return normalized


@dataclass(frozen=True)
class DynamicsTransformerConfig:
    """Configure the tiny DreamDojo-inspired latent dynamics transformer."""

    max_img_h: int
    max_img_w: int
    context_frames: int = DYNAMICS_FRAME_LAYOUT.context_frames
    target_frames: int = DYNAMICS_FRAME_LAYOUT.target_frames
    max_frames: int | None = None
    conditioning_frame_choices: tuple[int, ...] | None = None
    conditioning_frame_probabilities: tuple[float, ...] | None = None
    validation_conditioning_frame_choices: tuple[int, ...] | None = None
    open_rollout_context_frames: int | None = None
    open_rollout_stride_frames: int | None = None
    temporal_compression_ratio: int = DYNAMICS_FRAME_LAYOUT.temporal_compression_ratio
    in_channels: int = 32
    out_channels: int = 32
    patch_spatial: int = 2
    patch_temporal: int = 1
    model_channels: int = 256
    num_blocks: int = 4
    num_heads: int = 4
    concat_padding_mask: bool = False
    pos_emb_cls: str = "rope3d"
    pos_emb_learnable: bool = False
    pos_emb_interpolation: str = "crop"
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 64
    atten_backend: str = "torch"
    extra_per_block_abs_pos_emb: bool = False
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    use_learned_temporal_embedding: bool = False
    action_dim: int = 4
    action_conditioning_mode: str = "chunk_per_frame"
    zero_init_action_embedder: bool = False
    timestep_scale: float = 1.0
    conditional_frame_timestep: float = -1.0
    conditional_frame_sigma: float = 0.0
    dynamics_infer_steps: int = 20
    dynamics_train_timesteps: int = 1000
    dynamics_rf_shift: float = 5.0
    dynamics_video_condition_dropout: float = 0.0
    dynamics_guidance_scale: float = 0.0

    def __post_init__(self) -> None:
        """Validate the small DiT configuration."""

        resolved_max_frames = self.context_frames + self.target_frames
        if self.context_frames < 1:
            raise ValueError("context_frames must be positive.")
        if self.target_frames < 1:
            raise ValueError("target_frames must be positive.")
        if self.temporal_compression_ratio < 1:
            raise ValueError("temporal_compression_ratio must be positive.")
        if self.max_frames is None:
            object.__setattr__(self, "max_frames", resolved_max_frames)
        elif self.max_frames != resolved_max_frames:
            raise ValueError(
                f"max_frames={self.max_frames} must equal context_frames + target_frames "
                f"({self.context_frames} + {self.target_frames} = {resolved_max_frames})."
            )
        if self.in_channels != self.out_channels:
            raise ValueError("The RF DiT expects matching in/out channel counts.")
        if self.patch_spatial <= 0 or self.patch_temporal <= 0:
            raise ValueError("Patch sizes must be positive.")
        if self.max_img_h % self.patch_spatial != 0 or self.max_img_w % self.patch_spatial != 0:
            raise ValueError("Latent height and width must be divisible by patch_spatial.")
        if self.max_frames % self.patch_temporal != 0:
            raise ValueError("max_frames must be divisible by patch_temporal.")
        if self.model_channels % self.num_heads != 0:
            raise ValueError("model_channels must be divisible by num_heads.")
        if self.action_conditioning_mode != "chunk_per_frame":
            raise ValueError(
                "The DreamDojo-mechanics RF DiT only supports "
                "action_conditioning_mode='chunk_per_frame'."
            )
        if self.pos_emb_cls != "rope3d":
            raise ValueError("The RF DiT only supports pos_emb_cls='rope3d'.")
        if self.pos_emb_learnable:
            raise ValueError("The RF DiT only supports non-learnable RoPE.")
        if self.pos_emb_interpolation != "crop":
            raise ValueError("The RF DiT only supports crop interpolation.")
        if not self.use_adaln_lora:
            raise ValueError("The DreamDojo-mechanics RF DiT requires use_adaln_lora=True.")
        if self.concat_padding_mask:
            raise ValueError("concat_padding_mask is unsupported in the RF DiT.")
        if self.extra_per_block_abs_pos_emb:
            raise ValueError("extra_per_block_abs_pos_emb is unsupported in the RF DiT.")
        if self.atten_backend != "torch":
            raise ValueError("The RF DiT only supports the torch attention backend.")
        if self.use_learned_temporal_embedding:
            raise ValueError(
                "use_learned_temporal_embedding is unsupported in the DreamDojo-mechanics RF DiT."
            )
        if self.dynamics_infer_steps < 1:
            raise ValueError("dynamics_infer_steps must be positive.")
        if self.dynamics_train_timesteps < 2:
            raise ValueError("dynamics_train_timesteps must be at least 2.")
        if self.timestep_scale <= 0.0:
            raise ValueError("timestep_scale must be positive.")
        if self.conditional_frame_timestep < -1.0:
            raise ValueError("conditional_frame_timestep must be -1 or a non-negative value.")
        if not 0.0 <= self.conditional_frame_sigma <= 1.0:
            raise ValueError("conditional_frame_sigma must be between 0 and 1.")
        if not 0.0 <= self.dynamics_video_condition_dropout <= 1.0:
            raise ValueError("dynamics_video_condition_dropout must be between 0 and 1.")
        if self.dynamics_guidance_scale < 0.0:
            raise ValueError("dynamics_guidance_scale must be non-negative.")
        resolved_conditioning_frame_choices = _normalize_conditioning_frame_choices(
            context_frames=self.context_frames,
            target_frames=self.target_frames,
            conditioning_frame_choices=self.conditioning_frame_choices,
        )
        resolved_conditioning_frame_probabilities = _normalize_conditioning_frame_probabilities(
            conditioning_frame_probabilities=self.conditioning_frame_probabilities,
            conditioning_frame_choices=resolved_conditioning_frame_choices,
        )
        resolved_validation_conditioning_frame_choices = (
            _normalize_validation_conditioning_frame_choices(
                validation_conditioning_frame_choices=self.validation_conditioning_frame_choices,
                conditioning_frame_choices=resolved_conditioning_frame_choices,
            )
        )
        resolved_open_rollout_context_frames = (
            self.context_frames
            if self.open_rollout_context_frames is None
            else int(self.open_rollout_context_frames)
        )
        if resolved_open_rollout_context_frames not in resolved_conditioning_frame_choices:
            raise ValueError(
                "open_rollout_context_frames must be drawn from "
                f"{resolved_conditioning_frame_choices}, received "
                f"{resolved_open_rollout_context_frames}."
            )
        resolved_open_rollout_stride_frames = (
            None
            if self.open_rollout_stride_frames is None
            else int(self.open_rollout_stride_frames)
        )
        if (
            resolved_open_rollout_stride_frames is not None
            and (
                resolved_open_rollout_stride_frames < 1
                or resolved_open_rollout_stride_frames >= self.max_frames
            )
        ):
            raise ValueError("open_rollout_stride_frames must stay within [1, max_frames - 1].")
        object.__setattr__(
            self,
            "conditioning_frame_choices",
            resolved_conditioning_frame_choices,
        )
        object.__setattr__(
            self,
            "conditioning_frame_probabilities",
            resolved_conditioning_frame_probabilities,
        )
        object.__setattr__(
            self,
            "validation_conditioning_frame_choices",
            resolved_validation_conditioning_frame_choices,
        )
        object.__setattr__(
            self,
            "open_rollout_context_frames",
            resolved_open_rollout_context_frames,
        )
        object.__setattr__(
            self,
            "open_rollout_stride_frames",
            resolved_open_rollout_stride_frames,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the DiT config."""

        payload = asdict(self)
        payload["conditioning_frame_choices"] = list(self.conditioning_frame_choices)
        payload["conditioning_frame_probabilities"] = (
            None
            if self.conditioning_frame_probabilities is None
            else list(self.conditioning_frame_probabilities)
        )
        payload["validation_conditioning_frame_choices"] = list(
            self.validation_conditioning_frame_choices
        )
        payload["num_action_per_chunk"] = self.num_action_per_chunk
        payload["architecture_version"] = self.architecture_version
        return payload

    @property
    def num_action_per_chunk(self) -> int:
        """Return the DreamDojo-style number of transition actions in one frame chunk."""

        return int(self.temporal_compression_ratio) * (int(self.max_frames) - 1)

    @property
    def architecture_version(self) -> str:
        """Return the checkpointed backbone identifier for the DreamDojo-style DiT."""

        return DREAMDOJO_DYNAMICS_ARCHITECTURE_VERSION


@dataclass(frozen=True)
class DynamicsTrainingInputs:
    """Bundle one rectified-flow training sample for the latent DiT."""

    noisy_latent_video: torch.Tensor
    conditioning_latent_video: torch.Tensor
    target_velocity: torch.Tensor
    timesteps: torch.Tensor
    condition_mask: torch.Tensor
    actions: torch.Tensor
    target_sigmas: torch.Tensor
    num_conditional_frames: torch.Tensor
    use_video_condition: torch.Tensor


def _expand_frame_values(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast one per-batch scalar tensor across a video tensor."""

    while values.ndim < reference.ndim:
        values = values.unsqueeze(-1)
    return values


class RMSNorm(nn.Module):
    """Apply RMS normalization over the last tensor dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create the learnable RMS normalization layer."""

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        """Reset the learnable scale to the DreamDojo default."""

        nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unscaled RMS-normalized activations."""

        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension and rescale it."""

        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Timesteps(nn.Module):
    """Project scalar timesteps into sinusoidal embeddings."""

    def __init__(self, dim: int) -> None:
        """Store the requested embedding width."""

        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Return sinusoidal embeddings for a `(B, T)` timestep tensor."""

        if timesteps.ndim != 2:
            raise ValueError(f"Expected timesteps with shape (B, T), received {tuple(timesteps.shape)}.")
        in_dtype = timesteps.dtype
        flat = timesteps.flatten().float()
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            dtype=torch.float32,
            device=timesteps.device,
        )
        exponent = exponent / (half_dim - 0.0)
        embedding = torch.exp(exponent)
        embedding = flat[:, None] * embedding[None, :]
        embedding = torch.cat([torch.cos(embedding), torch.sin(embedding)], dim=-1)
        return rearrange(
            embedding.to(dtype=in_dtype),
            "(b t) d -> b t d",
            b=timesteps.shape[0],
            t=timesteps.shape[1],
        )


class TimestepEmbedding(nn.Module):
    """Convert sinusoidal timestep features into AdaLN conditioning vectors."""

    def __init__(self, in_dim: int, out_dim: int, use_adaln_lora: bool = False) -> None:
        """Build the timestep projection MLP."""

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_adaln_lora = use_adaln_lora
        self.fc1 = nn.Linear(in_dim, out_dim, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        if use_adaln_lora:
            self.fc2 = nn.Linear(out_dim, out_dim * 3, bias=False)
        else:
            self.fc2 = nn.Linear(out_dim, out_dim, bias=False)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the timestep embedding weights."""

        std = 1.0 / math.sqrt(self.in_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        nn.init.trunc_normal_(self.fc2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the timestep embedding and optional AdaLN-LoRA residual."""

        hidden = self.fc1(embeddings)
        hidden = self.activation(hidden)
        hidden = self.fc2(hidden)
        if self.use_adaln_lora:
            return embeddings, hidden
        return hidden, None


class Mlp(nn.Module):
    """Apply the DreamDojo action MLP used before AdaLN modulation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | None = None,
    ) -> None:
        """Create the two-layer action embedder."""

        super().__init__()
        resolved_hidden_features = (
            out_features * 4
            if hidden_features is None
            else int(hidden_features)
        )
        if resolved_hidden_features < 1:
            raise ValueError("hidden_features must be positive.")
        self.fc1 = nn.Linear(in_features, resolved_hidden_features)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(resolved_hidden_features, out_features)
        self.drop = nn.Dropout(0.0)
        self.hidden_features = resolved_hidden_features

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Return one action-conditioning embedding for each batch item."""

        hidden = self.fc1(actions)
        hidden = self.activation(hidden)
        hidden = self.drop(hidden)
        hidden = self.fc2(hidden)
        hidden = self.drop(hidden)
        return hidden

    def zero_output_projection(self) -> None:
        """Match DreamDojo's optional zero-init on the action output projection."""

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


class PatchEmbed(nn.Module):
    """Project spatio-temporal latent patches into transformer tokens."""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Create the patch embedding projection."""

        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        patch_dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(patch_dim, out_channels, bias=False),
        )
        self.patch_dim = patch_dim
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the patch embedding projection."""

        std = 1.0 / math.sqrt(self.patch_dim)
        nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings with shape `(B, T, H, W, D)`."""

        if video.ndim != 5:
            raise ValueError(f"Expected a 5D latent video, received {tuple(video.shape)}.")
        _, _, frames, height, width = video.shape
        if height % self.spatial_patch_size != 0 or width % self.spatial_patch_size != 0:
            raise ValueError("Video height and width must be divisible by the spatial patch size.")
        if frames % self.temporal_patch_size != 0:
            raise ValueError("Video frames must be divisible by the temporal patch size.")
        return self.proj(video)


class VideoRopePosition3DEmb(nn.Module):
    """Generate 3D rotary embeddings over temporal and spatial token grids."""

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float,
        w_extrapolation_ratio: float,
        t_extrapolation_ratio: float,
    ) -> None:
        """Store the maximum token-grid sizes and RoPE scaling factors."""

        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float32))
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim_h = head_dim // 6 * 2
        dim_w = dim_h
        dim_t = head_dim - 2 * dim_h
        if head_dim != dim_h + dim_w + dim_t:
            raise ValueError(f"bad dim: {head_dim} != {dim_h} + {dim_w} + {dim_t}")
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, dtype=torch.float32)[: (dim_h // 2)] / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, dtype=torch.float32)[: (dim_t // 2)] / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Refresh the cached RoPE sequences and frequency ranges."""

        dim_h = self._dim_h
        dim_t = self._dim_t
        self.seq = torch.arange(
            max(self.max_h, self.max_w, self.max_t),
            dtype=torch.float32,
            device=self.dim_spatial_range.device,
        )
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2, dtype=torch.float32, device=self.dim_spatial_range.device)[: (dim_h // 2)]
            / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2, dtype=torch.float32, device=self.dim_spatial_range.device)[: (dim_t // 2)]
            / dim_t
        )

    def forward(self, x_B_T_H_W_D: torch.Tensor) -> torch.Tensor:
        """Return DreamDojo-style RoPE frequencies for one embedded token grid."""

        if x_B_T_H_W_D.ndim != 5:
            raise ValueError(
                f"Expected embedded tokens with shape (B, T, H, W, D), received {tuple(x_B_T_H_W_D.shape)}."
            )
        _, frames, height, width, _ = x_B_T_H_W_D.shape
        if height > self.max_h or width > self.max_w or frames > self.max_t:
            raise ValueError("Requested token grid exceeds the configured RoPE capacity.")
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor
        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())
        half_emb_h = torch.outer(self.seq[:height], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:width], w_spatial_freqs)
        half_emb_t = torch.outer(self.seq[:frames], temporal_freqs)
        embedding = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=height, w=width),
                repeat(half_emb_h, "h d -> t h w d", t=frames, w=width),
                repeat(half_emb_w, "w d -> t h w d", t=frames, h=height),
            ]
            * 2,
            dim=-1,
        )
        return rearrange(embedding, "t h w d -> (t h w) 1 1 d").float()


def apply_rotary_position_embedding(
    x: torch.Tensor,
    rope_emb: torch.Tensor,
) -> torch.Tensor:
    """Rotate one query or key tensor with DreamDojo-style RoPE frequencies."""

    if x.ndim != 4:
        raise ValueError(f"Expected x with shape (B, S, H, D), received {tuple(x.shape)}.")
    if rope_emb.ndim != 4:
        raise ValueError(
            f"Expected rope_emb with shape (S, 1, 1, D) or (1, S, 1, D), received {tuple(rope_emb.shape)}."
        )
    if rope_emb.shape[0] == x.shape[1]:
        rope_emb = rearrange(rope_emb, "s one two d -> one s two d")
    elif rope_emb.shape[1] != x.shape[1]:
        raise ValueError(
            f"RoPE sequence length {rope_emb.shape[:2]} is incompatible with token sequence length {x.shape[1]}."
        )
    cos = rope_emb.cos().to(dtype=x.dtype)
    sin = rope_emb.sin().to(dtype=x.dtype)
    half_dim = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)
    return x * cos + rotated * sin


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Apply scaled dot-product attention to `[B, S, H, D]` tensors."""

    q_B_H_S_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_S_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_S_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    result_B_H_S_D = F.scaled_dot_product_attention(
        q_B_H_S_D,
        k_B_H_S_D,
        v_B_H_S_D,
        attn_mask=attn_mask,
    )
    if flatten_heads:
        return rearrange(result_B_H_S_D, "b h s d -> b s (h d)")
    return rearrange(result_B_H_S_D, "b h s d -> b s h d")


class Attention(nn.Module):
    """Apply DreamDojo-style self-attention over a flattened token sequence."""

    def __init__(self, query_dim: int, n_heads: int, head_dim: int) -> None:
        """Create the self-attention projections and RMS norms."""

        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.inner_dim = head_dim * n_heads
        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(self.inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Identity()
        self.attn_op = torch_attention_op

    def init_weights(self) -> None:
        """Initialize the attention projections and norm scales."""

        std = 1.0 / math.sqrt(self.query_dim)
        nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.inner_dim)
        nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        rope_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one token sequence into normalized Q/K/V tensors."""

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query, key, value = map(
            lambda tensor: rearrange(
                tensor,
                "b s (h d) -> b s h d",
                h=self.n_heads,
                d=self.head_dim,
            ),
            (query, key, value),
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_norm(value)
        if rope_emb is not None:
            query = apply_rotary_position_embedding(query, rope_emb)
            key = apply_rotary_position_embedding(key, rope_emb)
        return query, key, value

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention and project the concatenated heads back to model space."""

        attended = self.attn_op(query, key, value)
        return self.output_dropout(self.output_proj(attended))

    def forward(
        self,
        x: torch.Tensor,
        rope_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the attention-updated token sequence."""

        query, key, value = self.compute_qkv(x, rope_emb=rope_emb)
        return self.compute_attention(query, key, value)


class GPT2FeedForward(nn.Module):
    """Apply the DreamDojo two-layer GELU feed-forward network."""

    def __init__(self, model_channels: int, mlp_ratio: float = 4.0) -> None:
        """Create the two-layer feed-forward network."""

        super().__init__()
        hidden_channels = int(model_channels * mlp_ratio)
        self.fc1 = nn.Linear(model_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, model_channels, bias=False)
        self.activation = nn.GELU()
        self.model_channels = model_channels
        self.hidden_channels = hidden_channels

    def init_weights(self) -> None:
        """Initialize the feed-forward layers."""

        std = 1.0 / math.sqrt(self.model_channels)
        nn.init.trunc_normal_(self.fc1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.hidden_channels)
        nn.init.trunc_normal_(self.fc2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the MLP update for the token sequence."""

        return self.fc2(self.activation(self.fc1(x)))


class Block(nn.Module):
    """Apply one DreamDojo-style AdaLN self-attention plus MLP block."""

    def __init__(
        self,
        model_channels: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
    ) -> None:
        """Create the attention, MLP, and modulation paths."""

        super().__init__()
        self.model_channels = model_channels
        self.layer_norm_self_attn = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            model_channels,
            n_heads=num_heads,
            head_dim=model_channels // num_heads,
        )
        self.layer_norm_mlp = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(model_channels, mlp_ratio=mlp_ratio)
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * model_channels, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * model_channels, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 3 * model_channels, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 3 * model_channels, bias=False),
            )

    def reset_parameters(self) -> None:
        """Initialize the AdaLN modulation heads."""

        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.model_channels)
            nn.init.trunc_normal_(
                self.adaln_modulation_self_attn[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            nn.init.trunc_normal_(
                self.adaln_modulation_mlp[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        """Initialize modulation heads plus nested attention and MLP weights."""

        self.reset_parameters()
        self.self_attn.init_weights()
        self.mlp.init_weights()

    def _broadcast(self, values: torch.Tensor) -> torch.Tensor:
        """Expand one `(B, T, D)` tensor over the spatial token grid."""

        return rearrange(values, "b t d -> b t 1 1 d")

    def _modulate(
        self,
        x: torch.Tensor,
        norm: nn.LayerNorm,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Apply AdaLN modulation before one transformer sublayer."""

        return norm(x) * (1.0 + self._broadcast(scale)) + self._broadcast(shift)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        rope_emb: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the transformed token grid for one DiT block."""

        _, frames, height, width, _ = x.shape
        attn_modulation = self.adaln_modulation_self_attn(timestep_embedding)
        if adaln_lora is not None:
            attn_modulation = attn_modulation + adaln_lora
        shift_attn, scale_attn, gate_attn = attn_modulation.chunk(3, dim=-1)
        shift_attn = shift_attn.type_as(x)
        scale_attn = scale_attn.type_as(x)
        gate_attn = gate_attn.type_as(x)
        attn_input = self._modulate(x, self.layer_norm_self_attn, shift_attn, scale_attn)
        attn_input = rearrange(attn_input, "b t h w d -> b (t h w) d")
        attn_output = self.self_attn(attn_input, rope_emb=rope_emb)
        attn_output = rearrange(attn_output, "b (t h w) d -> b t h w d", t=frames, h=height, w=width)
        x = x + self._broadcast(gate_attn) * attn_output

        mlp_modulation = self.adaln_modulation_mlp(timestep_embedding)
        if adaln_lora is not None:
            mlp_modulation = mlp_modulation + adaln_lora
        shift_mlp, scale_mlp, gate_mlp = mlp_modulation.chunk(3, dim=-1)
        shift_mlp = shift_mlp.type_as(x)
        scale_mlp = scale_mlp.type_as(x)
        gate_mlp = gate_mlp.type_as(x)
        mlp_input = self._modulate(x, self.layer_norm_mlp, shift_mlp, scale_mlp)
        mlp_output = self.mlp(mlp_input)
        return x + self._broadcast(gate_mlp) * mlp_output


class FinalLayer(nn.Module):
    """Project transformer tokens back into latent-space patches."""

    def __init__(
        self,
        model_channels: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
    ) -> None:
        """Create the final AdaLN projection to the latent patch space."""

        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.layer_norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        patch_dim = spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels
        self.linear = nn.Linear(model_channels, patch_dim, bias=False)
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, model_channels * 2, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, model_channels * 2, bias=False),
            )

    def init_weights(self) -> None:
        """Initialize the final AdaLN projection."""

        std = 1.0 / math.sqrt(self.model_channels)
        nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            nn.init.trunc_normal_(
                self.adaln_modulation[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            nn.init.zeros_(self.adaln_modulation[1].weight)
        self.layer_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_lora: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-patch latent predictions."""

        if self.use_adaln_lora:
            if adaln_lora is None:
                raise ValueError("adaln_lora must be provided when use_adaln_lora=True.")
            modulation = self.adaln_modulation(timestep_embedding) + adaln_lora[:, :, : 2 * self.model_channels]
        else:
            modulation = self.adaln_modulation(timestep_embedding)
        shift, scale = modulation.chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d").type_as(x)
        scale = rearrange(scale, "b t d -> b t 1 1 d").type_as(x)
        x = self.layer_norm(x) * (1.0 + scale) + shift
        return self.linear(x)


class RectifiedFlowHelper:
    """Sample rectified-flow training noise levels and inference schedules."""

    def __init__(self, num_train_timesteps: int, shift: float) -> None:
        """Create the flow-matching training scheduler."""

        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.training_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

    def sample_training_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one uniform rectified-flow training time per batch item."""

        return torch.rand((batch_size,), device=device, dtype=dtype)

    def get_discrete_timestamps(
        self,
        train_time: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map one normalized training-time sample to scheduler timesteps."""

        indices = (train_time * self.training_scheduler.config.num_train_timesteps).long()
        indices = indices.clamp_max(self.training_scheduler.timesteps.shape[0] - 1)
        return self.training_scheduler.timesteps.to(device=device, dtype=dtype)[indices]

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Look up one sigma value for each discrete scheduler timestep."""

        schedule_timesteps = self.training_scheduler.timesteps.to(device=device, dtype=dtype)
        schedule_sigmas = self.training_scheduler.sigmas.to(device=device, dtype=dtype)
        flat_timesteps = timesteps.reshape(-1)
        step_indices = [(schedule_timesteps == timestep).nonzero().reshape(-1)[0] for timestep in flat_timesteps]
        gathered = schedule_sigmas[torch.stack(step_indices)]
        return gathered.view_as(timesteps)

    def interpolate(
        self,
        noise: torch.Tensor,
        clean: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate between Gaussian noise and clean latents."""

        sigma = _expand_frame_values(sigmas, clean)
        noisy = noise * sigma + clean * (1.0 - sigma)
        velocity = noise - clean
        return noisy, velocity

    def make_inference_schedule(
        self,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the timestep and sigma schedule for iterative RF sampling."""

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=self.shift,
        )
        scheduler.set_timesteps(num_steps, device=device)
        return (
            scheduler.timesteps.to(device=device, dtype=dtype),
            scheduler.sigmas.to(device=device, dtype=dtype),
        )

    def step(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma_from: torch.Tensor,
        sigma_to: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one latent sample from `sigma_from` to `sigma_to`."""

        delta = sigma_to - sigma_from
        return sample + _expand_frame_values(delta, sample) * velocity


class ActionConditionedDynamicsTransformer(nn.Module):
    """Run the tiny short-clip latent DiT with DreamDojo-style action-chunk conditioning."""

    def __init__(self, cfg: DynamicsTransformerConfig) -> None:
        """Build the patch embedder, RoPE, timestep path, and DiT blocks."""

        super().__init__()
        self.cfg = cfg
        self._num_action_per_latent_frame = cfg.temporal_compression_ratio
        self.x_embedder = PatchEmbed(
            spatial_patch_size=cfg.patch_spatial,
            temporal_patch_size=cfg.patch_temporal,
            in_channels=cfg.in_channels + 1,
            out_channels=cfg.model_channels,
        )
        self.build_pos_embed()
        self.t_embedder = nn.Sequential(
            Timesteps(cfg.model_channels),
            TimestepEmbedding(
                cfg.model_channels,
                cfg.model_channels,
                use_adaln_lora=cfg.use_adaln_lora,
            ),
        )
        self.t_embedding_norm = RMSNorm(cfg.model_channels, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.model_channels,
                    cfg.num_heads,
                    mlp_ratio=4.0,
                    use_adaln_lora=cfg.use_adaln_lora,
                    adaln_lora_dim=cfg.adaln_lora_dim,
                )
                for _ in range(cfg.num_blocks)
            ]
        )
        self.final_layer = FinalLayer(
            model_channels=cfg.model_channels,
            spatial_patch_size=cfg.patch_spatial,
            temporal_patch_size=cfg.patch_temporal,
            out_channels=cfg.out_channels,
            use_adaln_lora=cfg.use_adaln_lora,
            adaln_lora_dim=cfg.adaln_lora_dim,
        )
        action_embedder_in_features = cfg.action_dim * self._num_action_per_latent_frame
        action_embedder_hidden_features = cfg.model_channels * 4
        self.action_embedder_B_D = Mlp(
            action_embedder_in_features,
            cfg.model_channels,
            hidden_features=action_embedder_hidden_features,
        )
        self.action_embedder_B_3D = Mlp(
            action_embedder_in_features,
            cfg.model_channels * 3,
            hidden_features=action_embedder_hidden_features,
        )
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the DreamDojo-style DiT core and optional zero-init action heads."""

        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()
        if self.cfg.zero_init_action_embedder:
            self.action_embedder_B_D.zero_output_projection()
            self.action_embedder_B_3D.zero_output_projection()

    def build_pos_embed(self) -> None:
        """Create the RoPE embedder used by the DreamDojo-style DiT core."""

        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=self.cfg.model_channels // self.cfg.num_heads,
            len_h=self.cfg.max_img_h // self.cfg.patch_spatial,
            len_w=self.cfg.max_img_w // self.cfg.patch_spatial,
            len_t=self.cfg.max_frames // self.cfg.patch_temporal,
            h_extrapolation_ratio=self.cfg.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.cfg.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.cfg.rope_t_extrapolation_ratio,
        )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patchify the latent video and build its RoPE frequencies."""

        tokens = self.x_embedder(x_B_C_T_H_W)
        return tokens, self.pos_embedder(tokens).to(tokens.device)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Restore a patch tensor to `(B, C, T, H, W)` latent-video layout."""

        return rearrange(
            patches,
            "b t h w (p1 p2 r c) -> b c (t r) (h p1) (w p2)",
            p1=self.cfg.patch_spatial,
            p2=self.cfg.patch_spatial,
            r=self.cfg.patch_temporal,
            c=self.cfg.out_channels,
        )

    def _embed_actions(
        self,
        action: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return DreamDojo-style action embeddings for the configured conditioning mode."""

        action = action.to(dtype=dtype)
        num_actions = int(action.shape[1])
        flattened_action = rearrange(action, "b t d -> b 1 (t d)")
        action = rearrange(
            flattened_action,
            "b 1 (t d) -> b t d",
            t=num_actions // self._num_action_per_latent_frame,
        )
        action_emb_B_D = self.action_embedder_B_D(action)
        action_emb_B_3D = self.action_embedder_B_3D(action)
        zero_pad_action_emb_B_D = torch.zeros_like(action_emb_B_D[:, :1, :], device=action_emb_B_D.device)
        zero_pad_action_emb_B_3D = torch.zeros_like(
            action_emb_B_3D[:, :1, :],
            device=action_emb_B_3D.device,
        )
        return (
            torch.cat([zero_pad_action_emb_B_D, action_emb_B_D], dim=1),
            torch.cat([zero_pad_action_emb_B_3D, action_emb_B_3D], dim=1),
        )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the rectified-flow velocity over one short latent clip."""

        batch_size, _, frames, height, width = x_B_C_T_H_W.shape
        if frames != self.cfg.max_frames:
            raise ValueError(f"Expected {self.cfg.max_frames} frames, received {frames}.")
        if action is None:
            action = torch.zeros(
                batch_size,
                self.cfg.num_action_per_chunk,
                self.cfg.action_dim,
                device=x_B_C_T_H_W.device,
                dtype=x_B_C_T_H_W.dtype,
            )
        if action.ndim != 3:
            raise ValueError(
                f"Expected actions with shape (B, {self.cfg.num_action_per_chunk}, {self.cfg.action_dim}), "
                f"received {tuple(action.shape)}."
            )
        if action.shape[0] != batch_size:
            raise ValueError(f"Expected action batch size {batch_size}, received {action.shape[0]}.")
        if action.shape[1] != self.cfg.num_action_per_chunk:
            raise ValueError(
                f"Expected {self.cfg.num_action_per_chunk} action steps per chunk, "
                f"received {action.shape[1]}."
            )
        if action.shape[2] != self.cfg.action_dim:
            raise ValueError(
                f"Expected action dim {self.cfg.action_dim}, received {action.shape[2]}."
            )
        if condition_video_input_mask_B_C_T_H_W is None:
            condition_video_input_mask_B_C_T_H_W = torch.zeros(
                batch_size,
                1,
                frames,
                height,
                width,
                device=x_B_C_T_H_W.device,
                dtype=x_B_C_T_H_W.dtype,
            )
        model_input = torch.cat(
            [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.to(dtype=x_B_C_T_H_W.dtype)],
            dim=1,
        )
        timesteps_B_T = timesteps_B_T * self.cfg.timestep_scale
        tokens, rope_emb = self.prepare_embedded_sequence(model_input)
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timestep_embedding, adaln_lora = self.t_embedder(timesteps_B_T)
        if adaln_lora is None:
            raise ValueError("The DreamDojo-mechanics RF DiT requires AdaLN-LoRA conditioning.")
        action_emb_B_D, action_emb_B_3D = self._embed_actions(action, dtype=tokens.dtype)
        timestep_embedding = timestep_embedding + action_emb_B_D
        adaln_lora = adaln_lora + action_emb_B_3D
        timestep_embedding = self.t_embedding_norm(timestep_embedding)
        for block in self.blocks:
            tokens = block(tokens, timestep_embedding, rope_emb, adaln_lora=adaln_lora)
        patches = self.final_layer(tokens, timestep_embedding, adaln_lora=adaln_lora)
        return self.unpatchify(patches)


class RectifiedFlowDynamics(nn.Module):
    """Wrap the tiny DiT with RF training-input preparation and sampling helpers."""

    def __init__(self, cfg: DynamicsTransformerConfig) -> None:
        """Build the DiT backbone and the flow scheduler helper."""

        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.net = ActionConditionedDynamicsTransformer(cfg)
        self.flow = RectifiedFlowHelper(
            num_train_timesteps=cfg.dynamics_train_timesteps,
            shift=cfg.dynamics_rf_shift,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable dynamics configuration."""

        return self.cfg.to_dict()

    def _normalize_num_conditional_frames(
        self,
        num_conditional_frames: int | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        *,
        allow_unregistered: bool = False,
    ) -> torch.Tensor:
        """Validate one conditioning-frame count or one count per batch item."""

        if num_conditional_frames is None:
            choice_tensor = torch.as_tensor(
                self.cfg.conditioning_frame_choices,
                device=device,
                dtype=torch.long,
            )
            if self.cfg.conditioning_frame_probabilities is None:
                choice_indices = torch.randint(
                    low=0,
                    high=int(choice_tensor.shape[0]),
                    size=(batch_size,),
                    device=device,
                )
            else:
                probability_tensor = torch.as_tensor(
                    self.cfg.conditioning_frame_probabilities,
                    device=device,
                    dtype=torch.float32,
                )
                choice_indices = torch.multinomial(
                    probability_tensor,
                    num_samples=batch_size,
                    replacement=True,
                )
            return choice_tensor[choice_indices]
        if isinstance(num_conditional_frames, int):
            counts = torch.full((batch_size,), num_conditional_frames, device=device, dtype=torch.long)
        else:
            counts = num_conditional_frames.to(device=device, dtype=torch.long)
            if counts.ndim == 0:
                counts = counts.expand(batch_size)
            if counts.ndim != 1 or counts.shape[0] != batch_size:
                raise ValueError(
                    f"Expected num_conditional_frames with shape ({batch_size},), "
                    f"received {tuple(counts.shape)}."
                )
        if allow_unregistered:
            valid_choices = set(range(1, self.cfg.max_frames))
        else:
            valid_choices = set(self.cfg.conditioning_frame_choices)
        invalid = [int(count.item()) for count in counts if int(count.item()) not in valid_choices]
        if invalid:
            expected = (
                f"[1, {self.cfg.max_frames - 1}]"
                if allow_unregistered
                else f"{self.cfg.conditioning_frame_choices}"
            )
            raise ValueError(
                f"Unsupported conditioning frame counts {invalid}; expected choices from "
                f"{expected}."
            )
        return counts

    def make_condition_mask(
        self,
        latents: torch.Tensor,
        num_conditional_frames: int | torch.Tensor | None = None,
        *,
        allow_unregistered: bool = False,
    ) -> torch.Tensor:
        """Return a mask with the selected conditioning frames marked as known context."""

        if latents.ndim != 5:
            raise ValueError(f"Expected latent video shape (B, C, T, H, W), received {tuple(latents.shape)}.")
        batch_size, _, frames, height, width = latents.shape
        if frames != self.cfg.max_frames:
            raise ValueError(f"Expected {self.cfg.max_frames} latent frames, received {frames}.")
        counts = self._normalize_num_conditional_frames(
            num_conditional_frames,
            batch_size=batch_size,
            device=latents.device,
            allow_unregistered=allow_unregistered,
        )
        frame_indices = torch.arange(frames, device=latents.device, dtype=torch.long).view(1, 1, frames, 1, 1)
        mask = (frame_indices < counts.view(batch_size, 1, 1, 1, 1)).to(dtype=latents.dtype)
        mask = mask.expand(batch_size, 1, frames, height, width)
        return mask

    def recover_clean_latent_video(
        self,
        noisy_latent_video: torch.Tensor,
        predicted_velocity: torch.Tensor,
        target_sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Convert one RF velocity prediction back into a clean latent video estimate."""

        if noisy_latent_video.shape != predicted_velocity.shape:
            raise ValueError(
                "noisy_latent_video and predicted_velocity must share the same shape."
            )
        if target_sigmas.ndim != 1 or target_sigmas.shape[0] != noisy_latent_video.shape[0]:
            raise ValueError(
                "target_sigmas must have shape (B,) matching the latent batch size."
            )
        sigma = _expand_frame_values(target_sigmas, noisy_latent_video)
        return noisy_latent_video - sigma * predicted_velocity

    def make_zero_actions(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return one zero-valued DreamDojo-style action chunk."""

        return torch.zeros(
            batch_size,
            self.cfg.num_action_per_chunk,
            self.action_dim,
            device=device,
            dtype=dtype,
        )

    def _prepare_actions(
        self,
        actions: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Validate and cast one action chunk tensor for the current dynamics config."""

        if actions is None:
            return self.make_zero_actions(batch_size, device, dtype)
        if actions.ndim != 3:
            raise ValueError(
                f"Expected actions with shape (B, {self.cfg.num_action_per_chunk}, {self.action_dim}), "
                f"received {tuple(actions.shape)}."
            )
        if actions.shape[0] != batch_size:
            raise ValueError(f"Expected action batch size {batch_size}, received {actions.shape[0]}.")
        if actions.shape[1] != self.cfg.num_action_per_chunk:
            raise ValueError(
                f"Expected {self.cfg.num_action_per_chunk} action steps per chunk, "
                f"received {actions.shape[1]}."
            )
        if actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, received {actions.shape[2]}."
            )
        return actions.to(device=device, dtype=dtype)

    def _normalize_use_video_condition(
        self,
        use_video_condition: bool | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return one DreamDojo-style video-conditioning flag per batch item."""

        if use_video_condition is None:
            keep_probability = 1.0 - self.cfg.dynamics_video_condition_dropout
            return torch.rand(batch_size, device=device) < keep_probability
        if isinstance(use_video_condition, bool):
            return torch.full((batch_size,), use_video_condition, device=device, dtype=torch.bool)
        flags = use_video_condition.to(device=device, dtype=torch.bool)
        if flags.ndim == 0:
            flags = flags.expand(batch_size)
        if flags.ndim != 1 or flags.shape[0] != batch_size:
            raise ValueError(
                f"Expected use_video_condition with shape ({batch_size},), "
                f"received {tuple(flags.shape)}."
            )
        return flags

    def _apply_video_condition_dropout(
        self,
        conditioning_latent_video: torch.Tensor,
        use_video_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Port DreamDojo's dropped-video-conditioning behavior by zeroing pinned inputs."""

        video_condition_scale = use_video_condition.to(
            device=conditioning_latent_video.device,
            dtype=conditioning_latent_video.dtype,
        ).view(-1, 1, 1, 1, 1)
        return conditioning_latent_video * video_condition_scale

    def _apply_conditional_frame_sigma(
        self,
        conditioning_latent_video: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        target_velocity: torch.Tensor | None = None,
        reference_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add optional DreamDojo-style tiny noise to the conditioned latent prefix."""

        conditioning_sigma = self.cfg.conditional_frame_sigma
        if conditioning_sigma <= 0.0:
            return conditioning_latent_video
        if target_velocity is None and reference_noise is None:
            return conditioning_latent_video
        if target_velocity is not None and reference_noise is not None:
            raise ValueError("Provide either target_velocity or reference_noise, not both.")
        if target_velocity is None:
            target_velocity = reference_noise - conditioning_latent_video
        condition_video_mask = self._expand_condition_mask_to_channels(
            condition_mask,
            conditioning_latent_video.shape[1],
        ).to(dtype=conditioning_latent_video.dtype)
        return conditioning_latent_video + conditioning_sigma * target_velocity * condition_video_mask

    def _prepare_conditioning_state(
        self,
        conditioning_latent_video: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        target_velocity: torch.Tensor | None = None,
        reference_noise: torch.Tensor | None = None,
        use_video_condition: bool | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the repinned conditioning state with optional tiny sigma and dropout."""

        conditioning_state = self._apply_conditional_frame_sigma(
            conditioning_latent_video,
            condition_mask,
            target_velocity=target_velocity,
            reference_noise=reference_noise,
        )
        conditioning_flags = self._normalize_use_video_condition(
            use_video_condition,
            batch_size=conditioning_latent_video.shape[0],
            device=conditioning_latent_video.device,
        )
        conditioning_state = self._apply_video_condition_dropout(
            conditioning_state,
            conditioning_flags,
        )
        return conditioning_state, conditioning_flags

    def _apply_conditional_frame_timestep(
        self,
        timesteps: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Port DreamDojo's conditional-frame timestep override into the teacher path."""

        if self.cfg.conditional_frame_timestep < 0.0:
            return timesteps
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        condition_mask_B_T = condition_mask[:, 0, :, 0, 0].to(dtype=timesteps.dtype)
        timestep_cond = torch.full_like(timesteps, self.cfg.conditional_frame_timestep)
        return timestep_cond * condition_mask_B_T + timesteps * (1.0 - condition_mask_B_T)

    def _expand_condition_mask_to_channels(self, condition_mask: torch.Tensor, channels: int) -> torch.Tensor:
        """Broadcast the conditioning mask across every latent channel."""

        if condition_mask.ndim != 5:
            raise ValueError(
                "condition_mask must have shape (B, 1, T, H, W) before channel expansion."
            )
        return condition_mask.expand(-1, channels, -1, -1, -1)

    def _repin_conditioned_frames(
        self,
        noisy_latent_video: torch.Tensor,
        conditioning_latent_video: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace conditioned frames in `x_t` with their clean latent values."""

        condition_video_mask = self._expand_condition_mask_to_channels(
            condition_mask,
            noisy_latent_video.shape[1],
        ).to(dtype=noisy_latent_video.dtype)
        return conditioning_latent_video * condition_video_mask + noisy_latent_video * (1.0 - condition_video_mask)

    def _overwrite_conditioned_velocity(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Overwrite conditioned-frame velocity with the exact rectified-flow target."""

        condition_video_mask = self._expand_condition_mask_to_channels(
            condition_mask,
            predicted_velocity.shape[1],
        ).to(dtype=predicted_velocity.dtype)
        return target_velocity * condition_video_mask + predicted_velocity * (1.0 - condition_video_mask)

    def prepare_training_inputs(
        self,
        clean_latent_video: torch.Tensor,
        actions: torch.Tensor | None = None,
        num_conditional_frames: int | torch.Tensor | None = None,
        use_video_condition: bool | torch.Tensor | None = None,
    ) -> DynamicsTrainingInputs:
        """Build one full-clip RF teacher-training sample."""

        if clean_latent_video.ndim != 5:
            raise ValueError("clean_latent_video must have shape (B, C, T, H, W).")
        if clean_latent_video.shape[2] != self.cfg.max_frames:
            raise ValueError(
                f"Expected {self.cfg.max_frames} latent frames, received {clean_latent_video.shape[2]}."
            )
        batch_size = clean_latent_video.shape[0]
        train_time = self.flow.sample_training_time(
            batch_size=batch_size,
            device=clean_latent_video.device,
            dtype=clean_latent_video.dtype,
        )
        timesteps = self.flow.get_discrete_timestamps(
            train_time=train_time,
            device=clean_latent_video.device,
            dtype=clean_latent_video.dtype,
        )
        sigmas = self.flow.get_sigmas(
            timesteps=timesteps,
            device=clean_latent_video.device,
            dtype=clean_latent_video.dtype,
        )
        reference_noise = torch.randn_like(clean_latent_video)
        noisy_latent_video, target_velocity = self.flow.interpolate(
            noise=reference_noise,
            clean=clean_latent_video,
            sigmas=sigmas,
        )
        full_timesteps = timesteps.unsqueeze(1).expand(-1, self.cfg.max_frames)
        prepared_actions = self._prepare_actions(
            actions,
            batch_size=batch_size,
            device=clean_latent_video.device,
            dtype=clean_latent_video.dtype,
        )
        conditioning_counts = self._normalize_num_conditional_frames(
            num_conditional_frames,
            batch_size=batch_size,
            device=clean_latent_video.device,
        )
        conditioning_flags = self._normalize_use_video_condition(
            use_video_condition,
            batch_size=batch_size,
            device=clean_latent_video.device,
        )
        return DynamicsTrainingInputs(
            noisy_latent_video=noisy_latent_video,
            conditioning_latent_video=clean_latent_video,
            target_velocity=target_velocity,
            timesteps=full_timesteps,
            condition_mask=self.make_condition_mask(
                clean_latent_video,
                num_conditional_frames=conditioning_counts,
            ),
            actions=prepared_actions,
            target_sigmas=sigmas,
            num_conditional_frames=conditioning_counts,
            use_video_condition=conditioning_flags,
        )

    def sample_conditioned_latent_video(
        self,
        conditioning_latent_video: torch.Tensor,
        num_conditional_frames: int,
        actions: torch.Tensor | None = None,
        infer_steps: int | None = None,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Sample a full latent clip with DreamDojo-style CFG over video conditioning."""

        if conditioning_latent_video.ndim != 5:
            raise ValueError(
                "conditioning_latent_video must have shape (B, C, T, H, W) for RF sampling."
            )
        batch_size, channels, frames, height, width = conditioning_latent_video.shape
        if frames != self.cfg.max_frames:
            raise ValueError(
                f"Expected {self.cfg.max_frames} latent frames, received {conditioning_latent_video.shape[2]}."
            )
        conditioning_counts = self._normalize_num_conditional_frames(
            num_conditional_frames,
            batch_size=batch_size,
            device=conditioning_latent_video.device,
        )
        steps = self.cfg.dynamics_infer_steps if infer_steps is None else infer_steps
        if steps < 1:
            raise ValueError("infer_steps must be positive.")
        resolved_guidance_scale = (
            self.cfg.dynamics_guidance_scale if guidance_scale is None else guidance_scale
        )
        if resolved_guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative.")
        reference_noise = torch.randn(
            batch_size,
            channels,
            self.cfg.max_frames,
            height,
            width,
            device=conditioning_latent_video.device,
            dtype=conditioning_latent_video.dtype,
            generator=generator,
        )
        condition_mask = self.make_condition_mask(
            conditioning_latent_video,
            num_conditional_frames=conditioning_counts,
        )
        conditioning_state_in, _ = self._prepare_conditioning_state(
            conditioning_latent_video,
            condition_mask,
            reference_noise=reference_noise,
            use_video_condition=True,
        )
        latent_video = self._repin_conditioned_frames(
            noisy_latent_video=reference_noise.clone(),
            conditioning_latent_video=conditioning_state_in,
            condition_mask=condition_mask,
        )
        prepared_actions = self._prepare_actions(
            actions,
            batch_size=batch_size,
            device=conditioning_latent_video.device,
            dtype=conditioning_latent_video.dtype,
        )
        timesteps, sigmas = self.flow.make_inference_schedule(
            num_steps=steps,
            device=conditioning_latent_video.device,
            dtype=conditioning_latent_video.dtype,
        )
        for index, timestep in enumerate(timesteps):
            latent_video = self._repin_conditioned_frames(
                noisy_latent_video=latent_video,
                conditioning_latent_video=conditioning_state_in,
                condition_mask=condition_mask,
            )
            full_timesteps = torch.full(
                (batch_size, self.cfg.max_frames),
                float(timestep.item()),
                device=conditioning_latent_video.device,
                dtype=conditioning_latent_video.dtype,
            )
            conditioned_velocity = self.forward(
                noisy_latent_video=latent_video,
                timesteps=full_timesteps,
                condition_mask=condition_mask,
                actions=prepared_actions,
                conditioning_latent_video=conditioning_latent_video,
                reference_noise=reference_noise,
                use_video_condition=True,
            )
            if resolved_guidance_scale > 0.0:
                unconditioned_velocity = self.forward(
                    noisy_latent_video=latent_video,
                    timesteps=full_timesteps,
                    condition_mask=condition_mask,
                    actions=prepared_actions,
                    conditioning_latent_video=conditioning_latent_video,
                    reference_noise=reference_noise,
                    use_video_condition=False,
                )
                predicted_velocity = conditioned_velocity + resolved_guidance_scale * (
                    conditioned_velocity - unconditioned_velocity
                )
            else:
                predicted_velocity = conditioned_velocity
            latent_video = self.flow.step(
                sample=latent_video,
                velocity=predicted_velocity,
                sigma_from=sigmas[index],
                sigma_to=sigmas[index + 1],
            )
            latent_video = self._repin_conditioned_frames(
                noisy_latent_video=latent_video,
                conditioning_latent_video=conditioning_latent_video,
                condition_mask=condition_mask,
            )
        return latent_video

    def forward(
        self,
        noisy_latent_video: torch.Tensor,
        timesteps: torch.Tensor,
        condition_mask: torch.Tensor,
        actions: torch.Tensor | None,
        conditioning_latent_video: torch.Tensor,
        target_velocity: torch.Tensor | None = None,
        reference_noise: torch.Tensor | None = None,
        use_video_condition: bool | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict full-clip RF velocity with DreamDojo teacher conditioning semantics."""

        if target_velocity is not None and reference_noise is not None:
            raise ValueError("Provide either target_velocity or reference_noise, not both.")
        prepared_actions = self._prepare_actions(
            actions,
            batch_size=noisy_latent_video.shape[0],
            device=noisy_latent_video.device,
            dtype=noisy_latent_video.dtype,
        )
        conditioning_state_in, conditioning_flags = self._prepare_conditioning_state(
            conditioning_latent_video,
            condition_mask,
            target_velocity=target_velocity,
            reference_noise=reference_noise,
            use_video_condition=use_video_condition,
        )
        repinned_latent_video = self._repin_conditioned_frames(
            noisy_latent_video,
            conditioning_state_in,
            condition_mask,
        )
        effective_timesteps = self._apply_conditional_frame_timestep(timesteps, condition_mask)
        predicted_velocity = self.net(
            x_B_C_T_H_W=repinned_latent_video,
            timesteps_B_T=effective_timesteps,
            condition_video_input_mask_B_C_T_H_W=condition_mask,
            action=prepared_actions,
        )
        if target_velocity is not None:
            return self._overwrite_conditioned_velocity(
                predicted_velocity=predicted_velocity,
                target_velocity=target_velocity,
                condition_mask=condition_mask,
            )
        if reference_noise is not None:
            conditioned_velocity = reference_noise - conditioning_latent_video
            return self._overwrite_conditioned_velocity(
                predicted_velocity=predicted_velocity,
                target_velocity=conditioned_velocity,
                condition_mask=condition_mask,
            )

        return predicted_velocity

    def sample_next_latent(
        self,
        context_latent: torch.Tensor,
        actions: torch.Tensor | None = None,
        infer_steps: int | None = None,
        generator: torch.Generator | None = None,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Sample the next latent chunk from one supported clean latent context."""

        if context_latent.ndim != 5:
            raise ValueError(
                f"Expected context latents with shape (B, C, T, H, W), received {tuple(context_latent.shape)}."
            )
        context_frames = int(context_latent.shape[2])
        if context_frames not in self.cfg.conditioning_frame_choices:
            raise ValueError(
                f"Expected one of the supported context latent frame counts "
                f"{self.cfg.conditioning_frame_choices}, received {context_frames}."
            )
        batch_size, channels, _, height, width = context_latent.shape
        conditioning_latent_video = torch.cat(
            [
                context_latent,
                torch.zeros(
                    batch_size,
                    channels,
                    self.cfg.max_frames - context_frames,
                    height,
                    width,
                    device=context_latent.device,
                    dtype=context_latent.dtype,
                ),
            ],
            dim=2,
        )
        sampled_video = self.sample_conditioned_latent_video(
            conditioning_latent_video=conditioning_latent_video,
            num_conditional_frames=context_frames,
            actions=actions,
            infer_steps=infer_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )
        return sampled_video[:, :, context_frames:]
