"""Minimal rectified-flow DiT dynamics adapted from DreamDojo for short Wan-latent clips."""

from __future__ import annotations

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
class MinimalRFDiTConfig:
    """Configure the tiny DreamDojo-inspired latent dynamics transformer."""

    max_img_h: int
    max_img_w: int
    max_frames: int = 5
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    model_channels: int = 256
    num_blocks: int = 4
    num_heads: int = 4
    concat_padding_mask: bool = False
    pos_emb_cls: str = "rope3d"
    pos_emb_learnable: bool = False
    pos_emb_interpolation: str = "crop"
    use_adaln_lora: bool = False
    adaln_lora_dim: int = 64
    atten_backend: str = "torch"
    extra_per_block_abs_pos_emb: bool = False
    rope_h_extrapolation_ratio: float = 1.0
    rope_w_extrapolation_ratio: float = 1.0
    rope_t_extrapolation_ratio: float = 1.0
    action_dim: int = 4
    timestep_scale: float = 1.0
    dynamics_infer_steps: int = 16
    dynamics_train_timesteps: int = 1000
    dynamics_rf_shift: float = 5.0

    def __post_init__(self) -> None:
        """Validate the small DiT configuration."""

        if self.max_frames != 5:
            raise ValueError(
                "Minimal rectified-flow dynamics currently expects max_frames=5 "
                "for three context frames plus two target frames."
            )
        if self.in_channels != self.out_channels:
            raise ValueError("The minimal RF DiT expects matching in/out channel counts.")
        if self.patch_spatial <= 0 or self.patch_temporal <= 0:
            raise ValueError("Patch sizes must be positive.")
        if self.max_img_h % self.patch_spatial != 0 or self.max_img_w % self.patch_spatial != 0:
            raise ValueError("Latent height and width must be divisible by patch_spatial.")
        if self.max_frames % self.patch_temporal != 0:
            raise ValueError("max_frames must be divisible by patch_temporal.")
        if self.model_channels % self.num_heads != 0:
            raise ValueError("model_channels must be divisible by num_heads.")
        if self.pos_emb_cls != "rope3d":
            raise ValueError("The minimal RF DiT only supports pos_emb_cls='rope3d'.")
        if self.pos_emb_learnable:
            raise ValueError("The minimal RF DiT only supports non-learnable RoPE.")
        if self.pos_emb_interpolation != "crop":
            raise ValueError("The minimal RF DiT only supports crop interpolation.")
        if self.concat_padding_mask:
            raise ValueError("concat_padding_mask is unsupported in the minimal RF DiT.")
        if self.extra_per_block_abs_pos_emb:
            raise ValueError("extra_per_block_abs_pos_emb is unsupported in the minimal RF DiT.")
        if self.atten_backend != "torch":
            raise ValueError("The minimal RF DiT only supports the torch attention backend.")
        if self.dynamics_infer_steps < 1:
            raise ValueError("dynamics_infer_steps must be positive.")
        if self.dynamics_train_timesteps < 2:
            raise ValueError("dynamics_train_timesteps must be at least 2.")
        if self.timestep_scale <= 0.0:
            raise ValueError("timestep_scale must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the DiT config."""

        payload = asdict(self)
        payload["num_action_per_chunk"] = self.num_action_per_chunk
        return payload

    @property
    def context_frames(self) -> int:
        """Return the number of clean conditioning frames expected by the DiT."""

        return self.max_frames - self.target_frames

    @property
    def target_frames(self) -> int:
        """Return the number of predicted target frames in each latent clip."""

        return 2

    @property
    def num_action_per_chunk(self) -> int:
        """Return the DreamDojo-style number of transition actions in one frame chunk."""

        return self.max_frames - 1


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension and rescale it."""

        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight


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
        flat = timesteps.reshape(-1).float()
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(float(half_dim), 1.0)
        frequencies = torch.exp(exponent)
        angles = flat[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding.view(*timesteps.shape, self.dim).to(dtype=timesteps.dtype)


class TimestepEmbedding(nn.Module):
    """Convert sinusoidal timestep features into AdaLN conditioning vectors."""

    def __init__(self, in_dim: int, out_dim: int, use_adaln_lora: bool) -> None:
        """Build the timestep projection MLP."""

        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.SiLU()
        self.adaln = (
            nn.Linear(out_dim, out_dim * 3, bias=False)
            if use_adaln_lora
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the timestep embedding weights."""

        nn.init.trunc_normal_(self.fc1.weight, std=1.0 / math.sqrt(self.fc1.in_features))
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=1.0 / math.sqrt(self.fc2.in_features))
        nn.init.zeros_(self.fc2.bias)
        if self.adaln is not None:
            nn.init.zeros_(self.adaln.weight)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the timestep embedding and optional AdaLN-LoRA residual."""

        hidden = self.activation(self.fc1(embeddings))
        projected = self.fc2(hidden)
        if self.adaln is None:
            adaln = torch.zeros(
                (*projected.shape[:-1], projected.shape[-1] * 3),
                device=projected.device,
                dtype=projected.dtype,
            )
        else:
            adaln = self.adaln(hidden)
        return projected, adaln


class ActionEmbeddingMLP(nn.Module):
    """Embed a flattened action window with the DreamDojo MLP structure."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Create the DreamDojo action embedding MLP."""

        super().__init__()
        hidden_features = out_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the action MLP with the same fan-in scaling used elsewhere."""

        nn.init.trunc_normal_(self.fc1.weight, std=1.0 / math.sqrt(self.fc1.in_features))
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=1.0 / math.sqrt(self.fc2.in_features))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Return one action-conditioning embedding for each batch item."""

        hidden = self.fc1(actions)
        hidden = self.activation(hidden)
        hidden = self.drop(hidden)
        hidden = self.fc2(hidden)
        hidden = self.drop(hidden)
        return hidden


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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the patch embedding projection."""

        nn.init.trunc_normal_(self.proj[1].weight, std=1.0 / math.sqrt(self.patch_dim))

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
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        self.head_dim = head_dim
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.dim_h = (head_dim // 6) * 2
        self.dim_w = self.dim_h
        self.dim_t = head_dim - self.dim_h - self.dim_w
        self.h_theta = 10000.0 * self._ntk_factor(self.dim_h, h_extrapolation_ratio)
        self.w_theta = 10000.0 * self._ntk_factor(self.dim_w, w_extrapolation_ratio)
        self.t_theta = 10000.0 * self._ntk_factor(self.dim_t, t_extrapolation_ratio)

    def _ntk_factor(self, dim: int, ratio: float) -> float:
        """Return the DreamDojo-style NTK extrapolation factor."""

        if dim <= 2:
            return 1.0
        return ratio ** (dim / (dim - 2))

    def _angles(self, size: int, dim: int, theta: float, device: torch.device) -> torch.Tensor:
        """Return rotary angles for one axis."""

        if dim == 0:
            return torch.zeros(size, 0, device=device)
        positions = torch.arange(size, device=device, dtype=torch.float32)
        exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        frequencies = 1.0 / (theta ** exponents)
        return torch.outer(positions, frequencies)

    def forward(
        self,
        frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine RoPE tensors for the current token grid."""

        if frames > self.max_t or height > self.max_h or width > self.max_w:
            raise ValueError("Requested token grid exceeds the configured RoPE capacity.")
        angles_t = self._angles(frames, self.dim_t, self.t_theta, device)
        angles_h = self._angles(height, self.dim_h, self.h_theta, device)
        angles_w = self._angles(width, self.dim_w, self.w_theta, device)
        combined = torch.cat(
            [
                repeat(angles_t, "t d -> t h w d", h=height, w=width),
                repeat(angles_h, "h d -> t h w d", t=frames, w=width),
                repeat(angles_w, "w d -> t h w d", t=frames, h=height),
            ],
            dim=-1,
        ).reshape(frames * height * width, -1)
        return combined.cos().to(dtype=dtype), combined.sin().to(dtype=dtype)


def apply_rotary_position_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate query or key heads with 3D RoPE angles."""

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    """Apply full self-attention over the flattened latent token grid."""

    def __init__(self, model_channels: int, num_heads: int) -> None:
        """Build the attention projections for the DiT block."""

        super().__init__()
        self.model_channels = model_channels
        self.num_heads = num_heads
        self.head_dim = model_channels // num_heads
        self.q_proj = nn.Linear(model_channels, model_channels, bias=False)
        self.k_proj = nn.Linear(model_channels, model_channels, bias=False)
        self.v_proj = nn.Linear(model_channels, model_channels, bias=False)
        self.out_proj = nn.Linear(model_channels, model_channels, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the attention projections."""

        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.trunc_normal_(projection.weight, std=1.0 / math.sqrt(projection.in_features))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Return the attention-updated token sequence."""

        query = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        key = rearrange(self.k_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        value = rearrange(self.v_proj(x), "b l (h d) -> b h l d", h=self.num_heads)
        query = apply_rotary_position_embedding(self.q_norm(query), cos, sin)
        key = apply_rotary_position_embedding(self.k_norm(key), cos, sin)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        return self.out_proj(rearrange(attended, "b h l d -> b l (h d)"))


class FeedForward(nn.Module):
    """Apply the MLP half of one DiT block."""

    def __init__(self, model_channels: int, mlp_ratio: float = 4.0) -> None:
        """Create the two-layer feed-forward network."""

        super().__init__()
        hidden_channels = int(model_channels * mlp_ratio)
        self.fc1 = nn.Linear(model_channels, hidden_channels, bias=False)
        self.fc2 = nn.Linear(hidden_channels, model_channels, bias=False)
        self.activation = nn.GELU(approximate="tanh")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the feed-forward layers."""

        nn.init.trunc_normal_(self.fc1.weight, std=1.0 / math.sqrt(self.fc1.in_features))
        nn.init.trunc_normal_(self.fc2.weight, std=1.0 / math.sqrt(self.fc2.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the MLP update for the token sequence."""

        return self.fc2(self.activation(self.fc1(x)))


class AdaLNDiTBlock(nn.Module):
    """Apply one AdaLN-modulated self-attention plus MLP transformer block."""

    def __init__(self, model_channels: int, num_heads: int) -> None:
        """Create the attention, MLP, and modulation paths."""

        super().__init__()
        self.model_channels = model_channels
        self.attn_norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(model_channels, num_heads)
        self.mlp = FeedForward(model_channels)
        self.attn_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, model_channels * 3, bias=False),
        )
        self.mlp_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, model_channels * 3, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the AdaLN modulation heads."""

        nn.init.zeros_(self.attn_modulation[1].weight)
        nn.init.zeros_(self.mlp_modulation[1].weight)

    def _broadcast(self, values: torch.Tensor) -> torch.Tensor:
        """Expand one `(B, T, D)` tensor over the spatial token grid."""

        return rearrange(values, "b t d -> b t 1 1 d")

    def _modulate(self, x: torch.Tensor, norm: nn.LayerNorm, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN modulation before one transformer sublayer."""

        return norm(x) * (1.0 + self._broadcast(scale)) + self._broadcast(shift)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Return the transformed token grid for one DiT block."""

        batch_size, frames, height, width, _ = x.shape
        shift_attn, scale_attn, gate_attn = self.attn_modulation(timestep_embedding).chunk(3, dim=-1)
        attn_input = self._modulate(x, self.attn_norm, shift_attn, scale_attn)
        attn_input = rearrange(attn_input, "b t h w d -> b (t h w) d")
        attn_output = self.attn(attn_input, cos, sin)
        attn_output = rearrange(attn_output, "b (t h w) d -> b t h w d", t=frames, h=height, w=width)
        x = x + self._broadcast(gate_attn) * attn_output

        shift_mlp, scale_mlp, gate_mlp = self.mlp_modulation(timestep_embedding).chunk(3, dim=-1)
        mlp_input = self._modulate(x, self.mlp_norm, shift_mlp, scale_mlp)
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
    ) -> None:
        """Create the final AdaLN projection to the latent patch space."""

        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, model_channels * 2, bias=False),
        )
        patch_dim = spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels
        self.proj = nn.Linear(model_channels, patch_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the final AdaLN projection."""

        nn.init.zeros_(self.modulation[1].weight)
        nn.init.trunc_normal_(self.proj.weight, std=1.0 / math.sqrt(self.proj.in_features))

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        """Return per-patch latent predictions."""

        shift, scale = self.modulation(timestep_embedding).chunk(2, dim=-1)
        modulated = self.norm(x) * (1.0 + rearrange(scale, "b t d -> b t 1 1 d"))
        modulated = modulated + rearrange(shift, "b t d -> b t 1 1 d")
        return self.proj(modulated)


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


class ActionConditionedMinimalRFDiT(nn.Module):
    """Run the tiny short-clip latent DiT with DreamDojo-style action conditioning."""

    def __init__(self, cfg: MinimalRFDiTConfig) -> None:
        """Build the patch embedder, RoPE, timestep path, and DiT blocks."""

        super().__init__()
        self.cfg = cfg
        self.x_embedder = PatchEmbed(
            spatial_patch_size=cfg.patch_spatial,
            temporal_patch_size=cfg.patch_temporal,
            in_channels=cfg.in_channels + 1,
            out_channels=cfg.model_channels,
        )
        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=cfg.model_channels // cfg.num_heads,
            len_h=cfg.max_img_h // cfg.patch_spatial,
            len_w=cfg.max_img_w // cfg.patch_spatial,
            len_t=cfg.max_frames // cfg.patch_temporal,
            h_extrapolation_ratio=cfg.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=cfg.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=cfg.rope_t_extrapolation_ratio,
        )
        self.timesteps = Timesteps(cfg.model_channels)
        self.timestep_embedding = TimestepEmbedding(
            in_dim=cfg.model_channels,
            out_dim=cfg.model_channels,
            use_adaln_lora=cfg.use_adaln_lora,
        )
        self.t_embedding_norm = RMSNorm(cfg.model_channels)
        self.blocks = nn.ModuleList(
            [AdaLNDiTBlock(cfg.model_channels, cfg.num_heads) for _ in range(cfg.num_blocks)]
        )
        self.final_layer = FinalLayer(
            model_channels=cfg.model_channels,
            spatial_patch_size=cfg.patch_spatial,
            temporal_patch_size=cfg.patch_temporal,
            out_channels=cfg.out_channels,
        )
        action_features = cfg.num_action_per_chunk * cfg.action_dim
        self.action_embedder_B_D = ActionEmbeddingMLP(action_features, cfg.model_channels)
        self.action_embedder_B_3D = ActionEmbeddingMLP(action_features, cfg.model_channels * 3)

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
        timesteps_B_T = timesteps_B_T * self.cfg.timestep_scale
        model_input = torch.cat(
            [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.to(dtype=x_B_C_T_H_W.dtype)],
            dim=1,
        )
        tokens = self.x_embedder(model_input)
        _, token_frames, token_height, token_width, _ = tokens.shape
        cos, sin = self.pos_embedder(
            token_frames,
            token_height,
            token_width,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        timestep_features = self.timesteps(timesteps_B_T)
        timestep_embedding, adaln_lora = self.timestep_embedding(timestep_features)
        action = rearrange(action.to(dtype=tokens.dtype), "b t d -> b 1 (t d)")
        action_emb_B_D = self.action_embedder_B_D(action)
        action_emb_B_3D = self.action_embedder_B_3D(action)
        timestep_embedding = self.t_embedding_norm(timestep_embedding + action_emb_B_D)
        adaln_lora = adaln_lora + action_emb_B_3D
        del adaln_lora
        for block in self.blocks:
            tokens = block(tokens, timestep_embedding, cos, sin)
        patches = self.final_layer(tokens, timestep_embedding)
        return self.unpatchify(patches)


class MinimalRectifiedFlowDynamics(nn.Module):
    """Wrap the tiny DiT with RF training-input preparation and sampling helpers."""

    def __init__(self, cfg: MinimalRFDiTConfig) -> None:
        """Build the DiT backbone and the flow scheduler helper."""

        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.net = ActionConditionedMinimalRFDiT(cfg)
        self.flow = RectifiedFlowHelper(
            num_train_timesteps=cfg.dynamics_train_timesteps,
            shift=cfg.dynamics_rf_shift,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable dynamics configuration."""

        return self.cfg.to_dict()

    def make_condition_mask(self, latents: torch.Tensor) -> torch.Tensor:
        """Return a mask with every context frame marked as known conditioning."""

        if latents.ndim != 5:
            raise ValueError(f"Expected latent video shape (B, C, T, H, W), received {tuple(latents.shape)}.")
        batch_size, _, frames, height, width = latents.shape
        if frames != self.cfg.max_frames:
            raise ValueError(f"Expected {self.cfg.max_frames} latent frames, received {frames}.")
        mask = torch.zeros(batch_size, 1, frames, height, width, device=latents.device, dtype=latents.dtype)
        mask[:, :, : self.cfg.context_frames] = 1.0
        return mask

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
        return DynamicsTrainingInputs(
            noisy_latent_video=noisy_latent_video,
            conditioning_latent_video=clean_latent_video,
            target_velocity=target_velocity,
            timesteps=full_timesteps,
            condition_mask=self.make_condition_mask(clean_latent_video),
            actions=prepared_actions,
            target_sigmas=sigmas,
        )

    def forward(
        self,
        noisy_latent_video: torch.Tensor,
        timesteps: torch.Tensor,
        condition_mask: torch.Tensor,
        actions: torch.Tensor | None,
        conditioning_latent_video: torch.Tensor,
        target_velocity: torch.Tensor | None = None,
        reference_noise: torch.Tensor | None = None,
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
        repinned_latent_video = self._repin_conditioned_frames(
            noisy_latent_video,
            conditioning_latent_video,
            condition_mask,
        )
        predicted_velocity = self.net(
            x_B_C_T_H_W=repinned_latent_video,
            timesteps_B_T=timesteps,
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
    ) -> torch.Tensor:
        """Sample the next latent chunk from a clean three-frame latent context."""

        if context_latent.ndim != 5:
            raise ValueError(
                f"Expected context latents with shape (B, C, T, H, W), received {tuple(context_latent.shape)}."
            )
        if context_latent.shape[2] != self.cfg.context_frames:
            raise ValueError(
                f"Expected {self.cfg.context_frames} context latent frames, "
                f"received {context_latent.shape[2]}."
            )
        batch_size, channels, _, height, width = context_latent.shape
        steps = self.cfg.dynamics_infer_steps if infer_steps is None else infer_steps
        if steps < 1:
            raise ValueError("infer_steps must be positive.")
        target_slice = slice(self.cfg.context_frames, self.cfg.max_frames)
        reference_noise = torch.randn(
            batch_size,
            channels,
            self.cfg.max_frames,
            height,
            width,
            device=context_latent.device,
            dtype=context_latent.dtype,
            generator=generator,
        )
        conditioning_latent_video = torch.cat(
            [
                context_latent,
                torch.zeros(
                    batch_size,
                    channels,
                    self.cfg.target_frames,
                    height,
                    width,
                    device=context_latent.device,
                    dtype=context_latent.dtype,
                ),
            ],
            dim=2,
        )
        latent_video = self._repin_conditioned_frames(
            noisy_latent_video=reference_noise.clone(),
            conditioning_latent_video=conditioning_latent_video,
            condition_mask=self.make_condition_mask(conditioning_latent_video),
        )
        condition_mask = self.make_condition_mask(latent_video)
        prepared_actions = self._prepare_actions(
            actions,
            batch_size=batch_size,
            device=context_latent.device,
            dtype=context_latent.dtype,
        )
        timesteps, sigmas = self.flow.make_inference_schedule(
            num_steps=steps,
            device=context_latent.device,
            dtype=context_latent.dtype,
        )
        for index, timestep in enumerate(timesteps):
            latent_video = self._repin_conditioned_frames(
                noisy_latent_video=latent_video,
                conditioning_latent_video=conditioning_latent_video,
                condition_mask=condition_mask,
            )
            full_timesteps = torch.full(
                (batch_size, self.cfg.max_frames),
                float(timestep.item()),
                device=context_latent.device,
                dtype=context_latent.dtype,
            )
            predicted_velocity = self.forward(
                noisy_latent_video=latent_video,
                timesteps=full_timesteps,
                condition_mask=condition_mask,
                actions=prepared_actions,
                conditioning_latent_video=conditioning_latent_video,
                reference_noise=reference_noise,
            )
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
        return latent_video[:, :, target_slice]
