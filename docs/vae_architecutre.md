# Wan VAE Architecture

## Summary

The source of truth for the autoencoder is `world_model_v2/wan_vae.py`.
This repo now supports a single autoencoder backend: the Wan-style VAE used by `WorldModel`.
- `--mode ae_only` for Wan VAE reconstruction training
- `--mode dynamics_only` for latent dynamics training with the autoencoder frozen

The current deployed config comes from `WorldModel`, which instantiates:

- `dim = 64`
- `z_dim = 16`
- `dim_mult = (1, 2, 4, 4)`
- `num_res_blocks = 1`
- `attn_scales = ()`
- `temperal_downsample = (False, False, False)`
- `dropout = 0.0`

Important: the config field is spelled `temperal_downsample` in code. The doc keeps that spelling because it is part of the actual API and serialized config.

## What The VAE Does In This Repo

At a high level, the Wan VAE compresses RGB frames into 16-channel latent maps and reconstructs them back to RGB:

$$
x \;\xrightarrow{\text{encoder}}\; (\mu, \log \sigma^2)
\;\xrightarrow{\text{sample or mean}}\; z
\;\xrightarrow{\text{decoder}}\; \hat{x}
$$

For the default 128x128 setup used by the repo:

- input frame shape: `(B, 3, 128, 128)`
- posterior mean shape: `(B, 16, 16, 16)`
- posterior log-variance shape: `(B, 16, 16, 16)`
- reconstruction shape: `(B, 3, 128, 128)`

That is an 8x spatial downsample in height and width, so one frame goes from `3 x 128 x 128 = 49,152` scalars to `16 x 16 x 16 = 4,096` latent scalars, which is a 12x reduction in per-frame element count.

## Active Model Size

Measured from the instantiated default modules in the repo virtualenv:

- encoder params: `15,466,528`
- decoder params: `25,754,755`
- total VAE params: `41,221,283`
- parameter memory only:
  - about `157.25 MiB` in fp32
  - about `78.62 MiB` in fp16/bf16

The decoder is larger than the encoder because each decoder stage uses `num_res_blocks + 1` residual blocks, while each encoder stage uses `num_res_blocks`.

## Core Logic Blocks In `wan_vae.py`

### `WanVAEConfig`

Holds the backend hyperparameters and validates the shape assumptions:

- `dim_mult` must be non-empty
- `temperal_downsample` must have exactly `len(dim_mult) - 1` entries
- `spatial_downsample_factor()` returns the total spatial compression factor, which is `8` for the default config
- `to_dict()` produces the checkpoint-friendly serialized config

### `CausalConv3d`

Wraps `nn.Conv3d` but replaces normal symmetric temporal padding with explicit causal padding:

- spatial padding stays symmetric
- temporal padding is one-sided into the past
- future frames are never visible to the convolution

This matters if the 3D backbone is used on videos directly.

### `RMSNorm`

Channel-first RMS normalization with learned scale and optional bias:

- `images=True` creates 2D broadcast shapes for frame-wise attention tensors
- `images=False` creates 3D broadcast shapes for video tensors

### `AttentionBlock`

Per-frame spatial self-attention over a 5D tensor `(B, C, T, H, W)`:

- it flattens `(B, T)` together
- runs attention independently on each frame
- does not mix information across time
- the output projection is zero-initialized, so the block starts close to an identity residual update

This is important: the attention path is spatial-only, not temporal.

### `Resample`

Handles up/down sampling for 5D tensors:

- `downsample2d` and `upsample2d` change only height and width
- `downsample3d` and `upsample3d` change time as well
- `none` is an identity path

In the default repo config, only the 2D resampling modes are active because `temperal_downsample` is all `False`.

### `ResidualBlock`

A Wan-style residual block:

```text
RMSNorm -> SiLU -> causal 3D conv -> RMSNorm -> SiLU -> dropout -> causal 3D conv + shortcut
```

If the channel count changes, the shortcut uses a `1x1x1` causal convolution; otherwise it is an identity.

### `WanEncoder3d`

Builds the encoder backbone for video tensors:

1. `conv_in` maps RGB to the base feature width.
2. A stack of down blocks applies residual processing.
3. Optional attention can be inserted at selected spatial scales through `attn_scales`.
4. Each non-final stage downsamples spatially or spatio-temporally.
5. A middle bottleneck runs `ResidualBlock -> AttentionBlock -> ResidualBlock`.
6. `norm_out -> SiLU -> conv_out` produces `2 * z_dim` channels.
7. The output is split into $(\mu, \log \sigma^2)$.

For the default config, the encoder channel schedule is:

- `3 -> 64` via `conv_in`
- stage 0: `64 -> 64`, then downsample by 2
- stage 1: `64 -> 128`, then downsample by 2
- stage 2: `128 -> 256`, then downsample by 2
- stage 3: `256 -> 256`, no further downsample
- bottleneck at `256` channels
- output moments: `32` channels total, split into $16$ for $\mu$ and $16$ for $\log \sigma^2$

### `WanDecoder3d`

Mirrors the encoder and reconstructs RGB frames:

1. `conv_in` maps latent channels to the widest feature width.
2. A middle bottleneck again runs `ResidualBlock -> AttentionBlock -> ResidualBlock`.
3. Up blocks apply residual processing.
4. Each non-final stage upsamples spatially or spatio-temporally.
5. `norm_out -> SiLU -> conv_out -> sigmoid` returns RGB in `[0, 1]`.

For the default config, the decoder channel schedule is:

- `16 -> 256` via `conv_in`
- stage 0: `256 -> 256`, then upsample by 2
- stage 1: `256 -> 256`, then upsample by 2
- stage 2: `256 -> 128`, then upsample by 2
- stage 3: `128 -> 64`, no further upsample
- output: `3` RGB channels

### `WanVAEEncoder` and `WanVAEDecoder`

These are the image-facing wrappers used by the rest of the repo:

- `WanVAEEncoder` adds a singleton time dimension before calling `WanEncoder3d`, then squeezes time back out
- `WanVAEDecoder` does the inverse for latent image tensors

This lets the repo reuse one implementation for both image and video-shaped tensors.

### `kl_divergence_from_moments`

Computes the KL penalty from $(\mu, \log \sigma^2)$ against a unit Gaussian prior:

$$
\mathcal{L}_{\mathrm{KL}}
= \frac{1}{2}\,\operatorname{mean}\!\left(
\exp(\log \sigma^2) + \mu^2 - 1 - \log \sigma^2
\right)
$$

### `sample_posterior`

Implements the reparameterization trick:

$$
\sigma = \exp\!\left(\frac{1}{2}\log \sigma^2\right), \qquad
\varepsilon \sim \mathcal{N}(0, I), \qquad
z = \mu + \sigma \odot \varepsilon
$$

## Encoder And Decoder Flow

The actual default path is easiest to picture as:

```text
(B, 3, H, W)
  -> WanVAEEncoder
  -> unsqueeze to (B, 3, 1, H, W)
  -> 3D causal encoder backbone
  -> moments (B, 32, 1, H/8, W/8)
  -> split to μ / log σ²
  -> squeeze to (B, 16, H/8, W/8)

(B, 16, H/8, W/8)
  -> WanVAEDecoder
  -> unsqueeze to (B, 16, 1, H/8, W/8)
  -> 3D causal decoder backbone
  -> RGB logits
  -> sigmoid
  -> squeeze to (B, 3, H, W)
```

## Training Loss

In `ae_only` mode, the repo trains the Wan VAE with a KL-regularized reconstruction loss:

$$
\mathcal{L}_{\mathrm{AE}}
= \mathcal{L}_{\mathrm{recon}} + \beta_{\mathrm{KL}}\,\mathcal{L}_{\mathrm{KL}}
$$

where:

$$
\mathcal{L}_{\mathrm{KL}}
= \frac{1}{2}\,\operatorname{mean}\!\left(
\exp(\log \sigma^2) + \mu^2 - 1 - \log \sigma^2
\right)
$$

and the reconstruction term is the normalized weighted mixture used in
`Experiment.reconstruction_loss_terms(...)`:

$$
\mathcal{L}_{\mathrm{recon}}
=
\frac{
w_{\mathrm{mse}}\,\mathcal{L}_{\mathrm{mse}}
\, + \,
w_{\mathrm{l1}}\,\mathcal{L}_{\mathrm{l1}}
\, + \,
w_{\mathrm{edge}}\,\mathcal{L}_{\mathrm{edge}}
}{
w_{\mathrm{mse}} + w_{\mathrm{l1}} + w_{\mathrm{edge}}
}
$$

with:

$$
\mathcal{L}_{\mathrm{mse}}
=
\frac{1}{N}\sum_{i=1}^{N}(\hat{x}_i - x_i)^2
$$

$$
\mathcal{L}_{\mathrm{l1}}
=
\frac{1}{N}\sum_{i=1}^{N}\left|\hat{x}_i - x_i\right|
$$

$$
\partial_x u = u[..., :, 1:] - u[..., :, :-1]
\qquad\text{and}\qquad
\partial_y u = u[..., 1:, :] - u[..., :-1, :]
$$

$$
\mathcal{L}_{\mathrm{edge}}
= \frac{1}{2}\left(
\frac{1}{N_x}\sum_{i=1}^{N_x}\left|(\partial_x \hat{x})_i - (\partial_x x)_i\right|
+
\frac{1}{N_y}\sum_{i=1}^{N_y}\left|(\partial_y \hat{x})_i - (\partial_y x)_i\right|
\right)
$$

Here:

- `x` is the target image batch
- $\hat{x}$ is the reconstructed image batch
- `N` is the number of scalar elements in the image batch
- `N_x` and `N_y` are the number of scalar elements in the horizontal and vertical gradient tensors
- $\partial_x$ and $\partial_y$ are the finite-difference image gradients used in `finite_difference_gradients(...)`
- $\beta_{\mathrm{KL}}$ controls how strongly the posterior is pushed toward a unit Gaussian prior

Important practical detail:

- training uses `sample_posterior=True`, so reconstruction is computed from sampled latents
- validation uses `sample_posterior=False`, so reconstruction is computed from the posterior mean

### Best Pickplace VAE Loss Settings

The curated checkpoint in `outputs/best_vae_pickplace/` was trained with:

- `--recon-mse-weight 1.0`
- `--recon-l1-weight 1.0`
- `--recon-edge-weight 0.25`
- `--kl-beta 5e-5`

Plugging those values into the normalized reconstruction formula gives:

$$
\mathcal{L}_{\mathrm{recon}}
=
\frac{
1.0\,\mathcal{L}_{\mathrm{mse}}
\, + \,
1.0\,\mathcal{L}_{\mathrm{l1}}
\, + \,
0.25\,\mathcal{L}_{\mathrm{edge}}
}{
2.25
}
$$

and the full autoencoder objective for that run is:

$$
\mathcal{L}_{\mathrm{AE}}
=
\frac{
\mathcal{L}_{\mathrm{mse}}
\, + \,
\mathcal{L}_{\mathrm{l1}}
\, + \,
0.25\,\mathcal{L}_{\mathrm{edge}}
}{
2.25
}
+
5\times 10^{-5}\,\mathcal{L}_{\mathrm{KL}}
$$

So the effective reconstruction mix is about:

- `44.4%` MSE
- `44.4%` L1
- `11.1%` edge

with a relatively small KL coefficient. In practice, that biases training toward
pixel fidelity and sharp local detail while still keeping a light latent
regularization term.

## Inference Path

At inference time, the repo uses the deterministic posterior mean rather than sampling:

$$
z_{\mathrm{infer}} = \mu
$$

and reconstruction becomes:

$$
\hat{x}_{\mathrm{infer}} = D(\mu)
$$

This is the path used by `WorldModel.encode(..., deterministic=True)` and by
the sequence helpers that feed the dynamics model. For a frame sequence
$x_{1:T}$, the deployed path is:

$$
z_t = \mu(x_t), \qquad t = 1, \dots, T
$$

followed later by frame-wise decoding:

$$
\hat{x}_t = D(z_t)
$$

So during inference and rollout:

- no latent noise sample $\varepsilon$ is drawn
- no stochastic posterior sample $z = \mu + \sigma \odot \varepsilon$ is used
- the autoencoder contributes a deterministic latent representation based only on $\mu$
