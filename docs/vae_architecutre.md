# Wan VAE Architecture

## Summary

The source of truth for the autoencoder lives in `world_model_v2/wan_vae.py`, with the repo-facing integration in `world_model_v2/model.py`.

This repo now uses one default autoencoder path:

- a Wan2.1-style causal video tokenizer
- `8x` spatial compression
- `4x` temporal compression
- `z_dim = 32`

The world-model stack follows DreamDojo-style timing semantics end to end. The default latent layout is:

- `1` context latent frame
- `3` target latent frames

That corresponds to:

- `13` pixel frames per full training chunk
- `12` actions per chunk

## Active Default Config

`WorldModel` instantiates the tokenizer with:

- `dim = 96`
- `z_dim = 32`
- `dim_mult = (1, 2, 2, 4)`
- `num_res_blocks = 1`
- `attn_scales = ()`
- `temperal_downsample = (False, True, True)`
- `dropout = 0.0`

Important: the config field is spelled `temperal_downsample` in code and checkpoints, so the docs keep that spelling.

## Temporal Mapping

The tokenizer uses the Wan2.1 frame mapping:

$$
\text{latent\_T} = 1 + \left\lfloor \frac{\text{pixel\_T} - 1}{4} \right\rfloor
$$

and the inverse reconstruction length:

$$
\text{pixel\_T} = 1 + 4 \cdot (\text{latent\_T} - 1)
$$

For the default setup:

- `1` pixel frame encodes to `1` latent frame and decodes back to `1`
- `5` pixel frames encode to `2` latent frames and decode back to `5`
- `13` pixel frames encode to `4` latent frames and decode back to `13`

This is why the default dynamics chunk is `13` pixel frames wide even though the latent transformer sees `4` frames.

## Spatial Mapping

Spatial compression remains `8x`:

- input video: `(B, T, 3, H, W)`
- latent video: `(B, 32, T_latent, H/8, W/8)`
- decoded video: `(B, T, 3, H, W)`

At `128 x 128` resolution, a `13`-frame clip becomes a latent tensor of shape `(B, 32, 4, 16, 16)`.

## Parameter Count

Measured from the default instantiated modules in the repo virtualenv:

- encoder params: `28,213,824`
- decoder params: `43,374,915`
- total VAE params: `71,588,739`
- parameter memory only:
  - about `273.09 MiB` in fp32
  - about `136.55 MiB` in fp16 or bf16

## Core Components

### `WanVAEConfig`

`WanVAEConfig` owns the tokenizer hyperparameters and the frame-conversion helpers:

- `spatial_downsample_factor()`
- `temporal_downsample_factor()`
- `pixel_frames_to_latent_frames()`
- `latent_frames_to_pixel_frames()`
- `exact_latent_frames_for_pixels()`
- `to_dict()` and `from_dict()`

Those helpers are used by datasets, rollout code, validation, and checkpoint compatibility checks.

### `CausalConv3d`

The backbone uses causal temporal padding so each convolution only sees the current frame and the past.

### `Resample`

`Resample` implements Wan-style spatial and temporal up/downsampling. With the default `temperal_downsample = (False, True, True)`:

- the first downsample stage is spatial-only
- the second and third downsample stages compress time by `2x`
- the total temporal compression becomes `4x`

The decoder mirrors that schedule during reconstruction.

### `WanPosteriorEncoder`

`WanPosteriorEncoder` is the full video encoder used by the world model. It accepts 5D video tensors and runs the cached causal encode loop used by Wan-style temporal tokenizers.

This is the important behavior change relative to the old framewise path:

- sequence encoding no longer flattens `(B, T)` into independent images
- the encoder sees the whole clip as one causal video
- latent frame boundaries now match the tokenizer's real temporal compression

### `WanVideoDecoder`

`WanVideoDecoder` performs the matching cached causal decode loop. It reconstructs whole clips with the correct Wan frame counts.

For predicted videos, the repo now decodes:

- `context_latents + target_latents` together

and then crops away the context pixel frames after decoding. Target latents are not decoded in isolation, which keeps temporal alignment consistent with Wan2.1 and DreamDojo.

### Image Wrappers

Single-frame image encode/decode is still supported, but only as a thin wrapper:

- images are temporarily reshaped to a one-frame video
- all real sequence paths use the 5D video tokenizer directly

## World-Model Integration

`WorldModel` stores tokenizer metadata and latent normalization stats together:

- serialized Wan config
- `img_mean` and `img_std`
- `video_mean` and `video_std`

Image latents and video latents are normalized before they are passed into dynamics and unnormalized before decoding.

This matches the DreamDojo-style latent-stat handling used by the downstream transformer path.

## Dataset And Training Semantics

Autoencoder training now uses clip datasets rather than frame-only datasets.

The default AE clip size is the full latent training chunk mapped back to pixels:

$$
\text{pixel\_frames} = 1 + 4 \cdot ((1 + 3) - 1) = 13
$$

So the default AE batch element is a `13`-frame clip.

Dynamics training also works in pixel clips:

- context: `1` latent frame -> `1` pixel frame
- target: `3` latent frames -> `12` future actions and `12` future frame steps
- full encoded chunk: `13` pixel frames -> `4` latent frames

During training, the combined pixel clip is encoded once and then split in latent space. That avoids the frame-misalignment bug that happens when context and target windows are encoded separately.

## Validation And Rollout Semantics

Validation and rollout APIs are now pixel-facing:

- callers pass pixel frames and pixel-aligned actions
- the model converts those counts into latent lengths internally
- decoding expands predictions back to the correct pixel-frame counts

Examples:

- conditioning on `1` pixel frame produces the next `4` predicted pixel frames per chunk
- conditioning on `5` pixel frames means the tokenizer sees `2` latent context frames
- a default open-rollout chunk adds `12` actions and `12` new pixel frames

## Checkpoint Compatibility

Checkpoint metadata now includes the full tokenizer config and latent normalization stats.

That means:

- temporal Wan checkpoints can be resumed safely
- encoder/decoder loading requires matching tokenizer config and stats
- older spatial-only checkpoints fail fast with a clear incompatibility error instead of silently mis-shaping temporal data
