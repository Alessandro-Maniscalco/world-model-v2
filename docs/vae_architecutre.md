# Wan VAE Architecture

## What This File Explains

This document explains the exact behavior of the causal video VAE implemented in `world_model_v2/wan_vae.py` and how `world_model_v2/model.py` uses it.

The short answer to the main question is:

- a `13`-frame clip does **not** go through the encoder as one single dense temporal block
- it is split internally into `1 + 4 + 4 + 4` frames
- those chunks produce `4` latent frames: one latent for the very first pixel frame, then one latent for each later `4`-frame group
- the decoder mirrors that asymmetry: the first latent decodes to `1` frame, and each later latent decodes to `4` frames

So the mapping is:

$$
13\ \text{pixel frames} \longleftrightarrow 4\ \text{latent frames}
$$

but the latent frames are not symmetric in time:

$$
z_0 \leftrightarrow x_0,\qquad
z_1 \leftrightarrow x_{1:4},\qquad
z_2 \leftrightarrow x_{5:8},\qquad
z_3 \leftrightarrow x_{9:12}.
$$

Here `x_{a:b}` means pixel frames `a, a+1, ..., b`.

## Source Of Truth

The main implementation lives in:

- `world_model_v2/wan_vae.py`
- `world_model_v2/model.py`

The tests that verify the basic temporal mapping live in:

- `tests/test_wan_vae.py`

## Actual Defaults In Code

### Raw `WanVAEConfig()` defaults

`wan_vae.py` defines:

- `dim = 64`
- `z_dim = 64`
- `dim_mult = (1, 2, 4)`
- `num_res_blocks = 1`
- `attn_scales = ()`
- `temperal_downsample = (True, True)`
- `dropout = 0.0`

Important: the config key is spelled `temperal_downsample` in code and checkpoints.

### Effective `WorldModel()` defaults

`WorldModel()` builds:

```python
WanVAEConfig(z_dim=latent_channels)
```

with `latent_channels = 32` by default, so the effective default tokenizer in the model is:

- `dim = 64`
- `z_dim = 32`
- `dim_mult = (1, 2, 4)`
- `num_res_blocks = 1`
- `attn_scales = ()`
- `temperal_downsample = (True, True)`
- `dropout = 0.0`

## Compression Ratios

The config helpers define the total compression ratios.

### Spatial compression

The code uses:

$$
S = 2^{(\lvert \text{dim\_mult} \rvert - 1)}.
$$

With `dim_mult = (1, 2, 4)`:

$$
S = 2^{3 - 1} = 4.
$$

So the default spatial compression is `4x`, not `8x`.

If the input video has shape

$$
x \in \mathbb{R}^{B \times 3 \times T \times H \times W},
$$

then the latent tensor has spatial shape

$$
\frac{H}{4} \times \frac{W}{4}.
$$

Examples:

- `32 x 32 -> 8 x 8`
- `120 x 160 -> 30 x 40`

### Temporal compression

The code uses:

$$
R_t = 2^{\sum_i \mathbf{1}[\text{temperal\_downsample}_i]}.
$$

With `temperal_downsample = (True, True)`:

$$
R_t = 2^2 = 4.
$$

So the default temporal compression is `4x`, but it is a special causal `1 + 4k` mapping rather than a plain uniform stride-4 mapping.

## Exact Temporal Mapping

`WanVAEConfig.pixel_frames_to_latent_frames()` returns

$$
T_z^{\text{floor}} = 1 + \left\lfloor \frac{T_x - 1}{4} \right\rfloor
$$

and `latent_frames_to_pixel_frames()` returns

$$
T_x = 1 + 4(T_z - 1).
$$

For clips that the full encoder/decoder can represent exactly, we need

$$
T_x \in \{1, 5, 9, 13, 17, \dots\} = \{1 + 4k \mid k \ge 0\}.
$$

For those aligned lengths:

$$
T_z = 1 + \frac{T_x - 1}{4}.
$$

Examples:

- `1` pixel frame -> `1` latent frame
- `5` pixel frames -> `2` latent frames
- `9` pixel frames -> `3` latent frames
- `13` pixel frames -> `4` latent frames

## Important Practical Note About Non-Aligned Lengths

The helper formula uses a floor, but the actual chunked encoder/decoder path is only cleanly supported for `1 + 4k` pixel-frame counts.

Observed behavior in the repo virtualenv:

- `T = 1` works
- `T = 5` works
- `T = 9` works
- `T = 13` works
- `T = 2, 3, 4, 6, 7, 8` raise a runtime error in the current implementation

So the codebase correctly uses `exact_latent_frames_for_pixels()` when it needs a strict Wan-aligned context length.

## Tensor Conventions

The low-level VAE works with channel-first videos:

$$
(B, C, T, H, W).
$$

The higher-level world-model APIs usually accept image sequences in:

$$
(B, T, C, H, W)
$$

and convert them with `einops.rearrange`.

The main shapes are:

- pixel video: `(B, 3, T_x, H, W)`
- posterior mean/logvar: `(B, z_dim, T_z, H/4, W/4)`
- latent video: `(B, z_dim, T_z, H/4, W/4)`
- reconstructed video: `(B, 3, T_x, H, W)`

With the default `WorldModel()` and a `13 x 120 x 160` clip:

$$
(B, 3, 13, 120, 160) \to (B, 32, 4, 30, 40) \to (B, 3, 13, 120, 160).
$$

## Building Blocks

## `CausalConv3d`

`CausalConv3d` is a `Conv3d` with temporal padding only on the left, so output time `t` can only depend on the current and past inputs.

For temporal kernel size `k_t`, the idealized causal convolution is:

$$
y_t = \sum_{\tau=0}^{k_t-1} W_\tau x_{t-\tau}.
$$

In this file, the common temporal kernel is `3`, so each output time can see up to:

- the current time
- one step into the past
- two steps into the past

At chunk boundaries, cached past activations are concatenated so chunked execution matches full causal execution.

## `RMSNorm`

The code implements Wan-style RMS normalization with learned gain `\gamma` and optional bias `\beta`:

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{C}\sum_{c=1}^{C} x_c^2}
$$

$$
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\operatorname{RMS}(x)} + \beta.
$$

In the actual implementation, `F.normalize(..., dim=channel)` is multiplied by `\sqrt{C}`, which is equivalent to dividing by the root-mean-square across channels.

## `ResidualBlock`

Each residual block computes:

$$
h_1 = \operatorname{Conv3d}_{\text{causal}}(\operatorname{SiLU}(\operatorname{RMSNorm}(x)))
$$

$$
h_2 = \operatorname{Conv3d}_{\text{causal}}(\operatorname{Dropout}(\operatorname{SiLU}(\operatorname{RMSNorm}(h_1))))
$$

$$
y = h_2 + \operatorname{Shortcut}(x).
$$

If the channel count changes, the shortcut is a `1x1x1` causal convolution. Otherwise it is identity.

## `AttentionBlock`

Attention is spatial only, not temporal.

For each frame independently, the block flattens the spatial grid and computes single-head attention:

$$
Q, K, V = W_{qkv}(\operatorname{RMSNorm}(x_t))
$$

$$
\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

$$
y_t = W_o(\operatorname{Attn}(Q, K, V)) + x_t.
$$

Since attention is applied frame by frame, temporal mixing comes from causal convolutions and temporal resampling, not from attention.

## `Resample`

`Resample` handles spatial-only and spatio-temporal up/downsampling.

### Spatial downsample

Each frame is flattened into `(B*T, C, H, W)` and passed through a `Conv2d` with stride `2`, so:

$$
H \to \frac{H}{2}, \qquad W \to \frac{W}{2}.
$$

### Spatial upsample

Each frame is upsampled by nearest-neighbor and then passed through a `Conv2d`, so:

$$
H \to 2H, \qquad W \to 2W.
$$

### Temporal downsample (`downsample3d`)

The temporal part uses a causal `Conv3d` with:

- kernel `(3, 1, 1)`
- stride `(2, 1, 1)`

So once the cache is warm, time is reduced by roughly `2x`.

### Temporal upsample (`upsample3d`)

The temporal part uses a causal `Conv3d` that doubles channels from `C` to `2C`, then reshapes those channels into twice as many time steps:

$$
u \in \mathbb{R}^{B \times 2C \times T \times H \times W}
\longrightarrow
\tilde{u} \in \mathbb{R}^{B \times C \times 2T \times H \times W}.
$$

So once the cache is warm:

$$
T \to 2T.
$$

This reshape is the core reason a single later latent frame can expand into multiple pixel frames during decoding.

## Encoder Structure

## `WanPosteriorEncoder`

`WanPosteriorEncoder` is the real video encoder. It does not flatten frames into independent images.

It has:

- `self.temporal_window = cfg.temporal_downsample_factor()`
- a `WanEncoder3d` backbone
- a `1x1x1` causal `moments_conv` that outputs `2 * z_dim` channels

The two halves of the output are:

$$
[\mu, \log \sigma^2] \in \mathbb{R}^{B \times 2z \times T_z \times H/4 \times W/4}.
$$

Then the code returns:

$$
\mu,\ \log \sigma^2 \in \mathbb{R}^{B \times z \times T_z \times H/4 \times W/4}.
$$

## Public API vs Internal Chunking

At the top level, you call:

```python
model.encode_frame_sequence(images)
```

with `images` shaped `(B, T, C, H, W)`.

That does pass the full clip into `WanPosteriorEncoder.forward`, but inside `forward()` the video is split like this:

1. first chunk: the very first frame only
2. later chunks: windows of `temporal_window = 4` frames

For a `13`-frame clip:

$$
[x_0]\ [x_1,x_2,x_3,x_4]\ [x_5,x_6,x_7,x_8]\ [x_9,x_{10},x_{11},x_{12}].
$$

That is why `13` frames become `4` latent frames.

## Why The First Latent Represents One Frame

This is not learned dynamically. It is decided by the control flow in the encoder and the cached `Resample(..., mode=\"downsample3d\")` behavior.

For the first chunk:

- the encoder feeds only `x_0`
- each temporal downsample layer sees that its cache is empty
- instead of running the temporal stride-2 convolution, it stores the feature map in cache and leaves the temporal length unchanged

So the first chunk keeps temporal length `1` all the way through the encoder.

Observed trace for the default model:

$$
1 \to 1 \to 1
$$

across the two temporal downsample stages.

## Why Each Later Latent Represents Four Frames

For each later chunk, the input length is `4`.

The encoder now has cached features from the previous chunk. At each temporal downsample stage it prepends the last cached frame before applying the stride-2 temporal convolution.

For a later `4`-frame chunk:

### First temporal downsample stage

Input to the temporal conv is:

$$
1\ \text{cached frame} + 4\ \text{current frames} = 5\ \text{frames}.
$$

With kernel `3` and stride `2`, the temporal output length is:

$$
T_1 = \left\lfloor \frac{5 - 3}{2} \right\rfloor + 1 = 2.
$$

### Second temporal downsample stage

Again one cached frame is prepended:

$$
1 + 2 = 3.
$$

Applying the same kernel `3`, stride `2` temporal conv:

$$
T_2 = \left\lfloor \frac{3 - 3}{2} \right\rfloor + 1 = 1.
$$

So each later `4`-frame chunk becomes exactly one latent frame:

$$
4 \to 2 \to 1.
$$

This is why `13 = 1 + 4 + 4 + 4` pixel frames encode to `4 = 1 + 1 + 1 + 1` latent frames.

## Decoder Structure

## `WanVideoDecoder`

The decoder mirrors the encoder:

- `pre_decode_conv` first projects raw latents
- `WanDecoder3d` decodes them
- execution is chunked causally with caches

The forward path is:

1. decode the first latent frame alone
2. decode each later latent frame one at a time
3. concatenate the decoded pixel-frame chunks

For `4` latent frames, the decoder sees:

$$
[z_0]\ [z_1]\ [z_2]\ [z_3].
$$

## Why The First Decoded Latent Produces One Frame

In `Resample(..., mode=\"upsample3d\")`, the first time each temporal upsample layer is visited, the cache entry is `None`.

The code then:

- stores the sentinel value `"Rep"`
- skips the temporal upsample convolution for that first pass

So the first latent stays at temporal length `1` through both temporal upsample stages:

$$
1 \to 1 \to 1.
$$

That is why decoding a one-latent clip gives one pixel frame, not four.

## Why Later Decoded Latents Produce Four Frames

After the first pass, the temporal upsample caches are initialized.

Now each later latent frame goes through two temporal upsample stages.

Each stage does:

1. causal temporal convolution from `C` channels to `2C`
2. reshape `2C` channels into `2` time positions

So a single later latent frame evolves as:

$$
1 \to 2 \to 4.
$$

Observed trace for the default model:

$$
z_1:\ 1 \to 2 \to 4,\qquad
z_2:\ 1 \to 2 \to 4,\qquad
z_3:\ 1 \to 2 \to 4.
$$

This is the exact answer to:

> with 1 latent how do you produce 4 images?

You only get `4` images from a **later** latent frame in a sequence, after the decoder cache has already been initialized by an earlier latent frame.

A lone latent clip behaves differently:

- `[z_0]` decodes to `1` frame
- `[z_0, z_1]` decodes to `5` frames
- `[z_0, z_1, z_2, z_3]` decodes to `13` frames

So one target latent does not usually get decoded in isolation. It is decoded after the context latent(s) have warmed up the causal decoder state.

## Full 13-Frame Worked Example

Assume input video frames:

$$
x_0, x_1, x_2, \dots, x_{12}.
$$

### Encoding

The encoder chunks them as:

- chunk `0`: `[x_0]`
- chunk `1`: `[x_1, x_2, x_3, x_4]`
- chunk `2`: `[x_5, x_6, x_7, x_8]`
- chunk `3`: `[x_9, x_{10}, x_{11}, x_{12}]`

The output latents are:

- `z_0` from chunk `0`
- `z_1` from chunk `1`
- `z_2` from chunk `2`
- `z_3` from chunk `3`

So:

$$
(B, 3, 13, H, W) \to (B, z, 4, H/4, W/4).
$$

With the default `WorldModel()` at `120 x 160`:

$$
(B, 3, 13, 120, 160) \to (B, 32, 4, 30, 40).
$$

### Posterior moments and sampling

The encoder predicts:

$$
q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))
$$

where the code stores `log_var = \log \sigma^2`.

The reparameterization trick is:

$$
z = \mu + \sigma \odot \epsilon,\qquad
\sigma = \exp\!\left(\frac{1}{2}\log \sigma^2\right),\qquad
\epsilon \sim \mathcal{N}(0, I).
$$

The KL term in the file is:

$$
\operatorname{KL}(q(z \mid x)\,\|\,p(z))
= \frac{1}{2}\,\mathbb{E}\left[\exp(\log \sigma^2) + \mu^2 - 1 - \log \sigma^2\right].
$$

### Decoding

The decoder processes:

- `z_0` alone -> `1` frame
- `z_1` with warm cache -> `4` frames
- `z_2` with warm cache -> `4` frames
- `z_3` with warm cache -> `4` frames

So:

$$
(B, z, 4, H/4, W/4) \to (B, 3, 13, H, W).
$$

The reconstructed frame groups are:

- `z_0 -> x_0`
- `z_1 -> x_1, x_2, x_3, x_4`
- `z_2 -> x_5, x_6, x_7, x_8`
- `z_3 -> x_9, x_{10}, x_{11}, x_{12}`

That is the exact `13 -> 4 -> 13` schedule implemented by the current code.

## Why `decode_target_latents()` Concatenates Context And Target

`WorldModel.decode_target_latents()` does:

$$
\text{full\_latents} = [\text{context\_latents}, \text{target\_latents}]
$$

then decodes the full sequence and crops away the context pixel frames.

This is necessary because target latents are not temporally self-sufficient. The decoder needs the earlier latent frames to initialize its causal caches correctly.

Example with one context latent and one target latent:

- context latent count: `1`
- target latent count: `1`
- full latent sequence length: `2`
- decoded full pixel length: `5`
- crop away the first context pixel frame
- remaining target prediction length: `4`

So the operational rule is not:

> decode one target latent by itself and get four future frames

It is:

> decode the context latent(s) plus the target latent(s) together, then crop off the context frames.

## Image Wrappers

`WanVAEEncoder` and `WanVAEDecoder` are thin wrappers for 4D image tensors.

They just do:

- `unsqueeze(2)` to turn an image into a one-frame video
- run the same video VAE
- `squeeze(2)` on the way out

So for a single image:

$$
(B, 3, H, W) \leftrightarrow (B, z, H/4, W/4).
$$

This path always behaves like the first-frame case:

- one image -> one latent map
- one latent map -> one image

There is no `1 latent -> 4 images` behavior for the single-image wrapper.

## Stage-By-Stage Temporal Sizes In The Default Model

For the default `WorldModel()` configuration:

### Encoder

- first chunk: `1 -> 1 -> 1`
- later 4-frame chunk: `4 -> 2 -> 1`

### Decoder

- first latent chunk: `1 -> 1 -> 1`
- later latent chunk: `1 -> 2 -> 4`

This startup asymmetry is the core idea behind the Wan causal temporal tokenizer used here.

## Design Summary

The major design choices in this implementation are:

- The VAE is a true causal video tokenizer, not a framewise image VAE.
  Why: temporal compression and reconstruction depend on past context, so frames cannot be treated independently.

- The first temporal token is special.
  Why: at the start of a sequence there is no past context, so the model preserves the first frame as a singleton instead of pretending there were previous frames.

- Temporal compression is implemented with chunked cached execution.
  Why: this preserves causal semantics across long videos while keeping the code close to streaming Wan behavior.

- Target latents are decoded together with context latents and then cropped.
  Why: later latents need warmed decoder caches to expand into the correct number of future pixel frames.

## Bottom Line

For the current code:

- `13` aligned pixel frames become `4` latent frames
- the split is `1 + 4 + 4 + 4`, not `4 + 4 + 4 + 1`
- the first latent corresponds to the first pixel frame only
- each later latent corresponds to four later pixel frames
- a lone latent decodes to one frame
- a later latent can decode to four frames only when previous latent frames have already initialized the decoder cache
