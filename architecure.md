# Architecture

## Status

This document describes the current default architecture for the Stage-1
Interactive World Sim path as of March 28, 2026.

The repo is now structured to resemble the upstream
`interactive_world_sim` package layout, but the implemented model is still a
lightweight, plain-PyTorch Stage-1 system only.

Implemented:

- Stage 1 autoencoder training
- checkpoint save/load
- reconstruction visualization

Not implemented yet:

- Stage 2 latent dynamics training
- Stage 3 autoencoder finetuning
- upstream Hydra / Lightning / W&B stack
- upstream zarr-cache training backend

## Project goal

Build a small world-model pretraining path for the Interactive World Sim
dataset that is practical on the local `NVIDIA GeForce RTX 3080 Laptop GPU`
while keeping the repo structurally close to the upstream codebase.

Current target conditional reconstruction problem:

$$
\hat{x}_{\sigma_s} = D_\theta(x_{\sigma_t}; \sigma_t, \sigma_s, z),
\qquad
z = E_\phi(x)
$$

where:

- $`E_\phi`$ is a CNN image encoder
- $`D_\theta`$ is a consistency-style conditional decoder
- $`x_{\sigma_t}`$ is a noisy image at noise level $`\sigma_t`$
- $`x_{\sigma_s}`$ is a lower-noise target at noise level $`\sigma_s`$
- $`z`$ is a compact 2D latent grid

This follows the Stage-1 shape of the paper and upstream repo:

- encode RGB observations into a 2D latent
- decode from noisy image + latent conditioning
- train with a weighted denoising-style regression loss

## Dataset and sample shapes

Primary dataset:

- https://huggingface.co/datasets/yixuan1999/interactive-world-sim-data

Current default configuration from
[config.py](/home/amaniscalco/world-model-v2/world_model_v2/config.py):

- `task = "single_grasp"`
- `obs_keys = ("camera_1_color",)`
- `resolution = 128`
- `horizon = 1`
- `val_horizon = 200`
- `action_mode = "single_grasp"`

Raw episode observations in this dataset are RGB frames plus action vectors.
For the current task:

- raw image shape: $`480 \times 640 \times 3`$
- resized training image shape: $`128 \times 128 \times 3`$
- action shape per frame: $`4`$

The dataset loader at
[real_aloha_dataset.py](/home/amaniscalco/world-model-v2/world_model_v2/datasets/latent_dynamics/real_aloha_dataset.py)
exposes sequence-shaped samples:

```python
{
  "obs": {
    "camera_1_color": Tensor[T, C, H, W]
  },
  "action": Tensor[T, A],
  "episode_idx": Tensor[],
  "start_idx": Tensor[],
  "frame_idx": Tensor[T],
  "valid_length": Tensor[],
}
```

Current dimensions:

- $`T = 1`$ for training
- $`T = 200`$ for validation episodes
- $`C = 3`$
- $`H = W = 128`$
- $`A = 4`$

With batch size $`B`$ and number of views $`V = |\text{obs\_keys}|`$, the model
concatenates views along channels:

$$
O \in \mathbb{R}^{B \times T \times (3V) \times H \times W}
$$

For the current default:

$$
O \in \mathbb{R}^{B \times 1 \times 3 \times 128 \times 128}
$$

Inside training, the sequence is flattened across batch and time:

$$
X = \operatorname{reshape}(O) \in \mathbb{R}^{(B T) \times 3V \times 128 \times 128}
$$

## High-level model structure

The main Stage-1 model class is
[latent_world_model.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/latent_dynamics/latent_world_model.py):

$$
x \xrightarrow{E_\phi} z \xrightarrow{D_\theta(\cdot; t, s, z)} \hat{x}_{\sigma_s}
$$

It owns:

- encoder application
- latent normalization
- decoder application
- Stage-1 training loss
- iterative reconstruction
- validation preview generation

The structurally upstream-matched split is:

- `algorithms/latent_dynamics/latent_world_model.py`
- `algorithms/latent_dynamics/noise_scheduler.py`
- `algorithms/models/cnn_encoder.py`
- `algorithms/models/cm_decoder.py`
- `algorithms/models/blocks.py`
- `algorithms/models/embeddings.py`

## Encoder

The encoder lives in
[cnn_encoder.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/models/cnn_encoder.py).

It is a two-downsample CNN:

1. `Conv2d(image_channels, hidden_channels, 3, stride=1, padding=1)`
2. `SiLU`
3. `Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)`
4. `SiLU`
5. `Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)`
6. `SiLU`
7. `Conv2d(hidden_channels, latent_channels, 3, stride=2, padding=1)`

Let:

- $`C_{img} = 3V`$, V is the number of views
- $`C_h = \text{hidden\_channels}`$, controls the width of the CNN
- $`C_z = \text{latent\_channels}`$, depth of the latent space output

Then:

$$
E_\phi :
\mathbb{R}^{(BT) \times C_{img} \times 128 \times 128}
\to
\mathbb{R}^{(BT) \times C_z \times 32 \times 32}
$$

Default dimensions:

- $`V = 1`$
- $`C_{img} = 3`$
- $`C_h = 64`$ by config default
- $`C_z = 4`$ by config default

So the default latent tensor per frame is:

$$
z \in \mathbb{R}^{(BT) \times 4 \times 32 \times 32}
$$

The smoke run used:

- `hidden_channels = 32`
- `latent_channels = 4`
- `latent_dim = 64`

After encoding, the code applies channel-wise normalization:

$$
\tilde{z}_{b,:,h,w} =
\frac{z_{b,:,h,w}}
{\lVert z_{b,:,h,w} \rVert_2 + \varepsilon}
$$

where $`\varepsilon = 10^{-6}`$.

This is one of the clearest structural similarities to the upstream
`LatentWorldModel`.

## Decoder

The decoder lives in
[cm_decoder.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/models/cm_decoder.py).

It is a lightweight consistency-style conditional U-Net with:

- timestep embeddings
- FiLM-style residual conditioning
- skip connections
- multi-scale latent injection

### Timestep conditioning

For timestep indices $`t`$ and $`s`$, the model builds sinusoidal embeddings:

$$
e_t \in \mathbb{R}^{d},
\qquad
e_s \in \mathbb{R}^{d}
$$

where $`d = \text{latent\_dim}`$.

They are concatenated and projected:

$$
c = \operatorname{MLP}([e_t; e_s]) \in \mathbb{R}^{d}
$$

Default:

$$
d = 128
$$

Smoke run:

$$
d = 64
$$

### Decoder data flow

The decoder input is:

- noisy image: $`x \in \mathbb{R}^{(BT) \times C_{img} \times 128 \times 128}`$
- latent grid: $`\tilde{z} \in \mathbb{R}^{(BT) \times C_z \times 32 \times 32}`$
- conditioning vector: $`c \in \mathbb{R}^{(BT) \times d}`$

The latent is injected at three scales:

1. input scale: projected to $`C_{img}`$ channels and resized to `128x128`
2. mid scale: projected to $`C_h`$ channels and resized to `64x64`
3. low scale: projected to $`2C_h`$ channels and resized to `32x32`

With default one-view dimensions:

- input image channels: $`C_{img} = 3`$
- hidden channels: $`C_h = 64`$
- latent channels: $`C_z = 4`$

Shape flow:

$$
(BT, 3, 128, 128)
\xrightarrow{\text{down1}}
(BT, C_h, 64, 64)
\xrightarrow{\text{down2}}
(BT, 2C_h, 32, 32)
\xrightarrow{\text{mid}}
(BT, 2C_h, 32, 32)
\xrightarrow{\text{up1}}
(BT, C_h, 64, 64)
\xrightarrow{\text{up2}}
(BT, C_h, 128, 128)
\xrightarrow{\text{out}}
(BT, 3, 128, 128)
$$

For the current smoke configuration with $`C_h = 32`$:

$$
(BT, 3, 128, 128)
\to
(BT, 32, 64, 64)
\to
(BT, 64, 32, 32)
\to
(BT, 64, 32, 32)
\to
(BT, 32, 64, 64)
\to
(BT, 32, 128, 128)
\to
(BT, 3, 128, 128)
$$

The final layer uses `sigmoid`, so decoded outputs are in:

$$
\hat{x} \in [0,1]^{(BT) \times 3 \times 128 \times 128}
$$

## Noise schedule

The scheduler lives in
[noise_scheduler.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/latent_dynamics/noise_scheduler.py).

It defines a simple linear schedule:

$$
\sigma_i \in [\sigma_{\min}, \sigma_{\max}],
\qquad
i \in \{0, \dots, N-1\}
$$

where:

- $`N = \text{timesteps}`$
- $`\sigma_{\min} = 0.01`$
- $`\sigma_{\max} = 1.0`$

Default:

$$
N = 32
$$

Smoke run:

$$
N = 8
$$

Training samples a pair of timestep indices:

$$
t \sim \{1, \dots, N-1\},
\qquad
s \sim \{0, \dots, t-1\}
$$

Noise is added as:

$$
x_{\sigma_t} = x + \sigma_t \epsilon_t,
\qquad
x_{\sigma_s} = x + \sigma_s \epsilon_s
$$

Important note: unlike the idealized paper equation and the upstream intent,
the current local code samples the two Gaussian noise tensors independently.
So here $`\epsilon_t`$ and $`\epsilon_s`$ are not forced to be the same draw.

Loss weights are:

$$
w(t) = \frac{1}{\sigma_t^2 + 10^{-6}}
$$

## Stage-1 training objective

For one flattened image $`x`$:

1. encode:

$$
\tilde{z} = E_\phi(x)
$$

2. sample timesteps:

$$
t > s \ge 0
$$

3. build noisy source and lower-noise target:

$$
x_{\sigma_t} = x + \sigma_t \epsilon_t,
\qquad
x_{\sigma_s} = x + \sigma_s \epsilon_s
$$

4. predict the lower-noise target:

$$
\hat{x}_{\sigma_s} = D_\theta(x_{\sigma_t}; t, s, \tilde{z})
$$

The main weighted denoising loss is:

$$
\mathcal{L}_{recon}
=
w(t)\left\|
\hat{x}_{\sigma_s} - x_{\sigma_s}
\right\|_2^2
$$

The code also adds a direct clean reconstruction term:

$$
\hat{x}_0 = D_\theta(x; 0, 0, \tilde{z})
$$

$$
\mathcal{L}_{clean}
=
\left\|
\hat{x}_0 - x
\right\|_2^2
$$

Total implemented loss:

$$
\mathcal{L}
=
\mathcal{L}_{recon}
+
0.1\,\mathcal{L}_{clean}
$$

This is faithful to the current code, not to the full upstream training recipe.

## Reconstruction path

At inference or validation preview time, the model:

1. encodes the observation sequence
2. starts from either:
   - pure noise, or
   - a maximally noised version of the input image
3. iteratively denoises over a short schedule

For sequence input:

$$
O \in \mathbb{R}^{B \times T \times 3V \times 128 \times 128}
$$

the reconstruction output is:

$$
\hat{O} \in \mathbb{R}^{B \times T \times 3V \times 128 \times 128}
$$

For current validation:

- one episode
- one camera
- `T = 200`

so:

$$
\hat{O} \in \mathbb{R}^{1 \times 200 \times 3 \times 128 \times 128}
$$

The exported validation stats explicitly track:

- input frame count
- latent shape
- decoded frame count
- exported GIF frame count

## Training entrypoint and ownership

The canonical training entrypoint is:

```bash
source .venv/bin/activate
python -m world_model_v2.run
```

Ownership is split like this:

1. [run.py](/home/amaniscalco/world-model-v2/world_model_v2/run.py)
   - parses CLI args
   - builds nested config sections
   - launches the experiment
2. [latent_dynamics_experiment.py](/home/amaniscalco/world-model-v2/world_model_v2/experiments/latent_dynamics_experiment.py)
   - builds datasets and dataloaders
   - owns optimizer, checkpointing, and validation previews
3. [latent_world_model.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/latent_dynamics/latent_world_model.py)
   - owns model forward logic, loss, and reconstruction

## Matching upstream vs simplified pieces

Structurally matched:

- upstream-like package layout
- `LatentWorldModel` as the main algorithm object
- encoder and decoder moved into `algorithms/models`
- dataset under `datasets/latent_dynamics`
- experiment runner under `experiments`
- one canonical run entrypoint

Intentionally simplified:

- only `training_stage = 1`
- one-camera default path
- raw HDF5 only
- lightweight scheduler instead of the upstream DDPM/CTM stack
- lightweight consistency decoder instead of the larger upstream decoder
- no Stage 2 latent dynamics
- no Stage 3 finetuning

So the current repo is best described as:

$$
\text{upstream-shaped Stage-1 architecture}
\neq
\text{full upstream training system}
$$
