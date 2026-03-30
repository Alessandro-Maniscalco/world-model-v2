# Architecture

## Status

This document describes the current minimal faithful three-stage Interactive
World Sim path as of March 29, 2026.

Implemented:

- Stage 1 autoencoder training
- Stage 2 latent dynamics training
- Stage 3 decoder finetuning
- checkpoint save/load and stage bootstrapping
- Stage-1 reconstruction visualization
- Stage-2/3 rollout visualization

Still intentionally simplified relative to upstream:

- plain PyTorch instead of Hydra / Lightning / W&B
- raw HDF5 dataset path only
- lightweight latent dynamics module instead of the larger upstream stack
- no zarr-cache backend

## High-level pipeline

The repo now follows the same stage boundaries as the upstream
`interactive_world_sim` training flow while staying much smaller:

1. Stage 1 learns an encoder-decoder latent autoencoder.
2. Stage 2 freezes the Stage-1 autoencoder and learns latent dynamics.
3. Stage 3 freezes encoder and dynamics, then finetunes the decoder to be more
   robust to latent noise.

At a high level:

$$
z = E_\phi(x)
$$

$$
\hat{x}_{\sigma_s} = D_\theta(x_{\sigma_t}; t, s, z)
$$

$$
\hat{z}_{\sigma_s} = F_\psi(z_{\sigma_t}; t, s, a)
$$

where:

- $`E_\phi`$ is the CNN image encoder
- $`D_\theta`$ is the consistency-style decoder
- $`F_\psi`$ is the action-conditioned latent dynamics model
- $`a`$ is the action sequence

Inference for future prediction is:

$$
x_0 \xrightarrow{E_\phi} z_0 \xrightarrow{F_\psi, a_{1:T}} \hat{z}_{1:T}
\xrightarrow{D_\theta} \hat{x}_{1:T}
$$

## Dataset and shapes

Primary dataset:

- https://huggingface.co/datasets/yixuan1999/interactive-world-sim-data

The current default configuration from
[world_model_v2/config.py](/home/amaniscalco/world-model-v2/world_model_v2/config.py)
is still:

- `task = "single_grasp"`
- `obs_keys = ("camera_1_color",)`
- `resolution = 128`
- `action_dim = 4`

The dataset loader at
[world_model_v2/datasets/latent_dynamics/real_aloha_dataset.py](/home/amaniscalco/world-model-v2/world_model_v2/datasets/latent_dynamics/real_aloha_dataset.py)
returns sequence-shaped samples:

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

For the current task:

- image channels per view: `3`
- action dim: `4`
- Stage-1 default train horizon: `1`
- Stage-2 typical train horizon: `> 1`
- validation horizon: `val_horizon`

With batch size $`B`$, time $`T`$, and views $`V`$:

$$
O \in \mathbb{R}^{B \times T \times (3V) \times H \times W}
$$

The encoder downsamples twice, so with `128x128` inputs and one view:

$$
E_\phi :
\mathbb{R}^{(BT) \times 3 \times 128 \times 128}
\to
\mathbb{R}^{(BT) \times 4 \times 32 \times 32}
$$

The encoder output is channel-wise normalized before reuse by Stage 1, Stage 2,
or Stage 3.

## Components

### Encoder

The encoder in
[world_model_v2/algorithms/models/cnn_encoder.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/models/cnn_encoder.py)
is unchanged from Stage 1:

- two spatial downsampling convs
- output latent grid shape `(latent_channels, resolution / 4, resolution / 4)`
- channel-wise latent normalization after encoding

### Decoder

The decoder in
[world_model_v2/algorithms/models/cm_decoder.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/models/cm_decoder.py)
is the same lightweight consistency-style conditional U-Net:

- timestep-pair conditioning
- FiLM-style residual blocks
- multi-scale latent injection
- iterative denoising from noisy images or pure noise

Compactly, one decoder step can be written as:

$$
c_{t,s} = \operatorname{MLP}([\operatorname{emb}(t), \operatorname{emb}(s)])
$$

$$
h_0 = x_{\sigma_t} + U_0(z), \qquad
h_{\ell+1} = \operatorname{ResBlock}_\ell(h_\ell; c_{t,s}) + U_\ell(z)
$$

where each residual block is FiLM-modulated by the timestep-pair condition,
for example

$$
\operatorname{FiLM}(h; c_{t,s}) = \gamma(c_{t,s}) \odot h + \beta(c_{t,s}).
$$

The decoder output is then

$$
\hat{x}_{\sigma_s} = D_\theta(x_{\sigma_t}; t, s, z),
$$

and iterative sampling applies this step repeatedly along a descending schedule:

$$
x_{\sigma_{k+1}} = D_\theta(x_{\sigma_k}; \sigma_k, \sigma_{k+1}, z),
\qquad
\sigma_K > \cdots > \sigma_0 = 0,
$$

starting either from a heavily noised image or from pure noise.

### Latent dynamics

The Stage-2/3 dynamics model lives in
[world_model_v2/algorithms/models/cm_latent_dynamics.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/models/cm_latent_dynamics.py).

It keeps the same broad responsibilities as the upstream latent-dynamics stack,
but in a smaller local form:

- input/output shape: `(B, T, C_latent, H_latent, W_latent)`
- internal shape: `(B, C_latent, T, H_latent, W_latent)`
- spatial Conv3d residual blocks with kernels `(1, 3, 3)`
- timestep-pair embeddings from the shared sinusoidal helper
- per-frame action MLP conditioning
- FiLM conditioning inside residual blocks
- causal temporal self-attention for temporal mixing
- one spatial downsample stage, one bottleneck, one upsample stage, then a
  `1x1x1` output projection back to latent channels

This is the main “similar structure, lighter implementation” compromise versus
the upstream repo.

## Training stages

### Stage 1

Stage 1 optimizes the decoder-side denoising objective:

$$
z = E_\phi(x)
$$

$$
x_{\sigma_t} = x + \sigma_t \epsilon_t,\qquad
x_{\sigma_s} = x + \sigma_s \epsilon_s, \qquad
\sigma_t > \sigma_s \ge 0
$$

$$
\hat{x}_{\sigma_s} = D_\theta(x_{\sigma_t}; t, s, z)
$$

with weighted denoising loss plus a clean reconstruction term:

$$
\mathcal{L}_{stage1}
=
w(t)\lVert \hat{x}_{\sigma_s} - x_{\sigma_s} \rVert_2^2
+ 0.1 \lVert \hat{x}_0 - x \rVert_2^2
$$

In code, $`\hat{x}_0`$ comes from a second decoder call with the clean image
and zero timesteps, so the clean term is an explicit reconstruction anchor
rather than something inferred from the noisy-input pass.


### Stage 2

Stage 2 freezes encoder and decoder, encodes the full training sequence into
latents, and trains the dynamics model in a teacher-forced full-window mode
that is closer to the upstream `terminal_only` setup.

$$
z_{1:T} = E_\phi(x_{1:T})
$$

For a training window of length $`T`$, earlier frames are given matched low
noise while the last frame is trained as the terminal denoising target:

$$
z_{\sigma_t}, z_{\sigma_s}
=
\operatorname{NoisePair}(z_{1:T}; t_{1:T}, s_{1:T}),
$$

where for `terminal_only` with one dynamics denoising step:

$$
t_{1:T-1} = s_{1:T-1} \sim \text{low-noise}, \qquad
t_T = \sigma_{\max}, \qquad s_T = 0.
$$

The dynamics model then predicts the lower-noise latent sequence with aligned
action conditioning:

$$
\hat{z}_{\sigma_s} = F_\psi(z_{\sigma_t}; t_{1:T}, s_{1:T}, a_{1:T}).
$$

The implemented Stage-2 loss is:

$$
\mathcal{L}_{stage2}
=
w(t)\lVert \hat{z}_{\sigma_s} - z_{\sigma_s} \rVert_2^2
$$

with the current Stage-2 default using uniform weighting. This is still
teacher-forced training rather than full open-loop unrolling, but it now
matches the upstream Stage-2 data flow much more closely:

- the whole training window is passed through the dynamics model at once
- earlier context frames keep matched low noise with $`t = s`$
- the final frame is the terminal denoising target
- action conditioning stays aligned 1:1 with the latent window, including the
  target-step action slot

### Stage 3

Stage 3 loads a Stage-2 checkpoint, freezes encoder and dynamics, adds Gaussian
latent jitter, and finetunes only the decoder:

$$
z = E_\phi(x)
$$

$$
\tilde{z} = \operatorname{normalize}(z + \eta),
\qquad
\eta \sim \mathcal{N}(0, \sigma_{stage3}^2)
$$

$$
\hat{x}_{\sigma_s} = D_\theta(x_{\sigma_t}; t, s, \tilde{z})
$$

The loss is the same decoder-side denoising objective as Stage 1, but the
decoder is now being trained against noisy latent conditioning.

## Checkpoint dependency chain

The stage bootstrapping path is:

- Stage 1 checkpoint contains encoder + decoder
- Stage 2 loads Stage 1 via `load_ae`, copies encoder + decoder, freezes them,
  and trains dynamics
- Stage 3 loads Stage 2 via `load_ae`, copies encoder + dynamics + decoder,
  freezes encoder + dynamics, and finetunes decoder

This logic lives in
[world_model_v2/algorithms/latent_dynamics/latent_world_model.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/latent_dynamics/latent_world_model.py)
and is triggered by the experiment runner before optimizer creation.

## Noise scheduler

The scheduler in
[world_model_v2/algorithms/latent_dynamics/noise_scheduler.py](/home/amaniscalco/world-model-v2/world_model_v2/algorithms/latent_dynamics/noise_scheduler.py)
is still a lightweight linear sigma schedule, but it now works on both image
batches and sequence-shaped latent tensors.

It supports:

- timestep-pair sampling for arbitrary leading shapes
- Gaussian noise injection with broadcasting
- inverse-variance loss weights
- short descending schedules for iterative denoising

It is intentionally much simpler than the upstream DDPM / CTM helper stack.

## Validation and inference

### Stage 1 validation

Stage 1 validation reconstructs the observed sequence:

- encode frames into latents
- start from either max-noised input or pure noise
- iteratively decode back into images

The Stage-1 CLI remains:

- [world_model_v2/infer/reconstruct_episode.py](/home/amaniscalco/world-model-v2/world_model_v2/infer/reconstruct_episode.py)

### Stage 2 and Stage 3 validation

Stage 2 and Stage 3 validation use the rollout path:

1. encode the first observed frame into a seed latent
2. autoregress one future latent at a time
3. keep a sliding latent context capped by the configured train horizon
4. initialize each new target slot with scheduler-scaled Gaussian noise
5. decode predicted latents back into images from pure noise

The dedicated rollout CLI is:

- [world_model_v2/infer/predict_rollout.py](/home/amaniscalco/world-model-v2/world_model_v2/infer/predict_rollout.py)

Validation and inference stats now track:

- input frame count
- predicted frame count for rollout stages
- decoded frame count
- `dyn_loss_teacher_forced` in training logs for the next-step denoising loss
- `dyn_loss_rollout` as open-loop latent MSE on future frames
- latent shape
- exported GIF frame count
- rollout context size

## Key config knobs

The main stage-specific algorithm config fields are now:

- `training_stage`
- `load_ae`
- `action_dim`
- `infer_steps`
- `dyn_infer_steps`
- `dynamics_hidden_channels`
- `action_emb_dim`
- `dynamics_attention_heads`
- `mask_prev_action`
- `stage3_latent_noise_std`

The CLI entrypoint remains:

```bash
source .venv/bin/activate
python -m world_model_v2.run ...
```

## Matching upstream vs simplified pieces

Structurally matched:

- Stage 1 / 2 / 3 split
- checkpoint bootstrapping through `load_ae`
- encoder-decoder Stage-1 autoencoder
- separate latent dynamics module
- rollout-based future prediction
- stage-aware inference entrypoints

Intentionally simplified:

- plain-PyTorch runner and config
- smaller dynamics network
- lightweight scheduler
- raw HDF5 loader only
- one-camera default path

So the current repo is best described as:

$$
\text{minimal faithful 3-stage implementation}
\neq
\text{full upstream training system}
$$
