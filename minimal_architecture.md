# Minimal Architecture

## Status

This document describes the implemented minimal world-model path in
`world_model_v2/minimal/` as it exists in this repo now.

The purpose of this path is still a small, explicit, debuggable setup that can
overfit one short clip and let you inspect reconstructions, rollouts,
checkpoints, and plateau stopping without the complexity of the larger staged
world-model code.

Major design choices:

- one fixed clip instead of the full dataset
- one Wan-style VAE autoencoder backend
- one separate latent dynamics module
- no actions
- no joint AE+dynamics training mode
- KL-regularized VAE training in `ae_only`
- deterministic inference with posterior mean latents
- periodic validation with GIF/grid outputs
- validation-based plateau stopping by default

## Fixed data setup

By default the minimal path uses:

- task: `single_grasp`
- split: `val`
- episode: `0`
- camera: `camera_1_color`
- frames: `111..116` inclusive
- resolution: `128x128`

That gives:

- `6` total frames: `x_111, x_112, x_113, x_114, x_115, x_116`
- `5` one-step transitions:
  - `(x_111 -> x_112)`
  - `(x_112 -> x_113)`
  - `(x_113 -> x_114)`
  - `(x_114 -> x_115)`
  - `(x_115 -> x_116)`

There are no actions in this minimal path.

## Shared model

The shared model is:

$$
q_\phi(z_t \mid x_t)
$$

$$
\hat{z}_{t+1} = F_\psi(z_t)
$$

$$
\hat{x}_t = D_\theta(z_t)
$$

where:

- $q_\phi$ is the encoder posterior
- $F_\psi$ is the latent dynamics module
- $D_\theta$ is the decoder

### Wan-style VAE config

The Wan-style backend uses these internal defaults:

```python
cfg = dict(
    dim=64,
    z_dim=4,
    dim_mult=[1, 2, 4],
    num_res_blocks=1,
    attn_scales=[],
    temperal_downsample=[False, False],
    dropout=0.0,
)
```

Important notes:

- the Wan backend is adapted into this repo; it is not a verbatim copy of the
  upstream file
- the minimal implementation drops streaming/cache logic and pretrained latent
  normalization constants
- images are processed internally as single-frame videos with `T=1`

### Latent shape

The latent spatial shape is derived from the Wan config, not hard-coded.

For the default Wan config:

- spatial downsample factor = `2^(len(dim_mult)-1) = 4`
- `128x128 -> 32x32`
- latent shape = `B x 4 x 32 x 32`

In general:

$$
\text{latent height} = \frac{\text{resolution}}{\text{spatial downsample factor}}
$$

$$
\text{latent width} = \frac{\text{resolution}}{\text{spatial downsample factor}}
$$

The code validates that `resolution` is divisible by the Wan downsample
factor.

## Modes

The minimal path now supports only:

- `ae_only`
- `dynamics_only`

`joint` has been removed.

### `ae_only`

Trainable:

- encoder: trainable
- decoder: trainable
- dynamics: frozen

Dataset:

- all 6 frames independently

Training behavior:

1. encode `x_t` into posterior moments `mu_t, log_var_t`
2. sample `z_t` with reparameterization
3. decode `z_t`
4. optimize reconstruction plus KL

Training loss:

$$
\mathcal{L}_{ae}
=
\lVert D_\theta(z_t) - x_t \rVert_2^2
+ \beta \, \mathrm{KL}\left(q_\phi(z_t \mid x_t) \,\|\, \mathcal{N}(0, I)\right)
$$

Implemented as:

- `recon_mse = MSE(reconstructed, frame)`
- `kl_loss = KL(mu, log_var)`
- `ae_loss = recon_mse + kl_beta * kl_loss`

Default:

- `kl_beta = 1e-4`

Validation behavior:

- validation is deterministic
- encode with posterior mean `mu`
- decode `mu` directly
- metric of record is `ae_loss`

Saved validation stats include:

- `recon_mse`
- `kl_loss`
- `ae_loss`
- `input_frame_count = 6`
- `decoded_frame_count = 6`

Interpretation:

- if `recon_mse` stays high, the autoencoder itself is not reconstructing well
- if `kl_loss` dominates, the KL pressure is too strong for this tiny overfit
  setup

### `dynamics_only`

Trainable:

- encoder: frozen
- decoder: frozen
- dynamics: trainable

Required checkpoint loading:

- `--load-encoder-decoder` is required unless resuming from an existing
  `dynamics_only` run

Dataset:

- the 5 transitions `(x_t, x_{t+1})`

Training behavior:

- encode current and next frames with deterministic posterior mean `mu`
- keep both encoder and decoder frozen
- train only the latent dynamics model

Training loss:

$$
\mathcal{L}_{dyn}
=
\lVert F_\psi(\mu_t) - \mu_{t+1} \rVert_2^2
$$

This is implemented as latent-space MSE.

Validation rollout:

1. encode the seed frame `x_111` into deterministic latent `mu_111`
2. roll out latents autoregressively with the dynamics module
3. decode each predicted latent with the frozen decoder
4. compare predicted frames against frames `112..116`

Validation metric of record:

- `rollout_mse`

Saved validation stats include:

- `rollout_mse`
- `input_frame_count = 6`
- `predicted_frame_count = 6`
- `decoded_frame_count = 6`
- `seed_frames = 1`
- `loss_frames = 5`

## Checkpoints

Minimal checkpoints are versioned separately from the older staged world-model
checkpoints.

Current checkpoint kind:

- `kind = "world_model_v2_minimal_v2"`

Payload contains:

- `kind`
- `model_state`
- `optimizer_state`
- `scheduler_state`
- `step`
- `config`
- `mode`
- `clip_metadata`
- `best_metric`
- `ae_backend`
- `autoencoder`

For current checkpoints:

- `ae_backend = "wan"`
- `autoencoder["backend"] = "wan"`

Saved checkpoints:

- `checkpoints/last.pt`
- `checkpoints/best.pt`

Best metric selection:

- `ae_only` -> best `ae_loss`
- `dynamics_only` -> best `rollout_mse`

Checkpoint compatibility:

- Wan backend metadata is checked during partial loading and resume
- incompatible checkpoint metadata still fails fast instead of relying on shape
  errors

## Training loop and plateau stopping

The minimal runner validates every `validation_interval` steps and writes:

- `samples/step_xxxxxx/episode_0_grid.png`
- `samples/step_xxxxxx/episode_0.gif`
- `samples/step_xxxxxx/episode_0_stats.json`

It also supports validation-based plateau stopping by default.

Default early-stop settings:

- `early_stop_window_size = 1`
- `early_stop_patience_windows = 5`
- `early_stop_min_delta = 1e-10`
- `early_stop_warmup_steps = 300`

Early-stop metric:

- `ae_only` -> `ae_loss`
- `dynamics_only` -> `rollout_mse`

## CLI defaults

The minimal CLI entrypoint is:

```bash
python -m world_model_v2.minimal.run
```

Important defaults:

- `--mode ae_only`
- `--kl-beta 1e-4`
- `--data-root data/full`
- `--task single_grasp`
- `--split val`
- `--episode 0`
- `--camera camera_1_color`
- `--frame-start 111`
- `--frame-end 116`
- `--resolution 128`
- `--latent-channels 4`
- `--hidden-channels 64`
- `--batch-size 32`
- `--auto-batch-size` disabled by default
- `--lr 1e-4`
- `--max-steps 3000`
- `--validation-interval 100`
- `--checkpoint-interval 100`

## Practical commands

### Train the Wan-style VAE only

```bash
python -m world_model_v2.minimal.run \
  --mode ae_only \
  --auto-batch-size \
  --run-name ae_only_single_grasp_ep0_f111_116 \
  --output-dir outputs/minimal \
  --device cuda
```

### Train dynamics on a pretrained minimal AE

```bash
python -m world_model_v2.minimal.run \
  --mode dynamics_only \
  --auto-batch-size \
  --load-encoder-decoder outputs/minimal/ae_only_single_grasp_ep0_f111_116/checkpoints/best.pt \
  --run-name dynamics_only_single_grasp_ep0_f111_116 \
  --output-dir outputs/minimal \
  --device cuda
```

## Auto batch sizing

The minimal runner can optionally probe for the largest training batch size
that fits before training starts:

```bash
--auto-batch-size
```

Behavior:

- on CUDA, the runner probes batch sizes and keeps the largest size that fits a
  full forward/backward/optimizer-step dry run
- on CPU, the runner uses the full cached clip size because this minimal path
  only trains on one tiny clip
- the probe restores RNG state afterward so enabling auto batch sizing does not
  change the training randomness for the same seed
- if a real CUDA OOM still happens during training, the runner halves the batch
  size, rebuilds the train loader, and continues instead of crashing immediately
