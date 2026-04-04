# Minimal Architecture

## Status

This document describes the implemented minimal world-model path in
`world_model_v2/minimal/` as it exists in this repo now.

The purpose of this path is not broad generalization. It is a small, explicit,
debuggable setup that can overfit one short clip and let you inspect
reconstructions, rollouts, checkpoints, and plateau stopping without the
complexity of the larger staged world-model code.

Major design choices:

- one fixed clip instead of the full dataset
- one small deterministic `encoder + dynamics + decoder` model
- three modes that share the same submodules
- no actions
- no diffusion, no scheduler, no upstream-style multi-stage stack
- periodic validation with GIF/grid outputs
- validation-based plateau stopping by default
- image-space losses use MSE, which is the averaged squared L2 error

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
z_t = E_\phi(x_t)
$$

$$
\hat{z}_{t+1} = F_\psi(z_t)
$$

$$
\hat{x}_t = D_\theta(z_t)
$$

where:

- $E_\phi$ is the encoder
- $F_\psi$ is the latent dynamics module
- $D_\theta$ is the decoder

### Default dimensions

- input image shape: `B x 3 x 128 x 128`
- latent shape: `B x 4 x 32 x 32`
- output image shape: `B x 3 x 128 x 128`
- hidden channel width: `64`

Important note:

- “latent is `32x32`” means latent spatial resolution, not `32` latent channels
- the default latent channel count is `4`

### Parameter counts at default size

- encoder: `108,420`
- dynamics: `152,388`
- decoder: `172,227`
- total: `433,035`

## Block structures

### Encoder

The encoder is:

1. `Conv2d(3, 64, kernel_size=3, stride=1, padding=1)`
2. `SiLU`
3. `Conv2d(64, 64, kernel_size=4, stride=2, padding=1)`
4. `SiLU`
5. `Conv2d(64, 64, kernel_size=3, stride=1, padding=1)`
6. `SiLU`
7. `Conv2d(64, 4, kernel_size=4, stride=2, padding=1)`
8. `SiLU`

Shape flow:

- `128x128 -> 128x128`
- `128x128 -> 64x64`
- `64x64 -> 64x64`
- `64x64 -> 32x32`

So:

$$
E_\phi : \mathbb{R}^{B \times 3 \times 128 \times 128}
\to \mathbb{R}^{B \times 4 \times 32 \times 32}
$$

### Dynamics

The dynamics module is a residual latent predictor:

1. `input_proj = Conv2d(4, 64, kernel_size=3, padding=1)`
2. `SiLU`
3. residual block 1
4. `SiLU`
5. residual block 2
6. `SiLU`
7. `output_proj = Conv2d(64, 4, kernel_size=3, padding=1)`
8. residual update back to the input latent

Each residual block is:

1. `Conv2d(64, 64, kernel_size=3, padding=1)`
2. `SiLU`
3. `Conv2d(64, 64, kernel_size=3, padding=1)`
4. add skip connection

Compactly:

$$
h_t = \operatorname{Conv}_{3\times3}(z_t)
$$

$$
\Delta z_t = \operatorname{OutProj}(\operatorname{ResStack}(h_t))
$$

$$
\hat{z}_{t+1} = z_t + \Delta z_t
$$

So:

$$
F_\psi : \mathbb{R}^{B \times 4 \times 32 \times 32}
\to \mathbb{R}^{B \times 4 \times 32 \times 32}
$$

### Decoder

The decoder mirrors the encoder:

1. `Conv2d(4, 64, kernel_size=3, stride=1, padding=1)`
2. `SiLU`
3. `ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)`
4. `SiLU`
5. `Conv2d(64, 64, kernel_size=3, stride=1, padding=1)`
6. `SiLU`
7. `ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)`
8. `SiLU`
9. `Conv2d(64, 3, kernel_size=3, stride=1, padding=1)`
10. `sigmoid`

Shape flow:

- `32x32 -> 32x32`
- `32x32 -> 64x64`
- `64x64 -> 64x64`
- `64x64 -> 128x128`
- `128x128 -> 128x128`

So:

$$
D_\theta : \mathbb{R}^{B \times 4 \times 32 \times 32}
\to \mathbb{R}^{B \times 3 \times 128 \times 128}
$$

The sigmoid bounds outputs to `[0, 1]`.

## Modes

The same model is used in all modes. Only trainability and loss change.

### `ae_only`

Trainable:

- encoder: trainable
- dynamics: frozen
- decoder: trainable

Dataset:

- all 6 frames independently

Forward pass:

$$
z_t = E_\phi(x_t)
$$

$$
\hat{x}_t = D_\theta(z_t)
$$

Training loss:

$$
\mathcal{L}_{ae}
=
\lVert \hat{x}_t - x_t \rVert_2^2
$$

This is implemented as image-space MSE.

Validation:

- not open rollout
- each of the 6 frames is reconstructed independently
- metric of record is `recon_mse`

Validation formula:

$$
\mathcal{L}_{val,ae}
=
\frac{1}{6}\sum_{t=111}^{116}\lVert D_\theta(E_\phi(x_t)) - x_t \rVert_2^2
$$

Interpretation:

- this tells you how good the encoder/decoder pair is at reconstructing frames
- if outputs are blurry here, the autoencoder itself is blurry

### `joint`

Trainable:

- encoder: trainable
- dynamics: trainable
- decoder: trainable

Dataset:

- the 5 transitions `(x_t, x_{t+1})`

Forward pass:

$$
z_t = E_\phi(x_t)
$$

$$
\hat{x}_t = D_\theta(z_t)
$$

$$
\hat{z}_{t+1} = F_\psi(z_t)
$$

$$
\hat{x}_{t+1} = D_\theta(\hat{z}_{t+1})
$$

Training loss:

$$
\mathcal{L}_{joint}
=
\lVert \hat{x}_{t+1} - x_{t+1} \rVert_2^2
+ 0.25 \, \lVert \hat{x}_t - x_t \rVert_2^2
$$

This is implemented as:

- `pred_mse = MSE(predicted_next, next_frame)`
- `recon_mse = MSE(reconstructed, current_frame)`
- `loss = pred_mse + 0.25 * recon_mse`

Validation:

- open rollout from frame 111
- frame 111 is the seed
- frames 112..116 are predicted autoregressively
- metric of record is `rollout_mse`

### `dynamics_only`

Trainable:

- encoder: frozen
- dynamics: trainable
- decoder: frozen

Required checkpoint loading:

- `--load-encoder-decoder` is required unless resuming from an existing
  `dynamics_only` run

Dataset:

- the 5 transitions `(x_t, x_{t+1})`

Training forward pass:

$$
z_t = E_\phi(x_t)
$$

$$
z_{t+1} = E_\phi(x_{t+1})
$$

$$
\hat{z}_{t+1} = F_\psi(z_t)
$$

Both encoder calls are done under `torch.no_grad()` because the encoder is
frozen.

Training loss:

$$
\mathcal{L}_{dyn}
=
\lVert \hat{z}_{t+1} - z_{t+1} \rVert_2^2
$$

This is implemented as latent-space MSE.

Validation rollout:

First encode the seed frame:

$$
z_{111} = E_\phi(x_{111})
$$

Then roll out latents autoregressively:

$$
\hat{z}_{112} = F_\psi(z_{111})
$$

$$
\hat{z}_{113} = F_\psi(\hat{z}_{112})
$$

$$
\cdots
$$

$$
\hat{z}_{116} = F_\psi(\hat{z}_{115})
$$

Then decode each predicted latent with the frozen decoder:

$$
\hat{x}_{t} = D_\theta(\hat{z}_{t})
\qquad \text{for } t \in \{112,113,114,115,116\}
$$

Validation metric of record:

$$
\mathcal{L}_{rollout,MSE}
=
\frac{1}{5}\sum_{t=112}^{116}\lVert \hat{x}_t - x_t \rVert_2^2
$$

Interpretation:

- training is latent-only
- validation is image-space open rollout
- if latent MSE looks fine but rollout images drift or blur, the problem is in
  the learned latent transition behavior, not the frozen encoder/decoder alone

## Validation behavior

### `ae_only`

Validation output is reconstruction, not rollout.

The saved stats include:

- `mode = "ae_only"`
- `recon_mse`
- `input_frame_count = 6`
- `decoded_frame_count = 6`

### `joint` and `dynamics_only`

Validation output is open rollout.

The saved stats include:

- `mode = "joint"` or `mode = "dynamics_only"`
- `rollout_mse`
- `input_frame_count = 6`
- `predicted_frame_count = 6`
- `decoded_frame_count = 6`
- `seed_frames = 1`
- `loss_frames = 5`

The saved GIF and grid show:

- frame 111 as the ground-truth seed
- frames 112..116 as predictions

## Data loaders by mode

### `ae_only`

Training loader sample:

```python
{
  "frame": Tensor[3, 128, 128],
  "frame_idx": Tensor[],
  "episode_idx": Tensor[],
}
```

### `joint` and `dynamics_only`

Training loader sample:

```python
{
  "current_frame": Tensor[3, 128, 128],
  "next_frame": Tensor[3, 128, 128],
  "current_frame_idx": Tensor[],
  "next_frame_idx": Tensor[],
  "episode_idx": Tensor[],
}
```

### all modes

Validation loader sample:

```python
{
  "frames": Tensor[6, 3, 128, 128],
  "frame_idx": Tensor[6],
  "episode_idx": Tensor[],
}
```

## Checkpoints

Minimal checkpoints are separate from the older staged world-model checkpoints.

Checkpoint payload contains:

- `kind = "world_model_v2_minimal_v1"`
- `model_state`
- `optimizer_state`
- `scheduler_state`
- `step`
- `config`
- `mode`
- `clip_metadata`
- `best_metric`

Saved checkpoints:

- `checkpoints/last.pt`
- `checkpoints/best.pt`

Best metric selection:

- `ae_only` -> best `recon_mse`
- `joint` -> best `rollout_mse`
- `dynamics_only` -> best `rollout_mse`

Partial loading:

- `--load-encoder-decoder`
- `--load-dynamics`

Behavior:

- `joint`: can load either or both
- `ae_only`: can load encoder/decoder, cannot load dynamics
- `dynamics_only`: must load encoder/decoder unless resuming

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

- `ae_only` -> `recon_mse`
- `joint` -> `rollout_mse`
- `dynamics_only` -> `rollout_mse`

Plateau logic:

1. compute the validation metric every validation step
2. average over the configured validation window
3. compare to the best previous window
4. if the metric does not improve by at least `min_delta`, count one
   non-improving window
5. stop once non-improving windows reaches `early_stop_patience_windows`

Disabling early stop:

- set `early_stop_window_size = 0`, or
- set `early_stop_patience_windows = 0`

Important:

- validation-based early stopping requires `validation_interval > 0`

## CLI defaults

The minimal CLI entrypoint is:

```bash
python -m world_model_v2.minimal.run
```

Important defaults:

- `--mode joint`
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
- `--lr 1e-3`
- `--max-steps 3000`
- `--validation-interval 100`
- `--checkpoint-interval 100`

## What to look at when debugging

### If `ae_only` is blurry

The encoder/decoder itself is not reconstructing sharply enough. That is an
autoencoder problem, not a rollout problem.

Look at:

- `recon_mse`
- reconstruction GIFs from `ae_only`
- whether `128x128` is too hard for the current latent width or hidden width

### If `ae_only` is sharp but `joint` or `dynamics_only` drifts

The problem is likely in latent transition learning, not the decoder alone.

Look at:

- `rollout_mse`
- whether one-step training is enough for stable open-loop rollout
- whether the dynamics block is too small
- whether the latent space learned by `ae_only` is easy or hard to predict

### If `dynamics_only` does badly immediately

Check:

- was `--load-encoder-decoder` taken from a good `ae_only` checkpoint
- are the encoder and decoder actually frozen
- do the validation GIFs match the latent MSE trend

## Practical commands

### Train encoder/decoder only

```bash
python -m world_model_v2.minimal.run \
  --mode ae_only \
  --run-name ae_only_single_grasp_ep0_f111_116 \
  --output-dir outputs/minimal \
  --device cuda
```

### Train joint model

```bash
python -m world_model_v2.minimal.run \
  --mode joint \
  --run-name joint_single_grasp_ep0_f111_116 \
  --output-dir outputs/minimal \
  --device cuda
```

### Train dynamics only on a pretrained autoencoder

```bash
python -m world_model_v2.minimal.run \
  --mode dynamics_only \
  --load-encoder-decoder outputs/minimal/ae_only_single_grasp_ep0_f111_116/checkpoints/best.pt \
  --run-name dynamics_only_single_grasp_ep0_f111_116 \
  --output-dir outputs/minimal \
  --device cuda
```

## Bottom line

If you want to know whether the image model itself is blurry, run `ae_only`.

If you want to know whether the latent dynamics can roll forward from one seed
frame, run `dynamics_only` or `joint` and inspect the rollout GIFs and
`rollout_mse`.
