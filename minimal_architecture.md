# Minimal Architecture

## Status

This document describes the implemented minimal world-model path in
`world_model_v2/minimal/` as it exists in this repo now.

The goal of this path is still a small, explicit, debuggable setup that can
overfit short clips, save checkpoints and visualizations, and let you inspect
what the Wan VAE and the dynamics model are doing without the complexity of the
larger staged world-model code.

Major design choices:

- one Wan-style VAE backend
- one separate latent dynamics backend
- the dynamics backend is now a tiny rectified-flow DiT, not the old conv model
- no joint AE+dynamics training mode
- dormant action-conditioning structure is kept, but actions are multiplied by
  zero for now
- deterministic Wan posterior-mean latents are used for dynamics training and
  validation
- periodic validation exports side-by-side PNG and MP4 artifacts
- validation-based plateau stopping remains enabled by default

## Data setup

The minimal path is no longer hard-coded to one 6-frame debug clip.

The CLI can select:

- `interactive_world_sim` or `lerobot_metaworld`
- any episode
- any camera or the MetaWorld image column
- any frame slice through `--frame-start` and `--frame-end`

Important practical rule for the new dynamics backend:

- `dynamics_only` requires at least 5 frames in the selected clip
- for the intended first overfit, use exactly 5 frames so there is exactly 1
  three-context two-target window

Example first overfit target:

- frame slice `0..4`
- total frames: `5`
- windows: `1`
- frames 0, 1, and 2 are context
- frames 3 and 4 are the supervised targets

## Shared model

The shared per-frame autoencoding path is:

$$
\left(\mu_t, \log \sigma_t^2\right) = E_\phi(x_t)
$$

$$
z_t \sim q_\phi(z_t \mid x_t)
\quad \text{or} \quad
z_t = \mu_t
$$

$$
\hat{x}_t = D_\theta(z_t)
$$

The dynamics model then operates on one 5-frame latent clip:

$$
Z_0 = [z_0, z_1, z_2, z_3, z_4]
$$

with a binary conditioning mask

$$
m_t =
\begin{cases}
1 & t \in \{0,1,2\} \\
0 & t \in \{3,4\}
\end{cases}
$$

where frames `0,1,2` are context and frames `3,4` are the predicted targets.

### Wan-style VAE config

The Wan-style backend uses these internal defaults:

Parameter count for the current `z_dim=16` minimal Wan VAE:

- encoder: `15,466,528`
- decoder: `25,754,755`
- total VAE: `41,221,283`

```python
cfg = dict(
    dim=64,
    z_dim=16,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=1,
    attn_scales=[],
    temperal_downsample=[False, False, False],
    dropout=0.0,
)
```

Important notes:

- the Wan backend is adapted into this repo; it is not a verbatim upstream
  copy
- the minimal implementation drops streaming/cache logic and pretrained latent
  normalization constants
- images are processed internally as single-frame videos with `T=1`

### Latent shape

The latent spatial shape is derived from the Wan config, not hard-coded.

For the current Wan config:

- spatial downsample factor = `2^(len(dim_mult)-1) = 8`
- `128x128 -> 16x16`
- `240x240 -> 30x30`
- latent shape = `B x 16 x H_latent x W_latent`

In general:

$$
\text{latent height} = \frac{\text{image height}}{8}
$$

$$
\text{latent width} = \frac{\text{image width}}{8}
$$

The code validates that the image size is divisible by the Wan downsample
factor.

### RF DiT dynamics backend

The minimal dynamics backend is recorded as:

- `dynamics_backend = "rf_dit"`

The implemented DiT is intentionally tiny and DreamDojo-inspired:

Parameter count for the current minimal RF DiT:

- total RF DiT parameters: `7,724,288`

```python
cfg = dict(
    max_frames=5,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=256,
    num_blocks=4,
    num_heads=4,
    pos_emb_cls="rope3d",
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    use_adaln_lora=False,
    adaln_lora_dim=64,
    action_dim=4,
    timestep_scale=1.0,
    dynamics_infer_steps=16,
    dynamics_train_timesteps=1000,
    dynamics_rf_shift=5.0,
)
```

Implementation notes:

- the model operates on a 5-frame latent video
- a 1-channel condition mask is concatenated to the latent input
- frames 0, 1, and 2 are the clean context frames
- frames 3 and 4 are the predicted target frames
- RF interpolation is built over the full 5-frame latent clip, then the context
  frames are repinned before the DiT forward
- RoPE self-attention is used over the flattened `T x H x W` patch grid
- action MLPs are still present for structural compatibility, but their
  contribution is multiplied by zero
- for very tiny latent grids used in tests, `patch_spatial` falls back to `1`
  so the model can still be instantiated

Embedding flow:

1. concatenate the latent video and the 1-channel condition mask
2. patchify the `B x C x T x H x W` tensor into `B x T_p x H_p x W_p x D`
3. project each patch to `model_channels`
4. build one RoPE table over the flattened `T_p x H_p x W_p` token grid
5. sinusoidally embed the per-frame RF timesteps
6. pass those timestep features through `TimestepEmbedding`
7. normalize the timestep embedding with RMSNorm
8. compute action embeddings, but add them as `+ action * 0` for now

Transformer block structure:

- each block is AdaLN-modulated and has:
  - LayerNorm-free affine modulation from the timestep embedding
  - full self-attention over all latent patches
  - 3D RoPE applied to attention Q and K
  - residual add
  - modulated feedforward MLP
  - residual add
- there is currently no text cross-attention path in this minimal RF DiT

Initialization:

- patch, attention, MLP, timestep, action, and final projection weights use
  truncated normal initialization with scale about `1 / sqrt(fan_in)`
- AdaLN gate/modulation projections are zero-initialized so each transformer
  block starts close to an identity update
- this keeps the initial network stable while still allowing the patch embedder
  and final projection to produce non-zero velocity predictions

## Modes

The minimal path supports only:

- `ae_only`
- `dynamics_only`

`joint` has been removed.

### `ae_only`

Trainable:

- encoder: trainable
- decoder: trainable
- dynamics: frozen

Dataset:

- all selected frames independently

Training behavior:

1. encode `x_t` into posterior moments `mu_t, log_var_t`
2. sample `z_t` with reparameterization
3. decode `z_t`
4. optimize reconstruction plus KL

Training loss:

$$
\mathcal{L}_{ae}
=
\mathcal{L}_{recon}
+ \beta \, \mathrm{KL}\left(q_\phi(z_t \mid x_t) \,\|\, \mathcal{N}(0, I)\right)
$$

where `recon` is the weighted reconstruction loss configured by:

- `--recon-mse-weight`
- `--recon-l1-weight`
- `--recon-edge-weight`

Default:

- `kl_beta = 1e-4`

Validation behavior:

- validation is deterministic
- images are encoded with posterior mean `mu`
- `mu` is decoded directly
- metric of record is `ae_loss`

### `dynamics_only`

Trainable:

- encoder: frozen
- decoder: frozen
- dynamics: trainable

Required checkpoint loading:

- `--load-encoder-decoder` is required unless resuming from an existing
  `dynamics_only` run

Dataset:

- sliding 5-frame windows from the selected clip
- if the clip contains exactly 5 frames, the dataset contains exactly 1 sample

Training behavior:

1. encode three context frames and two target frames with deterministic
   posterior mean latents
2. build a clean latent video
   `[z_context_0, z_context_1, z_context_2, z_target_0, z_target_1]`
3. sample one rectified-flow training time per batch item and map it to one
   discrete timestep and one sigma
4. broadcast that same timestep across all 5 frames in the clip
5. sample Gaussian reference noise over the full 5-frame latent clip
6. build full-clip RF interpolation and target velocity
7. repin the context frames before the DiT forward
8. predict full-clip velocity with the DiT
9. overwrite the conditioned-frame velocity with the exact RF target
10. optimize full-clip MSE, which is identically zero on context frames because
    of the overwrite step

Let `M` denote the conditioning mask expanded across latent channels. The
training interpolation is:

$$
x_\sigma = \sigma \epsilon + (1 - \sigma) Z_0
$$

with RF target velocity

$$
v^\star = \epsilon - Z_0
$$

The DreamDojo-style repinning step before the DiT forward is:

$$
x_{\mathrm{in}} = M \odot Z_0 + (1 - M) \odot x_\sigma
$$

The DiT predicts full-clip velocity from the repinned clip and the mask:

$$
\hat{v} = f_\psi([x_{\mathrm{in}}, m], t)
$$

The conditioned-frame overwrite is:

$$
\tilde{v} = M \odot v^\star + (1 - M) \odot \hat{v}
$$

This makes the conditioned-frame loss exactly zero without an explicit loss
mask, because `\tilde{v} = v^\star` wherever `M = 1`.

Training loss:

$$
\mathcal{L}_{dyn}
=
\mathrm{MSE}\left(\tilde{v}, v^\star\right)
$$

The implementation logs:

- `latent_rf_mse`
- `target_sigma`

Validation rollout:

1. keep frames 0, 1, and 2 as the context frames in the exported preview
2. for each target chunk in the validation clip, encode the three ground-truth
   context frames into deterministic latents
3. build a 5-frame conditioning clip
   `[z_context_0, z_context_1, z_context_2, 0, 0]`
4. sample one fixed Gaussian reference-noise clip for the whole 5-frame latent
   state
5. initialize the latent state from that full noise clip and immediately repin
   the context frames
6. iteratively denoise the full 5-frame latent state with the RF DiT for
   `dynamics_infer_steps`
7. after every RF step, repin the context frames again so they remain exact
   conditioning latents
8. return only frames 3 and 4 from the final latent clip as the predicted
   target chunk
9. decode the predicted latent chunk with the frozen decoder
10. repeat independently for each future chunk using the corresponding
   ground-truth previous three frames as context

This is teacher-forced three-context two-target validation, not open-loop rollout.

If `x^{(k)}` is the current 5-frame latent state and `Z_{\mathrm{ctx}}` is the
conditioning clip `[z_0, z_1, z_2, 0, 0]`, the per-step sampling update is:

$$
x_{\mathrm{in}}^{(k)} = M \odot Z_{\mathrm{ctx}} + (1 - M) \odot x^{(k)}
$$

$$
\hat{v}^{(k)} = f_\psi([x_{\mathrm{in}}^{(k)}, m], t_k)
$$

$$
\tilde{v}^{(k)} = M \odot (\epsilon_{\mathrm{ref}} - Z_{\mathrm{ctx}})
+ (1 - M) \odot \hat{v}^{(k)}
$$

$$
x^{(k+1)} = x^{(k)} + (\sigma_{k+1} - \sigma_k)\tilde{v}^{(k)}
$$

followed by one more repinning step

$$
x^{(k+1)} \leftarrow M \odot Z_{\mathrm{ctx}} + (1 - M) \odot x^{(k+1)}.
$$

Validation metric of record:

- `next_frame_mse`

By default:

- `dynamics_infer_steps = 16`
- validation therefore integrates the target latent chunk across 16 RF sampling
  steps
- the target chunk is created from noise first, then refined step by step while
  the first three latent frames stay as conditioning latents
- the saved MP4 shows the first three frames as context and later frames as
  chunked predictions, not an autoregressive rollout

For a 5-frame overfit run, validation stats include:

- `input_frame_count = 5`
- `predicted_frame_count = 5`
- `decoded_frame_count = 5`
- `seed_frames = 3`
- `loss_frames = 2`
- `validation_style = "teacher_forced_three_context_two_target"`

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
- `dynamics_backend`
- `dynamics`

For current checkpoints:

- `ae_backend = "wan"`
- `autoencoder["backend"] = "wan"`
- `dynamics_backend = "rf_dit"`
- `dynamics["backend"] = "rf_dit"`

Saved checkpoints:

- `checkpoints/last.pt`
- `checkpoints/best.pt`

Best metric selection:

- `ae_only` -> best `ae_loss`
- `dynamics_only` -> best `next_frame_mse`

Checkpoint compatibility:

- Wan backend metadata is checked during partial loading and resume
- RF DiT backend metadata is checked during dynamics loading and resume
- old conv-dynamics checkpoints are rejected with a clear backend mismatch
  error

## Training loop and plateau stopping

The minimal runner validates every `validation_interval` steps and writes:

- `samples/step_xxxxxx/episode_0_grid.png`
- `samples/step_xxxxxx/episode_0.mp4`
- `samples/step_xxxxxx/episode_0_stats.json`

It also supports validation-based plateau stopping by default.

Default early-stop settings:

- `early_stop_window_size = 1`
- `early_stop_patience_windows = 5`
- `early_stop_min_delta = 1e-10`
- `early_stop_warmup_steps = 300`

Early-stop metric:

- `ae_only` -> `ae_loss`
- `dynamics_only` -> `next_frame_mse`

## CLI defaults

The minimal CLI entrypoint is:

```bash
python -m world_model_v2.minimal.run
```

Important defaults:

- `--mode ae_only`
- `--dataset-format interactive_world_sim`
- `--data-root data/full`
- `--task single_grasp`
- `--split val`
- `--episode 0`
- `--camera camera_1_color`
- `--frame-start` unset
- `--frame-end` unset
- `--resolution 128`
- `--latent-channels 16`
- `--hidden-channels 64`
- `--dynamics-infer-steps 16`
- `--dynamics-train-timesteps 1000`
- `--dynamics-rf-shift 5.0`
- `--batch-size 32`
- `--auto-batch-size` disabled by default
- `--lr 1e-4`
- `--max-steps 3000`
- `--validation-interval 100`
- `--checkpoint-interval 100`

## Practical commands

### Train the Wan-style VAE only

```bash
source .venv/bin/activate
python -m world_model_v2.minimal.run \
  --mode ae_only \
  --dataset-format lerobot_metaworld \
  --data-root data/full \
  --split train \
  --episode 0 \
  --metaworld-task-index 0 \
  --resolution 240 \
  --auto-batch-size \
  --run-name metaworld_task0_wan_ae_240 \
  --output-dir outputs/minimal \
  --device cuda
```

### Train the RF DiT dynamics model on a 5-frame overfit slice

```bash
source .venv/bin/activate
python -m world_model_v2.minimal.run \
  --mode dynamics_only \
  --dataset-format lerobot_metaworld \
  --data-root data/full \
  --split train \
  --episode 0 \
  --metaworld-task-index 0 \
  --frame-start 0 \
  --frame-end 4 \
  --resolution 240 \
  --batch-size 1 \
  --dynamics-infer-steps 16 \
  --dynamics-train-timesteps 1000 \
  --dynamics-rf-shift 5.0 \
  --load-encoder-decoder outputs/minimal/metaworld_task0_wan_ae_240/checkpoints/best.pt \
  --run-name metaworld_task0_rf_dit_f0_4 \
  --output-dir outputs/minimal \
  --device cuda
```

The same command was smoke-tested locally on CPU with:

- `--device cpu`
- `--max-steps 1`
- `--validation-interval 1`
- `--checkpoint-interval 1`

and it completed successfully.

## Initialization sanity check

On a fresh untrained `30 x 30` latent DiT, a quick random-input probe gave:

- noisy latent std about `1.00`
- patch token std about `0.99`
- timestep embedding std about `1.00`
- predicted velocity std about `1.01`

That is a good sign for training-time scale: the model is not obviously
exploding or collapsing at initialization.

One caveat:

- iterative sampling from an untrained model can still drift in variance because
  16 RF updates are composed in sequence
- this is expected before training and is less important than the training-time
  velocity scale, which looks healthy

## Auto batch sizing

The minimal runner can optionally probe for the largest training batch size
that fits before training starts:

```bash
--auto-batch-size
```

Behavior:

- on CUDA, the runner probes batch sizes and keeps the largest size that fits a
  full forward/backward/optimizer-step dry run
- on CPU, the runner uses the full dataset length instead of probing
- if a real CUDA OOM still happens during training, the runner halves the batch
  size, rebuilds the train loader, and continues instead of crashing
  immediately
