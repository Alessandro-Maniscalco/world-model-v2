# Dynamics Architecture

## Summary

The source of truth for the dynamics model is `world_model_v2/dynamics_transformer.py`.
This repo now supports a single latent-dynamics backend: a DreamDojo-mechanics DiT core wrapped by a local rectified-flow training and sampling shell.

- `--mode ae_only` trains the Wan VAE
- `--mode dynamics_only` trains the latent dynamics model with the autoencoder frozen

The small DreamDojo-style backbone used for current dynamics work is:

- `in_channels = 16`
- `out_channels = 16`
- `patch_spatial = 2`
- `patch_temporal = 1`
- `model_channels = 256`
- `num_blocks = 4`
- `num_heads = 4`
- `use_adaln_lora = True`
- `adaln_lora_dim = 64`
- `action_conditioning_mode = "chunk_per_frame"`

Important: the repo-level default frame layout helper still has `4 -> 1`, but the aligned controller recipe is `1 context latent -> 3 future latents`. The parameter counts below are for the DreamDojo-style small backbone and do not change between `4 -> 1` and `1 -> 3`, because the model does not keep trainable per-frame parameters.

## What The Dynamics Model Does In This Repo

At a high level, the model learns a rectified-flow velocity field over short latent video chunks:

$$
x_0,\; a,\; m,\; t
\;\xrightarrow{\text{RF noising + repinning}}\;
\tilde{x}_t
\;\xrightarrow{\text{DreamDojo-style DiT}}\;
\hat{v}_\theta
$$

where:

- $x_0$ is the clean latent video chunk
- $a$ is the aligned transition-action chunk
- $m$ is the conditioning mask over the known prefix
- $t$ is the rectified-flow timestep
- $\tilde{x}_t$ is the repinned noisy latent clip actually seen by the DiT
- $\hat{v}_\theta$ is the predicted rectified-flow velocity

For the aligned `1 -> 3` setup at `128 x 128` image resolution:

- input image clip shape: `(B, 4, 3, 128, 128)`
- latent clip shape after the Wan encoder: `(B, 16, 4, 16, 16)`
- DiT input clip shape: `(B, 16, 4, 16, 16)` plus one mask channel
- patch-embedded token grid: `(B, 4, 8, 8, 256)`
- flattened self-attention sequence length: `4 x 8 x 8 = 256`
- predicted velocity shape: `(B, 16, 4, 16, 16)`
- sampled future latent shape from one prediction step: `(B, 16, 3, 16, 16)`

## Active Model Size

Measured from the instantiated small DreamDojo-style dynamics module in the repo virtualenv:

- dynamics params: `6,658,816`
- parameter memory only:
  - about `25.40 MiB` in fp32
  - about `12.70 MiB` in fp16/bf16

For reference, the full `WorldModel` with Wan VAE plus this dynamics backbone contains:

- encoder params: `15,466,528`
- decoder params: `25,754,755`
- dynamics params: `6,658,816`
- total world-model params: `47,880,099`

So the dynamics module is about `13.9%` of the full world-model parameter count.

## Layer-By-Layer Parameter Breakdown

### Top-Level Dynamics Module

| module | params | notes |
| --- | ---: | --- |
| `x_embedder` | `17,408` | patchify + linear projection |
| `t_embedder` | `262,144` | sinusoid MLP head, AdaLN-LoRA contract |
| `t_embedding_norm` | `256` | RMSNorm scale only |
| `blocks[0]` | `917,632` | DreamDojo-style DiT block |
| `blocks[1]` | `917,632` | DreamDojo-style DiT block |
| `blocks[2]` | `917,632` | DreamDojo-style DiT block |
| `blocks[3]` | `917,632` | DreamDojo-style DiT block |
| `final_layer` | `65,536` | AdaLN-modulated latent-patch projection |
| `action_embedder_B_D` | `267,520` | per-frame action embedding into `D` |
| `action_embedder_B_3D` | `2,375,424` | per-frame action embedding into `3D` |
| `pos_embedder` | `0` | RoPE is buffer-only |
| `RectifiedFlowHelper` | `0` | scheduler helper, no trainable params |
| **total** | **`6,658,816`** |  |

### One DiT Block

Each of the 4 blocks contains:

| submodule | params | formula at `D = 256`, `L = 64` |
| --- | ---: | --- |
| self-attention total | `262,272` | `4D^2 + 2(D / H)` |
| MLP total | `524,288` | `8D^2` |
| AdaLN self-attn modulation | `65,536` | `D L + L (3D)` |
| AdaLN MLP modulation | `65,536` | `D L + L (3D)` |
| layer norms | `0` | affine disabled |
| **block total** | **`917,632`** |  |

The self-attention part breaks down further as:

| self-attention piece | params |
| --- | ---: |
| `q_proj` | `65,536` |
| `k_proj` | `65,536` |
| `v_proj` | `65,536` |
| `q_norm` | `64` |
| `k_norm` | `64` |
| `output_proj` | `65,536` |

### Action And Output Heads

| module | params | formula |
| --- | ---: | --- |
| `x_embedder` | `17,408` | `((16 + 1) \cdot 2 \cdot 2) \cdot 256` |
| `t_embedder[1]` | `262,144` | `256 \cdot 256 + 256 \cdot (3 \cdot 256)` |
| `final_layer.linear` | `16,384` | `256 \cdot (2 \cdot 2 \cdot 1 \cdot 16)` |
| `final_layer.adaln_modulation` | `49,152` | `256 \cdot 64 + 64 \cdot (2 \cdot 256)` |
| `action_embedder_B_D` | `267,520` | `4 \cdot 1024 + 1024 + 1024 \cdot 256 + 256` |
| `action_embedder_B_3D` | `2,375,424` | `4 \cdot 3072 + 3072 + 3072 \cdot 768 + 768` |

## Core Logic Blocks In `dynamics_transformer.py`

### `DynamicsTransformerConfig`

Holds the dynamics hyperparameters and validates the DreamDojo-style restricted path:

- only `chunk_per_frame` action conditioning is allowed
- `use_adaln_lora` must be enabled
- learned temporal embeddings are rejected
- only `rope3d` plus torch attention is supported
- `architecture_version = "dreamdojo_torch_small_v1"` is recorded for checkpoint compatibility

### `PatchEmbed`

`PatchEmbed` first concatenates the conditioning mask as one extra channel, then patchifies and linearly projects.

### `VideoRopePosition3DEmb`

Builds non-learned 3D RoPE frequencies over:

- time
- height
- width

The head dimension is split into temporal and spatial rotary bands, and the module returns frequency tensors used to rotate queries and keys before attention.

This module has no trainable parameters.

### `Timesteps`

Projects scalar timesteps into sinusoidal embeddings:

$$
\phi_i(t) = t \cdot 10000^{-i / (D/2)}
$$

$$
e(t) =
\left[
\cos(\phi_0(t)), \ldots, \cos(\phi_{D/2-1}(t)),
\sin(\phi_0(t)), \ldots, \sin(\phi_{D/2-1}(t))
\right]
$$

Important DreamDojo detail: the sinusoidal embedding itself is later reused directly as the base timestep embedding when AdaLN-LoRA is active.

### `TimestepEmbedding`

This is one of the most important DreamDojo contracts in the refactor.

For AdaLN-LoRA mode:

$$
h = W_2 \,\mathrm{SiLU}(W_1 e(t))
$$

but the return value is:

$$
\text{timestep\_embedding} = e(t), \qquad
\text{adaln\_lora} = h \in \mathbb{R}^{3D}
$$

So the second linear is not the main timestep embedding. It is the low-rank AdaLN residual that will be added inside every block and the final layer.

### `Mlp`

Used only for action conditioning.

For input action vector $a$:

$$
\psi(a) = W_2 \,\mathrm{GELU}(W_1 a + b_1) + b_2
$$

There are two copies:

- `action_embedder_B_D` produces one `D`-dimensional action embedding per transition
- `action_embedder_B_3D` produces one `3D`-dimensional action embedding per transition for the AdaLN-LoRA path

### `Attention`

The self-attention path is DreamDojo-style and torch-only:

$$
q = \mathrm{RMSNorm}(W_q x), \qquad
k = \mathrm{RMSNorm}(W_k x), \qquad
v = W_v x
$$

Then RoPE is applied to $q$ and $k$, and scaled dot-product attention is computed:

$$
\mathrm{Attn}(q,k,v)
=
\mathrm{softmax}\!\left(\frac{qk^\top}{\sqrt{d_h}}\right)v
$$

The head outputs are concatenated and projected:

$$
y = W_o \,\mathrm{ConcatHeads}(\mathrm{Attn}(q,k,v))
$$

This repo intentionally keeps only the self-attention branch. The original DreamDojo cross-attention machinery is not active here.

### `Block`

Each block contains:

- one AdaLN-modulated self-attention sublayer
- one AdaLN-modulated MLP sublayer

Given token grid $x$, timestep embedding $e_t$, and AdaLN-LoRA residual $\ell_t$:

$$
[\Delta s_{\text{attn}}, \Delta c_{\text{attn}}, g_{\text{attn}}]
=
\mathrm{AdaLN}_{\text{attn}}(e_t) + \ell_t
$$

$$
\bar{x}_{\text{attn}}
=
\mathrm{LN}(x)\odot(1+\Delta c_{\text{attn}}) + \Delta s_{\text{attn}}
$$

$$
x
\leftarrow
x + g_{\text{attn}} \odot \mathrm{SelfAttn}(\bar{x}_{\text{attn}})
$$

and then:

$$
[\Delta s_{\text{mlp}}, \Delta c_{\text{mlp}}, g_{\text{mlp}}]
=
\mathrm{AdaLN}_{\text{mlp}}(e_t) + \ell_t
$$

$$
\bar{x}_{\text{mlp}}
=
\mathrm{LN}(x)\odot(1+\Delta c_{\text{mlp}}) + \Delta s_{\text{mlp}}
$$

$$
x
\leftarrow
x + g_{\text{mlp}} \odot \mathrm{MLP}(\bar{x}_{\text{mlp}})
$$

### `FinalLayer`

The final layer maps DiT tokens back into latent patches:

$$
[\Delta s_f, \Delta c_f]
=
\mathrm{AdaLN}_{\text{final}}(e_t) + \ell_t[:2D]
$$

$$
\bar{x}
=
\mathrm{LN}(x)\odot(1+\Delta c_f) + \Delta s_f
$$

$$
\hat{v}_{\text{patch}} = W_{\text{out}} \bar{x}
$$

Those patches are then unpatchified back to `(B, C, T, H, W)`.

### `ActionConditionedDynamicsTransformer`

This module owns the actual DiT forward pass:

1. concatenate the condition mask as one input channel
2. patchify the latent video
3. build RoPE frequencies
4. compute DreamDojo timestep embedding and AdaLN-LoRA residual
5. compute per-transition action embeddings
6. prepend a zero action embedding on frame 0
7. add action embeddings to:
   - the timestep embedding
   - the AdaLN-LoRA residual
8. RMS-normalize the timestep embedding
9. run the stack of DiT blocks
10. run the final layer and unpatchify

If the transition actions are $a_0, \ldots, a_{T-2}$, then:

$$
u_i = \psi_D(a_i), \qquad
v_i = \psi_{3D}(a_i)
$$

$$
\bar{u} = [0, u_0, u_1, \ldots, u_{T-2}]
$$

$$
\bar{v} = [0, v_0, v_1, \ldots, v_{T-2}]
$$

$$
e_t \leftarrow \mathrm{RMSNorm}(e_t + \bar{u})
$$

$$
\ell_t \leftarrow \ell_t + \bar{v}
$$

This keeps action timing explicit and is the only supported conditioning mode.

### `RectifiedFlowHelper`

Provides the rectified-flow scheduler math:

training interpolation:

$$
x_t = \sigma \epsilon + (1-\sigma)x_0
$$

$$
v^\star = \epsilon - x_0
$$

solver step:

$$
x_{\sigma_{\text{to}}}
=
x_{\sigma_{\text{from}}}
+
(\sigma_{\text{to}} - \sigma_{\text{from}})\hat{v}_\theta
$$

This helper has no trainable parameters.

### `RectifiedFlowDynamics`

Wraps the DiT with repo-specific teacher conditioning and rollout behavior:

- full-clip RF input preparation
- condition-mask construction
- clean-frame repinning
- optional conditional sigma on the known prefix
- optional timestep override on conditioned frames
- optional conditioning dropout
- velocity overwrite on conditioned frames
- CFG over the video-conditioning path
- short-horizon latent sampling

The key repinning equation is:

$$
\tilde{x}_t = m \odot x_{\text{cond}} + (1-m)\odot x_t
$$

where $m$ is the binary condition mask broadcast across channels.

When classifier-free guidance is enabled:

$$
\hat{v}_{\text{cfg}}
=
\hat{v}_{\text{cond}}
+ s\left(\hat{v}_{\text{cond}} - \hat{v}_{\text{uncond}}\right)
$$

with guidance scale $s$.

## Forward Flow

For one aligned `1 -> 3` prediction step, the data path is:

```text
(B, 16, 1, 16, 16) context latent
  -> append 3 future-noise frames
  -> build full chunk (B, 16, 4, 16, 16)
  -> make condition mask over frame 0
  -> repin frame 0 clean
  -> concatenate mask channel to get 17 input channels
  -> patch embed to (B, 4, 8, 8, 256)
  -> flatten tokens to 256 positions for self-attention
  -> 4 DreamDojo-style DiT blocks
  -> final layer
  -> unpatchify to velocity (B, 16, 4, 16, 16)
  -> keep future part (B, 16, 3, 16, 16)
```

## Training Loss

In `dynamics_only` mode, the main training loss is plain MSE over the velocity target:

$$
\mathcal{L}_{\mathrm{RF}}
=
\operatorname{MSE}(\hat{v}_\theta,\; v^\star)
$$

but conditioned-frame velocity is overwritten with the exact RF target before the loss is computed. That means the learning signal is still shaped as a full-clip MSE, while the conditioned prefix behaves like teacher-forced clean context.

Optional self-forcing losses live above this backbone in `Experiment` and are controlled separately by the experiment config.

## Validation Boundary Checks

The validation/export path tracks the counts that most often go wrong in temporal pipelines:

- input frame count
- predicted frame count
- decoded frame count
- open-rollout predicted frame count
- open-rollout decoded frame count
- exported MP4 frame count

For the smoke-tested aligned `1 -> 3` run:

- latent chunk size during training: `T = 4`
- training action chunk size: `3`
- sampled future latent size: `T = 3`
- validation style: `teacher_forced_1_context_3_target`
- exported MP4 frame count matched the rollout target length

## Checkpoint Compatibility

The current dynamics architecture is versioned as:

- `architecture_version = "dreamdojo_torch_small_v1"`

Older dynamics checkpoints are intentionally not warm-start compatible unless they already match:

- dynamics backend
- architecture version
- frame layout
- action horizon
- action dimension
- action conditioning mode

This is deliberate. The repo now prefers a clear hard failure over silently reusing weights from an older simplified DiT that no longer matches the DreamDojo-style mechanics.
