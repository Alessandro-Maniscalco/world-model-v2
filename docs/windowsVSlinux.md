# Windows vs Linux Resume Notes

This note captures the working resume command for `saved_checkpoints/github/vae_pickplace_z32_8x.pt`, the Linux `batch_size=1` timing probe, the Windows `batch_size=2` observation, and the handoff prompt to use after rebooting into Windows.

## Exact Resume Command

This is the exact resume command that loaded the checkpoint correctly on Linux and advanced one real step.

```bash
source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m world_model_v2.run \
  --mode ae_only \
  --dataset-format lerobot_so101_base_sim_pickplace \
  --data-root data/so101_base_sim_pickplace_cache \
  --task single_grasp \
  --split train \
  --episode 0 \
  --train-all-episodes \
  --validation-split train \
  --validation-episode 0 \
  --resolution 120 \
  --height 120 \
  --width 160 \
  --wan-dim 128 \
  --latent-channels 32 \
  --wan-num-res-blocks 2 \
  --batch-size 1 \
  --dataloader-num-workers 1 \
  --lr 1e-5 \
  --max-steps 50000 \
  --validation-interval 500 \
  --checkpoint-interval 100 \
  --log-interval 10 \
  --dynamics-context-frames 1 \
  --dynamics-target-frames 3 \
  --kl-beta 1e-5 \
  --recon-mse-weight 1.0 \
  --recon-l1-weight 0.1 \
  --recon-edge-weight 0.05 \
  --recon-motion-weight 2.0 \
  --recon-motion-threshold 0.02 \
  --recon-motion-dilation-kernel-size 7 \
  --resume saved_checkpoints/github/vae_pickplace_z32_8x.pt \
  --run-name so101_all_episodes_ae_resume_from_github_vae_pickplace_z32_8x_bs1 \
  --output-dir outputs \
  --seed 7
```

## Why The Resume Command Needs Extra Wan Flags

The checkpoint does not match the current CLI defaults unless these flags are provided:

- `--wan-dim 128`
- `--wan-num-res-blocks 2`

Without them, resume fails with:

```text
ValueError: Checkpoint autoencoder config from saved_checkpoints/github/vae_pickplace_z32_8x.pt does not match the requested Wan temporal tokenizer config.
```

Checkpoint metadata confirmed:

- `mode = ae_only`
- `dataset_format = lerobot_so101_base_sim_pickplace`
- `train_all_episodes = True`
- `batch_size = 2` in the saved config
- `lr = 1e-5`
- `max_steps = 50000`
- `validation_interval = 500`
- `checkpoint_interval = 100`
- `step = 25100`
- autoencoder config:
  - `dim = 128`
  - `z_dim = 32`
  - `num_res_blocks = 2`
  - `dim_mult = [1, 2, 4]`
  - `temperal_downsample = [True, True]`

## Linux Benchmark: `batch_size=1`

The active Linux training run was stopped so the benchmark could run on a clean GPU. The profiling run resumed the checkpoint and measured `1` warmup step plus `3` steady-state steps with CUDA synchronization around host-to-device copy and the full optimizer step.

Steady-state averages:

- batch fetch on CPU: `0.00021s`
- host to GPU copy: `0.00051s`
- metrics append: `0.00089s`
- model forward + loss + backward + optimizer: `2.361s`
- total per step: `2.363s`
- peak CUDA allocated: `9.03 GiB`
- peak CUDA reserved: `9.21 GiB`

Warmup-only timings:

- first batch fetch: `0.148s`
- first train step: `2.742s`

Per-step timing records were written to:

- `outputs/temp_profile_resume_github_vae_pickplace_z32_8x_bs1/profile_metrics.jsonl`

Observed behavior on this Linux boot:

- resume worked cleanly with the command above
- `batch_size=2` loaded the checkpoint but OOMed on the first real training step on this GPU
- `batch_size=1` completed a real resumed step successfully

## Windows Observation: `batch_size=2`

Windows-side observation from the dual-boot run:

- batch fetch on CPU: about `0.84s`
- host to GPU copy: about `0.01s`
- metrics append: about `0.0008s`
- model forward + loss + backward + optimizer: about `29.5s` to `30s`

That makes the Windows run compute-bound, not dataloader-bound.

The reported `nvidia-smi` snapshot on Windows was:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 573.91                 Driver Version: 573.91         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------|
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   69C    P0             59W /  100W |   16127MiB /  16384MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------------------------------------------------------+
```

## Working Comparison

Useful high-level comparison:

- Linux `batch_size=1`: about `2.36s/step`
- Windows `batch_size=2`: about `30s/step`

Even accounting for the larger Windows batch size, the slowdown is far beyond what batch scaling alone should explain.

The strongest current hypothesis is that Windows is limited by the GPU execution path rather than the input pipeline. Likely suspects:

- WDDM/display overhead
- power or clock limits
- hybrid graphics / Optimus / MUX mode differences
- CUDA / cuDNN / PyTorch build mismatch
- Windows-specific driver behavior during long mixed-precision compute

## Windows Handoff Prompt

Paste this into the Windows-side session:

```text
Resume the VAE from `saved_checkpoints/github/vae_pickplace_z32_8x.pt` using the exact compatible Wan flags:
- `--wan-dim 128`
- `--wan-num-res-blocks 2`

Do not guess old defaults. First do a short profiling run, not a long training run.

Use the same training config as Linux, starting with `--batch-size 1`, and measure 1 warmup step + 3 steady-state steps with CUDA synchronization around copy and training. Report:
- warmup fetch time and warmup train-step time
- steady-state average batch fetch on CPU
- steady-state average host->GPU copy
- steady-state average metrics append
- steady-state average model forward + loss + backward + optimizer
- peak CUDA allocated and reserved memory
- PyTorch version, `torch.version.cuda`, cuDNN version
- whether training autocast is bf16 or fp16
- GPU clocks, power draw, P-state, and whether WDDM is active during the step

Known Linux baseline for the same resumed checkpoint at `batch_size=1`:
- fetch: 0.00021s steady-state, 0.148s warmup
- host->GPU: 0.00051s
- metrics append: 0.00089s
- train step: 2.361s steady-state, 2.742s warmup
- peak allocated: 9.03 GiB
- peak reserved: 9.21 GiB

Known Windows observation at `batch_size=2`:
- fetch about 0.84s
- host->GPU about 0.01s
- metrics append about 0.0008s
- train step about 29.5s to 30s
- GPU util 100%
- WDDM
- about 16.1/16.4 GiB VRAM used

Focus on why Windows compute is dramatically slower than Linux. Do not spend time tuning the dataloader unless steady-state fetch is actually large after warmup.
Check likely system causes first: WDDM/display path, power/performance mode, hybrid graphics/Optimus or MUX mode, clocks/power limits, and any PyTorch/CUDA/cuDNN mismatch.
After that, try `batch_size=2` only if `batch_size=1` looks healthy.
```

## Local Cleanup Status

After collecting the Linux `batch_size=1` timings:

- the active Linux training process was stopped
- the short profiling process exited cleanly
- no `python -m world_model_v2.run` training process was left running
