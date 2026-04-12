## Goal
Find the best DreamDojo-style dynamics setup for `1 context -> 3 inferred dynamics`, optimizing open-rollout quality and temporal consistency.

Start with 1 - 1 and seeing if actions work. Then move up.

## Dataset
--dataset-format lerobot_so101_base_sim_pickplace   --data-root data/so101_base_sim_pickplace_cache

## Episode trained
Episode 18, frames 0-63 as it has good action variation.

## Number of steps
Total frames seen should be XXXX. The number of steps is divided by the batch size hat doesn't OOM.

## Vae used
--load-encoder-decoder saved_checkpoints/vae_pickplace_z32_8x/best.pt
mp4: saved_checkpoints/vae_pickplace_z32_8x/episode_0.mp4

Validations:
Best `z32` / `8x` motion-weighted validation on episode 0 landed at step `15250`: `ae_loss=0.006813`, `recon_loss=0.006752`, `recon_mse=8.23e-5`, `motion_l1=1.34e-2`, `motion_mask_fraction=7.83%`.
Transform diagnostics for the new `z32` / `8x` VAE are in `outputs/checks/vae_pickplace_z32_8x_transforms/summary.json`.

Future: 16x downsamplling with z64.


## 1 - 1 tests
