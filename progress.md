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
trying 512/8/8/128
with 5 frames it works
now with 24 frames outputs/so101_action_probe_5frames_1to1_ep18_f60_64_z32_8x_bigdit512x8/samples/step_000500/episode_18.mp4
still very noisy
768/12/12/192 worked great 

1024/16/8/128. 1-3. did not work
outputs/so101_allvideos_fullvideo_1to3_z32_8x_dit1024x16_h8_l128_bs6/samples/step_005250/episode_18.mp4
The predicted future arm frames are usually closer to earlier GT frames than to the correct same-time GT frame.
When the arm is moving, the prediction is not just late, it is also blurred.
The background/box stay clean because the model handles static content well.

If you can afford one big architecture change, do the temporal-compressed VAE first.
Keep 1 latent ctx -> 3 latent target.
Add DreamDojo’s temporal consistency loss, because they explicitly say it reduces artifacts and improves controllability.
If you want action pretraining, make it a separate latent-action model, not just isolated training of the current action MLP.
After that, train the action pathway to map raw robot actions into that latent-action space, then finetune the full world model.


Now changed so VAE termoal downsample of 4x.