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
--load-encoder-decoder saved_checkpoints/github/vae_pickplace_z32_8x.pt
mp4: saved_checkpoints/github/vae_pickplace_z32_8x_episode_0.mp4

Validations:
Best `z32` / `8x` motion-weighted validation on episode 0 landed at step `15250`: `ae_loss=0.006813`, `recon_loss=0.006752`, `recon_mse=8.23e-5`, `motion_l1=1.34e-2`, `motion_mask_fraction=7.83%`.
Transform diagnostics for the new `z32` / `8x` VAE are in `outputs/checks/vae_pickplace_z32_8x_transforms/summary.json`.
The promoted checkpoint at `saved_checkpoints/github/vae_pickplace_z32_8x.pt` was copied from `outputs/so101_episode0_full_ae_resume_from_f113_137_crop_120x160_4xspatial_z32/checkpoints/best.pt`, while the previous file was archived under `saved_checkpoints/old/old_vae_pickplace_z32_8x/`.

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

  --kl-beta 1e-5 \
  --recon-mse-weight 1.0 \
  --recon-l1-weight 0.1 \
  --recon-edge-weight 0.05 \
  --recon-motion-weight 2.0 \
  --recon-motion-threshold 0.02 \
  --recon-motion-dilation-kernel-size 7 \


full episode 0, after seeing the 13 frame window 30 times works well
outputs/so101_episode0_full_ae_resume_from_f113_137_crop_120x160_4xspatial_z32/samples/step_016000/episode_0.mp4
stopped at validation not improving

  --lr 1e-5 \
  --kl-beta 1e-5 \
  --recon-mse-weight 1.0 \
  --recon-l1-weight 0.1 \
  --recon-edge-weight 0.05 \
  --recon-motion-weight 2.0 \
  --recon-motion-threshold 0.02 \
  --recon-motion-dilation-kernel-size 7 \



sudo apt install -y git-lfs
git lfs install
git clone https://github.com/Alessandro-Maniscalco/world-model-v2.git
cd world-model-v2
git lfs pull


to run for next training: 

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
  --latent-channels 32 \
  --batch-size 1 \
  --lr 1e-5 \
  --max-steps 200000 \
  --validation-interval 250 \
  --checkpoint-interval 250 \
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
  --run-name so101_all_episodes_ae_resume_from_episode0_crop_120x160_4xspatial_z32 \
  --output-dir outputs \
  --seed 7
