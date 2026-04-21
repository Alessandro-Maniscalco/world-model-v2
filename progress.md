## Goal
Find the best DreamDojo-style dynamics setup for `1 context -> 3 inferred dynamics`, optimizing open-rollout quality and temporal consistency.

## Dataset
--dataset-format lerobot_so101_base_sim_pickplace   --data-root data/so101_base_sim_pickplace_cache

## Frame size
Raw frame is 480x640.

Using 6 x 8 latent size., Crop out the static bottom and side regions to 384x512, then 4x downsample to 96x128 before the VAE. Smaller sizes are too blurry, consider increasing to 8x11 latents.
Various sized images: C:/Users/aless/world-model-v2/outputs/checks/so101_vae_size_sweep_ep0_frame0/so101_vae_size_sweep_contact_sheet.png

## Number of steps
The tests I do should be 2 epochs. 
Total frames are 28k. The number of steps is divided by the batch size x 2 epochs.

## VAE from scratch

full episode 0, after seeing the 13 frame window 30 times works well

trained on all episodes:

2x spatialdownsampling, 2x time downsampling.

.\.venv\Scripts\python.exe -m world_model_v2.run --mode ae_only --dataset-format lerobot_so101_base_sim_pickplace --data-root data/so101_base_sim_pickplace_cache --task single_grasp --split train --episode 0 --train-all-episodes --validation-split train --validation-episode 0 --resolution 208 --height 208 --width 276 --wan-dim 64 --latent-channels 16 --wan-num-res-blocks 1 --batch-size 3 --grad-accum-steps 1 --dataloader-num-workers 1 --lr 1e-4 --max-steps 80000 --validation-interval 250 --checkpoint-interval 50 --log-interval 10 --early-stop-patience-windows 20 --dynamics-context-frames 1 --dynamics-target-frames 3 --kl-beta 1e-5 --recon-mse-weight 1.0 --recon-l1-weight 0.1 --recon-edge-weight 0.1 --recon-motion-weight 0.75 --recon-motion-edge-weight 1.5 --recon-motion-threshold 0.01 --recon-motion-dilation-kernel-size 5 --resume outputs/so101_all_episodes_ae_208x276_d64_z16_r1_bs3/checkpoints/last.pt

archived AE checkpoint: saved_checkpoints\old\world_model_ae_only_20260418_183119\best.pt


## Pertrained VAE - WAN2.2 VAE

Works. [B, 48, Tlatent, H/16, W/16]
Tlatent = 1 + (NumFrames - 1) / 4

After full Wan compression, the latent clip is:
(B, 48, 2, 13, 17)

## DiT size

Standard Dream Dojo is model_channels=4096, num_blocks=28, num_heads=16, adaln_lora_dim=256

I tested: 384/6/6/96

I will increase to 768 / 8 / 6 / 128

## Now - DiT training
1ctx 1 infered, frames 110 to 114 works
outputs\world_model_dynamics_only_20260419_172629\samples\step_006000\episode_0.mp4

1 ctx 1 inferred, all frames, 20 epochs, 
can see that movement is followed, image very blurry
outputs\so101_all_frames_all_episodes_wan22_dit_384x6x6x96_bs8\samples\step_070500\episode_0.mp4

Reduced size, at 2 epochs image looks similar to larger size: 


## CLI
 --dynamics-validation-metric next_frame_mse , so that 
-- lr 1e-3