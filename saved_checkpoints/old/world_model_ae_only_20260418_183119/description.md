# world_model_ae_only_20260418_183119

2x spatialdownsampling, 2x time downsampling.

Archived from `outputs/world_model_ae_only_20260418_183119`.

Resumed from `outputs/so101_all_episodes_ae_208x276_d64_z16_r1_bs3/checkpoints/last.pt`.

Training command:

```powershell
.\.venv\Scripts\python.exe -m world_model_v2.run --mode ae_only --dataset-format lerobot_so101_base_sim_pickplace --data-root data/so101_base_sim_pickplace_cache --task single_grasp --split train --episode 0 --train-all-episodes --validation-split train --validation-episode 0 --resolution 208 --height 208 --width 276 --wan-dim 64 --latent-channels 16 --wan-num-res-blocks 1 --batch-size 3 --grad-accum-steps 1 --dataloader-num-workers 1 --lr 1e-4 --max-steps 80000 --validation-interval 250 --checkpoint-interval 50 --log-interval 10 --early-stop-patience-windows 20 --dynamics-context-frames 1 --dynamics-target-frames 3 --kl-beta 1e-5 --recon-mse-weight 1.0 --recon-l1-weight 0.1 --recon-edge-weight 0.1 --recon-motion-weight 0.75 --recon-motion-edge-weight 1.5 --recon-motion-threshold 0.01 --recon-motion-dilation-kernel-size 5 --resume outputs/so101_all_episodes_ae_208x276_d64_z16_r1_bs3/checkpoints/last.pt
```
