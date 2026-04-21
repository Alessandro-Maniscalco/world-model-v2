## Dynamics training process

scripts/check/benchmark_dynamics_batch_size.py use it to maximise batchize throughoutput in terms of samples per second. 1 step is fine.

## Fixed commands:

$WANVAE = Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--Wan-AI--Wan2.2-TI2V-5B\snapshots" -Recurse -Filter Wan2.2_VAE.pth | Select-Object -First 1 -ExpandProperty FullName

. .\.venv\Scripts\Activate.ps1

python -m world_model_v2.run `
  --mode dynamics_only `
  --dataset-format lerobot_so101_base_sim_pickplace `
  --data-root data/so101_base_sim_pickplace_cache `
  --task single_grasp `
  --split train `
  --episode 0 `
  --train-all-episodes `
  --validation-split train `
  --validation-episode 0 `
  --validation-max-frames 140 `
  --resolution XXX `
  --height XXX `
  --width XXX `
  --wan-dim XXX `
  --latent-channels XXX `
  --wan-num-res-blocks XXX `
  --hidden-channels 64 `
  --batch-size XX `
  --grad-accum-steps 1 `
  --dataloader-num-workers 1 `
  --lr XXXX `
  --lr-warmup-steps 200 `
  --optimizer-beta1 0.95 `
  --max-steps 879 `
  --validation-interval 250 `
  --checkpoint-interval 250 `
  --log-interval 10 `
  --early-stop-patience-windows 20 `
  --early-stop-warmup-steps 0 `
  --dynamics-context-frames 1 `
  --dynamics-target-frames X`
  --dynamics-model-channels XXX `
  --dynamics-num-blocks XXX `
  --dynamics-num-heads XXX `
  --dynamics-action-conditioning-mode chunk_per_frame `
  --dynamics-action-representation relative_delta `
  --dynamics-action-scale 20 `
  --dynamics-adaln-lora-dim XXX `
  --dynamics-infer-steps 35 `
  --dynamics-train-timesteps 1000 `
  --dynamics-rf-shift 5.0 `
  --dynamics-validation-metric next_frame_mse `
  --no-dynamics-run-open-rollout-validation `
  --load-encoder-decoder $WANVAE ` #improve
  --device cuda `
  --run-name  `

  --wandb `
  --wandb-project world-model-v2 `
  --wandb-group `
  --wandb-name  `
  --wandb-tags `
  --wandb-mode online
