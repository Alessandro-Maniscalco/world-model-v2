## Goal
 I am working on the dynamics training. I am following closely https://github.com/NVIDIA/DreamDojo design. DreamDojo uses 1 context latent and 3 inferred latents.

 To look into:
 First, the action-conditioned teacher really is a 1 latent ctx + 3 latent targets setup: in their 2B action-conditioned rectified-flow config they set min_num_conditional_frames = max_num_conditional_frames = 1, state_t = 1 + 12 // 4, and num_action_per_chunk = 12, so one clean latent is used to predict three future latents from a 12-action chunk with temporal compression ratio 4 (config). During denoising they strongly pin that context latent by replacing it with ground truth and forcing a tiny conditioning noise sigma_conditional = 1e-4 (model). My inference from the code is that the teacher “finds” the later frames jointly in one denoising pass, not by explicit within-chunk autoregression.

Second, DreamDojo does not stop there. Their docs say long-horizon stability comes from distillation into a causal student, with a warmup stage and then self-forcing to reduce error accumulation (DISTILL.md). The warmup model explicitly makes the network temporally causal (warmup), and the self-forcing config trains a causal student initialized from the teacher (self-forcing config). They also use chunk overlap for temporal consistency in autoregressive inference (config).

The other big difference is that their action-conditioned teacher is warm-started from a pretrained video model, not trained from scratch on the 1+3 task (post-train docs, load path in config). So the short answer is: DreamDojo gets away with 1->3 because it uses a strongly anchored pretrained teacher, joint denoising over the whole latent chunk, and then a separate causal self-forcing distillation stage for rollout stability.

## Progress

1 ctx 1 inferred, mp4: outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to1_f48_67/samples/step_001500/episode_0.mp4.
learned: works. Predicted frame is closer to current gt than previous gt, so it is not just copying the ctx frame. 
Test: all 19/19 predicted frames are closer to the current GT than the previous GT. Mean cropped-frame MSE was 3.76e-4 to gt[t] versus 1.73e-3 to gt[t-1]

1 ctx 2 inferred, mp4: outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_f48_67/samples/step_002000/episode_0.mp4
learned: first inferred moves correctly. Second inferred often lags and stays closer to the previous frame than the current gt.

3 frames, 1/2 cts 2/1 inferred, val 1 ctx 2 inferred, mp4: outputs/controller_dreamdojo_progressive/controller_mixed12_rfshift3_f48_67/samples/step_002000/episode_0.mp4
learned: mixed [1,2] conditioning with rf_shift=3 and infer_steps=32 did not remove the pattern. Arm motion appears, then one frame changes only slightly, then motion continues, which suggests the weak frame is still the second inferred frame.

## Latest Code Change

2026-04-08:
Implemented a DreamDojo-inspired causal self-forcing auxiliary loss for dynamics training.

Why:
The current 1 ctx -> 2 target teacher already predicts both future latents jointly, but the weak second inferred frame suggests the model is not being trained hard enough on the case where its first prediction becomes the next causal context. DreamDojo addresses this with warmup + self-forcing distillation. I added the cheapest clean approximation of that idea inside the current teacher-training codepath.

What changed:
- Added `--dynamics-self-forcing-loss-weight` to `world_model_v2.run`.
- In `Experiment._dynamics_only_training_step`, after the standard RF teacher loss, the code now optionally:
  - reconstructs the model's predicted clean latent chunk,
  - detaches the predicted prefix,
  - feeds that predicted prefix back in as a longer conditioning window (`ctx+1`, `ctx+2`, ...),
  - applies an auxiliary RF MSE only on the remaining later target frames.
- Added helper support for explicit conditioning masks outside the main sampled training choices so the auxiliary causal passes can use `2` conditioned frames even in a `1 -> 2` run.
- Threaded the new flag through `scripts/check/loop_dynamics_sweep.py` so controller sweeps can search it directly.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_dynamics_transformer.py tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py -q`
- `source .venv/bin/activate && pytest tests/test_experiment_runtime.py tests/test_model.py tests/test_model_runtime.py -q`
- Result: 98 tests passed.

Recommended next training direction:
- Re-run the focused `1 ctx 2 inferred` setup with `--dynamics-self-forcing-loss-weight 0.5`.
- If the second target improves but the first target regresses, sweep `0.25` and `1.0` next.

## Latest Result

1 ctx 2 inferred + self-forcing 0.5, best checkpoint:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_ft_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_ft_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_ft_f48_67/manual_best_open_rollout_check`

learned:
- The self-forcing loss did what it was supposed to do on the teacher-forced `1 -> 2` task.
- The weak second inferred frame improved a lot:
  - old `next_frame_mse_target_1`: `1.2786e-3`
  - new best `next_frame_mse_target_1`: `6.3489e-4`
- The first inferred frame became worse:
  - old `next_frame_mse_target_0`: `4.2226e-4`
  - new best `next_frame_mse_target_0`: `7.1931e-4`
- Overall teacher-forced `1 -> 2` MSE still improved:
  - old `next_frame_mse`: `8.2792e-4`
  - new best `next_frame_mse`: `6.7932e-4`
- Open-rollout improved only slightly:
  - old best `open_rollout_frame_mse`: `3.6188e-2`
  - new best `open_rollout_frame_mse`: `3.5676e-2`
- Open-rollout motion ratio also improved:
  - old `open_rollout_target_motion_ratio`: `6.84`
  - new best `open_rollout_target_motion_ratio`: `4.80`

Interpretation:
- The causal auxiliary loss is helping redistribute capacity from the first inferred frame toward the second inferred frame.
- That means the `1 ctx 2 inferred` formulation is not fundamentally broken in this codebase.
- But self-forcing alone is not enough; rollout quality is still far from the stronger `rf_shift=3` mixed-conditioning run.

Extra small check:
- Running the same best checkpoint at `infer_steps=32` instead of `50` improved open-rollout MSE slightly:
  - `infer_steps=50`: `3.5676e-2`
  - `infer_steps=32`: `3.4943e-2`
- So the next run should evaluate at `32` steps, not `50`.

Next search direction:
- Keep the clean `1 ctx 2 inferred` layout.
- Combine it with the better RF schedule from the mixed run: try `dynamics_rf_shift=3.0` on top of the current self-forcing checkpoint, and validate with `infer_steps=32`.

## Latest Result 2

1 ctx 2 inferred + self-forcing 0.5 + rf_shift 3.0 + infer_steps 32, best checkpoint:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/manual_best_open_rollout_check`

learned:
- This was the first clean `1 ctx -> 2 inferred` run that clearly moved the whole setup into the same regime as the stronger mixed-conditioning experiment.
- Best checkpoint happened early at step `250`, not at the end.
- Best open-rollout MSE improved a lot versus the previous clean `1 -> 2` runs:
  - baseline `1 -> 2`: `3.6188e-2`
  - `1 -> 2 + self-forcing`: `3.5676e-2`
  - `1 -> 2 + self-forcing + rf_shift=3`: `2.2398e-2`
- Teacher-forced quality also improved:
  - best `next_frame_mse`: `6.6285e-4`
  - best target-0 MSE: `7.1453e-4`
  - best target-1 MSE: `6.0542e-4`
- Open-rollout motion ratio improved strongly:
  - previous clean self-forcing run: `4.80`
  - new best: `3.16`

Comparison to the mixed-conditioning rf_shift=3 run:
- Mixed run still has the best tuned-span open rollout on this clip:
  - mixed `[1,2]` best `open_rollout_frame_mse`: `2.0849e-2`
  - clean `1 -> 2` best `open_rollout_frame_mse`: `2.2398e-2`
- But the clean `1 -> 2` run is now close, while having much better teacher-forced `1 -> 2` metrics than the mixed run.

Interpretation:
- The main blocker for clean `1 -> 2` was not the chunk layout by itself.
- The important ingredients were:
  - causal self-forcing pressure on the second inferred frame
  - the lower `rf_shift=3` schedule
  - validating/sampling at `infer_steps=32`
- The fact that the best checkpoint is at step `250` suggests this leg now wants short polish runs or early-stop-aware sweeps rather than longer full-length finetunes.

Next search direction:
- Start from the new best checkpoint and run a short low-LR polish leg.
- Keep the same `1 -> 2`, `self_forcing=0.5`, `rf_shift=3.0`, and `infer_steps=32` settings.
- Only change optimization pressure next: lower LR and shorter run length.

## Latest Result 3

Short low-LR polish from the clean `1 -> 2` best checkpoint:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_lr2e5_polish_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_lr2e5_polish_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_lr2e5_polish_f48_67/manual_best_open_rollout_check`

learned:
- This polish leg did not beat the parent checkpoint.
- Best checkpoint in the polish leg was step `400`.
- Best open-rollout MSE from the polish leg was `2.5395e-2`.
- That is worse than the parent clean `1 -> 2` best:
  - parent best `open_rollout_frame_mse`: `2.2398e-2`
  - polish best `open_rollout_frame_mse`: `2.5395e-2`
- Teacher-forced `1 -> 2` stayed strong, but rollout got worse again:
  - polish best `next_frame_mse`: `6.2933e-4`
  - polish best `next_frame_mse_target_0`: `7.0890e-4`
  - polish best `next_frame_mse_target_1`: `5.4093e-4`

Interpretation:
- Once the clean `1 -> 2` run reaches the good `rf_shift=3` regime, more plain low-LR polishing does not help rollout on this span.
- The objective is still pulling teacher-forced quality and open-loop quality in different directions.
- So the next search should not be another same-objective polish. The next search needs to change the balance of the objective again.

Next search direction:
- Keep the best checkpoint from `controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67`.
- Reduce self-forcing pressure from `0.5` to `0.25` and do a short run with frequent validation.
- Goal: keep most of the second-frame gain while giving the first inferred frame and rollout a chance to recover.
