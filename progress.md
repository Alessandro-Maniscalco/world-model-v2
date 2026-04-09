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

## Latest Result 4

Short clean `1 -> 2` continuation with lower self-forcing pressure:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce025_rfshift3_infer32_short_f48_67/checkpoints/best.pt`

learned:
- This did not help rollout. Best checkpoint was again early at step `250`, but the run landed far behind the parent `self_forcing=0.5` model on open rollout.
- Best open-rollout MSE regressed badly:
  - parent clean `1 -> 2` best: `2.2398e-2`
  - `self_forcing=0.25` short run best: `3.3411e-2`
- Teacher-forced `1 -> 2` metrics stayed strong:
  - best `next_frame_mse`: `6.1844e-4`
  - best target-0 MSE: `6.1040e-4`
  - best target-1 MSE: `6.2737e-4`

Interpretation:
- Reducing self-forcing pressure alone is the wrong axis.
- The model can still score well on teacher-forced `1 -> 2` while rollout gets much worse, so the remaining gap is now more about conditioning semantics than raw loss weight balance.
- That means the next change should stay with `self_forcing=0.5` and instead attack the teacher-conditioning mismatch versus DreamDojo.

## Latest Code Change 2

2026-04-08:
Implemented DreamDojo-style tiny conditioning noise for repinned context frames.

Why:
The local RF teacher had still been pinning conditioned frames perfectly clean during both training and sampling. DreamDojo's action-conditioned configs instead use `sigma_conditional=1e-4`, so their teacher never sees a mathematically exact clean prefix inside the denoising state. After the failed `self_forcing=0.25` run, this looked like the cleanest remaining mismatch to fix without changing the chunk layout or undoing the useful second-target pressure from `self_forcing=0.5`.

What changed:
- Added `--conditional-frame-sigma` to `world_model_v2.run`.
- Threaded the flag through `ExperimentConfig`, `WorldModel`, `scripts/check/loop_dynamics_sweep.py`, and `scripts/check/open_rollout_demo.py`.
- In `RectifiedFlowDynamics`, conditioned frames are now optionally repinned to `x_cond = x_clean + sigma_conditional * v` instead of always using exact clean latents.
- Applied the same tiny-sigma conditioning path in both teacher training and iterative sampling so rollout semantics match training semantics.
- Kept checkpoint compatibility: older checkpoints without this field still load with `conditional_frame_sigma=0.0`.

Design decision:
- I left `conditional_frame_timestep` unchanged at `-1.0` for the next run.
- Reason: the `self_forcing=0.25` experiment already showed the loss-weight axis was misleading, so the next study should isolate the conditioning-noise change by itself instead of moving two DreamDojo-mimic knobs at once.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_dynamics_transformer.py tests/test_experiment.py tests/test_run.py tests/test_model.py tests/test_loop_dynamics_sweep.py -q`
- `source .venv/bin/activate && pytest tests/test_experiment_runtime.py tests/test_model_runtime.py -q`
- Result: `102 passed` across both pytest slices.
- Compatibility smoke check:
  - `source .venv/bin/activate && python - <<'PY' ... build_model_from_checkpoint(...) ... PY`
  - Older best checkpoint `controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt` still loads and resolves `conditional_frame_sigma=0.0`.

Recommended next training direction:
- Start from `outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt`.
- Keep the current best clean recipe:
  - `1 ctx -> 2 inferred`
  - `self_forcing=0.5`
  - `rf_shift=3.0`
  - `infer_steps=32`
- Only add `--conditional-frame-sigma 1e-4`.
- Use a short frequent-validation run again, because the best clean checkpoint has kept appearing around step `250`.

## Latest Result 5

Short continuation from the clean best with DreamDojo-style `conditional_frame_sigma=1e-4`:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_condsig1e4_short_f48_67/checkpoints/best.pt`

learned:
- This change did not help the clean `1 -> 2` rollout regime.
- Best checkpoint was at step `200`.
- Best open-rollout MSE was worse than both the parent best and even the failed `self_forcing=0.25` short run:
  - parent clean best: `2.2398e-2`
  - `self_forcing=0.25` short run: `3.3411e-2`
  - `conditional_frame_sigma=1e-4` short run: `3.4155e-2`
- Teacher-forced `1 -> 2` stayed good:
  - best `next_frame_mse`: `6.3267e-4`
  - best target-0 MSE: `7.0201e-4`
  - best target-1 MSE: `5.5563e-4`

Interpretation:
- Matching DreamDojo's tiny conditional sigma was not the missing ingredient in this smaller setup.
- The run again improved teacher-forced metrics while hurting open-loop rollout, so the remaining gap still looks like a staging/objective issue rather than a tiny teacher-noise mismatch.

## Latest Code Change 3

2026-04-08:
Added explicit open-rollout overlap controls and staged self-forcing warmup support.

Why:
- DreamDojo docs mention chunk overlap during autoregressive inference, so I added a clean way to test overlap directly instead of assuming the current stride-2 rollout is optimal for a `1 -> 2` teacher.
- The bigger structural difference versus DreamDojo is still warmup first, self-forcing later. The repo had only a constant self-forcing weight from step `0`, so I added a warmup gate to let the model train as a plain teacher for an initial window before enabling the causal auxiliary loss.

What changed:
- Added `--dynamics-open-rollout-stride-frames` to the model/config/CLI/sweep/demo path.
- `WorldModel.rollout` can now append fewer frames than one sampled chunk predicts, so overlap experiments are explicit instead of being hard-coded to stride equal to the full target chunk.
- Added `--dynamics-self-forcing-warmup-steps` so self-forcing can stay off for the first N optimizer steps and then turn on automatically.
- Validation stats and rollout demo stats now record the effective rollout stride.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py tests/test_model.py tests/test_dynamics_transformer.py -q`
- `source .venv/bin/activate && pytest tests/test_experiment_runtime.py tests/test_model_runtime.py -q`
- Result: `111 passed`.

Focused local rollout check:
- I ran the parent clean best checkpoint with overlap stride `1` on the same `f48:67` span:
  - command: `source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --dynamics-open-rollout-stride-frames 1 --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/overlap_stride_checks --run-name stride1_f48_67`
  - result: `open_rollout_frame_mse = 3.5723e-2`
  - baseline same checkpoint with default stride-2 rollout: `2.2398e-2`

Interpretation:
- Overlap at inference is not the fix here. This teacher is using its joint `1 -> 2` head effectively when allowed to commit both frames; forcing stride-1 overlap makes rollout much worse on this clip.
- That leaves staged self-forcing as the cleaner remaining DreamDojo-inspired change to test next.

Recommended next training direction:
- Start from the clean no-self-forcing checkpoint `outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_f48_67/checkpoints/best.pt`.
- Keep the good rollout recipe:
  - `1 ctx -> 2 inferred`
  - `rf_shift=3.0`
  - `infer_steps=32`
  - validation metric `open_rollout_frame_mse`
- Enable DreamDojo-style staging instead of constant self-forcing from step `0`:
  - `dynamics_self_forcing_loss_weight=0.5`
  - `dynamics_self_forcing_warmup_steps=250`
- Do not use `conditional_frame_sigma` or overlap stride for that run.

## Latest Result 6

Warmup-then-self-forcing run from the clean no-self-forcing `1 -> 2` checkpoint:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_rfshift3_selfforce05_warm250_infer32_ft_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_rfshift3_selfforce05_warm250_infer32_ft_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_rfshift3_selfforce05_warm250_infer32_ft_f48_67/manual_best_open_rollout_check`

learned:
- This staging idea helped a lot versus the original clean no-self-forcing baseline, but it still did not beat the existing best constant-self-forcing recipe.
- Best checkpoint was exactly step `250`, which is the last pure warmup validation before self-forcing turns on.
- Best open-rollout MSE improved strongly versus the original clean baseline:
  - original clean `1 -> 2` baseline: `3.6188e-2`
  - warmup-best before self-forcing: `2.8402e-2`
- But it stayed clearly worse than the current best constant-self-forcing run:
  - constant `self_forcing=0.5` + `rf_shift=3` best: `2.2398e-2`
  - warmup-best before self-forcing: `2.8402e-2`
- Teacher-forced metrics at the warmup best looked like the old clean regime, not the stronger second-target regime:
  - best `next_frame_mse`: `7.9176e-4`
  - best target-0 MSE: `4.1987e-4`
  - best target-1 MSE: `1.2050e-3`

Interpretation:
- The important signal is that the best checkpoint happened exactly before self-forcing started.
- That means this single-run warmup gate is not enough by itself. The separate teacher-style warmup leg is useful, but turning on self-forcing in-place with the same optimizer state still hurts this setup.
- This is closer to DreamDojo than before, though: we now have an explicit warmup stage result, and it suggests the next thing to test is a separate second-stage finetune initialized from that warmup-best checkpoint, not a single run that switches objectives midstream.

## Latest Code Change 4

2026-04-08:
Added self-forcing warmup gating to the training loop and used it to isolate the warmup-stage best checkpoint.

Why:
- DreamDojo separates warmup and self-forcing into different stages.
- The repo previously only supported a constant self-forcing weight from step `0`.
- I added `--dynamics-self-forcing-warmup-steps` so we can train a pure teacher stage first, then enable the auxiliary causal loss later, and measure whether the best checkpoint comes before or after that transition.

What changed:
- Added `--dynamics-self-forcing-warmup-steps` to the CLI and sweep helper.
- `Experiment._dynamics_only_training_step` now computes an active self-forcing weight based on `current_step`.
- Metrics now log `active_self_forcing_loss_weight` so the switch point is visible in training traces.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py tests/test_model.py tests/test_dynamics_transformer.py -q`
- `source .venv/bin/activate && pytest tests/test_experiment_runtime.py tests/test_model_runtime.py -q`
- Result: `111 passed`.

Design decision:
- I am not removing the constant-self-forcing path, because it still gives the best observed rollout on this clip.
- The new warmup gate is a research control: it lets us cleanly separate "warmup helps" from "switching to self-forcing helps".

Recommended next training direction:
- Keep the warmup-best checkpoint from `controller_dreamdojo_progressive_1to2_rfshift3_selfforce05_warm250_infer32_ft_f48_67/checkpoints/best.pt` as the stage-1 result.
- Start a fresh second-stage finetune from that checkpoint with optimizer reset:
  - `self_forcing=0.5`
  - `self_forcing_warmup_steps=0`
  - lower LR than the warmup leg, because the objective is changing at init
- This is the closest clean analogue to DreamDojo's explicit warmup stage followed by self-forcing stage.

## Latest Result 7

Separate stage-2 self-forcing finetune from the warmup-best checkpoint:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_rfshift3_warmbest_selfforce05_lr2e5_short_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_rfshift3_warmbest_selfforce05_lr2e5_short_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_rfshift3_warmbest_selfforce05_lr2e5_short_f48_67/manual_best_open_rollout_check`

learned:
- This separate stage-2 run was worse than the warmup-best checkpoint it started from.
- Best checkpoint happened early at step `200`.
- Best open-rollout MSE regressed sharply:
  - warmup-best stage-1 result: `2.8402e-2`
  - separate stage-2 self-forcing finetune: `4.1677e-2`
- Teacher-forced metrics shifted only a little:
  - best `next_frame_mse`: `7.5179e-4`
  - best target-0 MSE: `6.3968e-4`
  - best target-1 MSE: `8.7635e-4`

Interpretation:
- A hard switch into `self_forcing=0.5` is too abrupt, even when done as a separate stage with reset optimizer and lower LR.
- The failure mode is now consistent across both in-place transition and separate stage-2 transition:
  - constant self-forcing from a strong `rf_shift=3` run can work,
  - but switching a warmup-style checkpoint straight to full self-forcing hurts rollout.
- So the next clean thing to test is not another binary on/off stage split. The next test needs a smooth self-forcing transition.

## Latest Code Change 5

2026-04-08:
Added linear self-forcing ramp support on top of the existing warmup gate.

Why:
- The latest stage-2 finetune showed that going directly from warmup-best to full `self_forcing=0.5` is too sharp.
- The training logic needed one more control: warmup decides when self-forcing is allowed to start, and ramp controls how fast it grows to the target weight once it starts.

What changed:
- Added `--dynamics-self-forcing-ramp-steps` to `world_model_v2.run` and `scripts/check/loop_dynamics_sweep.py`.
- `Experiment._active_dynamics_self_forcing_loss_weight()` now supports:
  - zero weight during warmup,
  - linear ramp from `0` to `dynamics_self_forcing_loss_weight`,
  - full weight after the ramp finishes.
- Added tests for negative-ramp rejection, CLI/sweep wiring, and ramped weight behavior inside `_dynamics_only_training_step`.

Design decision:
- I kept the warmup and ramp as separate knobs.
- Reason: they control different things:
  - warmup = when the auxiliary objective begins,
  - ramp = how sharply it takes over.
- Keeping them separate makes the search cleaner than baking one fixed schedule into the trainer.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py -q`
- `source .venv/bin/activate && pytest tests/test_experiment_runtime.py tests/test_model_runtime.py -q`
- Result: `87 passed` across both slices.

Recommended next training direction:
- Start again from the warmup-best checkpoint `controller_dreamdojo_progressive_1to2_rfshift3_selfforce05_warm250_infer32_ft_f48_67/checkpoints/best.pt`.
- Keep the same stage-2 setup as the failed run, except replace the abrupt full-weight switch with a ramp:
  - `self_forcing=0.5`
  - `self_forcing_warmup_steps=0`
  - `self_forcing_ramp_steps=250`
  - `lr=2e-5`
  - short frequent-validation run

## Latest Code Change 6

2026-04-09:
Replaced the self-forcing research path with a rollout-aligned option that uses the same context semantics as open-loop inference.

Why:
- After reading the current trainer carefully, the existing auxiliary loss turned out to be solving the wrong problem for the clean `1 -> 2` case.
- The old helper improves the second target by feeding back a predicted **expanded prefix** inside the same 3-frame chunk:
  - train on `1 ctx -> 2 targets`
  - auxiliary pass on `2 ctx -> 1 target`
- But open-loop rollout never uses that `2 ctx -> 1 target` pattern. With `conditioning_frame_choices=(1,)`, rollout always reuses a **single** context frame on the next chunk.
- That mismatch explains the recent pattern in the logs:
  - teacher-forced `1 -> 2` improved a lot
  - rollout improved only a little
- DreamDojo's distillation docs also point in this direction: self-forcing is about the model training on its own autoregressive predictions, not just a stronger within-chunk teacher target.

What changed:
- Added `--dynamics-self-forcing-mode` with:
  - `expanded_context` = the old within-chunk prefix expansion path
  - `rollout` = new same-context rollout auxiliary
- Added `--dynamics-self-forcing-rollout-chunks` to request extra future chunks for rollout self-forcing.
- Extended both `TransitionDataset` and `MetaWorldTransitionDataset` so dynamics batches can optionally carry:
  - `future_target_frames`
  - `future_actions`
- In rollout mode, `Experiment._dynamics_self_forcing_loss` now:
  - reuses the configured `open_rollout_context_frames`,
  - rolls the model forward on future chunks with the same action-window slicing used by `WorldModel.rollout`,
  - scores only future target frames from those extra chunks.
- Kept the old expanded-context path as the default so earlier runs remain reproducible.
- Threaded the new flags through `world_model_v2.run` and `scripts/check/loop_dynamics_sweep.py`.

Design decision:
- I intentionally tied rollout self-forcing to `open_rollout_context_frames` instead of adding a third context knob.
- Reason: the whole point of this study is to remove the training/inference semantic mismatch, so the auxiliary should follow the exact rollout context count that validation and demos already use.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_dataset.py tests/test_metaworld_dataset.py tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py -q`
- `source .venv/bin/activate && pytest tests/test_dynamics_transformer.py tests/test_model.py tests/test_model_runtime.py tests/test_experiment_runtime.py -q`
- Result: `136 passed` across both pytest slices.
- Real-checkpoint smoke run of the new objective path:
  - `source .venv/bin/activate && python -m world_model_v2.run --mode dynamics_only --dataset-format lerobot_metaworld --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --batch-size 1 --lr 2e-5 --max-steps 1 --validation-interval 1 --checkpoint-interval 0 --log-interval 1 --early-stop-window-size 0 --output-dir outputs/smoke --run-name rollout_self_forcing_smoke_1step --load-encoder-decoder outputs/minimal/metaworld_task0_wan_ae_240/checkpoints/best.pt --load-dynamics outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --dynamics-context-frames 1 --dynamics-target-frames 2 --dynamics-conditioning-frame-choices 1 --dynamics-conditioning-frame-probabilities 1.0 --dynamics-validation-conditioning-frame-choices 1 --dynamics-open-rollout-context-frames 1 --dynamics-self-forcing-loss-weight 0.5 --dynamics-self-forcing-mode rollout --dynamics-self-forcing-rollout-chunks 1 --dynamics-infer-steps 32 --dynamics-rf-shift 3.0 --dynamics-validation-metric open_rollout_frame_mse --device cuda`
  - Result: completed successfully and logged `latent_rf_self_forcing_rollout_mse_chunk1`, which confirms the new rollout auxiliary path runs end-to-end on real data/checkpoints.

Recommended next training direction:
- Start from the current best clean checkpoint:
  - `outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt`
- Keep the known-good recipe fixed:
  - `1 ctx -> 2 inferred`
  - `rf_shift=3.0`
  - `infer_steps=32`
  - validation metric `open_rollout_frame_mse`
  - batch size `9`
- Change only the auxiliary semantics:
  - `self_forcing_loss_weight=0.5`
  - `self_forcing_mode=rollout`
  - `self_forcing_rollout_chunks=1`
  - short low-LR continuation with frequent validation

Expectation:
- If this hypothesis is right, teacher-forced metrics may move only slightly, but open-rollout should respond more directly than it did under the old expanded-prefix auxiliary because the training target finally matches the `1 ctx` rollout path.

## Latest Result 8

Rollout-aligned self-forcing run from the clean best checkpoint:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rollout1_rfshift3_infer32_lr2e5_short_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rollout1_rfshift3_infer32_lr2e5_short_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rollout1_rfshift3_infer32_lr2e5_short_f48_67/manual_best_open_rollout_check`

learned:
- The new rollout-aligned auxiliary is real and trainable: the run completed cleanly, the best checkpoint was at step `350`, and the manual rollout check exactly matched the validation metric.
- It improved a lot over the earlier warmup-style and failed continuation legs, but it did **not** beat the existing best expanded-context self-forcing run.
- Best open-rollout MSE:
  - parent clean best with old expanded-context self-forcing: `2.2398e-2`
  - new rollout-self-forcing best: `2.4004e-2`
- Teacher-forced quality improved slightly versus the parent best:
  - parent `next_frame_mse`: `6.6285e-4`
  - rollout-self-forcing best `next_frame_mse`: `6.5196e-4`
- The target balance shifted:
  - parent best target-0 MSE: `7.1453e-4`
  - rollout-self-forcing best target-0 MSE: `5.9693e-4`
  - parent best target-1 MSE: `6.0542e-4`
  - rollout-self-forcing best target-1 MSE: `7.1310e-4`
- Open-rollout motion ratio improved a little versus the parent best:
  - parent best motion ratio: `3.16`
  - rollout-self-forcing best motion ratio: `3.00`

Interpretation:
- The rollout-aligned auxiliary is not a dead end. It changed behavior in a meaningful DreamDojo-like direction:
  - rollout motion got slightly more realistic,
  - teacher-forced average error got slightly better,
  - the best checkpoint appeared mid-run instead of immediately collapsing.
- But the loss is currently over-tilting the tradeoff:
  - it helps the generated sequence behave more like a usable rollout,
  - while giving back some of the old second-target sharpness that the expanded-context auxiliary had recovered.
- So the main conclusion is:
  - the semantic mismatch diagnosis was useful,
  - but `rollout self-forcing weight = 0.5` is too strong for this tiny setup.

Next search direction:
- Keep the new rollout auxiliary code.
- Start from the new rollout best checkpoint, not from scratch.
- Reduce rollout self-forcing pressure from `0.5` to `0.25` and run another short low-LR continuation.
- Goal: keep the better rollout motion ratio while recovering some of the open-rollout MSE gap to the parent best.

## Latest Result 9

Short continuation from the rollout-best checkpoint with lower rollout self-forcing pressure:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce025_rollout1_fromrolloutbest_rfshift3_infer32_lr2e5_short_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce025_rollout1_fromrolloutbest_rfshift3_infer32_lr2e5_short_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce025_rollout1_fromrolloutbest_rfshift3_infer32_lr2e5_short_f48_67/manual_best_open_rollout_check`

learned:
- Lowering rollout self-forcing from `0.5` to `0.25` helped teacher-forced metrics again, but it did not recover the rollout gap to the original expanded-context best.
- Best checkpoint was early at step `200`.
- Best open-rollout MSE:
  - rollout-self-forcing `0.5` best: `2.4004e-2`
  - rollout-self-forcing `0.25` best: `2.5747e-2`
  - original expanded-context best: `2.2398e-2`
- Teacher-forced metrics improved further:
  - rollout-self-forcing `0.5` best `next_frame_mse`: `6.5196e-4`
  - rollout-self-forcing `0.25` best `next_frame_mse`: `6.2929e-4`
  - rollout-self-forcing `0.25` best target-0 MSE: `5.4081e-4`
  - rollout-self-forcing `0.25` best target-1 MSE: `7.2760e-4`
- Open-rollout motion ratio moved back toward the expanded-context regime:
  - rollout-self-forcing `0.5` best: `3.00`
  - rollout-self-forcing `0.25` best: `3.21`
  - original expanded-context best: `3.16`

Interpretation:
- This confirms the pattern from the previous run:
  - smaller rollout weight improves teacher-forced numbers,
  - larger rollout weight helps rollout more directly.
- So rollout-only self-forcing is trading against the within-chunk second-target objective instead of replacing it.
- The clean next move is no longer another rollout-only sweep. The next move should combine both pressures:
  - keep the strong expanded-context auxiliary that gave the best `1 -> 2` checkpoint,
  - add a smaller rollout-aligned auxiliary on top.

## Latest Code Change 7

2026-04-09:
Added an optional rollout self-forcing auxiliary weight on top of the existing primary self-forcing mode.

Why:
- The experiments now separate cleanly into two behaviors:
  - expanded-context self-forcing gives the best overall open-rollout MSE and the strongest second-target recovery,
  - rollout self-forcing improves different rollout-facing signals but cannot beat the parent best by itself.
- DreamDojo effectively benefits from both kinds of pressure across teacher and student stages.
- The closest local analogue in this codebase is to keep the current primary auxiliary exactly as-is and add a second rollout-aligned loss with its own weight.

What changed:
- Added `--dynamics-rollout-self-forcing-loss-weight` to `world_model_v2.run` and `scripts/check/loop_dynamics_sweep.py`.
- Kept `--dynamics-self-forcing-mode` as the primary auxiliary selector:
  - `expanded_context` or `rollout`
- Added hybrid support implicitly:
  - use `dynamics-self-forcing-mode expanded_context`
  - keep `dynamics-self-forcing-loss-weight > 0`
  - add `dynamics-rollout-self-forcing-loss-weight > 0`
- Both self-forcing losses now share the same warmup/ramp schedule multiplier.
- Training metrics now report:
  - `active_rollout_self_forcing_loss_weight`
  - `latent_rf_rollout_self_forcing_mse`
  - `latent_rf_rollout_self_forcing_weighted_loss`
- Added config/CLI/test coverage and a real 1-step CUDA smoke run for the hybrid path.

Design decision:
- I did not add a separate second warmup/ramp schedule for rollout auxiliary.
- Reason: the immediate research question is about *loss composition*, not schedule search. Sharing one schedule keeps the next experiment interpretable.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_experiment.py tests/test_experiment_runtime.py tests/test_run.py tests/test_loop_dynamics_sweep.py -q`
- Result: `89 passed`.
- Hybrid-path smoke check:
  - `source .venv/bin/activate && python -m world_model_v2.run ... --dynamics-self-forcing-loss-weight 0.5 --dynamics-rollout-self-forcing-loss-weight 0.25 --dynamics-self-forcing-mode expanded_context --dynamics-self-forcing-rollout-chunks 1 ... --max-steps 1 ...`
  - Result: completed successfully on CUDA and logged both primary and rollout self-forcing metrics in the same training step.

Recommended next training direction:
- Start from the original clean best expanded-context checkpoint:
  - `outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt`
- Keep the known-good core recipe fixed:
  - `1 ctx -> 2 inferred`
  - `rf_shift=3.0`
  - `infer_steps=32`
  - validation metric `open_rollout_frame_mse`
  - `lr=2e-5`
- Run a short hybrid continuation:
  - expanded-context self-forcing weight `0.5`
  - rollout self-forcing auxiliary weight `0.25`
  - rollout chunks `1`
  - frequent validation

## Latest Result 10

Hybrid continuation from the expanded-context best with both auxiliaries enabled:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_plusrollout025_rfshift3_infer32_lr2e5_short_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_plusrollout025_rfshift3_infer32_lr2e5_short_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_plusrollout025_rfshift3_infer32_lr2e5_short_f48_67/manual_best_open_rollout_check_turn4`

learned:
- The hybrid loss still did not beat the original expanded-context best, and it also did not beat rollout-only `0.5`.
- Best checkpoint arrived immediately at step `50`, then degraded:
  - step `50` open-rollout MSE: `2.4634e-2`
  - step `100` open-rollout MSE: `3.1312e-2`
  - step `150` open-rollout MSE: `3.8475e-2`
  - step `200` open-rollout MSE: `6.6179e-2`
- The direct open-rollout check matched the saved validation exactly:
  - `input_frame_count=20`
  - `predicted_frame_count=20`
  - `decoded_frame_count=20`
  - `loss_frames=19`
  - `seed_frames=1`
  - `open_rollout_frame_mse=2.4633770808577538e-2`
- Comparison against the relevant baselines:
  - original expanded-context best: `2.2398e-2`
  - rollout-only `0.5` best: `2.4004e-2`
  - hybrid `0.5 + 0.25` best: `2.4634e-2`
  - rollout-only `0.25` continuation best: `2.5747e-2`

Interpretation:
- Adding rollout pressure on top of the expanded-context auxiliary is not wrong in principle, but enabling it immediately is too aggressive for a checkpoint that is already near a good teacher-forced optimum.
- The shape of the run matters more than the raw hybrid composition here:
  - step `50` is still usable,
  - later steps drift away,
  - that is the signature of an auxiliary that should be delayed or ramped separately instead of sharing the primary schedule.

## Latest Code Change 8

2026-04-09:
Added a separate warmup/ramp schedule for the rollout self-forcing auxiliary.

Why:
- The previous hybrid implementation made both auxiliaries share one schedule.
- The hybrid training result showed that the rollout auxiliary is the unstable part:
  - the best checkpoint appeared at the first validation window,
  - continued optimization under immediate rollout pressure made rollout quality worse instead of better.
- So the clean next change is not another weight sweep first. The clean change is to decouple *when* the rollout auxiliary turns on from the already-working expanded-context auxiliary.

What changed:
- Added rollout-auxiliary-only schedule config fields:
  - `dynamics_rollout_self_forcing_warmup_steps`
  - `dynamics_rollout_self_forcing_ramp_steps`
- Threaded those flags through:
  - `world_model_v2.run`
  - `scripts/check/loop_dynamics_sweep.py`
  - `ExperimentConfig`
- Refactored the schedule math so both objectives use the same helper but not the same config:
  - primary self-forcing keeps using `dynamics_self_forcing_warmup_steps` and `dynamics_self_forcing_ramp_steps`
  - rollout auxiliary now uses its own rollout-specific warmup/ramp pair
- Added tests for:
  - negative rollout warmup/ramp validation
  - CLI parsing
  - sweep-command forwarding
  - training-step behavior where the primary auxiliary is active while the rollout auxiliary is still warming up
  - training-step behavior where the rollout auxiliary ramps independently after warmup

Design decision:
- I kept the rollout schedule default at zero instead of implicitly inheriting the primary schedule.
- Reason: the bad behavior only appears when rollout pressure turns on too early. Making the rollout schedule explicit keeps the search space understandable and makes the next run interpretable from the command line.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py -q`
  - Result: `85 passed`.
- `source .venv/bin/activate && pytest tests/test_experiment_runtime.py -q`
  - Result: `10 passed`.
- Delayed-rollout CUDA smoke:
  - `source .venv/bin/activate && python -m world_model_v2.run --mode dynamics_only --dataset-format lerobot_metaworld --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --batch-size 1 --lr 2e-5 --max-steps 1 --validation-interval 1 --checkpoint-interval 0 --log-interval 1 --early-stop-window-size 0 --output-dir outputs/smoke --run-name rollout_aux_schedule_smoke_1step --load-encoder-decoder outputs/minimal/metaworld_task0_wan_ae_240/checkpoints/best.pt --load-dynamics outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --dynamics-context-frames 1 --dynamics-target-frames 2 --dynamics-conditioning-frame-choices 1 --dynamics-conditioning-frame-probabilities 1.0 --dynamics-validation-conditioning-frame-choices 1 --dynamics-open-rollout-context-frames 1 --dynamics-self-forcing-loss-weight 0.5 --dynamics-rollout-self-forcing-loss-weight 0.25 --dynamics-self-forcing-mode expanded_context --dynamics-self-forcing-rollout-chunks 1 --dynamics-rollout-self-forcing-warmup-steps 50 --dynamics-rollout-self-forcing-ramp-steps 100 --dynamics-infer-steps 32 --dynamics-rf-shift 3.0 --dynamics-validation-metric open_rollout_frame_mse --device cuda`
  - Result: completed successfully and logged `active_self_forcing_loss_weight=0.5` with `active_rollout_self_forcing_loss_weight=0.0` at step `1`, which confirms the rollout auxiliary now stays off during its own warmup window.

## Latest Result 11

Delayed-rollout hybrid continuation from the expanded-context best:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_plusrollout025_rwarm50_rramp100_rfshift3_infer32_lr2e5_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_plusrollout025_rwarm50_rramp100_rfshift3_infer32_lr2e5_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_plusrollout025_rwarm50_rramp100_rfshift3_infer32_lr2e5_f48_67/manual_best_open_rollout_check_turn5`

learned:
- Adding a delayed rollout auxiliary still did not beat the original expanded-context best.
- Best checkpoint was step `250`, and the direct open-rollout check matched validation:
  - `input_frame_count=20`
  - `predicted_frame_count=20`
  - `decoded_frame_count=20`
  - `loss_frames=19`
  - `seed_frames=1`
  - `open_rollout_frame_mse=2.54498440772295e-2`
- Validation trajectory:
  - step `50`: `2.8839e-2`
  - step `150`: `2.7650e-2`
  - step `250`: `2.5450e-2`
  - step `300`: `3.6581e-2`
- Best teacher-forced metrics were respectable but still not enough to close rollout:
  - `next_frame_mse=6.4695e-4`
  - target-0 MSE: `7.3469e-4`
  - target-1 MSE: `5.4947e-4`
- Comparison against the relevant baselines:
  - original expanded-context best: `2.2398e-2`
  - rollout-only `0.5` best: `2.4004e-2`
  - immediate hybrid `0.5 + 0.25`: `2.4634e-2`
  - delayed hybrid `0.5 + 0.25`, rollout warmup/ramp: `2.5450e-2`

Interpretation:
- The delayed rollout schedule removed the obvious immediate-collapse failure mode, but it still did not create a better policy than the simpler expanded-context objective.
- That makes the current conclusion pretty clear:
  - the main bottleneck is probably not *when* rollout pressure turns on,
  - it is more likely in the model's temporal signal itself.
- In the current local RF DiT:
  - there is no learned absolute temporal embedding,
  - there is no causal attention mask inside the chunk,
  - frame identity is mostly carried by 3D RoPE plus the binary condition mask and per-frame action conditioning.
- For `1 ctx -> 2 inferred`, that may be too weak a separator for target-0 versus target-1.

Next search direction:
- Stop spending more runs on rollout auxiliary variants for now.
- Keep the best proven training recipe fixed:
  - `1 ctx -> 2 inferred`
  - primary expanded-context self-forcing `0.5`
  - `rf_shift=3.0`
  - `infer_steps=32`
- Probe temporal-signal strength directly with the existing RoPE temporal scaling knob:
  - try `dynamics_rope_t_extrapolation_ratio=0.5`
- Rationale:
  - with only three temporal positions per chunk, the default `rope_t_extrapolation_ratio=1.0` may not separate the two inferred positions strongly enough.
  - lowering the ratio increases temporal phase separation in the current implementation, which is the cleanest no-architecture-change test of the positional-signal hypothesis.

Small check run inside Codex:
- `source .venv/bin/activate && python -m world_model_v2.run --mode dynamics_only --dataset-format lerobot_metaworld --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --batch-size 1 --lr 2e-5 --max-steps 1 --validation-interval 1 --checkpoint-interval 0 --log-interval 1 --early-stop-window-size 0 --output-dir outputs/smoke --run-name rope_t05_smoke_1step --load-encoder-decoder outputs/minimal/metaworld_task0_wan_ae_240/checkpoints/best.pt --load-dynamics outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --dynamics-context-frames 1 --dynamics-target-frames 2 --dynamics-conditioning-frame-choices 1 --dynamics-conditioning-frame-probabilities 1.0 --dynamics-validation-conditioning-frame-choices 1 --dynamics-open-rollout-context-frames 1 --dynamics-self-forcing-loss-weight 0.5 --dynamics-self-forcing-mode expanded_context --dynamics-self-forcing-warmup-steps 0 --dynamics-self-forcing-ramp-steps 0 --dynamics-infer-steps 32 --dynamics-rf-shift 3.0 --dynamics-rope-t-extrapolation-ratio 0.5 --dynamics-validation-metric open_rollout_frame_mse --device cuda`
  - Result: completed successfully, so the temporal-RoPE sweep command is ready for a full continuation run.

## Latest Result 12

Continuation from the expanded-context best with stronger temporal RoPE phase separation:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ropet05_lr2e5_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ropet05_lr2e5_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ropet05_lr2e5_f48_67/manual_best_open_rollout_check_turn6`

learned:
- Lowering `dynamics_rope_t_extrapolation_ratio` from `1.0` to `0.5` was a strong negative result.
- Best checkpoint was step `200`, and the direct open-rollout check matched validation:
  - `input_frame_count=20`
  - `predicted_frame_count=20`
  - `decoded_frame_count=20`
  - `loss_frames=19`
  - `seed_frames=1`
  - `open_rollout_frame_mse=3.5003338009119034e-2`
- Validation trajectory improved during training but never approached the older baseline:
  - step `50`: `4.7331e-2`
  - step `100`: `4.4447e-2`
  - step `150`: `3.9268e-2`
  - step `200`: `3.5003e-2`
  - step `300`: `3.5820e-2`
- Best teacher-forced numbers were decent, but rollout stayed much worse:
  - `next_frame_mse=6.3134e-4`
  - target-0 MSE: `7.0018e-4`
  - target-1 MSE: `5.5486e-4`
  - open-rollout motion ratio: `4.13`
- Comparison against the relevant baselines:
  - original expanded-context best: `2.2398e-2`
  - rollout-only `0.5` best: `2.4004e-2`
  - delayed hybrid best: `2.5450e-2`
  - `rope_t=0.5` continuation best: `3.5003e-2`

Interpretation:
- The temporal-signal hypothesis was not wrong in spirit, but the specific directional guess was wrong:
  - making temporal RoPE *more* aggressive did not help target separation,
  - it destabilized rollout badly.
- So the practical conclusion is:
  - default `rope_t=1.0` is already better than `0.5` for this checkpoint family,
  - if RoPE scaling matters here, the next sensible direction is the opposite one: a *larger* temporal ratio, not a smaller one.

Next search direction:
- Keep the best known recipe fixed:
  - `1 ctx -> 2 inferred`
  - expanded-context self-forcing `0.5`
  - `rf_shift=3.0`
  - `infer_steps=32`
- Probe the opposite temporal-RoPE direction with:
  - `dynamics_rope_t_extrapolation_ratio=2.0`
- Rationale:
  - `0.5` increased temporal phase separation and clearly hurt,
  - `2.0` is the clean opposite test using the same checkpoint and architecture,
  - it is still a no-code, checkpoint-compatible positional-signal experiment.

Small check run inside Codex:
- `source .venv/bin/activate && python -m world_model_v2.run --mode dynamics_only --dataset-format lerobot_metaworld --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --batch-size 1 --lr 2e-5 --max-steps 1 --validation-interval 1 --checkpoint-interval 0 --log-interval 1 --early-stop-window-size 0 --output-dir outputs/smoke --run-name rope_t20_smoke_1step --load-encoder-decoder outputs/minimal/metaworld_task0_wan_ae_240/checkpoints/best.pt --load-dynamics outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --dynamics-context-frames 1 --dynamics-target-frames 2 --dynamics-conditioning-frame-choices 1 --dynamics-conditioning-frame-probabilities 1.0 --dynamics-validation-conditioning-frame-choices 1 --dynamics-open-rollout-context-frames 1 --dynamics-self-forcing-loss-weight 0.5 --dynamics-self-forcing-mode expanded_context --dynamics-self-forcing-warmup-steps 0 --dynamics-self-forcing-ramp-steps 0 --dynamics-infer-steps 32 --dynamics-rf-shift 3.0 --dynamics-rope-t-extrapolation-ratio 2.0 --dynamics-validation-metric open_rollout_frame_mse --device cuda`
  - Result: completed successfully, so the opposite-direction temporal-RoPE run is ready.

## Latest Result 13

Continuation from the expanded-context best with weaker temporal RoPE phase separation:
`outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ropet20_lr2e5_f48_67/checkpoints/best.pt`

Validated with:
`source .venv/bin/activate && python scripts/check/open_rollout_demo.py --checkpoint outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ropet20_lr2e5_f48_67/checkpoints/best.pt --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --device cuda --output-dir outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ropet20_lr2e5_f48_67/manual_best_open_rollout_check_turn7`

learned:
- Increasing `dynamics_rope_t_extrapolation_ratio` from `1.0` to `2.0` was also a negative result.
- Best checkpoint was step `200`, and the direct open-rollout check matched validation:
  - `input_frame_count=20`
  - `predicted_frame_count=20`
  - `decoded_frame_count=20`
  - `loss_frames=19`
  - `seed_frames=1`
  - `open_rollout_frame_mse=3.330322727560997e-2`
- Validation trajectory:
  - step `50`: `4.4888e-2`
  - step `100`: `4.2885e-2`
  - step `150`: `3.9218e-2`
  - step `200`: `3.3303e-2`
  - step `300`: `3.6241e-2`
- Best teacher-forced metrics were still fine but rollout stayed far from the baseline:
  - `next_frame_mse=6.3393e-4`
  - target-0 MSE: `7.0383e-4`
  - target-1 MSE: `5.5626e-4`
  - open-rollout motion ratio: `4.06`

Interpretation:
- Both temporal-RoPE ratio directions were decisively worse than the default:
  - `rope_t=0.5`: `3.5003e-2`
  - `rope_t=2.0`: `3.3303e-2`
  - default `rope_t=1.0` parent best: `2.2398e-2`
- So the remaining temporal-signal issue is not fixable by simply stretching or compressing the existing RoPE frequencies.
- The next clean move is architectural:
  - keep RoPE as-is,
  - add an explicit learned temporal identity signal on top,
  - preserve checkpoint compatibility so the current best run can still be reused.

## Latest Code Change 9

2026-04-09:
Added an optional learned temporal embedding on top of the existing RoPE path, with backward-compatible loading from older dynamics checkpoints.

Why:
- The strongest remaining hypothesis is still temporal identity weakness inside the `1 ctx -> 2 inferred` chunk.
- The parameter-only RoPE sweeps gave a clear answer:
  - `rope_t=0.5` was much worse,
  - `rope_t=2.0` was also much worse.
- That means the problem is not just the frequency scale of the current RoPE basis.
- The next minimal architectural test is to keep the old RoPE path untouched and add a small learned per-frame temporal bias that can specialize target-0 versus target-1.

What changed:
- Added `use_learned_temporal_embedding` to `DynamicsTransformerConfig`.
- In `ActionConditionedDynamicsTransformer`, added an optional learned `temporal_pos_embed` tensor with shape:
  - `(1, max_frames // patch_temporal, 1, 1, model_channels)`
- The learned temporal embedding is added after patch embedding and before RoPE attention.
- Added `--dynamics-use-learned-temporal-embedding` to:
  - `world_model_v2.run`
  - `scripts/check/loop_dynamics_sweep.py`
- Threaded the flag through:
  - `ExperimentConfig`
  - `WorldModel`
  - `scripts/check/open_rollout_demo.py`
- Updated dynamics warm-start loading so older checkpoints can still load when this new parameter is enabled:
  - strict loading remains in effect for all existing weights,
  - only `net.temporal_pos_embed` is allowed to be missing from older checkpoints when the new flag is on.

Design decision:
- The new temporal embedding is zero-initialized.
- Reason:
  - the first full run should start exactly from the old best policy,
  - then learn only the extra temporal identity capacity during continuation,
  - without corrupting the working checkpoint at load time.

Small checks run inside Codex:
- `source .venv/bin/activate && pytest tests/test_dynamics_transformer.py tests/test_model.py tests/test_experiment.py tests/test_run.py tests/test_loop_dynamics_sweep.py -q`
  - Result: `115 passed`.
- Learned-temporal-embedding CUDA smoke:
  - `source .venv/bin/activate && python -m world_model_v2.run --mode dynamics_only --dataset-format lerobot_metaworld --data-root data/full --split train --episode 0 --metaworld-task-index 0 --frame-start 48 --frame-end 67 --resolution 240 --batch-size 1 --lr 2e-5 --max-steps 1 --validation-interval 1 --checkpoint-interval 0 --log-interval 1 --early-stop-window-size 0 --output-dir outputs/smoke --run-name learned_temporal_embed_smoke_1step --load-encoder-decoder outputs/minimal/metaworld_task0_wan_ae_240/checkpoints/best.pt --load-dynamics outputs/controller_dreamdojo_progressive/controller_dreamdojo_progressive_1to2_selfforce05_rfshift3_infer32_ft_f48_67/checkpoints/best.pt --dynamics-context-frames 1 --dynamics-target-frames 2 --dynamics-conditioning-frame-choices 1 --dynamics-conditioning-frame-probabilities 1.0 --dynamics-validation-conditioning-frame-choices 1 --dynamics-open-rollout-context-frames 1 --dynamics-self-forcing-loss-weight 0.5 --dynamics-self-forcing-mode expanded_context --dynamics-use-learned-temporal-embedding --dynamics-self-forcing-warmup-steps 0 --dynamics-self-forcing-ramp-steps 0 --dynamics-infer-steps 32 --dynamics-rf-shift 3.0 --dynamics-validation-metric open_rollout_frame_mse --device cuda`
  - Result: completed successfully, which confirms the old best checkpoint can warm start into the new architecture without load errors.
