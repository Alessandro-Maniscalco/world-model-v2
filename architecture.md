# Architecture

## Summary

This repo now exposes a single root package path: `world_model_v2`.

The supported training flow is:

- Wan VAE autoencoder training with `--mode ae_only`
- RF-DiT latent dynamics training with `--mode dynamics_only`
- Interactive World Sim and LeRobot MetaWorld MT50 dataset support
- Root CLI entrypoint: `python -m world_model_v2.run`

## Main Modules

- `world_model_v2/run.py`
  - CLI argument parsing and experiment startup.
- `world_model_v2/experiment.py`
  - `Experiment`, `ExperimentConfig`, checkpoint helpers, validation, and training loop.
- `world_model_v2/model.py`
  - `WorldModel` bundling the Wan encoder/decoder with RF-DiT dynamics.
- `world_model_v2/dynamics_transformer.py`
  - `DYNAMICS_FRAME_LAYOUT`, `DynamicsTransformerConfig`, and `RectifiedFlowDynamics`.
- `world_model_v2/dataset.py`
  - Interactive World Sim clip loading plus frame, transition, and validation datasets.
- `world_model_v2/metaworld_dataset.py`
  - MetaWorld repository access and dataset adapters.
- `world_model_v2/wan_vae.py`
  - Wan-style VAE building blocks.
- `world_model_v2/utils/checkpointing.py`
  - JSON and generic checkpoint I/O helpers.
- `world_model_v2/utils/visualization.py`
  - Grid and MP4 export helpers.

## Key Design Choices

- The repo no longer keeps a separate `minimal/` package. The former minimal path is now the package root.
- The canonical dynamics chunk length is 5 latent frames, defined once in `DYNAMICS_FRAME_LAYOUT`.
- Dataset windowing still slices fixed 5-frame chunks plus 4 aligned transition actions.
- Mixed DreamDojo-style teacher conditioning is active for dynamics training:
  - The same 5-frame chunk is trained with either 3 conditioned frames or 4 conditioned frames.
  - This means the model sees both `3 -> 2` and `4 -> 1` supervision during training.
  - The supported conditioning counts come from `DYNAMICS_FRAME_LAYOUT.conditioning_frame_choices`, currently `[3, 4]`.
- Only the Wan autoencoder backend and RF-DiT dynamics backend are supported.
- New checkpoints are written with the root package identity while still loading older `world_model_v2_minimal_v1/v2` checkpoints.

## Dynamics Path

- `world_model_v2/dynamics_transformer.py`
  - `DynamicsFrameLayout`
    - Single source of truth for:
      - `context_frames = 4`
      - `target_frames = 1`
      - `max_frames = 5`
      - `num_action_per_chunk = 4`
      - `conditioning_frame_choices = (3, 4)`
  - `RectifiedFlowDynamics`
    - Prepares full-clip rectified-flow training inputs.
    - Builds a condition mask over the first 3 or 4 frames, depending on the sampled teacher-conditioning count.
    - Concatenates that mask as one extra channel into the DiT input.
    - Runs DreamDojo-style teacher conditioning:
      - full-clip RF interpolation
      - clean context-frame repinning before every network forward
      - full-video velocity prediction
      - overwrite of conditioned-frame velocity with the exact RF target so loss can stay plain MSE over the full clip

## Important Dynamics Knobs

- `dynamics_action_conditioning_mode`
  - `chunk_per_frame`
    - Each transition action gets its own time-aligned embedding.
    - The model sees a zero action embedding on the first frame and one action embedding for each later frame in the 5-frame chunk.
    - This keeps action timing explicit and was the stronger mode in the controller search.
  - `global_chunk`
    - All 4 actions in the chunk are flattened into one shared embedding.
    - That shared action summary is broadcast across the whole clip.
    - This is closer to a coarse whole-chunk conditioning signal and underperformed `chunk_per_frame` in the controller run.

- `conditional_frame_timestep`
  - This does not decide whether conditioned frames are clean or noisy.
  - Clean conditioned inputs come from repinning: conditioned frames in `x_t` are replaced with the clean latent values before the DiT forward.
  - `conditional_frame_timestep` only decides what timestep label the conditioned frames receive.
  - `-1.0`
    - Sentinel meaning "do not override timesteps."
    - The conditioned frames keep the same sampled RF timestep labels as the rest of the clip.
  - `0.0`, `0.5`, or any non-negative value
    - Override the conditioned frames to use that fixed timestep value while leaving unconditioned target frames on the sampled RF timestep.
  - In the current search, `-1.0` outperformed nearby fixed overrides such as `0.0` and `-0.5`.

- `dynamics_video_condition_dropout`
  - Controls whether the model sometimes drops the clean video-conditioning path during training.
  - With `0.0`, the conditioned frames are always available for repinning.
  - With values above `0.0`, some training samples zero out the pinned inputs to make the model less dependent on conditioning.

- `dynamics_infer_steps`
  - Number of RF solver steps used during rollout and validation.
  - This is an inference-time sampler knob, not the training length.
  - More steps can improve fidelity but cost more runtime; fewer steps are faster but can degrade rollout quality.

- `lr`
  - Optimizer learning rate for the current training or fine-tuning run.
  - The controller used a pattern of:
    - larger LR for coarse exploration
    - smaller LR for checkpoint polishing
  - Very small LR polish runs were important near the best-performing checkpoints.

- `dynamics_rf_shift`
  - Shape parameter for the Diffusers `FlowMatchEulerDiscreteScheduler`.
  - Both training-time sigma lookup and inference-time sigma schedules use this shift.
  - The scheduler transforms sigma roughly as:
    - `sigma' = shift * sigma / (1 + (shift - 1) * sigma)`
  - Higher values skew the schedule toward higher-noise sigma values for more of the trajectory.
  - Lower values make the schedule less aggressively high-noise.
  - This is not a model-capacity knob and not a step-count knob; it changes the rectified-flow time/noise schedule itself.

- `seed`
  - Seeds Python, NumPy, PyTorch, and CUDA RNGs.
  - From scratch, this changes initialization and training randomness.
  - From a checkpoint, it still changes stochastic pieces such as sampled RF times, sampled noise, batch order, and any dropout behavior.
  - Seed changes are useful as robustness checks:
    - if a new seed reproduces the win, the configuration looks stable
    - if it does not, the result may be somewhat seed-sensitive

## What The Action Numbers Mean

- The current dynamics stack is configured for `action_dim = 4`.
- For the MetaWorld path, those 4 numbers are not per-joint motor commands.
- They should be read as an end-effector control:
  - `action[0] = dx`
  - `action[1] = dy`
  - `action[2] = dz`
  - `action[3] = gripper`
- This means the model is not asked to directly predict how each robot joint should move.
- Instead, the environment/controller turns that compact tool-space command into the full arm motion seen in the video.
- In practice, the visual arm trajectory is therefore a structured consequence of:
  - the current visible arm pose
  - the fixed robot geometry
  - the simulator/controller
  - the commanded end-effector delta plus gripper action
- One action row corresponds to one frame-to-frame transition, not to one standalone frame.
- For a 5-frame chunk:
  - frames: `f0 f1 f2 f3 f4`
  - actions: `a0 a1 a2 a3`
  - transition meaning:
    - `a0` causes `f0 -> f1`
    - `a1` causes `f1 -> f2`
    - `a2` causes `f2 -> f3`
    - `a3` causes `f3 -> f4`
- This is why `chunk_per_frame` prepends a zero action embedding on the first frame.
- There is no in-window action that caused `f0`; that missing action would have belonged to the previous chunk.
- So the effective per-frame action alignment is:
  - `f0` gets `0`
  - `f1` gets `a0`
  - `f2` gets `a1`
  - `f3` gets `a2`
  - `f4` gets `a3`
- This keeps the action timing explicit:
  - each later frame is conditioned by the action that produced it from the previous frame
  - the first frame stays purely as context because it has no preceding in-window transition action
- In `global_chunk` mode, that explicit timing is dropped:
  - the full `4 x 4 = 16` action numbers are flattened into one shared embedding
  - that single action summary is broadcast across the whole 5-frame clip
- During autoregressive rollout, the action windows slide forward with the predicted target frames.
- For a `4 -> 1` rollout:
  - step 0 uses actions `a0:a3`
  - step 1 uses actions `a1:a4`
  - step 2 uses actions `a2:a5`
- So each prediction step still receives the 4 transition actions aligned to the current 5-frame window.

## How Conditioning Actually Works

- The transition dataset yields 5-frame chunks with 4 aligned actions.
- Training first samples one RF training time per batch item and converts that to a sigma schedule for the whole chunk.
- The full latent clip is noised as:
  - `x_t = sigma * eps + (1 - sigma) * x0`
  - `v_target = eps - x0`
- The model then builds a condition mask over the first 3 or 4 frames.
- For those conditioned frames:
  - the clean latent values are repinned into the model input
  - the exact RF target velocity is restored in the loss target
- This means conditioned frames stay structurally "teacher-forced clean" in the input path while the target frames remain the real prediction problem.
- `conditional_frame_timestep` is layered on top of this and only changes the timestep embedding seen by the conditioned frames.

## How Training Works

- The transition dataset returns one fixed 5-frame chunk:
  - frames `t:t+4`
  - actions `t:t+3`
- In `dynamics_only` mode:
  - the Wan encoder maps the full 5-frame chunk into latents
  - `prepare_training_inputs(...)` samples one RF time per batch item
  - that one timestep is repeated across all 5 frames
  - the full latent clip is interpolated as:
    - `x_t = sigma * eps + (1 - sigma) * x0`
    - `v_target = eps - x0`
  - the condition mask marks either:
    - frames `0:3` for `3 -> 2`
    - frames `0:4` for `4 -> 1`
  - the model predicts velocity for the full 5-frame latent clip
  - training uses plain `F.mse_loss(predicted_velocity, target_velocity)` over the whole clip

## Prediction And Validation

- `WorldModel.predict_next_latent(...)` now accepts either:
  - 3 context latent frames, returning 2 predicted latent frames
  - 4 context latent frames, returning 1 predicted latent frame
- Validation now runs both teacher-forced checks on the same held-out clip:
  - primary metric: `3 -> 2`
  - auxiliary metric: `4 -> 1`
- Saved validation stats therefore include:
  - `next_frame_mse`
    - the primary `3 -> 2` metric
  - `next_frame_mse_4to1`
    - the auxiliary `4 -> 1` metric
  - `open_rollout_frame_mse`
    - fully autoregressive rollout error over the target portion of the clip
    - important for model selection because teacher-forced metrics can improve while open-loop behavior gets worse
  - `next_frame_mse_target_0`
  - `next_frame_mse_target_1`
  - `predicted_target_motion_l1`
  - `ground_truth_target_motion_l1`
  - `target_motion_ratio`

## Sweep And Selection Logic

- `scripts/check/loop_dynamics_sweep.py` is the controller-facing long-run sweep helper.
- It now supports:
  - action-conditioning mode selection
  - conditional-frame timestep forwarding
  - open-rollout-based checkpoint selection
  - fixed external open-rollout evaluation spans
  - aggregate multi-span selection metrics:
    - `eval_open_rollout_frame_mse`
      - single fixed external eval score
    - `eval_open_rollout_frame_mse_mean`
      - mean across multiple fixed eval spans
    - `eval_open_rollout_frame_mse_max`
      - worst-case score across multiple fixed eval spans
  - MetaWorld frame-span clamping/skipping by episode length
- This matters because "best on the tuned training clip" and "best across multiple rollout clips" are not always the same checkpoint.

## Controller Search Findings

- The controller searched mostly on MetaWorld episode 0 with the `48:67` clip as the main tuned span.
- Early key finding:
  - `global_chunk` action conditioning underperformed `chunk_per_frame`.
- Later key findings on the tuned `48:67` span:
  - `chunk_per_frame` was better than `global_chunk`.
  - `conditional_frame_timestep=-1.0` was better than nearby fixed overrides.
  - `dynamics_infer_steps=32` was better than nearby values like `24`, `28`, `30`, and later `40`.
  - Lowering `dynamics_rf_shift` from `5.0` to `3.0` helped.
  - Small-LR polish runs improved the best checkpoint further.
- Best tuned-span checkpoint found by that search:
  - `outputs/controller_loop_search_cuda/controller_chunkperframe_eval48_67_rfshift3_lr1e6_polish/checkpoints/best.pt`
  - tuned-span `open_rollout_frame_mse = 0.000112287`
- Important caveat:
  - that checkpoint was best on the tuned `48:67` span, not on every span.
- A later 4-span benchmark over:
  - `0:19`
  - `24:43`
  - `48:67`
  - `68:87`
  showed:
  - the tuned-span winner only won on `48:67`
  - the older pre-RF-shift checkpoint still had the best mean score across those four spans
- Best 4-span-mean checkpoint at that point:
  - `outputs/controller_loop_search_cuda/controller_chunkperframe_eval48_67_infer32_lr2e5_finetune/checkpoints/best.pt`
- Resulting design lesson:
  - single-span optimization is useful for rapid local progress
  - multi-span aggregate evaluation is necessary before calling a checkpoint the global best
  - the sweep helper therefore now supports mean and max multi-span open-rollout selection

## Current Goal

- Keep the root RF-DiT dynamics path DreamDojo-like while selecting checkpoints with open-rollout metrics that reflect true autoregressive behavior.
- Distinguish clearly between:
  - local best on the tuned clip
  - broader best across multiple validation spans
- Recommended warm-start pattern for new search legs:
  - Wan AE checkpoint for encoder/decoder
  - best checkpoint for the metric you actually care about
    - tuned-span work: local tuned-span winner
    - broader rollout quality: multi-span mean or max winner
