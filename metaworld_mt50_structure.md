# MetaWorld MT50 dataset analysis

## Goal
This note is meant to make the MetaWorld MT50 dataset concrete enough that we can later train an action-conditioned vision model in the style of [`interactive_world_sim`](https://github.com/WangYixuan12/interactive_world_sim) without guessing what the action and low-dimensional state channels mean.

## Dataset contract
- Dataset: [`lerobot/metaworld_mt50`](https://huggingface.co/datasets/lerobot/metaworld_mt50)
- Total episodes: 2500
- Total frames: 204806
- Total tasks: 49
- FPS: 80
- Observations per frame: one `observation.image` RGB view, `observation.state` with shape `[4]`, `observation.environment_state` with shape `[39]`, reward, success flag, task id/index, timestamps, and frame/episode indices.
- Important consistency check: `tasks.parquet` contains 49 task rows indexed `0..48`, so the dataset is internally consistent on task count.

## Repository layout
- Published parquet data shards: 492
- Published episode-metadata parquet files: 1
- Published MP4 video files: 0
- Dataset split declaration: `{'train': '0:2500'}`
- Data path pattern: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
- Video path pattern: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
- Example data shards: `data/chunk-000/file-000.parquet`, `data/chunk-000/file-001.parquet`, `data/chunk-000/file-002.parquet`
- Example video files: none published; the image bytes live directly in the parquet rows

## Row and metadata structure
- `data/chunk-*/file-*.parquet` stores frame rows. Each row contains `observation.image` as a struct with embedded encoded image bytes plus a source path, along with `observation.state`, `observation.environment_state`, `action`, reward, success, timestamps, and episode/frame indices.
- `meta/tasks.parquet` maps each `task_index` to its natural-language task name.
- `meta/episodes/chunk-000/file-000.parquet` maps every episode to the data shard that stores it and records exact row spans via `dataset_from_index` and `dataset_to_index`.
- The per-episode metadata also includes exact aggregated statistics for action, state, reward, success, image channels, timestamps, and indices, so the dataset can be audited without scanning all frame rows.

## High-level findings
- Average episode length is 81.92 frames, which is about 1.02 seconds at 80 FPS.
- Every task has 50 episodes, but the horizon varies a lot by task, from very short button/handle tasks to long manipulation tasks with some 500-frame rollouts.
- `observation.state[:3]` behaves like a compact end-effector position state, and `observation.state[3]` behaves like a gripper openness state.
- `action[:3]` behaves like Cartesian end-effector control, while `action[3]` behaves like a task-dependent gripper command. This is not documented explicitly in the dataset card, but it is strongly supported by the per-dimension ranges and row-level samples.
- `observation.environment_state` appears to contain privileged simulator state. In sampled rows, its first four values exactly match `observation.state`: True.

## Global statistics
- `observation.state` min/max/mean/std: [-0.506, 0.387, 0.043, 0.256] / [0.487, 0.904, 0.508, 1.000] / [0.014, 0.640, 0.162, 0.614] / [0.112, 0.103, 0.087, 0.300]
- `action` min/max/mean/std: [-11.475, -14.927, -17.279, -1.000] / [19.456, 12.469, 24.625, 1.000] / [0.255, 0.510, -0.127, 0.376] / [2.026, 2.007, 3.388, 0.710]
- `observation.environment_state` has shape `[39]`; its global min/max span is much wider because it packs simulator/object information beyond the low-dimensional policy state.

## `observation.state`
- Shape `[4]`. The first three channels stay within a compact workspace-sized range and move smoothly from frame to frame. The fourth channel stays within roughly `[0.26, 1.0]`.
- On sampled trajectories, the first three action channels are strongly correlated with `observation.state` deltas from one frame to the next in task 0. Estimated action to state-delta correlation rows for task 0 sample episode: `[[ 0.757 -0.239 -0.19   0.814]
 [ 0.822  0.383  0.039  0.545]
 [-0.209  0.417  0.873 -0.338]
 [-0.892 -0.136  0.11  -0.895]]`.
- Practical interpretation: `observation.state[0:3] ~= end-effector position`, `observation.state[3] ~= gripper openness`.
- Because `observation.state` also appears at the front of `observation.environment_state`, it is likely the policy-facing subset of the simulator state.

## `observation.environment_state`
- Shape `[39]`. This is privileged simulator state, not just robot proprioception.
- The first four values match `observation.state` in sampled rows.
- The remaining 35 values likely encode object/keypoint state and possibly repeated or lagged object state. That makes it useful for interpretation and debugging, but risky to use as a core input if the goal is an image-conditioned world model.
- Recommendation for later training: keep it for analysis, diagnostics, and optional auxiliary prediction targets; do not rely on it as the primary model input if you want to stay close to the `interactive_world_sim` setup.

## `action`
- Shape `[4]`.
- Channels 0-2 have large task-dependent continuous ranges and behave like Cartesian end-effector control.
- Channel 3 is not globally uniform. It is task-specific and often effectively fixed. Examples from exact per-task metadata:
  - Always `-1.0` in tasks like `Bypass a wall and press a button from the top`, `Push a button on the coffee machine`, `Open a drawer`, and `Press a handle down`.
  - Always `1.0` in tasks like `Press a button from the top`, `Rotate a dial 180 degrees`, `Open/close window`, and many handle/faucet tasks.
  - Variable in grasping/manipulation tasks like `Pick up a nut and place it onto a peg`, `Dunk the basketball into the basket`, `Insert a peg sideways`, and the puck tasks.
- This means the fourth action channel is better thought of as a gripper mode/control channel than as a generic continuous action dimension with one shared interpretation across all 49 tasks.
- Practical implication: if you later train across multiple tasks, per-dimension min-max normalization to `[-1, 1]` is fine, but the model still has to learn that `action[3]` means very different things across task families.

## Representative task samples
### Task 0: Pick up a nut and place it onto a peg
- Episodes: 50, frames: 4469, mean episode length: 89.38
- `action` min/max/mean: [-0.892, -0.037, -2.043, 0.000] / [1.054, 2.556, 1.761, 0.600] / [0.151, 0.369, -0.136, 0.499]
- `observation.state` min/max/mean: [0.005, 0.594, 0.064, 0.444] / [0.217, 0.845, 0.246, 1.000] / [0.106, 0.661, 0.174, 0.621]
- Sample frame `0`: state=[0.0054, 0.6006, 0.1942, 1.0000], action=[1.0542, -0.0139, -0.7520, 0.0000]
- Sample frame `1`: state=[0.0076, 0.5993, 0.1922, 1.0000], action=[1.0460, -0.0057, -0.7522, 0.0000]

### Task 3: Grasp the cover and close the box with it
- Episodes: 50, frames: 8511, mean episode length: 170.22
- `action` min/max/mean: [-4.868, -2.511, -4.361, 0.500] / [2.707, 7.036, 2.195, 1.000] / [-0.054, 0.512, -0.966, 0.783]
- `observation.state` min/max/mean: [-0.453, 0.454, 0.081, 0.280] / [0.434, 0.883, 0.219, 1.000] / [-0.041, 0.598, 0.158, 0.350]
- Sample frame `0`: state=[0.0045, 0.5999, 0.1948, 0.9999], action=[-0.2072, -1.8285, 0.1266, 0.5000]
- Sample frame `1`: state=[0.0041, 0.5974, 0.1948, 0.9910], action=[-0.2039, -1.8010, 0.1304, 0.5000]

### Task 24: Press a handle down
- Episodes: 50, frames: 1485, mean episode length: 29.70
- `action` min/max/mean: [-2.605, -0.899, -15.575, -1.000] / [2.267, 1.498, 4.422, -1.000] / [0.003, 0.140, -9.366, -1.000]
- `observation.state` min/max/mean: [-0.117, 0.549, 0.061, 1.000] / [0.104, 0.669, 0.293, 1.000] / [0.003, 0.612, 0.184, 1.000]
- Sample frame `0`: state=[0.0058, 0.6018, 0.1960, 1.0000], action=[1.1230, 0.8434, 4.3897, -1.0000]
- Sample frame `1`: state=[0.0086, 0.6036, 0.1989, 1.0000], action=[1.0939, 0.8329, 4.3321, -1.0000]

### Task 48: Push and close a window
- Episodes: 50, frames: 4044, mean episode length: 80.88
- `action` min/max/mean: [-2.536, -0.393, -6.629, 1.000] / [5.865, 9.302, 4.356, 1.000] / [0.499, 1.846, -0.622, 1.000]
- `observation.state` min/max/mean: [0.005, 0.401, 0.102, 0.264] / [0.258, 0.783, 0.385, 0.996] / [0.177, 0.637, 0.243, 0.365]
- Sample frame `0`: state=[0.0054, 0.4013, 0.1965, 0.9964], action=[0.8874, 6.8991, 4.3565, 1.0000]
- Sample frame `1`: state=[0.0077, 0.4042, 0.1992, 0.9785], action=[5.8646, 9.2345, 4.3370, 1.0000]

## Task coverage and difficulty proxies
### Longest average tasks
- Task 3: Grasp the cover and close the box with it -> mean length 170.22, max length 500
- Task 35: Insert a peg sideways -> mean length 166.64, max length 500
- Task 12: Pick a nut out of a peg -> mean length 134.72, max length 228
- Task 36: Unplug a peg sideways -> mean length 126.76, max length 500
- Task 40: Push the puck to a goal -> mean length 124.09, max length 500

### Shortest average tasks
- Task 24: Press a handle down -> mean length 29.70, max length 46
- Task 8: Push a button on the coffee machine -> mean length 36.24, max length 43
- Task 23: Press a handle down sideways -> mean length 41.18, max length 51
- Task 16: Unlock the door by rotating the lock counter-clockwise -> mean length 42.32, max length 61
- Task 34: Get a plate from the cabinet sideways -> mean length 45.78, max length 55

### Widest xyz action spans
- Task 25: Pull a handle up sideways -> xyz span [4.987, 3.985, 27.098]
- Task 35: Insert a peg sideways -> xyz span [17.107, 10.233, 11.222]
- Task 23: Press a handle down sideways -> xyz span [3.918, 4.014, 21.700]
- Task 13: Close a door with a revolving joint -> xyz span [19.277, 7.465, 4.789]
- Task 24: Press a handle down -> xyz span [4.872, 2.397, 19.998]

## Full task inventory
| Task | Episodes | Mean length | Gripper channel | Name |
| --- | ---: | ---: | --- | --- |
| 0 | 50 | 89.38 | variable | Pick up a nut and place it onto a peg |
| 1 | 50 | 109.90 | variable | Dunk the basketball into the basket |
| 2 | 50 | 120.24 | variable | Grasp the puck from one bin and place it into another bin |
| 3 | 50 | 170.22 | variable | Grasp the cover and close the box with it |
| 4 | 50 | 65.90 | fixed 1.0 | Press a button from the top |
| 5 | 50 | 58.16 | fixed -1.0 | Bypass a wall and press a button from the top |
| 6 | 50 | 62.20 | fixed 0.000 | Press a button |
| 7 | 50 | 73.80 | variable | Bypass a wall and press a button |
| 8 | 50 | 36.24 | fixed -1.0 | Push a button on the coffee machine |
| 9 | 50 | 79.30 | variable | Pull a mug from a coffee machine |
| 10 | 50 | 54.94 | variable | Push a mug under a coffee machine |
| 11 | 50 | 71.84 | fixed 1.0 | Rotate a dial 180 degrees |
| 12 | 50 | 134.72 | variable | Pick a nut out of a peg |
| 13 | 50 | 62.72 | fixed 1.0 | Close a door with a revolving joint |
| 14 | 50 | 82.78 | fixed -1.0 | Lock the door by rotating the lock clockwise |
| 15 | 50 | 99.76 | fixed 1.0 | Open a door with a revolving joint |
| 16 | 50 | 42.32 | fixed 1.0 | Unlock the door by rotating the lock counter-clockwise |
| 17 | 50 | 60.60 | variable | Insert the gripper into a hole |
| 18 | 50 | 78.46 | fixed 1.0 | Push and close a drawer |
| 19 | 50 | 88.86 | fixed -1.0 | Open a drawer |
| 20 | 50 | 59.12 | fixed 1.0 | Rotate the faucet counter-clockwise |
| 21 | 50 | 65.28 | fixed 1.0 | Rotate the faucet clockwise |
| 22 | 50 | 65.30 | variable | Hammer a screw on the wall |
| 23 | 50 | 41.18 | fixed 1.0 | Press a handle down sideways |
| 24 | 50 | 29.70 | fixed -1.0 | Press a handle down |
| 25 | 50 | 79.98 | variable | Pull a handle up sideways |
| 26 | 50 | 122.68 | fixed 1.0 | Pull a handle up |
| 27 | 50 | 78.02 | fixed 1.0 | Pull a lever down 90 degrees |
| 28 | 50 | 82.64 | variable | Pick a puck, bypass a wall and place the puck |
| 29 | 50 | 117.38 | variable | Pick up a puck from a hole |
| 30 | 50 | 53.12 | variable | Pick and place a puck to a goal |
| 31 | 50 | 51.10 | fixed -1.0 | Slide a plate into a cabinet |
| 32 | 50 | 47.68 | fixed 1.0 | Slide a plate into a cabinet sideways |
| 33 | 50 | 65.20 | fixed -1.0 | Get a plate from the cabinet |
| 34 | 50 | 45.78 | fixed 1.0 | Get a plate from the cabinet sideways |
| 35 | 50 | 166.64 | variable | Insert a peg sideways |
| 36 | 50 | 126.76 | variable | Unplug a peg sideways |
| 37 | 50 | 106.16 | fixed 1.0 | Kick a soccer into the goal |
| 38 | 50 | 75.32 | variable | Grasp a stick and push a box using the stick |
| 39 | 50 | 109.38 | variable | Grasp a stick and pull a box with the stick |
| 40 | 100 | 124.09 | variable | Push the puck to a goal |
| 41 | 50 | 80.94 | variable | Bypass a wall and push a puck to a goal |
| 42 | 50 | 47.02 | fixed 0.000 | Reach a goal position |
| 43 | 50 | 46.32 | fixed 0.000 | Bypass a wall and reach a goal |
| 44 | 50 | 98.50 | variable | Pick and place a puck onto a shelf |
| 45 | 50 | 88.80 | variable | Sweep a puck into a hole |
| 46 | 50 | 87.56 | variable | Sweep a puck off the table |
| 47 | 50 | 87.16 | fixed 1.0 | Push and open a window |
| 48 | 50 | 80.88 | fixed 1.0 | Push and close a window |

## What this means for later training
- If the goal is to stay close to `https://github.com/WangYixuan12/interactive_world_sim`, the cleanest later setup is `image + action` as the core model input, with `observation.state` used as a debugging target or auxiliary predictor.
- Normalize each action dimension independently to `[-1, 1]`, but remember that `action[3]` is effectively task-coded gripper behavior and is not globally continuous in the same way as the xyz channels.
- For single-task-first training, choose a task family with variable gripper behavior if you want the model to actually learn grasp/open-close structure. Tasks with fixed `action[3]` are simpler but teach less about contact-rich control.
- Because the average episode is short and the dataset runs at 80 FPS, later training should consider temporal downsampling or larger action chunks to avoid wasting model capacity on nearly redundant adjacent frames.

## Training hook in this repo
- The minimal Wan VAE path in this repo can now read MT50 directly through `world_model_v2/minimal/run.py` with `--dataset-format lerobot_metaworld`.
```bash
source .venv/bin/activate
python -m world_model_v2.minimal.run \
  --mode ae_only \
  --dataset-format lerobot_metaworld \
  --metaworld-task-index 0 \
  --split train \
  --train-all-episodes \
  --validation-split val \
  --validation-episode 0 \
  --resolution 128 \
  --batch-size 32 \
  --max-steps 3000 \
  --run-name metaworld_mt50_task0_wan_ae
```
- Important local behavior: MT50 only publishes a train split, so this repo treats both `train` and `val` requests as aliases for the same upstream split and uses `validation_episode` only to pick which train episode to preview.

## Recommended first tasks if the aim is understanding
- Good variable-gripper tasks: task 0 `Pick up a nut and place it onto a peg`, task 1 `Dunk the basketball into the basket`, task 2 `Grasp the puck from one bin and place it into another bin`, task 35 `Insert a peg sideways`.
- Good fixed-gripper tasks: task 24 `Press a handle down`, task 47 `Push and open a window`, task 48 `Push and close a window`.

## Sources
- https://huggingface.co/datasets/lerobot/metaworld_mt50
- https://huggingface.co/datasets/lerobot/metaworld_mt50/raw/main/README.md
- https://huggingface.co/datasets/lerobot/metaworld_mt50/resolve/main/meta/info.json
- https://huggingface.co/datasets/lerobot/metaworld_mt50/resolve/main/meta/stats.json
- https://huggingface.co/datasets/lerobot/metaworld_mt50/resolve/main/meta/tasks.parquet?download=true
- https://huggingface.co/datasets/lerobot/metaworld_mt50/resolve/main/meta/episodes/chunk-000/file-000.parquet?download=true
- https://github.com/WangYixuan12/interactive_world_sim

