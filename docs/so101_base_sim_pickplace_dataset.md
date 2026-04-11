# SO101 Base Sim Pickplace Dataset

## Summary

This note describes the local LeRobot dataset `davidlinjiahao/lerobot_so101_base_sim_pickplace`, which this repo loads through `world_model_v2/lerobot_video_dataset.py`.

The analysis below is based on:

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- all `105` episode parquet files
- spot checks of exported episode MP4s under `outputs/so101_base_pickplace_dataset_examples/`

The short version is:

- this dataset contains a single repeated task
- the task is a right-to-left pick-and-place
- the action vector is `6D`
- the action vector is not a delta action
- instead, `action[t]` is exactly the next frame's first six state values
- the last six state values encode cube position plus a goal-relative offset

Important: the metadata does not publish human-readable action names, so the per-dimension semantic labels below are partly inferred from the measured state-action relationships.

## Dataset Facts

From `meta/info.json` and `meta/episodes.jsonl`:

- dataset id: `davidlinjiahao/lerobot_so101_base_sim_pickplace`
- episodes: `105`
- total frames: `28,747`
- FPS: `30`
- image stream: `observation.images.front`
- image shape: `480 x 640 x 3`
- action shape: `6`
- state shape: `12`
- task count: `1`

Episode length statistics:

- min: `144` frames, about `4.8s`
- mean: `273.78` frames, about `9.13s`
- median: `262` frames, about `8.73s`
- max: `584` frames, about `19.47s`

## It Is A Single-Task Dataset

`meta/tasks.jsonl` contains only one task index:

```json
{"task_index": 0, "task": 0}
```

So from the dataset metadata alone, every episode belongs to the same task.

Video spot checks agree with that:

- the cube starts on the right side of the table
- the open box is on the left side of the table
- the robot reaches to the right, grasps the cube, and places it into the left box

This is visible in the exported examples under `outputs/so101_base_pickplace_dataset_examples/`, especially `frame_contact_sheet.png`.

## Observation Layout

The dataset provides:

- `observation.images.front`: front RGB video
- `observation.state`: a `12D` float vector
- `action`: a `6D` float vector

The most useful structural fact from the parquet rows is:

```text
action[t] == observation.state[t + 1][:6]
```

exactly, across the full dataset. The measured mean absolute error is `0.0` on every action dimension.

That tells us:

- `observation.state[:6]` is the robot-control state for the six commanded channels
- `action` is the next commanded robot state for those same six channels
- the action is an absolute target vector, not a delta vector

## Interpreting `observation.state[6:12]`

The last six state values have a very clean algebraic structure:

```text
observation.state[6:9] + observation.state[9:12] = [0.4, -0.18, 0.42]
```

for every frame in the dataset, up to float noise.

That strongly suggests the following interpretation:

- `state[6:9]`: cube position `(x, y, z)`
- `state[9:12]`: goal position minus cube position `(goal_xyz - cube_xyz)`

This interpretation matches the videos:

- early in an episode, the cube is on the right and `state[9:12]` points leftward toward the box
- late in an episode, the cube is inside or near the box and `state[9:12]` is close to zero

So the fixed goal position appears to be:

```text
goal_xyz = [0.4, -0.18, 0.42]
```

## What The `6D` Action Likely Means

Because `action[t]` becomes the first six robot-state values at `t + 1`, the safest interpretation is:

- `action[0:6]` are six absolute robot-control targets
- channels `0` through `4` behave like continuous arm-joint targets
- channel `5` is likely the gripper or finger-opening command

Why channel `5` is likely the gripper:

- it is much more bimodal than the other channels
- `55.7%` of all values are `<= -0.15`
- only `6.6%` of values are in the middle band `(-0.15, 0.15]`
- `19.8%` of values are `> 0.5`

That looks much more like an open/close control channel than a continuously sweeping rotary joint.

I do not have an official joint-name mapping in the dataset metadata, so I will refer to the six channels as:

- `joint_target_0`
- `joint_target_1`
- `joint_target_2`
- `joint_target_3`
- `joint_target_4`
- `gripper_like_target_5`

The `gripper_like` name is an inference, not a documented field from the dataset.

## Action Value Analysis

Full-dataset action statistics:

| dim | interpretation | min | mean | std | p50 | p95 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `joint_target_0` | `-1.2174` | `-0.2835` | `0.5353` | `-0.3789` | `0.7047` | `1.0380` |
| 1 | `joint_target_1` | `-1.7453` | `-0.2536` | `0.8954` | `0.0166` | `0.8269` | `1.3957` |
| 2 | `joint_target_2` | `-1.6900` | `-0.0086` | `0.8843` | `-0.2897` | `1.6900` | `1.6900` |
| 3 | `joint_target_3` | `-1.6495` | `0.8223` | `1.0784` | `1.3343` | `1.6181` | `1.6381` |
| 4 | `joint_target_4` | `-1.7744` | `-0.3033` | `0.3749` | `-0.2608` | `0.2555` | `0.5547` |
| 5 | `gripper_like_target_5` | `-0.1745` | `0.0903` | `0.3370` | `-0.1729` | `0.6692` | `1.0067` |

### What These Ranges Suggest

- `joint_target_0`: a medium-range arm joint with both negative and positive travel, but biased negative.
- `joint_target_1`: a large-range arm joint spanning almost the full negative-to-positive range of the robot's motion.
- `joint_target_2`: another large-range joint that spends a noticeable amount of time at its upper limit. Its `p95` and `max` are both `1.69`.
- `joint_target_3`: the most positive-biased major joint. Median is already `1.3343`, so the dataset spends much of its time with this joint bent into a fairly consistent working posture.
- `joint_target_4`: a smaller-range fine-adjustment joint relative to dims `1` to `3`.
- `gripper_like_target_5`: mostly near a closed-like value around `-0.1745`, but with a substantial positive/open regime up to `1.0067`.

## How Fast The Actions Move

The actions are mostly smooth from frame to frame.

Mean and `p95` absolute per-step change:

| dim | mean abs step | p95 abs step | max abs step |
| --- | ---: | ---: | ---: |
| 0 | `0.0123` | `0.0583` | `0.6128` |
| 1 | `0.0209` | `0.0763` | `2.8247` |
| 2 | `0.0217` | `0.0914` | `2.9742` |
| 3 | `0.0258` | `0.1384` | `3.0608` |
| 4 | `0.0081` | `0.0381` | `0.8683` |
| 5 | `0.0131` | `0.0971` | `0.5413` |

Interpretation:

- most steps are small, so this is not a jerky random-action dataset
- dims `1` to `3` carry the largest arm motion
- dim `4` is the quietest joint
- dim `5` changes less often than a free-swinging joint, which again fits a gripper-like open/close role

The large max jumps on dims `1` to `3` are rare and likely happen at episode boundaries or during rapid replanning transitions.

## A Concrete State Example

Episode `0`, first frame:

```text
robot state      = [-0.1756, -1.7245,  1.6900, -1.6124, -0.0188, -0.1745]
cube_xyz         = [ 0.4000,  0.1800,  0.4325]
goal_minus_cube  = [ 0.0000, -0.3600, -0.0125]
goal_xyz         = [ 0.4000, -0.1800,  0.4200]
```

Episode `0`, last frame:

```text
robot state      = [-0.1950, -1.7245,  1.6900, -1.6281,  0.0927, -0.1682]
cube_xyz         = [ 0.4087, -0.1793,  0.4375]
goal_minus_cube  = [-0.0087, -0.0007, -0.0175]
goal_xyz         = [ 0.4000, -0.1800,  0.4200]
```

So the cube starts to the right of the goal box and ends very close to the fixed goal location.

## Practical Takeaways For This Repo

- This is a clean single-task dataset, so it is good for controlled reconstruction or dynamics experiments without cross-task confounds.
- The action space is low-dimensional and highly structured, which should make action-conditioned next-frame prediction easier than a heterogeneous multi-task robot dataset.
- Since the goal position is fixed, much of the learning problem is about consistent arm motion and grasp timing, not goal generalization.
- If we want broader generalization pressure, this dataset alone is not enough, because it does not vary task identity or goal location.

## Bottom Line

This dataset is best understood as:

- one robot
- one camera
- one cube
- one box
- one repeated right-to-left pick-and-place task
- six absolute control targets per frame

The strongest data-backed semantic claim is that the action vector is the next-step six-channel robot target state. The strongest inferred semantic claim is that the last action channel is the gripper command and the last six state channels are `cube_xyz` plus `goal_xyz - cube_xyz`.
