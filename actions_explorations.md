# Interactive World Sim dataset setup

## What changed

I switched the project focus from `lerobot/metaworld_mt50` to [`yixuan1999/interactive-world-sim-data`](https://huggingface.co/datasets/yixuan1999/interactive-world-sim-data).

It is organized as task folders with `train/` and `val/` splits, raw `episode_*.hdf5` files, and in some cases prebuilt `cache.zarr.zip` files that `interactive_world_sim` can load directly.

## Local data root

The dataset was downloaded under:

`data/full`

Current local footprint:

- about `12G` on disk

There is also a small local manifest at:

- `data/full/DOWNLOAD_MANIFEST.json`

## What is downloaded right now

### Fully usable right now

- `single_grasp/train/cache.zarr.zip`
- `single_grasp/train/episode_0.hdf5`
- full `single_grasp/val/episode_0.hdf5` through `episode_9.hdf5`

This makes `single_grasp` the only task in the current local checkout that is honestly ready for train/val use right away:

- train can use the prebuilt cache
- val has the full raw split, so a loader can build its own cache if needed

### Downloaded for inspection / visualization

For the other task families I downloaded one raw train episode and one raw val episode so the format is inspectable and the visualization script works:

- `pusht`
- `bimanual_sweep`
- `bimanual_rope`
- `bimanual_box`
- `single_chain_in_box`

These are good enough for exploration, camera inspection, and debugging the file format.

They are **not** full training downloads yet, because most of those folders do not expose a prebuilt `cache.zarr.zip` upstream and I only downloaded `episode_0.hdf5` from each split.

## Important practical fact

`interactive_world_sim` uses the `real_aloha_dataset` loader for this dataset family.

That loader works like this:

- if `train/cache.zarr.zip` or `val/cache.zarr.zip` exists, it loads that directly
- otherwise it scans all `episode_*.hdf5` files in the split and builds a cache

So for a task to be truly training-ready, you need one of:

1. a published `cache.zarr.zip` for that split
2. the full raw HDF5 split downloaded locally

Right now:

- `single_grasp/train` has the published cache
- `single_grasp/val` has the full raw split
- the other tasks are still sample-only locally

## On-disk file format

Each raw episode file looks like:

- top-level keys: `action`, `joint_action`, `obs`, `timestamp`

Inside `obs`:

- `images`
- `joint_pos`
- `full_joint_pos`
- `ee_pos`
- `world_t_robot_base`

Inside `obs/images`:

- `camera_0_color`
- `camera_1_color`
- `camera_0_intrinsics`
- `camera_1_intrinsics`
- `camera_0_extrinsics`
- `camera_1_extrinsics`

For the RGB observations:

- shape is `(T, 480, 640, 3)`
- dtype is `uint8`
- both `camera_0_color` and `camera_1_color` are present in the sampled tasks

For `single_grasp/val/episode_0.hdf5` specifically:

- `action` shape is `(200, 4)`. Used for training. It is `(x, y, z, gripper)`, where `x, y, z` are absolute end-effector targets in the dataset world frame, not image-relative and not deltas. `0,0,0,0` is not a special rest pose; it is just the world origin plus a zero gripper value, and this dataset does not operate near that point. In the inspected episode, the reachable region is roughly `x in [0.09, 0.30]`, `y in [-0.03, 0.17]`, `z in [0.12, 0.20]`. gripper values around `1.64` (opened) and `0.37` (closed) are raw gripper joint values
- `joint_action` shape is `(200, 7)`
- `joint_pos` shape is `(200, 7)`
- `full_joint_pos` shape is `(200, 8)`
- `ee_pos` shape is `(200, 7)`
- `world_t_robot_base` shape is `(200, 1, 4, 4)`. This is the transform from robot-base coordinates into the dataset world frame, which is why the action `x, y, z` values are expressed in that world coordinate system.
- mean timestamp step is about `0.1001s`, so this split runs at roughly `10 Hz`

For the bimanual tasks:

- `joint_pos` is usually `(T, 14)`
- `full_joint_pos` is usually `(T, 16)`
- `world_t_robot_base` is usually `(T, 2, 4, 4)`

## Task inventory

From the downloaded sample files and the upstream `interactive_world_sim` code:

- `single_grasp`
  - action shape: `4`
  - cameras: `camera_0_color`, `camera_1_color`
  - action mode: `single_grasp`
  - local status: usable now

- `pusht`
  - action shape: `4`
  - cameras: `camera_0_color`, `camera_1_color`
  - README training examples use action mode: `bimanual_push`
  - local status: sample only

- `bimanual_sweep`
  - action shape: `4`
  - cameras: `camera_0_color`, `camera_1_color`
  - codebase inference: action mode is `bimanual_sweep_v2`
  - local status: sample only

- `bimanual_rope`
  - action shape: `8`
  - cameras: `camera_0_color`, `camera_1_color`
  - action mode: `bimanual_rope`
  - local status: sample only

- `bimanual_box`
  - action shape: `14`
  - cameras: `camera_0_color`, `camera_1_color`
  - action mode: `bimanual_box`
  - local status: sample only

- `single_chain_in_box`
  - action shape: `4`
  - cameras: `camera_0_color`, `camera_1_color`
  - action mode: `single_chain_in_box`
  - local status: sample only

## What to use first

If the goal is to start actually using this dataset now, use:

- `data/full/single_grasp`

Why:

- it already has a train cache
- it has the full raw validation split locally
- it is one of the core tasks used in the upstream repo
- it has the simplest 4-D action space

## How to inspect the local inventory

```bash
source .venv/bin/activate
python scripts/check/visualize_interactive_world_sim_data.py list --data-root data/full
```

## Visualization script

I added:

- `scripts/check/visualize_interactive_world_sim_data.py`

Supported commands:

- `list`
- `episode-grid`
- `episode-gif`
- `task-sheet`

### Example: grid from one episode

```bash
source .venv/bin/activate
python scripts/check/visualize_interactive_world_sim_data.py episode-grid \
  --data-root data/full \
  --task bimanual_rope \
  --split val \
  --episode 0 \
  --camera camera_0_color \
  --frames 9 \
  --output /tmp/bimanual_rope_grid.png
```

### Example: GIF from one episode

```bash
source .venv/bin/activate
python scripts/check/visualize_interactive_world_sim_data.py episode-gif \
  --data-root data/full \
  --task single_grasp \
  --split val \
  --episode 0 \
  --camera camera_1_color \
  --frames 24 \
  --duration-ms 120 \
  --output /tmp/single_grasp.gif
```

### Example: contact sheet across episodes

```bash
source .venv/bin/activate
python scripts/check/visualize_interactive_world_sim_data.py task-sheet \
  --data-root data/full \
  --task single_grasp \
  --split val \
  --camera camera_1_color \
  --frame-index 50 \
  --limit 6 \
  --output /tmp/single_grasp_sheet.png
```

Both `episode-grid` and `task-sheet` were tested successfully.

## Download script

I added:

- `scripts/check/download_interactive_world_sim_data.py`

Default behavior:

- download one raw train episode and one raw val episode per task
- try to download `cache.zarr.zip` where the dataset actually publishes one

Useful command:

```bash
source .venv/bin/activate
python scripts/check/download_interactive_world_sim_data.py
```

If you want to fully hydrate a split later, use `--full-splits`.

Example:

```bash
source .venv/bin/activate
python scripts/check/download_interactive_world_sim_data.py \
  --full-splits single_grasp/val
```

## If you want another task to become truly training-ready

For `pusht`, `bimanual_sweep`, `bimanual_rope`, `bimanual_box`, or `single_chain_in_box`, do one of:

1. download a published cache if that task/split exposes one later
2. download the full raw split with `episode_*.hdf5` files

Then the upstream loader can build its cache locally.

In practice, the clean next step would be one of:

- fully hydrate `pusht/train` and `pusht/val`
- fully hydrate `bimanual_rope/train` and `bimanual_rope/val`
- stay with `single_grasp` first

## Source references

- Dataset: https://huggingface.co/datasets/yixuan1999/interactive-world-sim-data
- Repo: https://github.com/WangYixuan12/interactive_world_sim
- README download section: https://raw.githubusercontent.com/WangYixuan12/interactive_world_sim/main/README.md
