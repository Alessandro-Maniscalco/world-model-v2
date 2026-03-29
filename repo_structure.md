# Repo Structure

## Main package

Root package:

- `world_model_v2/`

This is now organized to structurally resemble the upstream
`interactive_world_sim` repo while staying much smaller.

## Config

- `world_model_v2/config.py`

Contains nested dataclass config sections:

- `DatasetConfig`
- `AlgorithmConfig`
- `ExperimentConfig`
- `RunConfig`

These replace the role of upstream Hydra config composition, but in plain
Python.

## Algorithms

- `world_model_v2/algorithms/latent_dynamics/latent_world_model.py`

Primary Stage-1 model class:

- `LatentWorldModel`

Responsibilities:

- Stage-1 loss computation
- reconstruction
- validation preview stats

- `world_model_v2/algorithms/latent_dynamics/noise_scheduler.py`

Lightweight scheduler for:

- timestep sampling
- noise injection
- weighted reconstruction loss
- iterative denoising schedules

- `world_model_v2/algorithms/models/cnn_encoder.py`

Small CNN image encoder.

- `world_model_v2/algorithms/models/cm_decoder.py`

Consistency-style decoder with latent conditioning.

- `world_model_v2/algorithms/models/blocks.py`

Shared U-Net-like blocks:

- residual blocks
- down blocks
- up blocks

- `world_model_v2/algorithms/models/embeddings.py`

Sinusoidal timestep embeddings.

## Datasets

- `world_model_v2/datasets/latent_dynamics/real_aloha_dataset.py`

Raw-HDF5 dataset loader for Interactive World Sim data.

Responsibilities:

- read `episode_*.hdf5`
- resize images to configured resolution
- expose sequence-shaped `obs` and `action`
- return train windows and validation episodes
- compute action min/max stats

This is structurally modeled after the upstream `real_aloha_dataset`, but uses
only raw HDF5 right now.

## Experiments

- `world_model_v2/experiments/latent_dynamics_experiment.py`

Plain PyTorch runner for Stage-1 training.

Responsibilities:

- seed setup
- dataset and dataloader construction
- model and optimizer setup
- checkpoint saving
- validation preview export
- metrics logging

## Inference

- `world_model_v2/infer/reconstruct_episode.py`

Loads a checkpoint, reconstructs one episode, and writes:

- side-by-side grid
- side-by-side GIF
- stats JSON

## Utils

- `world_model_v2/utils/checkpointing.py`

Helpers for:

- JSON config writes
- JSONL metrics logging
- checkpoint save/load

- `world_model_v2/utils/visualization.py`

Helpers for:

- annotated frames
- side-by-side contact sheets
- GIF export

## CLI and Scripts

- `world_model_v2/run.py`

Single canonical training entrypoint:

```bash
source .venv/bin/activate
python -m world_model_v2.run ...
```

- `scripts/check/visualize_stage1_reconstruction.py`

Manual smoke-check wrapper for checkpoint reconstruction.

- `scripts/check/visualize_interactive_world_sim_data.py`

Raw dataset inspection and visualization.

- `scripts/check/download_interactive_world_sim_data.py`

Dataset download helper.

## Tests

Tests live in `tests/` and cover:

- config serialization
- embeddings and blocks
- encoder and decoder
- dataset loading
- scheduler behavior
- `LatentWorldModel`
- experiment runner
- run entrypoint
- reconstruction export
- visualization helpers

## Current scope

Implemented:

- Stage 1 autoencoder training
- checkpoint reload
- reconstruction visualization

Not implemented yet:

- Stage 2 latent dynamics training
- Stage 3 decoder finetuning
- upstream zarr/cache path
- Hydra / Lightning / W&B stack
