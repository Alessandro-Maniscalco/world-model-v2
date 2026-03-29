"""Download the Interactive World Sim dataset in a training-ready local layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


DATASET_ID = "yixuan1999/interactive-world-sim-data"
ALL_TASKS = [
    "pusht",
    "single_grasp",
    "bimanual_sweep",
    "bimanual_rope",
    "bimanual_box",
    "single_chain_in_box",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        default="data/full",
        help="Directory where the dataset should be materialized.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=ALL_TASKS,
        help="Tasks to download. Defaults to all task folders in the dataset.",
    )
    parser.add_argument(
        "--sample-episodes-per-split",
        type=int,
        default=1,
        help=(
            "Number of raw episode_*.hdf5 files to download per split for visualization."
            " The training-ready cache.zarr.zip files are always downloaded."
        ),
    )
    parser.add_argument(
        "--full-splits",
        nargs="*",
        default=[],
        help=(
            "Explicit split folders to download completely, for example"
            " `single_grasp/val` or `bimanual_rope/train`."
        ),
    )
    return parser.parse_args()


def expand_full_split_patterns(full_splits: list[str]) -> list[str]:
    """Expand split folders into explicit episode file paths."""

    if not full_splits:
        return []
    repo_files = list_repo_files(DATASET_ID, repo_type="dataset")
    patterns: list[str] = []
    for split_prefix in full_splits:
        prefix = f"{split_prefix}/"
        patterns.extend(
            file_path
            for file_path in repo_files
            if file_path.startswith(prefix) and file_path.endswith(".hdf5")
        )
    return sorted(set(patterns))


def build_patterns(
    tasks: list[str], sample_episodes_per_split: int, full_splits: list[str]
) -> list[str]:
    """Build the allow-pattern list for snapshot_download."""

    patterns: list[str] = []
    for task in tasks:
        patterns.append(f"{task}/train/cache.zarr.zip")
        patterns.append(f"{task}/val/cache.zarr.zip")
        for split in ("train", "val"):
            for episode_idx in range(sample_episodes_per_split):
                patterns.append(f"{task}/{split}/episode_{episode_idx}.hdf5")
    patterns.extend(expand_full_split_patterns(full_splits))
    return sorted(set(patterns))


def write_manifest(
    local_dir: Path,
    tasks: list[str],
    sample_episodes_per_split: int,
    full_splits: list[str],
) -> Path:
    """Write a small manifest describing the local download policy."""

    manifest_path = local_dir / "DOWNLOAD_MANIFEST.json"
    manifest = {
        "dataset_id": DATASET_ID,
        "tasks": tasks,
        "sample_episodes_per_split": sample_episodes_per_split,
        "full_splits": full_splits,
        "notes": [
            "cache.zarr.zip is sufficient for training with interactive_world_sim loaders when a task publishes that cache upstream.",
            "Some task folders may only expose raw HDF5 episodes upstream. Those tasks need a full raw split download before they are truly training-ready.",
            "sample episode_*.hdf5 files are included so local visualization scripts can inspect raw episodes.",
            "To fetch more raw episodes, rerun this script with a larger --sample-episodes-per-split value.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    """Download the selected task folders into the local data directory."""

    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    patterns = build_patterns(
        args.tasks, args.sample_episodes_per_split, args.full_splits
    )
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        ignore_patterns=["*.lock", "*cache_bak.zarr.zip"],
        local_dir=str(local_dir),
    )
    manifest_path = write_manifest(
        local_dir=local_dir,
        tasks=args.tasks,
        sample_episodes_per_split=args.sample_episodes_per_split,
        full_splits=args.full_splits,
    )
    print(f"Downloaded dataset subset to {local_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
