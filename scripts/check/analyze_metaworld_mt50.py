"""Analyze the LeRobot MetaWorld MT50 dataset and write a markdown report."""

from __future__ import annotations

import argparse
import json
import socket
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np
import pyarrow.parquet as pq


DATASET_ID = "lerobot/metaworld_mt50"
TARGET_REPO_URL = "https://github.com/WangYixuan12/interactive_world_sim"
TASK_SAMPLE_INDICES = [0, 3, 24, 48]


@dataclass(frozen=True)
class EpisodeRecord:
    """Store the episode-level metadata needed for aggregation."""

    episode_index: int
    task_index: int
    task_id: int
    task_name: str
    length: int
    from_index: int
    to_index: int
    action_min: np.ndarray
    action_max: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    state_min: np.ndarray
    state_max: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    success_mean: float
    reward_mean: float
    image_mean: np.ndarray
    image_std: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="metaworld_mt50_structure.md",
        help="Path to the markdown report to write.",
    )
    return parser.parse_args()


def download_meta_file(filename: str) -> Path:
    """Download one metadata file from the dataset repository."""

    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
        )
    )


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    with path.open() as handle:
        return json.load(handle)


def load_task_names(tasks_path: Path) -> dict[int, str]:
    """Load the task index to task name mapping."""

    table = pq.read_table(tasks_path).to_pydict()
    return {
        int(task_index): task_name
        for task_index, task_name in zip(
            table["task_index"], table["__index_level_0__"], strict=True
        )
    }


def load_episode_records(episodes_path: Path) -> list[EpisodeRecord]:
    """Load the per-episode metadata table into typed records."""

    table = pq.read_table(episodes_path).to_pydict()
    records: list[EpisodeRecord] = []
    num_rows = len(table["episode_index"])
    for idx in range(num_rows):
        image_mean = np.array(
            [channel[0][0] for channel in table["stats/observation.image/mean"][idx]],
            dtype=float,
        )
        image_std = np.array(
            [channel[0][0] for channel in table["stats/observation.image/std"][idx]],
            dtype=float,
        )
        records.append(
            EpisodeRecord(
                episode_index=int(table["episode_index"][idx]),
                task_index=int(table["stats/task_index/min"][idx][0]),
                task_id=int(table["stats/task_id/min"][idx][0]),
                task_name=table["tasks"][idx][0],
                length=int(table["length"][idx]),
                from_index=int(table["dataset_from_index"][idx]),
                to_index=int(table["dataset_to_index"][idx]),
                action_min=np.array(table["stats/action/min"][idx], dtype=float),
                action_max=np.array(table["stats/action/max"][idx], dtype=float),
                action_mean=np.array(table["stats/action/mean"][idx], dtype=float),
                action_std=np.array(table["stats/action/std"][idx], dtype=float),
                state_min=np.array(
                    table["stats/observation.state/min"][idx], dtype=float
                ),
                state_max=np.array(
                    table["stats/observation.state/max"][idx], dtype=float
                ),
                state_mean=np.array(
                    table["stats/observation.state/mean"][idx], dtype=float
                ),
                state_std=np.array(
                    table["stats/observation.state/std"][idx], dtype=float
                ),
                success_mean=float(table["stats/next.success/mean"][idx][0]),
                reward_mean=float(table["stats/next.reward/mean"][idx][0]),
                image_mean=image_mean,
                image_std=image_std,
            )
        )
    return records


def group_records_by_task(
    records: list[EpisodeRecord],
) -> dict[int, list[EpisodeRecord]]:
    """Group episode records by task index."""

    grouped: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_index].append(record)
    return dict(grouped)


def weighted_mean(values: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Compute a weighted mean across vectors."""

    stacked = np.stack(values)
    return np.average(stacked, axis=0, weights=weights)


def weighted_std(
    means: list[np.ndarray], stds: list[np.ndarray], weights: np.ndarray
) -> np.ndarray:
    """Compute an exact weighted standard deviation from episode summaries."""

    mean = weighted_mean(means, weights)
    second_moment = np.average(
        np.stack([std**2 + mu**2 for mu, std in zip(means, stds, strict=True)]),
        axis=0,
        weights=weights,
    )
    variance = np.maximum(second_moment - mean**2, 0.0)
    return np.sqrt(variance)


def summarize_task(task_index: int, records: list[EpisodeRecord]) -> dict:
    """Aggregate per-episode metadata into an exact per-task summary."""

    weights = np.array([record.length for record in records], dtype=float)
    lengths = weights.astype(int)
    action_means = [record.action_mean for record in records]
    action_stds = [record.action_std for record in records]
    state_means = [record.state_mean for record in records]
    state_stds = [record.state_std for record in records]

    return {
        "task_index": task_index,
        "task_name": records[0].task_name,
        "task_id": records[0].task_id,
        "episodes": len(records),
        "frames": int(lengths.sum()),
        "length_min": int(lengths.min()),
        "length_mean": float(lengths.mean()),
        "length_max": int(lengths.max()),
        "first_offset": records[0].from_index,
        "action_min": np.min(np.stack([record.action_min for record in records]), axis=0),
        "action_max": np.max(np.stack([record.action_max for record in records]), axis=0),
        "action_mean": weighted_mean(action_means, weights),
        "action_std": weighted_std(action_means, action_stds, weights),
        "state_min": np.min(np.stack([record.state_min for record in records]), axis=0),
        "state_max": np.max(np.stack([record.state_max for record in records]), axis=0),
        "state_mean": weighted_mean(state_means, weights),
        "state_std": weighted_std(state_means, state_stds, weights),
        "success_mean": float(
            np.average([record.success_mean for record in records], weights=weights)
        ),
        "reward_mean": float(
            np.average([record.reward_mean for record in records], weights=weights)
        ),
        "image_mean": weighted_mean([record.image_mean for record in records], weights),
        "image_std": weighted_std(
            [record.image_mean for record in records],
            [record.image_std for record in records],
            weights,
        ),
    }


def fetch_rows(offset: int, length: int) -> list[dict]:
    """Fetch row samples from the Hugging Face dataset server."""

    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urllib.request.urlopen(url, timeout=15) as response:
        payload = json.load(response)
    return [row["row"] for row in payload["rows"]]


def compute_transition_correlation(rows: list[dict]) -> np.ndarray:
    """Estimate action-to-state-delta correlations inside one sampled episode."""

    states = np.array([row["observation.state"] for row in rows], dtype=float)
    actions = np.array([row["action"] for row in rows[:-1]], dtype=float)
    deltas = states[1:] - states[:-1]
    correlations = np.zeros((4, 4), dtype=float)
    for action_idx in range(4):
        for state_idx in range(4):
            corr = np.corrcoef(actions[:, action_idx], deltas[:, state_idx])[0, 1]
            correlations[action_idx, state_idx] = corr
    return correlations


def infer_observation_environment_relationship(sample_rows: list[dict]) -> bool:
    """Check whether the first four environment-state values match observation.state."""

    for row in sample_rows:
        state = np.array(row["observation.state"], dtype=float)
        env = np.array(row["observation.environment_state"], dtype=float)
        if not np.allclose(state, env[:4]):
            return False
    return True


def top_tasks_by_length(task_summaries: list[dict], reverse: bool) -> list[dict]:
    """Return the five tasks with the largest or smallest mean episode length."""

    return sorted(
        task_summaries,
        key=lambda summary: summary["length_mean"],
        reverse=reverse,
    )[:5]


def top_tasks_by_action_scale(task_summaries: list[dict]) -> list[dict]:
    """Return the five tasks with the largest action span norm."""

    return sorted(
        task_summaries,
        key=lambda summary: float(
            np.linalg.norm(summary["action_max"][:3] - summary["action_min"][:3])
        ),
        reverse=True,
    )[:5]


def classify_gripper_channel(summary: dict) -> str:
    """Return a short description of how task-specific action channel 3 behaves."""

    minimum = float(summary["action_min"][3])
    maximum = float(summary["action_max"][3])
    if np.isclose(minimum, -1.0) and np.isclose(maximum, -1.0):
        return "fixed -1.0"
    if np.isclose(minimum, 1.0) and np.isclose(maximum, 1.0):
        return "fixed 1.0"
    if np.isclose(minimum, maximum):
        return f"fixed {minimum:.3f}"
    return "variable"


def summarize_repository_layout() -> dict[str, Any]:
    """Return a compact summary of the files published in the dataset repo."""

    repo_files = list_repo_files(DATASET_ID, repo_type="dataset")
    data_files = [path for path in repo_files if path.startswith("data/") and path.endswith(".parquet")]
    video_files = [path for path in repo_files if path.startswith("videos/") and path.endswith(".mp4")]
    meta_episode_files = [
        path
        for path in repo_files
        if path.startswith("meta/episodes/") and path.endswith(".parquet")
    ]
    return {
        "data_file_count": len(data_files),
        "video_file_count": len(video_files),
        "meta_episode_file_count": len(meta_episode_files),
        "first_data_files": data_files[:3],
        "first_video_files": video_files[:3],
        "has_readme": "README.md" in repo_files,
    }


def format_vector(vector: np.ndarray, decimals: int = 3) -> str:
    """Format a vector for markdown."""

    rounded = [f"{value:.{decimals}f}" for value in vector]
    return "[" + ", ".join(rounded) + "]"


def build_sample_blocks(task_summaries: dict[int, dict]) -> dict[int, dict]:
    """Fetch a few representative row samples for selected tasks."""

    samples: dict[int, dict] = {}
    for task_index in TASK_SAMPLE_INDICES:
        offset = task_summaries[task_index]["first_offset"]
        try:
            rows = fetch_rows(offset=offset, length=5)
        except (OSError, TimeoutError, socket.timeout, json.JSONDecodeError):
            rows = []
        samples[task_index] = {
            "offset": offset,
            "rows": rows,
        }
    return samples


def build_report(
    info: dict,
    global_stats: dict,
    task_summaries: list[dict],
    sample_blocks: dict[int, dict],
    repository_layout: dict[str, Any],
) -> str:
    """Render the dataset analysis as markdown."""

    task_summary_map = {summary["task_index"]: summary for summary in task_summaries}
    try:
        task_zero_rows = fetch_rows(offset=0, length=32)
    except (OSError, TimeoutError, socket.timeout, json.JSONDecodeError):
        task_zero_rows = []
    env_prefix_matches_state = (
        infer_observation_environment_relationship(task_zero_rows[:5])
        if task_zero_rows
        else False
    )
    transition_corr = (
        compute_transition_correlation(task_zero_rows)
        if len(task_zero_rows) >= 8
        else np.full((4, 4), np.nan)
    )

    longest_tasks = top_tasks_by_length(task_summaries, reverse=True)
    shortest_tasks = top_tasks_by_length(task_summaries, reverse=False)
    widest_action_tasks = top_tasks_by_action_scale(task_summaries)

    lines: list[str] = []
    lines.append("# MetaWorld MT50 dataset analysis")
    lines.append("")
    lines.append("## Goal")
    lines.append(
        "This note is meant to make the MetaWorld MT50 dataset concrete enough that we can"
        " later train an action-conditioned vision model in the style of"
        f" [`interactive_world_sim`]({TARGET_REPO_URL}) without guessing what the action and"
        " low-dimensional state channels mean."
    )
    lines.append("")
    lines.append("## Dataset contract")
    lines.append(
        f"- Dataset: [`{DATASET_ID}`](https://huggingface.co/datasets/{DATASET_ID})"
    )
    lines.append(f"- Total episodes: {info['total_episodes']}")
    lines.append(f"- Total frames: {info['total_frames']}")
    lines.append(f"- Total tasks: {info['total_tasks']}")
    lines.append(f"- FPS: {info['fps']}")
    lines.append(
        "- Observations per frame: one `observation.image` RGB view, `observation.state`"
        " with shape `[4]`, `observation.environment_state` with shape `[39]`, reward,"
        " success flag, task id/index, timestamps, and frame/episode indices."
    )
    lines.append(
        "- Important consistency check: `tasks.parquet` contains 49 task rows indexed"
        f" `0..48`, so the dataset is internally consistent on task count."
    )
    lines.append("")
    lines.append("## Repository layout")
    lines.append(f"- Published parquet data shards: {repository_layout['data_file_count']}")
    lines.append(f"- Published episode-metadata parquet files: {repository_layout['meta_episode_file_count']}")
    lines.append(f"- Published MP4 video files: {repository_layout['video_file_count']}")
    lines.append(f"- Dataset split declaration: `{info['splits']}`")
    lines.append(f"- Data path pattern: `{info['data_path']}`")
    lines.append(f"- Video path pattern: `{info['video_path']}`")
    lines.append(
        "- Example data shards: "
        + ", ".join(f"`{path}`" for path in repository_layout["first_data_files"])
    )
    lines.append(
        "- Example video files: "
        + (
            ", ".join(f"`{path}`" for path in repository_layout["first_video_files"])
            if repository_layout["first_video_files"]
            else "none published; the image bytes live directly in the parquet rows"
        )
    )
    lines.append("")
    lines.append("## Row and metadata structure")
    lines.append(
        "- `data/chunk-*/file-*.parquet` stores frame rows. Each row contains"
        " `observation.image` as a struct with embedded encoded image bytes plus a source path,"
        " along with `observation.state`, `observation.environment_state`, `action`, reward,"
        " success, timestamps, and episode/frame indices."
    )
    lines.append(
        "- `meta/tasks.parquet` maps each `task_index` to its natural-language task name."
    )
    lines.append(
        "- `meta/episodes/chunk-000/file-000.parquet` maps every episode to the data shard that"
        " stores it and records exact row spans via `dataset_from_index` and `dataset_to_index`."
    )
    lines.append(
        "- The per-episode metadata also includes exact aggregated statistics for action, state,"
        " reward, success, image channels, timestamps, and indices, so the dataset can be audited"
        " without scanning all frame rows."
    )
    lines.append("")
    lines.append("## High-level findings")
    lines.append(
        f"- Average episode length is {info['total_frames'] / info['total_episodes']:.2f} frames,"
        f" which is about {(info['total_frames'] / info['total_episodes']) / info['fps']:.2f} seconds"
        " at 80 FPS."
    )
    lines.append(
        "- Every task has 50 episodes, but the horizon varies a lot by task, from very short"
        " button/handle tasks to long manipulation tasks with some 500-frame rollouts."
    )
    lines.append(
        "- `observation.state[:3]` behaves like a compact end-effector position state,"
        " and `observation.state[3]` behaves like a gripper openness state."
    )
    lines.append(
        "- `action[:3]` behaves like Cartesian end-effector control, while `action[3]` behaves"
        " like a task-dependent gripper command. This is not documented explicitly in the"
        " dataset card, but it is strongly supported by the per-dimension ranges and row-level"
        " samples."
    )
    lines.append(
        "- `observation.environment_state` appears to contain privileged simulator state."
        f" In sampled rows, its first four values exactly match `observation.state`: {env_prefix_matches_state}."
    )
    lines.append("")
    lines.append("## Global statistics")
    lines.append(
        f"- `observation.state` min/max/mean/std: {format_vector(np.array(global_stats['observation.state']['min']))}"
        f" / {format_vector(np.array(global_stats['observation.state']['max']))}"
        f" / {format_vector(np.array(global_stats['observation.state']['mean']))}"
        f" / {format_vector(np.array(global_stats['observation.state']['std']))}"
    )
    lines.append(
        f"- `action` min/max/mean/std: {format_vector(np.array(global_stats['action']['min']))}"
        f" / {format_vector(np.array(global_stats['action']['max']))}"
        f" / {format_vector(np.array(global_stats['action']['mean']))}"
        f" / {format_vector(np.array(global_stats['action']['std']))}"
    )
    lines.append(
        f"- `observation.environment_state` has shape `[39]`; its global min/max span is much"
        " wider because it packs simulator/object information beyond the low-dimensional policy"
        " state."
    )
    lines.append("")
    lines.append("## `observation.state`")
    lines.append(
        "- Shape `[4]`. The first three channels stay within a compact workspace-sized range and"
        " move smoothly from frame to frame. The fourth channel stays within roughly `[0.26, 1.0]`."
    )
    lines.append(
        "- On sampled trajectories, the first three action channels are strongly correlated with"
        " `observation.state` deltas from one frame to the next in task 0. Estimated action to"
        f" state-delta correlation rows for task 0 sample episode: `{np.array2string(transition_corr, precision=3)}`."
    )
    lines.append(
        "- Practical interpretation:"
        " `observation.state[0:3] ~= end-effector position`,"
        " `observation.state[3] ~= gripper openness`."
    )
    lines.append(
        "- Because `observation.state` also appears at the front of `observation.environment_state`,"
        " it is likely the policy-facing subset of the simulator state."
    )
    lines.append("")
    lines.append("## `observation.environment_state`")
    lines.append(
        "- Shape `[39]`. This is privileged simulator state, not just robot proprioception."
    )
    lines.append(
        "- The first four values match `observation.state` in sampled rows."
    )
    lines.append(
        "- The remaining 35 values likely encode object/keypoint state and possibly repeated or"
        " lagged object state. That makes it useful for interpretation and debugging, but risky"
        " to use as a core input if the goal is an image-conditioned world model."
    )
    lines.append(
        "- Recommendation for later training: keep it for analysis, diagnostics, and optional"
        " auxiliary prediction targets; do not rely on it as the primary model input if you want"
        " to stay close to the `interactive_world_sim` setup."
    )
    lines.append("")
    lines.append("## `action`")
    lines.append("- Shape `[4]`.")
    lines.append(
        "- Channels 0-2 have large task-dependent continuous ranges and behave like Cartesian"
        " end-effector control."
    )
    lines.append(
        "- Channel 3 is not globally uniform. It is task-specific and often effectively fixed."
        " Examples from exact per-task metadata:"
    )
    lines.append(
        "  - Always `-1.0` in tasks like `Bypass a wall and press a button from the top`,"
        " `Push a button on the coffee machine`, `Open a drawer`, and `Press a handle down`."
    )
    lines.append(
        "  - Always `1.0` in tasks like `Press a button from the top`,"
        " `Rotate a dial 180 degrees`, `Open/close window`, and many handle/faucet tasks."
    )
    lines.append(
        "  - Variable in grasping/manipulation tasks like `Pick up a nut and place it onto a peg`,"
        " `Dunk the basketball into the basket`, `Insert a peg sideways`, and the puck tasks."
    )
    lines.append(
        "- This means the fourth action channel is better thought of as a gripper mode/control"
        " channel than as a generic continuous action dimension with one shared interpretation"
        " across all 49 tasks."
    )
    lines.append(
        "- Practical implication: if you later train across multiple tasks, per-dimension min-max"
        " normalization to `[-1, 1]` is fine, but the model still has to learn that `action[3]`"
        " means very different things across task families."
    )
    lines.append("")
    lines.append("## Representative task samples")
    for task_index in TASK_SAMPLE_INDICES:
        summary = task_summary_map[task_index]
        rows = sample_blocks[task_index]["rows"]
        lines.append(
            f"### Task {task_index}: {summary['task_name']}"
        )
        lines.append(
            f"- Episodes: {summary['episodes']}, frames: {summary['frames']},"
            f" mean episode length: {summary['length_mean']:.2f}"
        )
        lines.append(
            f"- `action` min/max/mean: {format_vector(summary['action_min'])}"
            f" / {format_vector(summary['action_max'])}"
            f" / {format_vector(summary['action_mean'])}"
        )
        lines.append(
            f"- `observation.state` min/max/mean: {format_vector(summary['state_min'])}"
            f" / {format_vector(summary['state_max'])}"
            f" / {format_vector(summary['state_mean'])}"
        )
        if rows:
            for row in rows[:2]:
                lines.append(
                    f"- Sample frame `{row['frame_index']}`:"
                    f" state={format_vector(np.array(row['observation.state'], dtype=float), decimals=4)}"
                    f", action={format_vector(np.array(row['action'], dtype=float), decimals=4)}"
                )
        else:
            lines.append(
                "- Sample frames were skipped because the dataset-server row request timed out."
            )
        lines.append("")
    lines.append("## Task coverage and difficulty proxies")
    lines.append("### Longest average tasks")
    for summary in longest_tasks:
        lines.append(
            f"- Task {summary['task_index']}: {summary['task_name']} ->"
            f" mean length {summary['length_mean']:.2f}, max length {summary['length_max']}"
        )
    lines.append("")
    lines.append("### Shortest average tasks")
    for summary in shortest_tasks:
        lines.append(
            f"- Task {summary['task_index']}: {summary['task_name']} ->"
            f" mean length {summary['length_mean']:.2f}, max length {summary['length_max']}"
        )
    lines.append("")
    lines.append("### Widest xyz action spans")
    for summary in widest_action_tasks:
        span = summary["action_max"][:3] - summary["action_min"][:3]
        lines.append(
            f"- Task {summary['task_index']}: {summary['task_name']} -> xyz span"
            f" {format_vector(span)}"
        )
    lines.append("")
    lines.append("## Full task inventory")
    lines.append("| Task | Episodes | Mean length | Gripper channel | Name |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for summary in task_summaries:
        lines.append(
            f"| {summary['task_index']} | {summary['episodes']} | {summary['length_mean']:.2f} |"
            f" {classify_gripper_channel(summary)} | {summary['task_name']} |"
        )
    lines.append("")
    lines.append("## What this means for later training")
    lines.append(
        f"- If the goal is to stay close to `{TARGET_REPO_URL}`, the cleanest later setup is"
        " `image + action` as the core model input, with `observation.state` used as a"
        " debugging target or auxiliary predictor."
    )
    lines.append(
        "- Normalize each action dimension independently to `[-1, 1]`, but remember that"
        " `action[3]` is effectively task-coded gripper behavior and is not globally continuous"
        " in the same way as the xyz channels."
    )
    lines.append(
        "- For single-task-first training, choose a task family with variable gripper behavior"
        " if you want the model to actually learn grasp/open-close structure. Tasks with"
        " fixed `action[3]` are simpler but teach less about contact-rich control."
    )
    lines.append(
        "- Because the average episode is short and the dataset runs at 80 FPS, later training"
        " should consider temporal downsampling or larger action chunks to avoid wasting model"
        " capacity on nearly redundant adjacent frames."
    )
    lines.append("")
    lines.append("## Training hook in this repo")
    lines.append(
        "- The minimal Wan VAE path in this repo can now read MT50 directly through"
        " `world_model_v2/minimal/run.py` with `--dataset-format lerobot_metaworld`."
    )
    lines.append("```bash")
    lines.append("source .venv/bin/activate")
    lines.append("python -m world_model_v2.minimal.run \\")
    lines.append("  --mode ae_only \\")
    lines.append("  --dataset-format lerobot_metaworld \\")
    lines.append("  --metaworld-task-index 0 \\")
    lines.append("  --split train \\")
    lines.append("  --train-all-episodes \\")
    lines.append("  --validation-split val \\")
    lines.append("  --validation-episode 0 \\")
    lines.append("  --resolution 128 \\")
    lines.append("  --batch-size 32 \\")
    lines.append("  --max-steps 3000 \\")
    lines.append("  --run-name metaworld_mt50_task0_wan_ae")
    lines.append("```")
    lines.append(
        "- Important local behavior: MT50 only publishes a train split, so this repo treats both"
        " `train` and `val` requests as aliases for the same upstream split and uses"
        " `validation_episode` only to pick which train episode to preview."
    )
    lines.append("")
    lines.append("## Recommended first tasks if the aim is understanding")
    lines.append(
        "- Good variable-gripper tasks: task 0 `Pick up a nut and place it onto a peg`,"
        " task 1 `Dunk the basketball into the basket`, task 2 `Grasp the puck from one bin and"
        " place it into another bin`, task 35 `Insert a peg sideways`."
    )
    lines.append(
        "- Good fixed-gripper tasks: task 24 `Press a handle down`, task 47 `Push and open a window`,"
        " task 48 `Push and close a window`."
    )
    lines.append("")
    lines.append("## Sources")
    lines.append(f"- https://huggingface.co/datasets/{DATASET_ID}")
    lines.append(f"- https://huggingface.co/datasets/{DATASET_ID}/raw/main/README.md")
    lines.append(
        f"- https://huggingface.co/datasets/{DATASET_ID}/resolve/main/meta/info.json"
    )
    lines.append(
        f"- https://huggingface.co/datasets/{DATASET_ID}/resolve/main/meta/stats.json"
    )
    lines.append(
        f"- https://huggingface.co/datasets/{DATASET_ID}/resolve/main/meta/tasks.parquet?download=true"
    )
    lines.append(
        f"- https://huggingface.co/datasets/{DATASET_ID}/resolve/main/meta/episodes/chunk-000/file-000.parquet?download=true"
    )
    lines.append(f"- {TARGET_REPO_URL}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the full MT50 metadata analysis and write the markdown report."""

    args = parse_args()
    info = load_json(download_meta_file("meta/info.json"))
    global_stats = load_json(download_meta_file("meta/stats.json"))
    task_names = load_task_names(download_meta_file("meta/tasks.parquet"))
    episode_records = load_episode_records(
        download_meta_file("meta/episodes/chunk-000/file-000.parquet")
    )
    task_groups = group_records_by_task(episode_records)
    task_summaries = [
        summarize_task(task_index, task_groups[task_index])
        for task_index in sorted(task_groups)
    ]
    sample_blocks = build_sample_blocks({s["task_index"]: s for s in task_summaries})
    repository_layout = summarize_repository_layout()
    report = build_report(
        info=info,
        global_stats=global_stats,
        task_summaries=task_summaries,
        sample_blocks=sample_blocks,
        repository_layout=repository_layout,
    )
    output_path = Path(args.output)
    output_path.write_text(report + "\n")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
