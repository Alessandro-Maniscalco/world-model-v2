"""Endlessly cycle RF-DiT dynamics runs and score their validation artifacts.

source .venv/bin/activate
python scripts/check/loop_dynamics_sweep.py \
  --vae-checkpoint outputs/metaworld_task0_wan_ae_240/checkpoints/best.pt \
  --device cuda

source .venv/bin/activate
python scripts/check/loop_dynamics_sweep.py \
  --frame-span 0:31 \
  --frame-span 32:95 \
  --infer-steps 16 \
  --max-cycles 1 \
  --max-steps 1000 \
  --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.dynamics_transformer import DYNAMICS_FRAME_LAYOUT
from world_model_v2.metaworld_dataset import MetaWorldRepository

DEFAULT_VAE_CHECKPOINT = REPO_ROOT / "outputs" / "best_vae" / "checkpoints" / "best.pt"
LEGACY_DEFAULT_VAE_CHECKPOINT = (
    REPO_ROOT / "outputs" / "minimal" / "metaworld_task0_wan_ae_240" / "checkpoints" / "best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_FRAME_SPANS = ("0:31", "32:95", "96:159")
DEFAULT_SELECTION_METRIC = "worst_case_next_frame_mse"
EVAL_OPEN_ROLLOUT_SELECTION_METRIC = "eval_open_rollout_frame_mse"
EVAL_OPEN_ROLLOUT_MEAN_SELECTION_METRIC = "eval_open_rollout_frame_mse_mean"
EVAL_OPEN_ROLLOUT_MAX_SELECTION_METRIC = "eval_open_rollout_frame_mse_max"
OPEN_ROLLOUT_CONSISTENCY_SELECTION_METRIC = "open_rollout_consistency_score"
EVAL_OPEN_ROLLOUT_CONSISTENCY_SELECTION_METRIC = "eval_open_rollout_consistency_score"
EVAL_OPEN_ROLLOUT_CONSISTENCY_MEAN_SELECTION_METRIC = (
    "eval_open_rollout_consistency_score_mean"
)
EVAL_OPEN_ROLLOUT_CONSISTENCY_MAX_SELECTION_METRIC = (
    "eval_open_rollout_consistency_score_max"
)
_TEACHER_FORCED_NEXT_FRAME_MSE_PATTERN = re.compile(r"^next_frame_mse(?:_\d+to\d+)?$")
SELECTION_METRIC_CHOICES = (
    DEFAULT_SELECTION_METRIC,
    "next_frame_mse",
    "next_frame_mse_1to1",
    "next_frame_mse_1to2",
    "next_frame_mse_1to3",
    "next_frame_mse_4to1",
    "open_rollout_frame_mse",
    OPEN_ROLLOUT_CONSISTENCY_SELECTION_METRIC,
    EVAL_OPEN_ROLLOUT_SELECTION_METRIC,
    EVAL_OPEN_ROLLOUT_MEAN_SELECTION_METRIC,
    EVAL_OPEN_ROLLOUT_MAX_SELECTION_METRIC,
    EVAL_OPEN_ROLLOUT_CONSISTENCY_SELECTION_METRIC,
    EVAL_OPEN_ROLLOUT_CONSISTENCY_MEAN_SELECTION_METRIC,
    EVAL_OPEN_ROLLOUT_CONSISTENCY_MAX_SELECTION_METRIC,
)


@dataclass(frozen=True)
class SweepSpec:
    """Describe one dynamics run configuration in the endless sweep."""

    frame_start: int
    frame_end: int
    infer_steps: int
    batch_size: int
    max_steps: int
    lr: float

    @property
    def name(self) -> str:
        """Return a compact identifier for one sweep configuration."""

        return f"f{self.frame_start}_{self.frame_end}_infer{self.infer_steps}_bs{self.batch_size}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the endless dynamics sweep helper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", default=str(DEFAULT_VAE_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-prefix", default="loop_rf_dit_action_teacher_1to3")
    parser.add_argument("--dataset-format", default="lerobot_metaworld", choices=["lerobot_metaworld", "interactive_world_sim"])
    parser.add_argument("--data-root", default="data/full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--metaworld-task-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-span", action="append", default=None)
    parser.add_argument("--infer-steps", action="append", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--dynamics-train-timesteps", type=int, default=1000)
    parser.add_argument("--dynamics-rf-shift", type=float, default=5.0)
    parser.add_argument(
        "--dynamics-context-frames",
        type=int,
        default=DYNAMICS_FRAME_LAYOUT.context_frames,
    )
    parser.add_argument(
        "--dynamics-target-frames",
        type=int,
        default=DYNAMICS_FRAME_LAYOUT.target_frames,
    )
    parser.add_argument("--dynamics-conditioning-frame-choices", default=None)
    parser.add_argument("--dynamics-conditioning-frame-probabilities", default=None)
    parser.add_argument("--dynamics-validation-conditioning-frame-choices", default=None)
    parser.add_argument("--dynamics-open-rollout-context-frames", type=int, default=None)
    parser.add_argument("--dynamics-open-rollout-stride-frames", type=int, default=None)
    parser.add_argument("--dynamics-model-channels", type=int, default=256)
    parser.add_argument("--dynamics-num-blocks", type=int, default=4)
    parser.add_argument("--dynamics-num-heads", type=int, default=4)
    parser.add_argument(
        "--dynamics-action-conditioning-mode",
        default="chunk_per_frame",
        choices=["chunk_per_frame"],
    )
    parser.add_argument("--dynamics-zero-init-action-embedder", action="store_true")
    parser.add_argument(
        "--dynamics-use-adaln-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dynamics-adaln-lora-dim", type=int, default=64)
    parser.add_argument("--dynamics-rope-t-extrapolation-ratio", type=float, default=1.0)
    parser.add_argument(
        "--dynamics-use-learned-temporal-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--conditional-frame-timestep",
        type=float,
        default=-1.0,
        help="Forward the conditioning-frame timestep used by dynamics conditioning.",
    )
    parser.add_argument(
        "--conditional-frame-sigma",
        type=float,
        default=0.0,
        help="Optional tiny noise sigma applied to conditioned frames before repinning.",
    )
    parser.add_argument(
        "--dynamics-self-forcing-loss-weight",
        type=float,
        default=0.0,
        help="Optional DreamDojo-inspired causal self-forcing auxiliary loss weight.",
    )
    parser.add_argument(
        "--dynamics-rollout-self-forcing-loss-weight",
        type=float,
        default=0.0,
        help="Optional additional rollout-aligned self-forcing loss weight used on top of the primary mode.",
    )
    parser.add_argument(
        "--dynamics-self-forcing-mode",
        choices=["expanded_context", "rollout"],
        default="expanded_context",
        help="Choose between the legacy expanded-prefix auxiliary and same-context rollout self-forcing.",
    )
    parser.add_argument(
        "--dynamics-self-forcing-warmup-steps",
        type=int,
        default=0,
        help="Optional number of optimizer steps to keep self-forcing disabled before enabling it.",
    )
    parser.add_argument(
        "--dynamics-self-forcing-ramp-steps",
        type=int,
        default=0,
        help="Optional number of optimizer steps used to ramp self-forcing from zero to the target weight.",
    )
    parser.add_argument(
        "--dynamics-rollout-self-forcing-warmup-steps",
        type=int,
        default=0,
        help="Optional number of optimizer steps to keep the rollout self-forcing auxiliary disabled before enabling it.",
    )
    parser.add_argument(
        "--dynamics-rollout-self-forcing-ramp-steps",
        type=int,
        default=0,
        help="Optional number of optimizer steps used to ramp the rollout self-forcing auxiliary from zero to the target weight.",
    )
    parser.add_argument(
        "--dynamics-self-forcing-rollout-chunks",
        type=int,
        default=0,
        help="Optional number of extra rollout chunks used by same-context self-forcing.",
    )
    parser.add_argument(
        "--dynamics-validation-metric",
        default="next_frame_mse",
        choices=[
            "next_frame_mse",
            "open_rollout_frame_mse",
            "open_rollout_consistency_score",
        ],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--eval-open-rollout-frame-span",
        action="append",
        default=None,
        help="Optional fixed `start:end` clip used for cross-run open-rollout evaluation. Repeat to score multiple spans.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Number of full sweep cycles to run. Use 0 to loop forever.",
    )
    parser.add_argument(
        "--max-next-frame-mse",
        type=float,
        default=0.03,
        help="Runs above this value are marked as failed quality checks even if training completed.",
    )
    parser.add_argument(
        "--selection-metric",
        default=DEFAULT_SELECTION_METRIC,
        choices=list(SELECTION_METRIC_CHOICES),
        help="Metric used to decide which successful checkpoint becomes the warm-start best.",
    )
    parser.add_argument("--load-dynamics", default="")
    parser.add_argument(
        "--warmstart-previous-best",
        action="store_true",
        help="Load the previous successful sweep checkpoint into the next run.",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    if args.vae_checkpoint == str(DEFAULT_VAE_CHECKPOINT) and not DEFAULT_VAE_CHECKPOINT.exists():
        if LEGACY_DEFAULT_VAE_CHECKPOINT.exists():
            args.vae_checkpoint = str(LEGACY_DEFAULT_VAE_CHECKPOINT)
    return args


def configured_dynamics_max_frames(args: argparse.Namespace) -> int:
    """Return the total latent-frame chunk length requested for the sweep."""

    return int(args.dynamics_context_frames) + int(args.dynamics_target_frames)


def parse_frame_span(value: str) -> tuple[int, int]:
    """Parse one `start:end` frame span string."""

    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected frame span 'start:end', received {value!r}.")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame span {value!r}.")
    return start, end


def build_sweep_specs(args: argparse.Namespace) -> list[SweepSpec]:
    """Build the ordered list of sweep runs to execute repeatedly."""

    specs: list[SweepSpec] = []
    max_valid_frame_end = resolve_max_valid_frame_end(args)
    required_frames = configured_dynamics_max_frames(args)
    frame_spans = list(DEFAULT_FRAME_SPANS) if args.frame_span is None else list(args.frame_span)
    infer_steps_list = [16] if args.infer_steps is None else [int(value) for value in args.infer_steps]
    for frame_span in frame_spans:
        frame_start, frame_end = parse_frame_span(frame_span)
        if max_valid_frame_end is not None:
            if frame_start > max_valid_frame_end:
                continue
            frame_end = min(frame_end, max_valid_frame_end)
            if frame_end - frame_start + 1 < required_frames:
                continue
        for infer_steps in infer_steps_list:
            specs.append(
                SweepSpec(
                    frame_start=frame_start,
                    frame_end=frame_end,
                    infer_steps=int(infer_steps),
                    batch_size=int(args.batch_size),
                    max_steps=int(args.max_steps),
                    lr=float(args.lr),
                )
            )
    return specs


def resolve_max_valid_frame_end(args: argparse.Namespace) -> int | None:
    """Return the highest valid frame index for the selected dataset slice."""

    if args.dataset_format != "lerobot_metaworld":
        return None
    repository = MetaWorldRepository(
        data_root=args.data_root,
        repo_id="lerobot/metaworld_mt50",
        cache_dir=None,
    )
    record = repository.episode_record(
        episode=int(args.episode),
        task_index=int(args.metaworld_task_index),
    )
    return int(record.length) - 1


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def timestamp_slug() -> str:
    """Return a filesystem-friendly UTC-ish timestamp slug."""

    return time.strftime("%Y%m%d_%H%M%S")


def build_run_name(run_prefix: str, spec: SweepSpec, cycle_index: int, run_index: int) -> str:
    """Return a unique run name for one sweep item."""

    return f"{run_prefix}_{spec.name}_c{cycle_index:03d}_r{run_index:03d}_{timestamp_slug()}"


def build_command(
    args: argparse.Namespace,
    spec: SweepSpec,
    run_name: str,
    load_dynamics: Path | None,
) -> list[str]:
    """Build the training command for one sweep item."""

    command = [
        sys.executable,
        "-m",
        "world_model_v2.run",
        "--mode",
        "dynamics_only",
        "--dataset-format",
        args.dataset_format,
        "--data-root",
        args.data_root,
        "--split",
        args.split,
        "--episode",
        str(args.episode),
        "--frame-start",
        str(spec.frame_start),
        "--frame-end",
        str(spec.frame_end),
        "--resolution",
        str(args.resolution),
        "--batch-size",
        str(spec.batch_size),
        "--lr",
        str(spec.lr),
        "--max-steps",
        str(spec.max_steps),
        "--validation-interval",
        str(args.validation_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--early-stop-window-size",
        "0",
        "--conditional-frame-timestep",
        str(args.conditional_frame_timestep),
        "--conditional-frame-sigma",
        str(args.conditional_frame_sigma),
        "--dynamics-self-forcing-loss-weight",
        str(args.dynamics_self_forcing_loss_weight),
        "--dynamics-rollout-self-forcing-loss-weight",
        str(args.dynamics_rollout_self_forcing_loss_weight),
        "--dynamics-self-forcing-mode",
        str(args.dynamics_self_forcing_mode),
        "--dynamics-self-forcing-warmup-steps",
        str(args.dynamics_self_forcing_warmup_steps),
        "--dynamics-self-forcing-ramp-steps",
        str(args.dynamics_self_forcing_ramp_steps),
        "--dynamics-rollout-self-forcing-warmup-steps",
        str(args.dynamics_rollout_self_forcing_warmup_steps),
        "--dynamics-rollout-self-forcing-ramp-steps",
        str(args.dynamics_rollout_self_forcing_ramp_steps),
        "--dynamics-self-forcing-rollout-chunks",
        str(args.dynamics_self_forcing_rollout_chunks),
        "--dynamics-infer-steps",
        str(spec.infer_steps),
        "--dynamics-train-timesteps",
        str(args.dynamics_train_timesteps),
        "--dynamics-rf-shift",
        str(args.dynamics_rf_shift),
        *(
            ["--dynamics-use-learned-temporal-embedding"]
            if args.dynamics_use_learned_temporal_embedding
            else []
        ),
        "--dynamics-context-frames",
        str(args.dynamics_context_frames),
        "--dynamics-target-frames",
        str(args.dynamics_target_frames),
        "--dynamics-model-channels",
        str(args.dynamics_model_channels),
        "--dynamics-num-blocks",
        str(args.dynamics_num_blocks),
        "--dynamics-num-heads",
        str(args.dynamics_num_heads),
        "--dynamics-action-conditioning-mode",
        str(args.dynamics_action_conditioning_mode),
        "--dynamics-adaln-lora-dim",
        str(args.dynamics_adaln_lora_dim),
        "--dynamics-rope-t-extrapolation-ratio",
        str(args.dynamics_rope_t_extrapolation_ratio),
        "--dynamics-validation-metric",
        str(args.dynamics_validation_metric),
        "--load-encoder-decoder",
        args.vae_checkpoint,
        "--run-name",
        run_name,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.dynamics_conditioning_frame_choices is not None:
        command.extend(
            [
                "--dynamics-conditioning-frame-choices",
                str(args.dynamics_conditioning_frame_choices),
            ]
        )
    if args.dynamics_conditioning_frame_probabilities is not None:
        command.extend(
            [
                "--dynamics-conditioning-frame-probabilities",
                str(args.dynamics_conditioning_frame_probabilities),
            ]
        )
    if args.dynamics_validation_conditioning_frame_choices is not None:
        command.extend(
            [
                "--dynamics-validation-conditioning-frame-choices",
                str(args.dynamics_validation_conditioning_frame_choices),
            ]
        )
    if args.dynamics_open_rollout_context_frames is not None:
        command.extend(
            [
                "--dynamics-open-rollout-context-frames",
                str(args.dynamics_open_rollout_context_frames),
            ]
        )
    if args.dynamics_open_rollout_stride_frames is not None:
        command.extend(
            [
                "--dynamics-open-rollout-stride-frames",
                str(args.dynamics_open_rollout_stride_frames),
            ]
        )
    if args.dynamics_zero_init_action_embedder:
        command.append("--dynamics-zero-init-action-embedder")
    if args.dynamics_use_adaln_lora:
        command.append("--dynamics-use-adaln-lora")
    else:
        command.append("--no-dynamics-use-adaln-lora")
    if args.dataset_format == "lerobot_metaworld":
        command.extend(["--metaworld-task-index", str(args.metaworld_task_index)])
    else:
        command.extend(["--task", args.task])
    if load_dynamics is not None:
        command.extend(["--load-dynamics", str(load_dynamics)])
    return command


def build_open_rollout_eval_command(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    frame_start: int,
    frame_end: int,
) -> list[str] | None:
    """Build the fixed-span open-rollout evaluation command when configured."""

    if args.dataset_format != "lerobot_metaworld":
        return None
    command = [
        sys.executable,
        "scripts/check/open_rollout_demo.py",
        "--checkpoint",
        str(checkpoint_path),
        "--data-root",
        args.data_root,
        "--split",
        args.split,
        "--episode",
        str(args.episode),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
        "--resolution",
        str(args.resolution),
        "--device",
        args.device,
        "--output-dir",
        str(output_dir),
        "--run-name",
        f"open_rollout_eval_f{frame_start}_{frame_end}",
    ]
    if args.dynamics_open_rollout_stride_frames is not None:
        command.extend(
            [
                "--dynamics-open-rollout-stride-frames",
                str(args.dynamics_open_rollout_stride_frames),
            ]
        )
    command.extend(["--metaworld-task-index", str(args.metaworld_task_index)])
    return command


def _eval_open_rollout_spans(args: argparse.Namespace) -> list[tuple[int, int]]:
    """Return the configured fixed-span open-rollout evaluation windows."""

    raw_spans = list(args.eval_open_rollout_frame_span or [])
    return [parse_frame_span(str(value)) for value in raw_spans]


def _frame_span_slug(frame_start: int, frame_end: int) -> str:
    """Return a metric-safe slug for one evaluation span."""

    return f"f{frame_start}_{frame_end}"


def run_training_command(command: list[str]) -> int:
    """Execute one training command and return its exit code."""

    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def find_latest_stats_path(run_dir: Path) -> Path:
    """Return the newest saved validation stats file for one run."""

    stats_paths = sorted(
        run_dir.glob("samples/step_*/episode_0_stats.json"),
        key=lambda path: path.parent.name,
    )
    if not stats_paths:
        raise FileNotFoundError(f"No validation stats found under {run_dir}.")
    return stats_paths[-1]


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def probe_mp4_frame_count(video_path: Path) -> int | None:
    """Return the MP4 frame count when `ffprobe` is available."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "json",
            str(video_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return None
    raw_value = streams[0].get("nb_read_packets")
    if raw_value in {None, "N/A"}:
        return None
    return int(raw_value)


def summarize_grid_image(grid_path: Path) -> dict[str, float]:
    """Return simple image statistics for the saved side-by-side grid."""

    image = Image.open(grid_path).convert("RGB")
    stat = ImageStat.Stat(image)
    mean_std = sum(float(value) for value in stat.stddev) / len(stat.stddev)
    mean_mean = sum(float(value) for value in stat.mean) / len(stat.mean)
    return {
        "grid_channel_mean": mean_mean,
        "grid_channel_std": mean_std,
    }


def _extract_selection_metric(stats: dict[str, Any], metric_name: str) -> float:
    """Return the numeric score used to rank successful dynamics runs."""

    if metric_name == DEFAULT_SELECTION_METRIC:
        return _worst_case_teacher_forced_next_frame_mse(stats)
    return float(stats.get(metric_name, math.inf))


def _teacher_forced_next_frame_mse_values(stats: dict[str, Any]) -> list[float]:
    """Return all aggregate teacher-forced next-frame MSE values found in one stats blob."""

    values: list[float] = []
    for key, value in stats.items():
        if not _TEACHER_FORCED_NEXT_FRAME_MSE_PATTERN.fullmatch(key):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _worst_case_teacher_forced_next_frame_mse(stats: dict[str, Any]) -> float:
    """Return the worst aggregate teacher-forced next-frame MSE across validated layouts."""

    values = _teacher_forced_next_frame_mse_values(stats)
    if not values:
        return math.inf
    return max(values)


def evaluate_run(run_dir: Path, max_next_frame_mse: float) -> dict[str, Any]:
    """Score one finished run from its saved validation stats and artifacts."""

    stats_path = find_latest_stats_path(run_dir)
    stats = load_json(stats_path)
    grid_path = stats_path.with_name("episode_0_grid.png")
    video_path = stats_path.with_name("episode_0.mp4")
    grid_summary = summarize_grid_image(grid_path) if grid_path.exists() else {}
    ffprobe_frame_count = probe_mp4_frame_count(video_path) if video_path.exists() else None
    next_frame_mse = float(stats.get("next_frame_mse", math.inf))
    worst_case_next_frame_mse = _worst_case_teacher_forced_next_frame_mse(stats)

    issues: list[str] = []
    if not video_path.exists():
        issues.append("missing_mp4")
    if not grid_path.exists():
        issues.append("missing_grid")
    predicted_frame_count = int(stats.get("predicted_frame_count", -1))
    exported_video_frame_count = int(stats.get("exported_video_frame_count", -1))
    if predicted_frame_count != exported_video_frame_count:
        issues.append("predicted_vs_exported_frame_count_mismatch")
    if ffprobe_frame_count is not None and ffprobe_frame_count != exported_video_frame_count:
        issues.append("ffprobe_frame_count_mismatch")
    if not math.isfinite(next_frame_mse):
        issues.append("non_finite_next_frame_mse")
    if not math.isfinite(worst_case_next_frame_mse):
        issues.append("non_finite_worst_case_next_frame_mse")
    if next_frame_mse > max_next_frame_mse:
        issues.append("next_frame_mse_above_threshold")
    if worst_case_next_frame_mse > max_next_frame_mse:
        issues.append("worst_case_next_frame_mse_above_threshold")

    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    return {
        "run_dir": str(run_dir),
        "stats_path": str(stats_path),
        "video_path": str(video_path),
        "grid_path": str(grid_path),
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_exists": best_checkpoint.exists(),
        "ffprobe_frame_count": ffprobe_frame_count,
        "issues": issues,
        "passed": not issues,
        "worst_case_next_frame_mse": worst_case_next_frame_mse,
        **grid_summary,
        **stats,
    }


def evaluate_open_rollout(
    args: argparse.Namespace,
    run_dir: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Run a fixed-span open-rollout evaluation and merge its stats into the run record."""

    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    if not best_checkpoint.exists():
        return evaluation
    eval_spans = _eval_open_rollout_spans(args)
    if not eval_spans:
        return evaluation
    output_dir = run_dir / "open_rollout_eval"
    merged: dict[str, Any] = dict(evaluation)
    span_scores: list[float] = []
    span_consistency_scores: list[float] = []
    span_issue = False
    for frame_start, frame_end in eval_spans:
        command = build_open_rollout_eval_command(
            args,
            checkpoint_path=best_checkpoint,
            output_dir=output_dir,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        if command is None:
            return evaluation
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        span_slug = _frame_span_slug(frame_start, frame_end)
        if completed.returncode != 0:
            merged[f"eval_open_rollout_{span_slug}_exit_code"] = int(completed.returncode)
            merged[f"eval_open_rollout_{span_slug}_error"] = (
                completed.stderr.strip() or completed.stdout.strip()
            )
            span_issue = True
            continue
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        if payload.get("open_rollout_frame_mse") is not None:
            span_scores.append(float(payload["open_rollout_frame_mse"]))
        if payload.get("open_rollout_consistency_score") is not None:
            span_consistency_scores.append(float(payload["open_rollout_consistency_score"]))
        if len(eval_spans) == 1:
            prefixed_payload = {
                f"eval_{key}": value
                for key, value in payload.items()
                if key not in {"checkpoint", "device", "validation_style"}
            }
            merged.update(
                {
                    "eval_open_rollout_checkpoint": payload.get("checkpoint"),
                    "eval_open_rollout_device": payload.get("device"),
                    "eval_open_rollout_validation_style": payload.get("validation_style"),
                    **prefixed_payload,
                }
            )
        span_prefixed_payload = {
            f"eval_open_rollout_{span_slug}_{key}": value
            for key, value in payload.items()
            if key not in {"checkpoint", "device", "validation_style"}
        }
        merged.update(
            {
                f"eval_open_rollout_{span_slug}_checkpoint": payload.get("checkpoint"),
                f"eval_open_rollout_{span_slug}_device": payload.get("device"),
                f"eval_open_rollout_{span_slug}_validation_style": payload.get("validation_style"),
                **span_prefixed_payload,
            }
        )
    if span_scores:
        mean_score = float(sum(span_scores) / len(span_scores))
        max_score = float(max(span_scores))
        merged["eval_open_rollout_frame_mse_mean"] = mean_score
        merged["eval_open_rollout_frame_mse_max"] = max_score
        if len(eval_spans) > 1:
            merged["eval_open_rollout_frame_mse"] = mean_score
    if span_consistency_scores:
        mean_score = float(sum(span_consistency_scores) / len(span_consistency_scores))
        max_score = float(max(span_consistency_scores))
        merged["eval_open_rollout_consistency_score_mean"] = mean_score
        merged["eval_open_rollout_consistency_score_max"] = max_score
        if len(eval_spans) > 1:
            merged["eval_open_rollout_consistency_score"] = mean_score
    if span_issue:
        merged["issues"] = [*list(merged.get("issues", [])), "eval_open_rollout_failed"]
        merged["passed"] = False
    return merged


def maybe_update_best_run(
    current_best: dict[str, Any] | None,
    evaluation: dict[str, Any],
    selection_metric: str,
) -> dict[str, Any] | None:
    """Return the updated best-run record when the new evaluation is eligible."""

    if not evaluation.get("passed") or not evaluation.get("best_checkpoint_exists"):
        return current_best
    selection_score = _extract_selection_metric(evaluation, selection_metric)
    if not math.isfinite(selection_score):
        return current_best
    candidate = {
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "best_checkpoint": str(evaluation["best_checkpoint"]),
        "run_dir": str(evaluation["run_dir"]),
        "run_name": Path(str(evaluation["run_dir"])).name,
    }
    if current_best is None or selection_score < float(current_best["selection_score"]):
        return candidate
    return current_best


def sleep_between_runs(seconds: float) -> None:
    """Sleep between runs when requested."""

    if seconds <= 0:
        return
    time.sleep(seconds)


def main() -> None:
    """Run the endless dynamics sweep."""

    args = parse_args()
    specs = build_sweep_specs(args)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / f"{args.run_prefix}_summary.jsonl"
    best_run_path = output_dir / f"{args.run_prefix}_best.json"
    initial_warmstart = Path(args.load_dynamics) if args.load_dynamics else None
    warmstart_checkpoint = initial_warmstart
    best_run: dict[str, Any] | None = None
    cycle_index = 0
    run_index = 0

    while True:
        cycle_index += 1
        for spec in specs:
            run_index += 1
            run_name = build_run_name(args.run_prefix, spec, cycle_index, run_index)
            run_dir = output_dir / run_name
            command = build_command(args, spec, run_name, warmstart_checkpoint)
            started_at = time.time()
            exit_code = run_training_command(command)
            finished_at = time.time()
            record: dict[str, Any] = {
                "cycle_index": cycle_index,
                "run_index": run_index,
                "spec": asdict(spec),
                "run_name": run_name,
                "command": command,
                "exit_code": exit_code,
                "started_at_unix": started_at,
                "finished_at_unix": finished_at,
                "duration_seconds": finished_at - started_at,
            }
            if exit_code == 0:
                try:
                    evaluation = evaluate_run(run_dir, max_next_frame_mse=args.max_next_frame_mse)
                    evaluation = evaluate_open_rollout(args, run_dir, evaluation)
                    record["evaluation"] = evaluation
                    best_run = maybe_update_best_run(
                        current_best=best_run,
                        evaluation=evaluation,
                        selection_metric=args.selection_metric,
                    )
                    record["best_run"] = best_run
                    if best_run is not None:
                        write_path = {
                            **best_run,
                            "updated_by_run_index": run_index,
                            "updated_by_cycle_index": cycle_index,
                        }
                        best_run_path.parent.mkdir(parents=True, exist_ok=True)
                        best_run_path.write_text(
                            json.dumps(write_path, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    if args.warmstart_previous_best and best_run is not None:
                        warmstart_checkpoint = Path(str(best_run["best_checkpoint"]))
                except Exception as error:  # pragma: no cover - manual sweep safeguard
                    record["evaluation_error"] = f"{type(error).__name__}: {error}"
                    if args.stop_on_failure:
                        append_jsonl(summary_path, record)
                        raise
            else:
                if args.stop_on_failure:
                    append_jsonl(summary_path, record)
                    raise SystemExit(exit_code)
            append_jsonl(summary_path, record)
            print(json.dumps(record, sort_keys=True))
            sleep_between_runs(args.sleep_seconds)
        if args.max_cycles > 0 and cycle_index >= args.max_cycles:
            break


if __name__ == "__main__":
    main()
