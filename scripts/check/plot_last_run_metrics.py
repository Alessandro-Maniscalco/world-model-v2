"""Plot a compact subset of run metrics.

source .venv/bin/activate
python scripts/check/plot_last_run_metrics.py
python scripts/check/plot_last_run_metrics.py --run-dir outputs/dynamics_only_single_grasp_ep0_f111_116_from_last
python scripts/check/plot_last_run_metrics.py --run-dir outputs/so101_base_pickplace_wan_ae_240x320_1224 --min-step 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_OUTPUT_NAME = "metrics_validation_plot.png"
MAX_TRAIN_METRICS = 1
MAX_VALIDATION_METRICS = 1
TRAIN_METRIC_PRIORITY = (
    "loss",
    "latent_rf_total_loss",
    "latent_rf_mse",
)
VALIDATION_TOTAL_LOSS_PRIORITY = (
    "loss",
    "latent_rf_total_loss",
    "ae_loss",
    "latent_rf_mse",
)
VALIDATION_METRIC_PRIORITY = (
    "ae_loss",
    "next_frame_mse",
    "worst_case_next_frame_mse",
    "open_rollout_consistency_score",
    "open_rollout_frame_mse",
    "next_latent_mse",
    "target_motion_ratio",
    "open_rollout_target_motion_ratio",
    "predicted_target_motion_l1",
    "open_rollout_predicted_target_motion_l1",
)
VALIDATION_METADATA_KEYS = {
    "episode",
    "elapsed_run_seconds",
    "input_frame_count",
    "decoded_frame_count",
    "predicted_frame_count",
    "exported_video_frame_count",
    "seed_frames",
    "loss_frames",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the metrics plotting helper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="",
        help="Run directory containing metrics.jsonl. Defaults to the newest run under outputs/.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Destination PNG path. Defaults to <run-dir>/metrics_validation_plot.png.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help="Only plot records at or after this step.",
    )
    return parser.parse_args()


def find_latest_run_dir(output_root: Path) -> Path:
    """Return the run directory containing the newest metrics.jsonl file."""

    metrics_paths = sorted(
        output_root.rglob("metrics.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics.jsonl files found under {output_root}.")
    return metrics_paths[-1].parent


def load_metrics_records(metrics_path: Path) -> list[dict[str, Any]]:
    """Load every JSONL metrics record from disk."""

    with metrics_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_records_min_step(
    records: list[dict[str, Any]],
    min_step: int,
) -> list[dict[str, Any]]:
    """Keep run metadata plus step-scoped records at or after the minimum step."""

    if min_step <= 0:
        return list(records)

    filtered_records: list[dict[str, Any]] = []
    for record in records:
        step = record.get("step")
        if step is None:
            filtered_records.append(record)
            continue
        if int(step) >= min_step:
            filtered_records.append(record)
    return filtered_records


def is_numeric(value: Any) -> bool:
    """Return whether a value should be plotted as a scalar metric."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_validation_metadata_key(key: str) -> bool:
    """Return whether a validation payload key should be treated as metadata."""

    if key in VALIDATION_METADATA_KEYS:
        return True
    return any(
        fragment in key
        for fragment in ("frame_count", "loss_frames", "seed_frames", "episode_count")
    )


def collect_training_series(records: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Collect numeric training metrics keyed by metric name."""

    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        if "step" not in record:
            continue
        step = int(record["step"])
        for key, value in record.items():
            if key == "step" or not is_numeric(value):
                continue
            series.setdefault(key, []).append((step, float(value)))
    return series


def collect_validation_series(records: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Collect numeric validation metrics keyed by metric name."""

    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        validation = record.get("validation")
        if not isinstance(validation, dict) or "step" not in record:
            continue
        step = int(record["step"])
        for key, value in validation.items():
            if is_validation_metadata_key(key) or not is_numeric(value):
                continue
            series.setdefault(key, []).append((step, float(value)))
    return series


def collect_stop_step(records: list[dict[str, Any]]) -> int | None:
    """Return the recorded early-stop step when present."""

    for record in reversed(records):
        stopped = record.get("stopped")
        if isinstance(stopped, dict) and record.get("step") is not None:
            return int(record["step"])
    return None


def extract_run_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the run config recorded in the run_start event when present."""

    run_start = next((record["run_start"] for record in records if "run_start" in record), {})
    return run_start.get("config", {}) if isinstance(run_start, dict) else {}


def select_metric_names(
    series: dict[str, list[tuple[int, float]]],
    preferred_metrics: tuple[str, ...],
    *,
    max_metrics: int,
    primary_metric: str | None = None,
) -> tuple[str, ...]:
    """Return a compact ordered subset of series names to plot."""

    selected: list[str] = []
    if primary_metric and primary_metric in series:
        selected.append(primary_metric)
    for key in preferred_metrics:
        if key in series and key not in selected:
            selected.append(key)
    if selected:
        return tuple(selected[:max_metrics])
    ranked_keys = sorted(series, key=lambda key: (-len(series[key]), key))
    return tuple(ranked_keys[:max_metrics])


def select_training_metric_names(
    series: dict[str, list[tuple[int, float]]],
) -> tuple[str, ...]:
    """Return the single training loss curve to plot."""

    return select_metric_names(
        series,
        TRAIN_METRIC_PRIORITY,
        max_metrics=MAX_TRAIN_METRICS,
    )


def select_validation_metric_names(
    records: list[dict[str, Any]],
    series: dict[str, list[tuple[int, float]]],
) -> tuple[str, ...]:
    """Return one validation loss-like curve, preferring an aggregate loss when present."""

    for key in VALIDATION_TOTAL_LOSS_PRIORITY:
        if key in series:
            return (key,)
    config = extract_run_config(records)
    primary_metric = str(config.get("dynamics_validation_metric", "")).strip() or None
    return select_metric_names(
        series,
        VALIDATION_METRIC_PRIORITY,
        max_metrics=MAX_VALIDATION_METRICS,
        primary_metric=primary_metric,
    )


def summarize_latest_metrics(
    training_series: dict[str, list[tuple[int, float]]],
    validation_series: dict[str, list[tuple[int, float]]],
    training_metric_names: tuple[str, ...],
    validation_metric_names: tuple[str, ...],
) -> dict[str, float]:
    """Return the latest plotted value for each selected metric."""

    summary: dict[str, float] = {}
    for key in training_metric_names:
        values = training_series.get(key, [])
        if values:
            summary[key] = values[-1][1]
    for key in validation_metric_names:
        values = validation_series.get(key, [])
        if values:
            summary[f"validation.{key}"] = values[-1][1]
    return summary


def build_plot(
    run_dir: Path,
    records: list[dict[str, Any]],
    output_path: Path,
    *,
    min_step: int = 0,
) -> dict[str, Any]:
    """Render the training and validation curves to a PNG file."""

    training_series = collect_training_series(records)
    validation_series = collect_validation_series(records)
    training_metric_names = select_training_metric_names(training_series)
    validation_metric_names = select_validation_metric_names(records, validation_series)
    if not training_metric_names and not validation_metric_names:
        raise ValueError(f"No plottable metrics found in {run_dir / 'metrics.jsonl'}.")

    fig, ax_train = plt.subplots(figsize=(12, 7))
    ax_val = ax_train.twinx()

    train_colors = plt.get_cmap("tab10")
    for index, key in enumerate(training_metric_names):
        values = training_series.get(key, [])
        if not values:
            continue
        steps = [step for step, _ in values]
        scalars = [value for _, value in values]
        linewidth = 2.2 if key == "loss" else 1.4
        alpha = 0.95 if key == "loss" else 0.7
        ax_train.plot(
            steps,
            scalars,
            label=f"train:{key}",
            color=train_colors(index % 10),
            linewidth=linewidth,
            alpha=alpha,
        )

    val_colors = plt.get_cmap("Dark2")
    for index, key in enumerate(validation_metric_names):
        values = validation_series.get(key, [])
        if not values:
            continue
        steps = [step for step, _ in values]
        scalars = [value for _, value in values]
        ax_val.plot(
            steps,
            scalars,
            label=f"val:{key}",
            color=val_colors(index % 8),
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=2.0,
            alpha=0.95,
        )

    stop_step = collect_stop_step(records)
    if stop_step is not None:
        ax_train.axvline(
            stop_step,
            color="#444444",
            linestyle=":",
            linewidth=1.5,
            label=f"stop@{stop_step}",
        )

    config = extract_run_config(records)
    run_start = next((record["run_start"] for record in records if "run_start" in record), {})
    mode = str(config.get("mode", run_start.get("mode", "unknown")))
    title = f"{run_dir.name} ({mode})"
    if min_step > 0:
        title = f"{title} steps >= {min_step}"
    ax_train.set_title(title)
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Training metrics")
    ax_val.set_ylabel("Validation metrics")
    ax_train.grid(True, alpha=0.25)

    handles_train, labels_train = ax_train.get_legend_handles_labels()
    handles_val, labels_val = ax_val.get_legend_handles_labels()
    fig.legend(
        handles_train + handles_val,
        labels_train + labels_val,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    summary = summarize_latest_metrics(
        training_series,
        validation_series,
        training_metric_names,
        validation_metric_names,
    )
    return {
        "min_step": min_step,
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.jsonl"),
        "output_path": str(output_path),
        "stop_step": stop_step,
        "latest": summary,
    }


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Resolve the requested run directory or fall back to the newest run."""

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        return run_dir.resolve()
    return find_latest_run_dir(DEFAULT_OUTPUT_ROOT).resolve()


def resolve_output_path(args: argparse.Namespace, run_dir: Path) -> Path:
    """Resolve the destination PNG path for the plot."""

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        return output_path.resolve()
    return (run_dir / DEFAULT_OUTPUT_NAME).resolve()


def main() -> None:
    """Run the metrics plotting CLI and print a compact summary."""

    args = parse_args()
    run_dir = resolve_run_dir(args)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics at {metrics_path}.")
    output_path = resolve_output_path(args, run_dir)
    records = filter_records_min_step(load_metrics_records(metrics_path), args.min_step)
    result = build_plot(run_dir, records, output_path, min_step=args.min_step)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
