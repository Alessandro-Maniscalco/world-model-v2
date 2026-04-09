"""Plot a compact subset of run metrics.

source .venv/bin/activate
python scripts/check/plot_last_run_metrics.py
python scripts/check/plot_last_run_metrics.py --run-dir outputs/dynamics_only_single_grasp_ep0_f111_116_from_last
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
TRAIN_METRICS_TO_PLOT = ("loss",)
VALIDATION_METRICS_TO_PLOT = ("ae_loss",)
VALIDATION_METADATA_KEYS = {
    "episode",
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


def is_numeric(value: Any) -> bool:
    """Return whether a value should be plotted as a scalar metric."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def collect_training_series(records: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Collect the selected numeric training metrics keyed by metric name."""

    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        if "loss" not in record or "step" not in record:
            continue
        step = int(record["step"])
        for key, value in record.items():
            if key not in TRAIN_METRICS_TO_PLOT or not is_numeric(value):
                continue
            series.setdefault(key, []).append((step, float(value)))
    return series


def collect_validation_series(records: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    """Collect the selected numeric validation metrics keyed by metric name."""

    series: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        validation = record.get("validation")
        if not isinstance(validation, dict) or "step" not in record:
            continue
        step = int(record["step"])
        for key, value in validation.items():
            if (
                key in VALIDATION_METADATA_KEYS
                or key not in VALIDATION_METRICS_TO_PLOT
                or not is_numeric(value)
            ):
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


def summarize_latest_metrics(
    training_series: dict[str, list[tuple[int, float]]],
    validation_series: dict[str, list[tuple[int, float]]],
) -> dict[str, float]:
    """Return the latest plotted value for each selected metric."""

    summary: dict[str, float] = {}
    for key in TRAIN_METRICS_TO_PLOT:
        values = training_series.get(key, [])
        if values:
            summary[key] = values[-1][1]
    for key in VALIDATION_METRICS_TO_PLOT:
        values = validation_series.get(key, [])
        if values:
            summary[f"validation.{key}"] = values[-1][1]
    return summary


def build_plot(
    run_dir: Path,
    records: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Render the training and validation curves to a PNG file."""

    training_series = collect_training_series(records)
    validation_series = collect_validation_series(records)
    if not training_series and not validation_series:
        raise ValueError(f"No plottable metrics found in {run_dir / 'metrics.jsonl'}.")

    fig, ax_train = plt.subplots(figsize=(12, 7))
    ax_val = ax_train.twinx()

    train_colors = plt.get_cmap("tab10")
    for index, key in enumerate(TRAIN_METRICS_TO_PLOT):
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
    for index, key in enumerate(VALIDATION_METRICS_TO_PLOT):
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

    run_start = next((record["run_start"] for record in records if "run_start" in record), {})
    config = run_start.get("config", {}) if isinstance(run_start, dict) else {}
    mode = str(config.get("mode", run_start.get("mode", "unknown")))
    ax_train.set_title(f"{run_dir.name} ({mode})")
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

    summary = summarize_latest_metrics(training_series, validation_series)
    return {
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
    result = build_plot(run_dir, load_metrics_records(metrics_path), output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
