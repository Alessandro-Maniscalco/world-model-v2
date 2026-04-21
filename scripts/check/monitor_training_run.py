"""Monitor one local training run and auto-resume it when needed.

This smoke-check helper watches one run directory on a fixed cadence, records
basic health snapshots, and relaunches `world_model_v2.run` from `last.pt`
when the tracked process exits before reaching `max_steps`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StepSnapshot:
    """Capture the latest observable run step and write timestamp."""

    step: int
    metrics_mtime: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one monitor session."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--stalled-intervals-before-restart", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file into a dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_latest_step(metrics_path: Path) -> StepSnapshot:
    """Return the latest logged training or validation step from one metrics file."""

    latest_step = 0
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    continue
                raw_step = record.get("step")
                if isinstance(raw_step, int):
                    latest_step = max(latest_step, raw_step)
    mtime = metrics_path.stat().st_mtime if metrics_path.exists() else 0.0
    return StepSnapshot(step=int(latest_step), metrics_mtime=float(mtime))


def _powershell_json(command: str) -> list[dict[str, Any]]:
    """Run one PowerShell expression that emits JSON and return a normalized list."""

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            command,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = completed.stdout.strip()
    if not payload:
        return []
    parsed = json.loads(payload)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def find_matching_python_processes(run_name: str) -> list[dict[str, Any]]:
    """Return running python processes whose command line references the run name."""

    escaped_run_name = run_name.replace("'", "''")
    command = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.Name -match '^python(\\.exe|w\\.exe)?$' } "
        f"| Where-Object {{ $_.CommandLine -like '*--run-name {escaped_run_name}*' }} "
        "| Select-Object ProcessId, CommandLine "
        "| ConvertTo-Json -Compress"
    )
    return _powershell_json(command)


def terminate_processes(processes: list[dict[str, Any]]) -> None:
    """Terminate the provided processes by PID when they are still present."""

    for process in processes:
        pid = process.get("ProcessId")
        if not isinstance(pid, int):
            continue
        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(pid),
                "/T",
                "/F",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )


def build_resume_command(config: dict[str, Any], run_dir: Path) -> list[str]:
    """Build the `world_model_v2.run` resume command for one saved config."""

    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    command: list[str] = [
        sys.executable,
        "-m",
        "world_model_v2.run",
        "--mode",
        str(config["mode"]),
        "--dataset-format",
        str(config["dataset_format"]),
        "--data-root",
        str(config["data_root"]),
        "--task",
        str(config["task"]),
        "--split",
        str(config["split"]),
        "--episode",
        str(config["episode"]),
        "--validation-split",
        str(config["validation_split"]),
        "--validation-episode",
        str(config["validation_episode"]),
        "--resolution",
        str(config["resolution"]),
        "--height",
        str(config["height"]),
        "--width",
        str(config["width"]),
        "--wan-dim",
        str(config["wan_dim"]),
        "--latent-channels",
        str(config["latent_channels"]),
        "--wan-num-res-blocks",
        str(config["wan_num_res_blocks"]),
        "--hidden-channels",
        str(config["hidden_channels"]),
        "--batch-size",
        str(config["batch_size"]),
        "--grad-accum-steps",
        str(config["gradient_accumulation_steps"]),
        "--dataloader-num-workers",
        str(config["dataloader_num_workers"]),
        "--lr",
        str(config["lr"]),
        "--lr-warmup-steps",
        str(config["lr_warmup_steps"]),
        "--optimizer-beta1",
        str(config["optimizer_beta1"]),
        "--max-steps",
        str(config["max_steps"]),
        "--validation-interval",
        str(config["validation_interval"]),
        "--checkpoint-interval",
        str(config["checkpoint_interval"]),
        "--log-interval",
        str(config["log_interval"]),
        "--early-stop-patience-windows",
        str(config["early_stop_patience_windows"]),
        "--early-stop-warmup-steps",
        str(config["early_stop_warmup_steps"]),
        "--dynamics-context-frames",
        str(config["dynamics_context_frames"]),
        "--dynamics-target-frames",
        str(config["dynamics_target_frames"]),
        "--dynamics-model-channels",
        str(config["dynamics_model_channels"]),
        "--dynamics-num-blocks",
        str(config["dynamics_num_blocks"]),
        "--dynamics-num-heads",
        str(config["dynamics_num_heads"]),
        "--dynamics-action-conditioning-mode",
        str(config["dynamics_action_conditioning_mode"]),
        "--dynamics-adaln-lora-dim",
        str(config["dynamics_adaln_lora_dim"]),
        "--dynamics-infer-steps",
        str(config["dynamics_infer_steps"]),
        "--dynamics-train-timesteps",
        str(config["dynamics_train_timesteps"]),
        "--dynamics-rf-shift",
        str(config["dynamics_rf_shift"]),
        "--dynamics-validation-metric",
        str(config["dynamics_validation_metric"]),
        "--resume",
        str(checkpoint_path),
        "--device",
        str(config["device"]),
        "--run-name",
        str(config["run_name"]),
    ]
    if bool(config.get("train_all_episodes", False)):
        command.append("--train-all-episodes")
    if not bool(config.get("dynamics_run_open_rollout_validation", False)):
        command.append("--no-dynamics-run-open-rollout-validation")
    if config.get("frame_start") is not None:
        command.extend(["--frame-start", str(config["frame_start"])])
    if config.get("frame_end") is not None:
        command.extend(["--frame-end", str(config["frame_end"])])
    if config.get("validation_max_frames") is not None:
        command.extend(["--validation-max-frames", str(config["validation_max_frames"])])
    return command


def launch_resume(command: list[str], run_dir: Path) -> int:
    """Launch one detached resume command and return its PID."""

    stdout_path = run_dir / "monitor_resume_stdout.log"
    stderr_path = run_dir / "monitor_resume_stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a",
        encoding="utf-8",
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
    return int(process.pid)


def monitor_loop(run_dir: Path, interval_seconds: float, stalled_intervals_before_restart: int) -> None:
    """Monitor one run forever and relaunch it when it exits early or stalls repeatedly."""

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    monitor_log_path = run_dir / "monitor_log.jsonl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    config = load_json(config_path)
    run_name = str(config["run_name"])
    max_steps = int(config["max_steps"])
    previous_snapshot = read_latest_step(metrics_path)
    stalled_intervals = 0
    append_jsonl(
        monitor_log_path,
        {
            "event": "monitor_started",
            "interval_seconds": float(interval_seconds),
            "max_steps": max_steps,
            "run_name": run_name,
            "starting_step": previous_snapshot.step,
        },
    )
    while True:
        time.sleep(interval_seconds)
        snapshot = read_latest_step(metrics_path)
        processes = find_matching_python_processes(run_name)
        alive = len(processes) > 0
        progressed = snapshot.step > previous_snapshot.step or snapshot.metrics_mtime > previous_snapshot.metrics_mtime
        if snapshot.step >= max_steps:
            append_jsonl(
                monitor_log_path,
                {
                    "alive": alive,
                    "event": "run_completed",
                    "step": snapshot.step,
                },
            )
            return
        if alive and progressed:
            stalled_intervals = 0
            append_jsonl(
                monitor_log_path,
                {
                    "alive": True,
                    "event": "healthy",
                    "process_count": len(processes),
                    "step": snapshot.step,
                },
            )
            previous_snapshot = snapshot
            continue
        if alive and not progressed:
            stalled_intervals += 1
            append_jsonl(
                monitor_log_path,
                {
                    "alive": True,
                    "event": "stalled_interval",
                    "process_count": len(processes),
                    "stalled_intervals": stalled_intervals,
                    "step": snapshot.step,
                },
            )
            if stalled_intervals < stalled_intervals_before_restart:
                previous_snapshot = snapshot
                continue
            terminate_processes(processes)
        command = build_resume_command(config, run_dir)
        pid = launch_resume(command, run_dir)
        stalled_intervals = 0
        append_jsonl(
            monitor_log_path,
            {
                "alive": alive,
                "event": "resume_launched",
                "pid": pid,
                "step": snapshot.step,
            },
        )
        previous_snapshot = snapshot


def main() -> None:
    """Run the long-lived training monitor."""

    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    interval_seconds = max(float(args.interval_minutes), 0.1) * 60.0
    stalled_intervals_before_restart = max(int(args.stalled_intervals_before_restart), 1)
    monitor_loop(
        run_dir=run_dir,
        interval_seconds=interval_seconds,
        stalled_intervals_before_restart=stalled_intervals_before_restart,
    )


if __name__ == "__main__":
    main()
