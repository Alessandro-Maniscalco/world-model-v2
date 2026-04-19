r"""Profile one short resumed training run step-by-step.

This smoke-check helper resumes a real world-model training checkpoint,
measures one warmup step plus a small number of steady-state steps, and
records per-step timing breakdowns for:

- iterator construction
- batch fetch on CPU
- host-to-device copy
- model forward + loss + backward + optimizer
- metrics append

It also samples `nvidia-smi` during the run so Windows/Linux comparisons can
include clocks, power, P-state, and driver model information.

Example
-------
& .\.venv\Scripts\Activate.ps1
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python scripts/check/profile_resumed_training_step.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.experiment import Experiment, ExperimentConfig
from world_model_v2.utils.checkpointing import append_jsonl, save_json


DEFAULT_RESUME = REPO_ROOT / "saved_checkpoints" / "github" / "vae_pickplace_z32_8x.pt"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "so101_base_sim_pickplace_cache"
DEFAULT_OUTPUT_SUBDIR = "temp_profile_resume_github_vae_pickplace_z32_8x_bs1_windows"
GPU_QUERY_FIELDS = (
    "timestamp",
    "name",
    "driver_model.current",
    "pstate",
    "power.draw",
    "clocks.current.sm",
    "clocks.max.sm",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the short resumed-step profiler."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=str(DEFAULT_RESUME))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--gpu-poll-seconds", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build the exact baseline experiment config used for the comparison."""

    return ExperimentConfig(
        mode="ae_only",
        dataset_format="lerobot_so101_base_sim_pickplace",
        data_root=str(Path(args.data_root)),
        task="single_grasp",
        split="train",
        episode=0,
        train_all_episodes=True,
        validation_split="train",
        validation_episode=0,
        resolution=120,
        height=120,
        width=160,
        wan_dim=128,
        latent_channels=32,
        wan_num_res_blocks=2,
        batch_size=max(int(args.batch_size), 1),
        gradient_accumulation_steps=max(int(args.grad_accum_steps), 1),
        dataloader_num_workers=max(int(args.dataloader_num_workers), 0),
        lr=1e-5,
        max_steps=50000,
        validation_interval=500,
        checkpoint_interval=100,
        log_interval=10,
        dynamics_context_frames=1,
        dynamics_target_frames=3,
        kl_beta=1e-5,
        recon_mse_weight=1.0,
        recon_l1_weight=0.1,
        recon_edge_weight=0.05,
        recon_motion_weight=2.0,
        recon_motion_threshold=0.02,
        recon_motion_dilation_kernel_size=7,
        resume=str(Path(args.resume)),
        run_name=str(args.output_subdir),
        output_dir=str(Path(args.output_dir)),
        seed=7,
        device=str(args.device),
    )


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize the active CUDA stream when profiling GPU work."""

    if device.type == "cuda":
        torch.cuda.synchronize()


def query_gpu_state() -> dict[str, Any] | None:
    """Return one parsed `nvidia-smi` snapshot, or `None` if unavailable."""

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) != len(GPU_QUERY_FIELDS):
        return {"raw": line[0]}
    record: dict[str, Any] = dict(zip(GPU_QUERY_FIELDS, parts, strict=True))
    for key in (
        "power.draw",
        "clocks.current.sm",
        "clocks.max.sm",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
    ):
        try:
            record[key] = float(record[key])
        except (TypeError, ValueError):
            pass
    return record


def sample_gpu_state(
    *,
    samples: list[dict[str, Any]],
    stop_event: threading.Event,
    poll_seconds: float,
) -> None:
    """Poll `nvidia-smi` until asked to stop and append each sample."""

    while not stop_event.is_set():
        snapshot = query_gpu_state()
        if snapshot is not None:
            samples.append(snapshot)
        stop_event.wait(max(poll_seconds, 0.1))


def profile_experiment(
    experiment: Experiment,
    *,
    warmup_steps: int,
    measured_steps: int,
    output_dir: Path,
    gpu_poll_seconds: float,
) -> dict[str, Any]:
    """Run the short profiling loop and return a structured summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_metrics_path = output_dir / "profile_metrics.jsonl"
    gpu_samples_path = output_dir / "gpu_samples.jsonl"
    summary_path = output_dir / "summary.json"
    if profile_metrics_path.exists():
        profile_metrics_path.unlink()
    if gpu_samples_path.exists():
        gpu_samples_path.unlink()

    total_steps = max(int(warmup_steps), 0) + max(int(measured_steps), 0)
    iterator_started_at = time.perf_counter()
    iterator = iter(experiment.train_loader)
    iterator_init_s = time.perf_counter() - iterator_started_at
    synchronize_if_cuda(experiment.device)
    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    gpu_samples: list[dict[str, Any]] = []
    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None
    if experiment.device.type == "cuda":
        monitor_thread = threading.Thread(
            target=sample_gpu_state,
            kwargs={
                "samples": gpu_samples,
                "stop_event": stop_event,
                "poll_seconds": gpu_poll_seconds,
            },
            daemon=True,
        )
        monitor_thread.start()

    per_step: list[dict[str, Any]] = []
    max_allocated = 0.0
    max_reserved = 0.0
    try:
        for profile_index in range(total_steps):
            phase = "warmup" if profile_index < warmup_steps else "steady_state"
            step_wall_started_at = time.perf_counter()
            batch: dict[str, Any] | None = None

            fetch_started_at = time.perf_counter()
            experiment.model.train()
            fetch_s = 0.0
            copy_s = 0.0
            if experiment._gradient_accumulation_steps() == 1:
                batch = next(iterator)
                fetch_s = time.perf_counter() - fetch_started_at

                synchronize_if_cuda(experiment.device)
                copy_started_at = time.perf_counter()
                batch = experiment._move_batch_to_device(batch)
                synchronize_if_cuda(experiment.device)
                copy_s = time.perf_counter() - copy_started_at

                synchronize_if_cuda(experiment.device)
                train_started_at = time.perf_counter()
                loss_dict = experiment._execute_training_step(batch)
                synchronize_if_cuda(experiment.device)
                train_s = time.perf_counter() - train_started_at
            else:
                fetch_s = time.perf_counter() - fetch_started_at
                synchronize_if_cuda(experiment.device)
                train_started_at = time.perf_counter()
                loss_dict, iterator = experiment._execute_accumulated_training_step(iterator)
                synchronize_if_cuda(experiment.device)
                train_s = time.perf_counter() - train_started_at

            experiment.current_step += 1
            metric_record = {
                "step": int(experiment.current_step),
                "loss": float(loss_dict["loss"].detach().cpu()),
            }
            for key, value in loss_dict.items():
                if key == "loss":
                    continue
                metric_record[key] = float(value.detach().cpu())

            record = {
                "profile_index": profile_index,
                "phase": phase,
                "step": int(experiment.current_step),
                "fetch_s": fetch_s,
                "copy_s": copy_s,
                "train_s": train_s,
            }
            append_started_at = time.perf_counter()
            append_jsonl(profile_metrics_path, record | {"metric_record": metric_record})
            append_s = time.perf_counter() - append_started_at
            record["append_s"] = append_s
            record["total_s"] = time.perf_counter() - step_wall_started_at

            if experiment.device.type == "cuda":
                max_allocated = max(max_allocated, float(torch.cuda.max_memory_allocated()))
                max_reserved = max(max_reserved, float(torch.cuda.max_memory_reserved()))
            per_step.append(record)

            del loss_dict
            if batch is not None:
                del batch
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=max(gpu_poll_seconds * 4.0, 1.0))
        if gpu_samples:
            for sample in gpu_samples:
                append_jsonl(gpu_samples_path, sample)
        shutdown_iterator = getattr(experiment, "_shutdown_dataloader_iterator", None)
        if callable(shutdown_iterator):
            shutdown_iterator(iterator)

    warmup_records = [record for record in per_step if record["phase"] == "warmup"]
    steady_records = [record for record in per_step if record["phase"] == "steady_state"]

    def mean_or_none(key: str, records: list[dict[str, Any]]) -> float | None:
        """Return the arithmetic mean of one numeric key when records exist."""

        if not records:
            return None
        return float(statistics.fmean(float(record[key]) for record in records))

    summary: dict[str, Any] = {
        "iterator_init_s": iterator_init_s,
        "warmup": warmup_records[0] if warmup_records else None,
        "steady_state": {
            "fetch_s": mean_or_none("fetch_s", steady_records),
            "copy_s": mean_or_none("copy_s", steady_records),
            "train_s": mean_or_none("train_s", steady_records),
            "append_s": mean_or_none("append_s", steady_records),
            "total_s": mean_or_none("total_s", steady_records),
            "count": len(steady_records),
        },
        "peak_cuda_allocated_gib": max_allocated / (1024**3) if max_allocated else 0.0,
        "peak_cuda_reserved_gib": max_reserved / (1024**3) if max_reserved else 0.0,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "autocast_dtype": (
                str(experiment.training_autocast_dtype).replace("torch.", "")
                if experiment.training_autocast_dtype is not None
                else None
            ),
        },
        "gpu": summarize_gpu_samples(gpu_samples),
        "output_paths": {
            "profile_metrics": str(profile_metrics_path),
            "gpu_samples": str(gpu_samples_path),
            "summary": str(summary_path),
        },
    }
    save_json(summary_path, summary)
    return summary


def summarize_gpu_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compress raw `nvidia-smi` snapshots into a small readable summary."""

    if not samples:
        return {"samples": 0}
    driver_models = sorted(
        {
            str(sample.get("driver_model.current"))
            for sample in samples
            if sample.get("driver_model.current") is not None
        }
    )
    pstates = sorted(
        {
            str(sample.get("pstate"))
            for sample in samples
            if sample.get("pstate") is not None
        }
    )
    power_draws = [
        float(sample["power.draw"])
        for sample in samples
        if isinstance(sample.get("power.draw"), (int, float))
    ]
    sm_clocks = [
        float(sample["clocks.current.sm"])
        for sample in samples
        if isinstance(sample.get("clocks.current.sm"), (int, float))
    ]
    max_sm_clocks = [
        float(sample["clocks.max.sm"])
        for sample in samples
        if isinstance(sample.get("clocks.max.sm"), (int, float))
    ]
    gpu_utils = [
        float(sample["utilization.gpu"])
        for sample in samples
        if isinstance(sample.get("utilization.gpu"), (int, float))
    ]
    memory_used = [
        float(sample["memory.used"])
        for sample in samples
        if isinstance(sample.get("memory.used"), (int, float))
    ]
    summary: dict[str, Any] = {
        "samples": len(samples),
        "driver_models": driver_models,
        "pstates": pstates,
    }
    if power_draws:
        summary["power_draw_w"] = {
            "min": min(power_draws),
            "mean": float(statistics.fmean(power_draws)),
            "max": max(power_draws),
        }
    if sm_clocks:
        summary["sm_clock_mhz"] = {
            "min": min(sm_clocks),
            "mean": float(statistics.fmean(sm_clocks)),
            "max": max(sm_clocks),
        }
    if max_sm_clocks:
        summary["sm_clock_max_mhz"] = max(max_sm_clocks)
    if gpu_utils:
        summary["gpu_util_percent"] = {
            "min": min(gpu_utils),
            "mean": float(statistics.fmean(gpu_utils)),
            "max": max(gpu_utils),
        }
    if memory_used:
        summary["memory_used_mib"] = {
            "min": min(memory_used),
            "mean": float(statistics.fmean(memory_used)),
            "max": max(memory_used),
        }
    summary["first_sample"] = samples[0]
    summary["last_sample"] = samples[-1]
    return summary


def main() -> None:
    """Run the resumed-step profiler and print the final summary JSON."""

    args = parse_args()
    config = build_config(args)
    experiment = Experiment(config)
    output_dir = Path(args.output_dir) / args.output_subdir
    save_json(output_dir / "requested_config.json", config.to_dict())
    summary = profile_experiment(
        experiment,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        output_dir=output_dir,
        gpu_poll_seconds=args.gpu_poll_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
