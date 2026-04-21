"""Benchmark one short RF-DiT throughput step across candidate batch sizes.

This smoke-check helper loads a real `dynamics_only` experiment, runs a single
measured training step for each requested batch size, and reports the fastest
throughput snapshot without relying on `--auto-batch-size`.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.experiment import Experiment, ExperimentConfig
from world_model_v2.utils.checkpointing import append_jsonl, save_json


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_OUTPUT_SUBDIR = "temp_dynamics_batch_benchmark"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "world_model_ae_only_20260418_183119" / "checkpoints" / "best.pt"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "so101_base_sim_pickplace_cache"
DEFAULT_BATCH_SIZES = "1,2,4,8,12,16,20,24,28,32,40,48,64"
DEFAULT_GRAD_ACCUM_STEPS = "1"
DEFAULT_WARMUP_STEPS = 0
DEFAULT_MEASURED_STEPS = 1


def parse_positive_int_list(value: str, *, name: str) -> tuple[int, ...]:
    """Parse one comma-separated positive-integer list."""

    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one value for {name}.")
    if any(item < 1 for item in values):
        raise ValueError(f"{name} must contain only positive integers.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the dynamics batch-size benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-encoder-decoder", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-accum-steps-list", default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--dataloader-num-workers", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=DEFAULT_MEASURED_STEPS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--stop-after-oom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-format", default="lerobot_so101_base_sim_pickplace")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--train-all-episodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-split", default="train")
    parser.add_argument("--validation-episode", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=208)
    parser.add_argument("--height", type=int, default=208)
    parser.add_argument("--width", type=int, default=276)
    parser.add_argument("--wan-dim", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--wan-num-res-blocks", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--early-stop-patience-windows", type=int, default=20)
    parser.add_argument("--dynamics-context-frames", type=int, default=1)
    parser.add_argument("--dynamics-target-frames", type=int, default=3)
    parser.add_argument("--dynamics-patch-spatial", type=int, default=1)
    parser.add_argument("--dynamics-model-channels", type=int, default=256)
    parser.add_argument("--dynamics-num-blocks", type=int, default=4)
    parser.add_argument("--dynamics-num-heads", type=int, default=4)
    parser.add_argument("--dynamics-action-conditioning-mode", default="chunk_per_frame")
    parser.add_argument("--dynamics-action-representation", default="dataset_default")
    parser.add_argument("--dynamics-action-scale", type=float, default=20.0)
    parser.add_argument("--dynamics-adaln-lora-dim", type=int, default=64)
    parser.add_argument("--dynamics-infer-steps", type=int, default=16)
    parser.add_argument("--dynamics-train-timesteps", type=int, default=1000)
    parser.add_argument("--dynamics-rf-shift", type=float, default=5.0)
    parser.add_argument("--dynamics-validation-metric", default="next_frame_mse")
    parser.add_argument(
        "--dynamics-run-open-rollout-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, *, batch_size: int, run_name: str) -> ExperimentConfig:
    """Build one short `dynamics_only` config for a single batch size."""

    return ExperimentConfig(
        mode="dynamics_only",
        dataset_format=str(args.dataset_format),
        data_root=str(Path(args.data_root)),
        task=str(args.task),
        split=str(args.split),
        episode=int(args.episode),
        train_all_episodes=bool(args.train_all_episodes),
        validation_split=str(args.validation_split),
        validation_episode=int(args.validation_episode),
        resolution=int(args.resolution),
        height=int(args.height),
        width=int(args.width),
        wan_dim=int(args.wan_dim),
        latent_channels=int(args.latent_channels),
        wan_num_res_blocks=int(args.wan_num_res_blocks),
        hidden_channels=int(args.hidden_channels),
        batch_size=max(int(batch_size), 1),
        gradient_accumulation_steps=max(int(args.grad_accum_steps), 1),
        dataloader_num_workers=max(int(args.dataloader_num_workers), 0),
        lr=float(args.lr),
        lr_warmup_steps=max(int(args.lr_warmup_steps), 0),
        max_steps=int(args.max_steps),
        validation_interval=int(args.validation_interval),
        checkpoint_interval=int(args.checkpoint_interval),
        log_interval=int(args.log_interval),
        early_stop_patience_windows=int(args.early_stop_patience_windows),
        dynamics_context_frames=int(args.dynamics_context_frames),
        dynamics_target_frames=int(args.dynamics_target_frames),
        dynamics_patch_spatial=int(args.dynamics_patch_spatial),
        dynamics_model_channels=int(args.dynamics_model_channels),
        dynamics_num_blocks=int(args.dynamics_num_blocks),
        dynamics_num_heads=int(args.dynamics_num_heads),
        dynamics_action_conditioning_mode=str(args.dynamics_action_conditioning_mode),
        dynamics_action_representation=str(args.dynamics_action_representation),
        dynamics_action_scale=float(args.dynamics_action_scale),
        dynamics_adaln_lora_dim=int(args.dynamics_adaln_lora_dim),
        dynamics_infer_steps=int(args.dynamics_infer_steps),
        dynamics_train_timesteps=int(args.dynamics_train_timesteps),
        dynamics_rf_shift=float(args.dynamics_rf_shift),
        dynamics_validation_metric=str(args.dynamics_validation_metric),
        dynamics_run_open_rollout_validation=bool(args.dynamics_run_open_rollout_validation),
        load_encoder_decoder=str(Path(args.load_encoder_decoder)),
        run_name=run_name,
        output_dir=str(Path(args.output_dir)),
        seed=int(args.seed),
        device=str(args.device),
    )


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize the active CUDA stream when needed."""

    if device.type == "cuda":
        torch.cuda.synchronize()


def mean_or_none(records: list[dict[str, float]], key: str) -> float | None:
    """Return the arithmetic mean for one record field."""

    if not records:
        return None
    return float(statistics.fmean(float(record[key]) for record in records))


def clear_cuda_memory() -> None:
    """Release as much cached CUDA state as possible between runs."""

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def is_oom(error: RuntimeError) -> bool:
    """Return whether the runtime error looks like a CUDA OOM."""

    return "out of memory" in str(error).lower()


def benchmark_one_batch_size(
    args: argparse.Namespace,
    *,
    batch_size: int,
    grad_accum_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Benchmark one candidate batch size and return a structured result."""

    run_name = f"{args.output_subdir}_bs{batch_size}_ga{grad_accum_steps}"
    config = build_config(args, batch_size=batch_size, run_name=run_name)
    config.gradient_accumulation_steps = max(int(grad_accum_steps), 1)
    experiment: Experiment | None = None
    iterator: Any = None
    profile_records: list[dict[str, float]] = []
    timings_path = output_dir / f"batch_size_{batch_size}_grad_accum_{grad_accum_steps}.jsonl"
    try:
        experiment = Experiment(config)
        if experiment.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        iterator = iter(experiment.train_loader)
        total_steps = max(int(args.warmup_steps), 0) + max(int(args.measured_steps), 0)
        for profile_index in range(total_steps):
            phase = "warmup" if profile_index < int(args.warmup_steps) else "steady_state"
            batch: dict[str, Any] | None = None
            step_started_at = time.perf_counter()
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
            record = {
                "batch_size": float(batch_size),
                "copy_s": float(copy_s),
                "fetch_s": float(fetch_s),
                "grad_accum_steps": float(grad_accum_steps),
                "loss": float(loss_dict["loss"].detach().cpu()),
                "phase": phase,
                "profile_index": float(profile_index),
                "step": float(experiment.current_step),
                "total_s": float(time.perf_counter() - step_started_at),
                "train_s": float(train_s),
            }
            append_jsonl(timings_path, record)
            profile_records.append(record)
            del loss_dict
            if batch is not None:
                del batch
        steady_state_records = [
            record for record in profile_records if record["phase"] == "steady_state"
        ]
        steady_total_s = mean_or_none(steady_state_records, "total_s")
        effective_batch_size = int(experiment._effective_train_batch_size())
        layout = config.dynamics_frame_layout()
        samples_per_second = (
            None if steady_total_s in {None, 0.0} else effective_batch_size / float(steady_total_s)
        )
        result = {
            "batch_size": int(batch_size),
            "effective_batch_size": effective_batch_size,
            "grad_accum_steps": int(grad_accum_steps),
            "status": "ok",
            "steady_state": {
                "count": len(steady_state_records),
                "copy_s": mean_or_none(steady_state_records, "copy_s"),
                "fetch_s": mean_or_none(steady_state_records, "fetch_s"),
                "total_s": steady_total_s,
                "train_s": mean_or_none(steady_state_records, "train_s"),
            },
            "samples_per_second": samples_per_second,
            "latent_frames_per_second": (
                None if samples_per_second is None else samples_per_second * layout.max_frames
            ),
            "pixel_frames_per_second": (
                None
                if samples_per_second is None
                else samples_per_second * layout.max_pixel_frames
            ),
            "peak_cuda_allocated_gib": (
                float(torch.cuda.max_memory_allocated()) / (1024**3)
                if experiment.device.type == "cuda"
                else 0.0
            ),
            "peak_cuda_reserved_gib": (
                float(torch.cuda.max_memory_reserved()) / (1024**3)
                if experiment.device.type == "cuda"
                else 0.0
            ),
            "timings_path": str(timings_path),
        }
        return result
    except RuntimeError as error:
        if not is_oom(error):
            raise
        if experiment is not None:
            try:
                experiment._cleanup_after_cuda_oom()
            except RuntimeError:
                pass
        return {
            "batch_size": int(batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "status": "oom",
            "error": str(error),
            "timings_path": str(timings_path),
        }
    finally:
        if experiment is not None and iterator is not None:
            shutdown_iterator = getattr(experiment, "_shutdown_dataloader_iterator", None)
            if callable(shutdown_iterator):
                try:
                    shutdown_iterator(iterator)
                except RuntimeError:
                    pass
        del iterator
        del experiment
        clear_cuda_memory()


def main() -> None:
    """Run the full batch-size sweep and print a compact JSON summary."""

    args = parse_args()
    batch_sizes = parse_positive_int_list(str(args.batch_sizes), name="batch_sizes")
    grad_accum_steps_list = parse_positive_int_list(
        str(args.grad_accum_steps_list),
        name="grad_accum_steps_list",
    )
    output_dir = Path(args.output_dir) / str(args.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "requested_args.json", vars(args))
    results: list[dict[str, Any]] = []
    for grad_accum_steps in grad_accum_steps_list:
        for batch_size in batch_sizes:
            result = benchmark_one_batch_size(
                args,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                output_dir=output_dir,
            )
            append_jsonl(output_dir / "results.jsonl", result)
            print(json.dumps(result, sort_keys=True))
            results.append(result)
            if result["status"] == "oom" and bool(args.stop_after_oom):
                break
    successful = [result for result in results if result["status"] == "ok"]
    best_result = max(
        successful,
        key=lambda result: (
            float(result["samples_per_second"]),
            int(result["batch_size"]),
        ),
        default=None,
    )
    summary = {
        "best_result": best_result,
        "results": results,
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
