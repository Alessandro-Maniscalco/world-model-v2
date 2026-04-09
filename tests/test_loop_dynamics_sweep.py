"""Tests for the dynamics sweep helper script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def _load_loop_dynamics_sweep_module() -> object:
    """Load the sweep helper module directly from its script path."""

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check" / "loop_dynamics_sweep.py"
    spec = importlib.util.spec_from_file_location("loop_dynamics_sweep", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loop_dynamics_sweep = _load_loop_dynamics_sweep_module()


def test_extract_selection_metric_uses_worst_case_next_frame_mse() -> None:
    """The default sweep ranking should penalize the weaker teacher-forced horizon."""

    score = loop_dynamics_sweep._extract_selection_metric(
        {"next_frame_mse": 0.001, "next_frame_mse_4to1": 0.004},
        loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
    )
    assert score == 0.004


def test_extract_selection_metric_can_use_open_rollout_frame_mse() -> None:
    """Sweep ranking should optionally optimize the open-rollout validation metric."""

    score = loop_dynamics_sweep._extract_selection_metric(
        {"next_frame_mse": 0.001, "open_rollout_frame_mse": 0.02},
        "open_rollout_frame_mse",
    )
    assert score == 0.02


def test_extract_selection_metric_can_use_eval_open_rollout_frame_mse() -> None:
    """Sweep ranking should optionally use a fixed external open-rollout score."""

    score = loop_dynamics_sweep._extract_selection_metric(
        {"eval_open_rollout_frame_mse": 0.015},
        loop_dynamics_sweep.EVAL_OPEN_ROLLOUT_SELECTION_METRIC,
    )
    assert score == 0.015


def test_extract_selection_metric_can_use_eval_open_rollout_frame_mse_mean() -> None:
    """Sweep ranking should optionally use the aggregated eval open-rollout mean."""

    score = loop_dynamics_sweep._extract_selection_metric(
        {"eval_open_rollout_frame_mse_mean": 0.012},
        loop_dynamics_sweep.EVAL_OPEN_ROLLOUT_MEAN_SELECTION_METRIC,
    )
    assert score == 0.012


def test_extract_selection_metric_can_use_eval_open_rollout_frame_mse_max() -> None:
    """Sweep ranking should optionally use the aggregated eval open-rollout max."""

    score = loop_dynamics_sweep._extract_selection_metric(
        {"eval_open_rollout_frame_mse_max": 0.019},
        loop_dynamics_sweep.EVAL_OPEN_ROLLOUT_MAX_SELECTION_METRIC,
    )
    assert score == 0.019


def test_maybe_update_best_run_replaces_only_with_better_success() -> None:
    """Warm-start selection should keep the best successful checkpoint so far."""

    current_best = loop_dynamics_sweep.maybe_update_best_run(
        current_best=None,
        evaluation={
            "passed": True,
            "best_checkpoint_exists": True,
            "best_checkpoint": "/tmp/run_a/checkpoints/best.pt",
            "run_dir": "/tmp/run_a",
            "next_frame_mse": 0.005,
            "next_frame_mse_4to1": 0.004,
        },
        selection_metric=loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
    )
    assert current_best is not None
    assert current_best["run_name"] == "run_a"
    assert current_best["selection_score"] == 0.005

    kept_best = loop_dynamics_sweep.maybe_update_best_run(
        current_best=current_best,
        evaluation={
            "passed": True,
            "best_checkpoint_exists": True,
            "best_checkpoint": "/tmp/run_b/checkpoints/best.pt",
            "run_dir": "/tmp/run_b",
            "next_frame_mse": 0.006,
            "next_frame_mse_4to1": 0.003,
        },
        selection_metric=loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
    )
    assert kept_best == current_best

    updated_best = loop_dynamics_sweep.maybe_update_best_run(
        current_best=current_best,
        evaluation={
            "passed": True,
            "best_checkpoint_exists": True,
            "best_checkpoint": "/tmp/run_c/checkpoints/best.pt",
            "run_dir": "/tmp/run_c",
            "next_frame_mse": 0.002,
            "next_frame_mse_4to1": 0.0025,
        },
        selection_metric=loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
    )
    assert updated_best is not None
    assert updated_best["run_name"] == "run_c"
    assert updated_best["selection_score"] == 0.0025


def test_maybe_update_best_run_ignores_failed_or_missing_checkpoint_runs() -> None:
    """Failed evaluations should never replace the current best checkpoint."""

    best_run = {
        "selection_metric": loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
        "selection_score": 0.001,
        "best_checkpoint": "/tmp/run_a/checkpoints/best.pt",
        "run_dir": "/tmp/run_a",
        "run_name": "run_a",
    }
    assert (
        loop_dynamics_sweep.maybe_update_best_run(
            current_best=best_run,
            evaluation={
                "passed": False,
                "best_checkpoint_exists": True,
                "best_checkpoint": "/tmp/run_b/checkpoints/best.pt",
                "run_dir": "/tmp/run_b",
                "next_frame_mse": 0.0,
            },
            selection_metric=loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
        )
        == best_run
    )
    assert (
        loop_dynamics_sweep.maybe_update_best_run(
            current_best=best_run,
            evaluation={
                "passed": True,
                "best_checkpoint_exists": False,
                "best_checkpoint": "/tmp/run_c/checkpoints/best.pt",
                "run_dir": "/tmp/run_c",
                "next_frame_mse": 0.0,
            },
            selection_metric=loop_dynamics_sweep.DEFAULT_SELECTION_METRIC,
        )
        == best_run
    )


def test_parse_args_falls_back_to_legacy_default_vae_checkpoint() -> None:
    """The sweep helper should use the legacy minimal AE checkpoint when the root one is absent."""

    with patch.object(
        loop_dynamics_sweep.Path,
        "exists",
        new=lambda self: self == loop_dynamics_sweep.LEGACY_DEFAULT_VAE_CHECKPOINT,
    ), patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    assert args.vae_checkpoint == str(loop_dynamics_sweep.LEGACY_DEFAULT_VAE_CHECKPOINT)


def test_build_command_preserves_open_rollout_validation_metric() -> None:
    """The sweep helper should forward the requested validation metric to training runs."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_validation_metric = "open_rollout_frame_mse"
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=53,
            frame_end=57,
            infer_steps=50,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-validation-metric" in command
    metric_index = command.index("--dynamics-validation-metric")
    assert command[metric_index + 1] == "open_rollout_frame_mse"


def test_build_command_preserves_conditional_frame_timestep() -> None:
    """The sweep helper should forward conditional-frame timestep explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.conditional_frame_timestep = 0.0
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=50,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--conditional-frame-timestep" in command
    timestep_index = command.index("--conditional-frame-timestep")
    assert command[timestep_index + 1] == "0.0"


def test_build_command_preserves_conditional_frame_sigma() -> None:
    """The sweep helper should forward conditional-frame sigma explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.conditional_frame_sigma = 1e-4
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--conditional-frame-sigma" in command
    sigma_index = command.index("--conditional-frame-sigma")
    assert command[sigma_index + 1] == "0.0001"


def test_build_command_preserves_self_forcing_loss_weight() -> None:
    """The sweep helper should forward the causal self-forcing weight explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_self_forcing_loss_weight = 0.35
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-self-forcing-loss-weight" in command
    weight_index = command.index("--dynamics-self-forcing-loss-weight")
    assert command[weight_index + 1] == "0.35"


def test_build_command_preserves_rollout_self_forcing_loss_weight() -> None:
    """The sweep helper should forward the rollout self-forcing auxiliary weight explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_rollout_self_forcing_loss_weight = 0.2
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-rollout-self-forcing-loss-weight" in command
    weight_index = command.index("--dynamics-rollout-self-forcing-loss-weight")
    assert command[weight_index + 1] == "0.2"


def test_build_command_preserves_self_forcing_mode() -> None:
    """The sweep helper should forward the selected self-forcing mode explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_self_forcing_mode = "rollout"
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-self-forcing-mode" in command
    mode_index = command.index("--dynamics-self-forcing-mode")
    assert command[mode_index + 1] == "rollout"


def test_build_command_preserves_learned_temporal_embedding_flag() -> None:
    """The sweep helper should forward the learned temporal embedding flag explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_use_learned_temporal_embedding = True
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-use-learned-temporal-embedding" in command


def test_build_command_preserves_self_forcing_warmup_steps() -> None:
    """The sweep helper should forward self-forcing warmup steps explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_self_forcing_warmup_steps = 125
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-self-forcing-warmup-steps" in command
    warmup_index = command.index("--dynamics-self-forcing-warmup-steps")
    assert command[warmup_index + 1] == "125"


def test_build_command_preserves_self_forcing_ramp_steps() -> None:
    """The sweep helper should forward self-forcing ramp steps explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_self_forcing_ramp_steps = 200
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-self-forcing-ramp-steps" in command
    ramp_index = command.index("--dynamics-self-forcing-ramp-steps")
    assert command[ramp_index + 1] == "200"


def test_build_command_preserves_rollout_self_forcing_warmup_steps() -> None:
    """The sweep helper should forward rollout self-forcing warmup steps explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_rollout_self_forcing_warmup_steps = 50
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-rollout-self-forcing-warmup-steps" in command
    warmup_index = command.index("--dynamics-rollout-self-forcing-warmup-steps")
    assert command[warmup_index + 1] == "50"


def test_build_command_preserves_rollout_self_forcing_ramp_steps() -> None:
    """The sweep helper should forward rollout self-forcing ramp steps explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_rollout_self_forcing_ramp_steps = 100
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-rollout-self-forcing-ramp-steps" in command
    ramp_index = command.index("--dynamics-rollout-self-forcing-ramp-steps")
    assert command[ramp_index + 1] == "100"


def test_build_command_preserves_self_forcing_rollout_chunks() -> None:
    """The sweep helper should forward rollout self-forcing chunk counts explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_self_forcing_rollout_chunks = 2
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-self-forcing-rollout-chunks" in command
    chunks_index = command.index("--dynamics-self-forcing-rollout-chunks")
    assert command[chunks_index + 1] == "2"


def test_build_command_preserves_open_rollout_stride_frames() -> None:
    """The sweep helper should forward open-rollout stride frames explicitly."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_open_rollout_stride_frames = 1
    command = loop_dynamics_sweep.build_command(
        args=args,
        spec=loop_dynamics_sweep.SweepSpec(
            frame_start=48,
            frame_end=67,
            infer_steps=32,
            batch_size=1,
            max_steps=10,
            lr=1e-4,
        ),
        run_name="smoke",
        load_dynamics=None,
    )
    assert "--dynamics-open-rollout-stride-frames" in command
    stride_index = command.index("--dynamics-open-rollout-stride-frames")
    assert command[stride_index + 1] == "1"


def test_build_open_rollout_eval_command_uses_fixed_eval_span() -> None:
    """The sweep helper should build a fixed-span rollout-eval command when requested."""

    with patch.object(
        sys,
        "argv",
        ["loop_dynamics_sweep.py", "--eval-open-rollout-frame-span", "48:67"],
    ):
        args = loop_dynamics_sweep.parse_args()
    command = loop_dynamics_sweep.build_open_rollout_eval_command(
        args,
        checkpoint_path=Path("/tmp/best.pt"),
        output_dir=Path("/tmp/eval"),
        frame_start=48,
        frame_end=67,
    )
    assert command is not None
    assert "scripts/check/open_rollout_demo.py" in command
    assert "--frame-start" in command
    assert "--frame-end" in command
    assert command[command.index("--frame-start") + 1] == "48"
    assert command[command.index("--frame-end") + 1] == "67"


def test_build_open_rollout_eval_command_preserves_overlap_stride() -> None:
    """The rollout-eval command should preserve the configured overlap stride."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py"]):
        args = loop_dynamics_sweep.parse_args()
    args.dynamics_open_rollout_stride_frames = 1
    command = loop_dynamics_sweep.build_open_rollout_eval_command(
        args,
        checkpoint_path=Path("/tmp/best.pt"),
        output_dir=Path("/tmp/eval"),
        frame_start=48,
        frame_end=67,
    )
    assert command is not None
    assert "--dynamics-open-rollout-stride-frames" in command
    stride_index = command.index("--dynamics-open-rollout-stride-frames")
    assert command[stride_index + 1] == "1"


def test_evaluate_open_rollout_aggregates_multiple_spans() -> None:
    """The sweep helper should aggregate multi-span open-rollout metrics."""

    with patch.object(
        sys,
        "argv",
        [
            "loop_dynamics_sweep.py",
            "--eval-open-rollout-frame-span",
            "0:19",
            "--eval-open-rollout-frame-span",
            "48:67",
        ],
    ):
        args = loop_dynamics_sweep.parse_args()
    run_dir = Path("/tmp/fake_run")
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    with patch.object(
        loop_dynamics_sweep.Path,
        "exists",
        new=lambda self: self == best_checkpoint,
    ), patch.object(
        loop_dynamics_sweep.subprocess,
        "run",
        side_effect=[
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "checkpoint": str(best_checkpoint),
                        "device": "cpu",
                        "validation_style": "open_rollout_autoregressive",
                        "open_rollout_frame_mse": 0.02,
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "checkpoint": str(best_checkpoint),
                        "device": "cpu",
                        "validation_style": "open_rollout_autoregressive",
                        "open_rollout_frame_mse": 0.01,
                    }
                ),
                stderr="",
            ),
        ],
    ):
        evaluation = loop_dynamics_sweep.evaluate_open_rollout(
            args=args,
            run_dir=run_dir,
            evaluation={"passed": True, "issues": []},
        )
    assert evaluation["eval_open_rollout_frame_mse"] == 0.015
    assert evaluation["eval_open_rollout_frame_mse_mean"] == 0.015
    assert evaluation["eval_open_rollout_frame_mse_max"] == 0.02
    assert evaluation["eval_open_rollout_f0_19_open_rollout_frame_mse"] == 0.02
    assert evaluation["eval_open_rollout_f48_67_open_rollout_frame_mse"] == 0.01


def test_build_sweep_specs_clamps_metaworld_frame_end_to_episode_length() -> None:
    """The sweep helper should clamp overlong MT50 frame spans before launching runs."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py", "--frame-span", "32:95"]):
        args = loop_dynamics_sweep.parse_args()
    with patch.object(loop_dynamics_sweep, "resolve_max_valid_frame_end", return_value=87):
        specs = loop_dynamics_sweep.build_sweep_specs(args)
    assert len(specs) == 1
    assert specs[0].frame_start == 32
    assert specs[0].frame_end == 87


def test_build_sweep_specs_skips_metaworld_spans_starting_past_episode_end() -> None:
    """The sweep helper should skip spans whose start already exceeds episode length."""

    with patch.object(sys, "argv", ["loop_dynamics_sweep.py", "--frame-span", "96:159"]):
        args = loop_dynamics_sweep.parse_args()
    with patch.object(loop_dynamics_sweep, "resolve_max_valid_frame_end", return_value=87):
        specs = loop_dynamics_sweep.build_sweep_specs(args)
    assert specs == []


def test_loop_dynamics_sweep_script_bootstraps_repo_imports() -> None:
    """The sweep script should run as a script path without import failures."""

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check/loop_dynamics_sweep.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "loop_dynamics_sweep.py" in result.stdout
