"""Tests for the standalone dynamics batch-size benchmark helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_benchmark_module() -> object:
    """Load the batch-size benchmark helper directly from its script path."""

    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check"
        / "benchmark_dynamics_batch_size.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_dynamics_batch_size", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_dynamics_batch_size = _load_benchmark_module()


def test_parse_args_defaults_to_one_measured_step_without_warmup() -> None:
    """The default benchmark should stay fast by skipping warmup and extra repeats."""

    args = benchmark_dynamics_batch_size.parse_args([])

    assert args.warmup_steps == 0
    assert args.measured_steps == 1
