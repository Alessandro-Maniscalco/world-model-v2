"""Checkpoint helpers shared by training and inference entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a log file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    step: int,
    config: dict[str, Any],
    normalization_stats: dict[str, Any],
) -> None:
    """Save a training checkpoint with metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": config,
        "normalization_stats": normalization_stats,
    }
    torch.save(payload, output_path)


def load_checkpoint(path: str | Path, device: torch.device | str) -> dict[str, Any]:
    """Load a saved checkpoint onto a target device."""

    return torch.load(Path(path), map_location=device, weights_only=False)
