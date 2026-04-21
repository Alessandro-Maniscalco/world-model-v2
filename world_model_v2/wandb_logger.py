"""Optional Weights & Biases logging helpers for world-model training runs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from world_model_v2.utils.checkpointing import save_json


WANDB_RUN_METADATA_FILENAME = "wandb_run.json"


@dataclass(frozen=True)
class WandbResumeMetadata:
    """Describe one persisted W&B run identity for same-directory resumes."""

    entity: str
    project: str
    run_id: str


@dataclass(frozen=True)
class WandbInitOutcome:
    """Describe one resolved W&B initialization result."""

    active_mode: str
    fallback_error: str | None
    run: Any


def _normalize_tag_tuple(tags: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return stripped non-empty W&B tags or `None` when no tags remain."""

    if tags is None:
        return None
    normalized = tuple(tag.strip() for tag in tags if tag.strip())
    return normalized or None


def _import_wandb_module() -> Any:
    """Import and return the optional `wandb` module."""

    return importlib.import_module("wandb")


def _load_wandb_resume_metadata(path: Path) -> WandbResumeMetadata | None:
    """Load one saved W&B run identity from disk when it is well formed."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    entity = payload.get("entity")
    project = payload.get("project")
    run_id = payload.get("run_id")
    if not all(isinstance(value, str) for value in (entity, project, run_id)):
        return None
    if not project or not run_id:
        return None
    return WandbResumeMetadata(
        entity=str(entity),
        project=str(project),
        run_id=str(run_id),
    )


def _save_wandb_resume_metadata(
    path: Path,
    *,
    entity: str,
    project: str,
    run_id: str,
) -> None:
    """Persist one W&B run identity for later same-directory resumes."""

    save_json(
        path,
        {
            "entity": entity,
            "project": project,
            "run_id": run_id,
        },
    )


def _resume_checkpoint_run_dir(resume_checkpoint: str) -> Path | None:
    """Infer the original run directory from one checkpoint path when possible."""

    normalized = resume_checkpoint.strip()
    if not normalized:
        return None
    checkpoint_path = Path(normalized)
    if checkpoint_path.parent.name != "checkpoints":
        return None
    return checkpoint_path.parent.parent


def _resolve_resume_metadata(
    *,
    run_dir: Path,
    resume_checkpoint: str,
) -> WandbResumeMetadata | None:
    """Return saved W&B identity from the current or resumed source run directory."""

    local_metadata = _load_wandb_resume_metadata(run_dir / WANDB_RUN_METADATA_FILENAME)
    if local_metadata is not None:
        return local_metadata
    checkpoint_run_dir = _resume_checkpoint_run_dir(resume_checkpoint)
    if checkpoint_run_dir is None:
        return None
    return _load_wandb_resume_metadata(checkpoint_run_dir / WANDB_RUN_METADATA_FILENAME)


def _coerce_wandb_scalar(value: Any) -> bool | int | float | None:
    """Return one finite scalar value suitable for W&B metric logging."""

    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        resolved = float(value)
        if not math.isfinite(resolved):
            return None
        return resolved
    return None


def _coerce_wandb_summary_value(value: Any) -> bool | int | float | str | list[Any] | None:
    """Return one W&B-summary-safe value or `None` when unsupported."""

    if isinstance(value, str):
        return value
    scalar = _coerce_wandb_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, (list, tuple)):
        resolved_items: list[Any] = []
        for item in value:
            if isinstance(item, str):
                resolved_items.append(item)
                continue
            scalar_item = _coerce_wandb_scalar(item)
            if scalar_item is None:
                return None
            resolved_items.append(scalar_item)
        return resolved_items
    return None


def _wandb_error_types(wandb: Any) -> tuple[type[Exception], ...]:
    """Return the W&B exception types that justify online-to-offline fallback."""

    errors = getattr(wandb, "errors", None)
    if errors is None:
        return ()
    resolved_types: list[type[Exception]] = []
    for name in ("AuthenticationError", "CommError"):
        error_type = getattr(errors, name, None)
        if isinstance(error_type, type) and issubclass(error_type, Exception):
            resolved_types.append(error_type)
    return tuple(resolved_types)


def _is_retryable_online_init_error(wandb: Any, error: Exception) -> bool:
    """Return whether one W&B init failure should fall back to offline mode."""

    known_types = _wandb_error_types(wandb)
    if known_types and isinstance(error, known_types):
        return True
    message = str(error).lower()
    return (
        "401" in message
        or "permission_error" in message
        or "not logged in" in message
        or "authentication" in message
    )


def _format_wandb_init_error(mode: str, error: Exception) -> str:
    """Return a concise user-facing W&B initialization error message."""

    return (
        f"W&B initialization failed in {mode!r} mode: {error}. "
        "Run `wandb login --relogin` to refresh credentials, or retry with "
        "`--wandb-mode offline` to keep training and sync later."
    )


def _initialize_wandb_run(
    wandb: Any,
    *,
    mode: str,
    init_kwargs: dict[str, Any],
) -> WandbInitOutcome:
    """Initialize one W&B run, retrying offline when online auth fails."""

    try:
        return WandbInitOutcome(
            active_mode=mode,
            fallback_error=None,
            run=wandb.init(**init_kwargs),
        )
    except Exception as error:
        if mode != "online" or not _is_retryable_online_init_error(wandb, error):
            raise RuntimeError(_format_wandb_init_error(mode, error)) from error
        fallback_kwargs = dict(init_kwargs)
        fallback_kwargs["mode"] = "offline"
        print(
            json.dumps(
                {
                    "active_mode": "offline",
                    "error": str(error),
                    "event": "wandb_init_fell_back_to_offline",
                    "requested_mode": mode,
                },
                sort_keys=True,
            )
        )
        try:
            run = wandb.init(**fallback_kwargs)
        except Exception as fallback_error:
            raise RuntimeError(_format_wandb_init_error("offline", fallback_error)) from fallback_error
        return WandbInitOutcome(
            active_mode="offline",
            fallback_error=str(error),
            run=run,
        )


class WandbRunLogger:
    """Wrap one active W&B run with scalar-only logging helpers."""

    def __init__(self, run: Any) -> None:
        """Store the active W&B run handle."""

        self._run = run

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        project: str,
        entity: str,
        group: str,
        name: str,
        tags: tuple[str, ...] | None,
        mode: str,
        config: dict[str, Any],
        run_dir: str | Path,
        resume_checkpoint: str = "",
        run_id: str = "",
    ) -> WandbRunLogger | None:
        """Create one W&B run when logging is enabled."""

        if not enabled:
            return None
        wandb = _import_wandb_module()
        resolved_run_dir = Path(run_dir)
        metadata_path = resolved_run_dir / WANDB_RUN_METADATA_FILENAME
        normalized_entity = entity.strip()
        normalized_group = group.strip()
        normalized_tags = _normalize_tag_tuple(tags)
        normalized_run_id = run_id.strip()
        init_kwargs: dict[str, Any] = {
            "config": config,
            "dir": str(resolved_run_dir),
            "mode": mode,
            "name": name,
            "project": project,
        }
        if normalized_entity:
            init_kwargs["entity"] = normalized_entity
        if normalized_group:
            init_kwargs["group"] = normalized_group
        if normalized_tags is not None:
            init_kwargs["tags"] = list(normalized_tags)
        saved_metadata = _resolve_resume_metadata(
            run_dir=resolved_run_dir,
            resume_checkpoint=resume_checkpoint,
        )
        if normalized_run_id:
            init_kwargs["id"] = normalized_run_id
            init_kwargs["resume"] = "allow"
        elif (
            saved_metadata is not None
            and saved_metadata.project == project
            and saved_metadata.entity == normalized_entity
        ):
            init_kwargs["id"] = saved_metadata.run_id
            init_kwargs["resume"] = "allow"
        init_outcome = _initialize_wandb_run(
            wandb,
            mode=mode,
            init_kwargs=init_kwargs,
        )
        run = init_outcome.run
        run_id = getattr(run, "id", None)
        if isinstance(run_id, str) and run_id:
            _save_wandb_resume_metadata(
                metadata_path,
                entity=normalized_entity,
                project=project,
                run_id=run_id,
            )
        logger = cls(run)
        logger.update_summary(
            "run",
            {
                "active_wandb_mode": init_outcome.active_mode,
                "metrics_path": str(resolved_run_dir / "metrics.jsonl"),
                "requested_wandb_mode": mode,
                "resume_checkpoint": resume_checkpoint,
                "run_dir": str(resolved_run_dir),
                "wandb_online_init_error": init_outcome.fallback_error or "",
            },
        )
        return logger

    def log_training_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log one training metrics record at a specific optimizer step."""

        self._log_prefixed_scalars("train", step, metrics)

    def log_validation_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log one validation metrics record and refresh the validation summary."""

        self._log_prefixed_scalars("validation", step, metrics)
        self.update_summary("validation", metrics)

    def log_early_stop_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log one early-stop record and refresh the early-stop summary."""

        self._log_prefixed_scalars("early_stop", step, metrics)
        self.update_summary("early_stop", metrics)

    def update_summary(self, prefix: str, values: dict[str, Any]) -> None:
        """Write supported flat values into the W&B run summary."""

        summary = getattr(self._run, "summary", None)
        if summary is None:
            return
        for key, value in values.items():
            resolved = _coerce_wandb_summary_value(value)
            if resolved is None:
                continue
            summary[f"{prefix}/{key}"] = resolved

    def finish(self) -> None:
        """Close the active W&B run when one is open."""

        finish = getattr(self._run, "finish", None)
        if callable(finish):
            finish()

    def _log_prefixed_scalars(self, prefix: str, step: int, payload: dict[str, Any]) -> None:
        """Log one prefixed scalar payload to W&B."""

        logged_payload: dict[str, bool | int | float] = {}
        for key, value in payload.items():
            if key == "step":
                continue
            resolved = _coerce_wandb_scalar(value)
            if resolved is None:
                continue
            logged_payload[f"{prefix}/{key}"] = resolved
        if not logged_payload:
            return
        self._run.log(logged_payload, step=int(step))
