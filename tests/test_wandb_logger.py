"""Tests for the optional Weights & Biases logging helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from world_model_v2 import wandb_logger
from world_model_v2.wandb_logger import WandbRunLogger


class FakeWandbRun:
    """Capture one fake W&B run's logged metrics and summary state."""

    def __init__(self, run_id: str) -> None:
        """Store one stable fake run identifier."""

        self.id = run_id
        self.log_calls: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.finished = False

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        """Record one fake log payload."""

        self.log_calls.append({"payload": dict(payload), "step": int(step)})

    def finish(self) -> None:
        """Mark the fake run as finished."""

        self.finished = True


class FakeWandbModule:
    """Return fake W&B runs and keep track of init arguments."""

    def __init__(self) -> None:
        """Initialize the fake module state."""

        self.init_calls: list[dict[str, Any]] = []
        self.runs: list[FakeWandbRun] = []
        self.errors = type(
            "FakeErrors",
            (),
            {
                "AuthenticationError": type("AuthenticationError", (Exception,), {}),
                "CommError": type("CommError", (Exception,), {}),
            },
        )

    def init(self, **kwargs: Any) -> FakeWandbRun:
        """Create one fake run for the requested init arguments."""

        self.init_calls.append(dict(kwargs))
        run_id = str(kwargs.get("id") or f"generated-{len(self.runs) + 1}")
        run = FakeWandbRun(run_id)
        self.runs.append(run)
        return run


class FlakyOnlineFakeWandbModule(FakeWandbModule):
    """Fail the first online init so the logger must retry offline."""

    def __init__(self) -> None:
        """Initialize the fake module state and failure counter."""

        super().__init__()
        self.online_failures = 0

    def init(self, **kwargs: Any) -> FakeWandbRun:
        """Raise one retryable online init error before succeeding offline."""

        self.init_calls.append(dict(kwargs))
        if kwargs.get("mode") == "online" and self.online_failures == 0:
            self.online_failures += 1
            raise self.errors.CommError("returned error 401: user is not logged in")
        run_id = str(kwargs.get("id") or f"generated-{len(self.runs) + 1}")
        run = FakeWandbRun(run_id)
        self.runs.append(run)
        return run


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch, fake_module: FakeWandbModule) -> None:
    """Replace the lazy `wandb` import with one fake module."""

    monkeypatch.setattr(
        wandb_logger,
        "_import_wandb_module",
        lambda: fake_module,
    )


def test_wandb_run_logger_logs_prefixed_scalars_and_updates_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper should log prefixed scalar metrics and track summary fields."""

    fake_module = FakeWandbModule()
    _install_fake_wandb(monkeypatch, fake_module)

    logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="",
        group="debug",
        name="ae-smoke",
        tags=("ae", "smoke"),
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=tmp_path / "run",
        resume_checkpoint="",
    )

    assert logger is not None
    logger.log_training_metrics(
        3,
        {
            "elapsed_run_seconds": 0.5,
            "loss": 1.25,
            "note": "skip-me",
            "non_finite": float("nan"),
        },
    )
    logger.log_validation_metrics(
        3,
        {
            "ae_loss": 0.4,
            "is_best_checkpoint": True,
            "validation_episodes": [0, 1],
            "validation_style": "teacher_forced",
        },
    )
    logger.finish()

    assert fake_module.init_calls == [
        {
            "config": {"mode": "ae_only"},
            "dir": str(tmp_path / "run"),
            "group": "debug",
            "mode": "offline",
            "name": "ae-smoke",
            "project": "wm-v2",
            "tags": ["ae", "smoke"],
        }
    ]
    assert fake_module.runs[0].log_calls == [
        {
            "payload": {
                "train/elapsed_run_seconds": 0.5,
                "train/loss": 1.25,
            },
            "step": 3,
        },
        {
            "payload": {
                "validation/ae_loss": 0.4,
                "validation/is_best_checkpoint": True,
            },
            "step": 3,
        },
    ]
    assert fake_module.runs[0].summary["run/run_dir"] == str(tmp_path / "run")
    assert fake_module.runs[0].summary["validation/validation_style"] == "teacher_forced"
    assert fake_module.runs[0].summary["validation/validation_episodes"] == [0, 1]
    assert fake_module.runs[0].finished is True
    assert (tmp_path / "run" / "wandb_run.json").exists()


def test_wandb_run_logger_reuses_saved_run_id_for_same_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper should resume the saved W&B run ID inside the same run directory."""

    fake_module = FakeWandbModule()
    _install_fake_wandb(monkeypatch, fake_module)
    run_dir = tmp_path / "shared_run"

    first_logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="team",
        group="",
        name="resume-source",
        tags=None,
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=run_dir,
        resume_checkpoint="",
    )
    assert first_logger is not None
    first_logger.finish()

    second_logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="team",
        group="",
        name="resume-target",
        tags=None,
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=run_dir,
        resume_checkpoint="checkpoint.pt",
    )

    assert second_logger is not None
    assert fake_module.init_calls[1]["id"] == fake_module.runs[0].id
    assert fake_module.init_calls[1]["resume"] == "allow"


def test_wandb_run_logger_reuses_checkpoint_source_run_id_for_new_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resumes should reuse the source checkpoint run ID even in a fresh output directory."""

    fake_module = FakeWandbModule()
    _install_fake_wandb(monkeypatch, fake_module)
    source_run_dir = tmp_path / "source_run"
    source_checkpoint = source_run_dir / "checkpoints" / "last.pt"
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint.write_bytes(b"checkpoint")

    first_logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="team",
        group="",
        name="resume-source",
        tags=None,
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=source_run_dir,
        resume_checkpoint="",
    )
    assert first_logger is not None
    first_logger.finish()

    resumed_run_dir = tmp_path / "resumed_elsewhere"
    resumed_logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="team",
        group="",
        name="resume-target",
        tags=None,
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=resumed_run_dir,
        resume_checkpoint=str(source_checkpoint),
    )

    assert resumed_logger is not None
    assert fake_module.init_calls[1]["id"] == fake_module.runs[0].id
    assert fake_module.init_calls[1]["resume"] == "allow"
    saved_resume_metadata = (resumed_run_dir / "wandb_run.json").read_text(encoding="utf-8")
    assert fake_module.runs[0].id in saved_resume_metadata


def test_wandb_run_logger_explicit_run_id_overrides_saved_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit run ID should take priority over any saved metadata."""

    fake_module = FakeWandbModule()
    _install_fake_wandb(monkeypatch, fake_module)
    run_dir = tmp_path / "explicit_override"

    first_logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="team",
        group="",
        name="first",
        tags=None,
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=run_dir,
        resume_checkpoint="",
    )
    assert first_logger is not None
    first_logger.finish()

    override_logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="team",
        group="",
        name="override",
        tags=None,
        mode="offline",
        config={"mode": "ae_only"},
        run_dir=run_dir,
        resume_checkpoint="",
        run_id="manual-run-id",
    )

    assert override_logger is not None
    assert fake_module.init_calls[1]["id"] == "manual-run-id"
    assert fake_module.init_calls[1]["resume"] == "allow"


def test_wandb_run_logger_falls_back_to_offline_after_online_auth_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retryable online init failures should fall back to offline logging."""

    fake_module = FlakyOnlineFakeWandbModule()
    _install_fake_wandb(monkeypatch, fake_module)

    logger = WandbRunLogger.create(
        enabled=True,
        project="wm-v2",
        entity="",
        group="debug",
        name="online-then-offline",
        tags=("ae",),
        mode="online",
        config={"mode": "ae_only"},
        run_dir=tmp_path / "fallback_run",
        resume_checkpoint="",
    )

    assert logger is not None
    captured = capsys.readouterr()
    assert '"event": "wandb_init_fell_back_to_offline"' in captured.out
    assert fake_module.init_calls[0]["mode"] == "online"
    assert fake_module.init_calls[1]["mode"] == "offline"
    assert fake_module.runs[0].summary["run/requested_wandb_mode"] == "online"
    assert fake_module.runs[0].summary["run/active_wandb_mode"] == "offline"
    assert "not logged in" in fake_module.runs[0].summary["run/wandb_online_init_error"]
