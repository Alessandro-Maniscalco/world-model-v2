"""Tests for the checkpoint-triggered training restart helper."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


def _load_restart_helper_module() -> object:
    """Load the restart helper module directly from its script path."""

    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check"
        / "restart_training_after_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("restart_training_after_checkpoint", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


restart_training_after_checkpoint = _load_restart_helper_module()


@pytest.mark.skipif(os.name != "nt", reason="Windows command-line splitting only applies on Windows.")
def test_windows_command_line_to_argv_splits_python_module_command() -> None:
    """Windows argv parsing should preserve quoted executables and module arguments."""

    argv = restart_training_after_checkpoint.windows_command_line_to_argv(
        '"C:\\Users\\aless\\world-model-v2\\.venv\\Scripts\\python.exe" '
        "-m world_model_v2.run --run-name so101"
    )

    assert argv == [
        "C:\\Users\\aless\\world-model-v2\\.venv\\Scripts\\python.exe",
        "-m",
        "world_model_v2.run",
        "--run-name",
        "so101",
    ]


def test_choose_target_process_prefers_top_level_parent() -> None:
    """Auto-selection should choose the top-level run over worker children."""

    root = restart_training_after_checkpoint.RunProcess(
        pid=8040,
        parent_pid=19040,
        command_line="python -m world_model_v2.run",
        argv=("python", "-m", "world_model_v2.run"),
    )
    child = restart_training_after_checkpoint.RunProcess(
        pid=1588,
        parent_pid=8040,
        command_line="python -m world_model_v2.run",
        argv=("python", "-m", "world_model_v2.run"),
    )

    chosen = restart_training_after_checkpoint.choose_target_process([root, child])

    assert chosen == root


def test_resolve_checkpoint_path_prefers_existing_resume_flag(tmp_path: Path) -> None:
    """Existing `--resume` flags should define the watched checkpoint path."""

    checkpoint_path = tmp_path / "outputs" / "run_a" / "checkpoints" / "last.pt"
    argv = (
        "python",
        "-m",
        "world_model_v2.run",
        "--resume",
        str(checkpoint_path),
        "--run-name",
        "ignored_run_name",
    )

    resolved = restart_training_after_checkpoint.resolve_checkpoint_path(argv, tmp_path)

    assert resolved == checkpoint_path.resolve()


def test_resolve_checkpoint_path_falls_back_to_run_name_and_output_dir(tmp_path: Path) -> None:
    """Fresh runs should derive `last.pt` from the configured output directory and run name."""

    argv = (
        "python",
        "-m",
        "world_model_v2.run",
        "--run-name",
        "my_run",
        "--output-dir",
        "outputs",
    )

    resolved = restart_training_after_checkpoint.resolve_checkpoint_path(argv, tmp_path)

    assert resolved == (tmp_path / "outputs" / "my_run" / "checkpoints" / "last.pt").resolve()


def test_build_resumed_command_replaces_resume_and_conflicting_load_flags(tmp_path: Path) -> None:
    """Restarted commands should use the fresh resume path and drop incompatible load flags."""

    checkpoint_path = tmp_path / "outputs" / "run_b" / "checkpoints" / "last.pt"
    argv = (
        str(tmp_path / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "world_model_v2.run",
        "--load-encoder-decoder",
        "encoder.pt",
        "--load-dynamics",
        "dyn.pt",
        "--resume",
        "stale.pt",
        "--run-name",
        "run_b",
    )

    resumed = restart_training_after_checkpoint.build_resumed_command(argv, checkpoint_path)

    assert "--load-encoder-decoder" not in resumed
    assert "--load-dynamics" not in resumed
    assert resumed[-2:] == ["--resume", str(checkpoint_path)]
    assert resumed[:3] == [str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-m", "world_model_v2.run"]


def test_wait_for_next_checkpoint_returns_after_signature_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The watcher should return once the checkpoint file changes and stabilizes."""

    checkpoint_path = tmp_path / "last.pt"
    checkpoint_path.write_text("before", encoding="utf-8")
    signatures = iter(
        [
            (6, 1),
            (6, 1),
            (7, 2),
            (7, 2),
            (7, 2),
        ]
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "file_signature",
        lambda _path: next(signatures),
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "process_is_alive",
        lambda _pid: True,
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint.time,
        "sleep",
        lambda _seconds: None,
    )

    resolved = restart_training_after_checkpoint.wait_for_next_checkpoint(
        8040,
        checkpoint_path=checkpoint_path,
        poll_seconds=0.01,
    )

    assert resolved == (7, 2)


def test_main_watches_then_relaunches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The helper should watch, stop, and relaunch the run using the fresh checkpoint."""

    root_process = restart_training_after_checkpoint.RunProcess(
        pid=8040,
        parent_pid=19040,
        command_line='"python" -m world_model_v2.run --run-name so101 --output-dir outputs',
        argv=(
            "python",
            "-m",
            "world_model_v2.run",
            "--run-name",
            "so101",
            "--output-dir",
            "outputs",
        ),
    )
    relaunched_commands: list[list[str]] = []
    stopped_pids: list[int] = []

    class DummyProcess:
        """Expose one synthetic relaunched pid."""

        pid = 9901

    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "pid": None,
                "poll_seconds": 0.1,
                "repo_root": str(tmp_path),
            },
        )(),
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "list_world_model_run_processes",
        lambda: [root_process],
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "wait_for_next_checkpoint",
        lambda pid, checkpoint_path, poll_seconds: (123, 456),
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "stop_process_tree",
        lambda pid: stopped_pids.append(pid),
    )
    monkeypatch.setattr(
        restart_training_after_checkpoint,
        "relaunch_command",
        lambda command, repo_root: relaunched_commands.append(command) or DummyProcess(),
    )

    restart_training_after_checkpoint.main()

    checkpoint_path = (tmp_path / "outputs" / "so101" / "checkpoints" / "last.pt").resolve()
    assert stopped_pids == [8040]
    assert relaunched_commands == [
        [
            "python",
            "-m",
            "world_model_v2.run",
            "--run-name",
            "so101",
            "--output-dir",
            "outputs",
            "--resume",
            str(checkpoint_path),
        ]
    ]
    captured = capsys.readouterr()
    assert '"event": "watching_checkpoint_for_restart"' in captured.out
    assert '"event": "relaunched_training"' in captured.out
