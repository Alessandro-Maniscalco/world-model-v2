"""Tests for the Codex session controller helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _load_codex_session_controller_module() -> object:
    """Load the controller helper module directly from its script path."""

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check" / "codex_session_controller.py"
    spec = importlib.util.spec_from_file_location("codex_session_controller", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codex_session_controller = _load_codex_session_controller_module()


def _make_args(output_dir: Path) -> SimpleNamespace:
    """Build a minimal args object for controller unit tests."""

    return SimpleNamespace(
        codex_bin="codex",
        dangerously_bypass_approvals_and_sandbox=False,
        goal="maximize architecture quality",
        max_iterations=2,
        max_output_chars=12000,
        model="",
        output_dir=str(output_dir),
        prompt="controller instructions",
        prompt_file="",
        sandbox="danger-full-access",
        skip_git_repo_check=False,
    )


def _codex_success_stream(message: dict[str, object], thread_id: str) -> str:
    """Build a minimal successful Codex JSONL stream for one turn."""

    events = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": json.dumps(message),
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file into a list of dictionaries."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_codex_turn_retries_capacity_error_and_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Transient provider failures should retry and then return the successful response."""

    commands: list[list[str]] = []
    sleep_calls: list[int] = []
    responses = [
        subprocess.CompletedProcess(
            args=["codex"],
            returncode=1,
            stdout=json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n",
            stderr="Selected model is at capacity. Please try a different model.\n",
        ),
        subprocess.CompletedProcess(
            args=["codex"],
            returncode=0,
            stdout=_codex_success_stream(
                {
                    "summary": "retry succeeded",
                    "training_command": "",
                    "fatal": False,
                    "fatal_reason": "",
                },
                thread_id="thread-123",
            ),
            stderr="",
        ),
    ]

    def fake_run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
        """Return one mocked subprocess result per controller attempt."""

        commands.append(command)
        return responses[len(commands) - 1]

    monkeypatch.setattr(codex_session_controller, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(codex_session_controller.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    thread_id, response = codex_session_controller.run_codex_turn(
        args=_make_args(tmp_path),
        schema_path=tmp_path / "schema.json",
        prompt="turn prompt",
        iteration=6,
        output_dir=tmp_path,
        summary_path=tmp_path / "summary.jsonl",
        thread_id=None,
    )

    assert thread_id == "thread-123"
    assert response.summary == "retry succeeded"
    assert sleep_calls == [15]
    assert len(commands) == 2
    assert "resume" not in commands[0]
    assert "resume" in commands[1]
    assert (tmp_path / "turn_006_attempt_01.log").exists()
    assert (tmp_path / "turn_006_attempt_02.log").exists()
    assert (tmp_path / "turn_006.log").read_text(encoding="utf-8") == _codex_success_stream(
        {
            "summary": "retry succeeded",
            "training_command": "",
            "fatal": False,
            "fatal_reason": "",
        },
        thread_id="thread-123",
    )
    summary_records = _read_jsonl(tmp_path / "summary.jsonl")
    assert summary_records == [
        {
            "attempt": 1,
            "backoff_seconds": 15,
            "event": "codex_turn_retry",
            "iteration": 6,
            "retry_reason": "Selected model is at capacity",
            "thread_id": "thread-123",
        }
    ]


def test_run_codex_turn_stops_after_retry_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent transient provider failures should stop after the configured retries."""

    commands: list[list[str]] = []
    sleep_calls: list[int] = []

    def fake_run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
        """Always return the same retryable provider failure."""

        commands.append(command)
        return subprocess.CompletedProcess(
            args=["codex"],
            returncode=1,
            stdout="",
            stderr="Selected model is at capacity. Please try a different model.\n",
        )

    monkeypatch.setattr(codex_session_controller, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(codex_session_controller.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(RuntimeError, match="Codex turn 4 failed"):
        codex_session_controller.run_codex_turn(
            args=_make_args(tmp_path),
            schema_path=tmp_path / "schema.json",
            prompt="turn prompt",
            iteration=4,
            output_dir=tmp_path,
            summary_path=tmp_path / "summary.jsonl",
            thread_id="thread-existing",
        )

    assert len(commands) == 6
    assert sleep_calls == [15, 30, 60, 120, 240]
    assert (tmp_path / "turn_004_attempt_06.log").exists()
    summary_records = _read_jsonl(tmp_path / "summary.jsonl")
    assert len(summary_records) == 5
    assert summary_records[-1]["attempt"] == 5
    assert summary_records[-1]["backoff_seconds"] == 240


def test_run_codex_turn_does_not_retry_missing_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed successful output should fail immediately without retries."""

    commands: list[list[str]] = []
    sleep_calls: list[int] = []

    def fake_run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
        """Return a successful process result without an agent message."""

        commands.append(command)
        return subprocess.CompletedProcess(
            args=["codex"],
            returncode=0,
            stdout=json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(codex_session_controller, "run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(codex_session_controller.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(RuntimeError, match="returned no agent message"):
        codex_session_controller.run_codex_turn(
            args=_make_args(tmp_path),
            schema_path=tmp_path / "schema.json",
            prompt="turn prompt",
            iteration=2,
            output_dir=tmp_path,
            summary_path=tmp_path / "summary.jsonl",
            thread_id=None,
        )

    assert len(commands) == 1
    assert sleep_calls == []
    assert not (tmp_path / "summary.jsonl").exists()


def test_build_resume_prompt_marks_failed_training_command() -> None:
    """Failed training commands should be called out explicitly in the resume prompt."""

    prompt = codex_session_controller.build_resume_prompt(
        result=codex_session_controller.TrainingCommandResult(
            training_command="python train.py",
            exit_code=7,
            duration_seconds=12.5,
            stdout="oops",
            stderr="traceback",
            stdout_path="stdout.log",
            stderr_path="stderr.log",
        ),
        repo_changes=["M world_model_v2/run.py"],
        max_output_chars=200,
        iteration=3,
        goal="maximize architecture quality",
    )

    assert "Remember the goal:\nmaximize architecture quality" in prompt
    assert "failed" in prompt
    assert "Exit code: 7" in prompt
    assert "not automatically fatal" in prompt
    assert 'Training command summary: "python train.py"' in prompt
    assert 'Stdout log: "stdout.log"' in prompt
    assert 'Stderr log: "stderr.log"' in prompt
    assert "oops" in prompt
    assert "traceback" in prompt
    assert "M world_model_v2/run.py" in prompt


def test_build_resume_prompt_keeps_same_shape_without_training_result() -> None:
    """Resume prompts should still restate the goal and use explicit empty run fields."""

    prompt = codex_session_controller.build_resume_prompt(
        result=None,
        repo_changes=[],
        max_output_chars=200,
        iteration=5,
        goal="maximize architecture quality",
    )

    assert "Remember the goal:\nmaximize architecture quality" in prompt
    assert "No external training command was run after the previous turn." in prompt
    assert 'Training command summary: ""' in prompt
    assert "Exit code: null" in prompt
    assert "Duration seconds: null" in prompt
    assert 'Stdout log: ""' in prompt
    assert 'Stderr log: ""' in prompt
    assert "```" in prompt
    assert "[empty]" in prompt


def test_build_controller_prompt_includes_instructions_and_goal() -> None:
    """The initial controller prompt should carry both instructions and the persistent goal."""

    prompt = codex_session_controller.build_controller_prompt(
        prompt="inspect code and search broadly",
        goal="find the best architecture",
    )

    assert "Session instructions:\ninspect code and search broadly" in prompt
    assert "Remember the goal:\nfind the best architecture" in prompt


def test_summarize_output_for_prompt_compacts_json_lines() -> None:
    """JSONL outputs should shed bulky path fields before prompting Codex again."""

    output = "\n".join(
        [
            json.dumps(
                {
                    "frame_start": 0,
                    "frame_end": 19,
                    "label": "current_best",
                    "open_rollout_frame_mse": 0.123,
                    "output_dir": "/tmp/very/long/path",
                }
            ),
            json.dumps({"count": 1}),
        ]
    )

    summary = codex_session_controller.summarize_output_for_prompt(output, max_chars=400, max_lines=12)

    assert '"span": "0:19"' in summary
    assert '"label": "current_best"' in summary
    assert '"count": 1' in summary
    assert "output_dir" not in summary


def test_summarize_repo_code_changes_detects_add_modify_delete() -> None:
    """Repo snapshot diffs should describe added, modified, and deleted files."""

    changes = codex_session_controller.summarize_repo_code_changes(
        before={
            "world_model_v2/a.py": "old-a",
            "scripts/check/b.py": "same-b",
            "tests/c.py": "old-c",
        },
        after={
            "world_model_v2/a.py": "new-a",
            "scripts/check/b.py": "same-b",
            "tests/d.py": "new-d",
        },
    )

    assert changes == [
        "D tests/c.py",
        "A tests/d.py",
        "M world_model_v2/a.py",
    ]


def test_capture_repo_code_snapshot_ignores_outputs_and_cache_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repo snapshots should only track code-oriented files."""

    (tmp_path / "world_model_v2").mkdir()
    (tmp_path / "scripts" / "check").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "world_model_v2" / "__pycache__").mkdir()
    (tmp_path / "world_model_v2" / "module.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "scripts" / "check" / "helper.py").write_text("print('helper')\n", encoding="utf-8")
    (tmp_path / "tests" / "test_helper.py").write_text("assert True\n", encoding="utf-8")
    (tmp_path / "outputs" / "artifact.txt").write_text("ignore me\n", encoding="utf-8")
    (tmp_path / "world_model_v2" / "__pycache__" / "module.pyc").write_bytes(b"ignore")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    monkeypatch.setattr(codex_session_controller, "REPO_ROOT", tmp_path)

    snapshot = codex_session_controller.capture_repo_code_snapshot()

    assert sorted(snapshot) == [
        "requirements.txt",
        "scripts/check/helper.py",
        "tests/test_helper.py",
        "world_model_v2/module.py",
    ]


def test_main_allows_training_command_without_code_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Long runs should still execute even when the current turn made no repo code edits."""

    prompts: list[str] = []
    training_calls: list[tuple[str, int]] = []
    responses = [
        codex_session_controller.ControllerResponse(
            summary="analysis only",
            training_command="python train.py",
            fatal=False,
            fatal_reason="",
        ),
        codex_session_controller.ControllerResponse(
            summary="second turn",
            training_command="",
            fatal=False,
            fatal_reason="",
        ),
    ]

    def fake_run_codex_turn(
        args: SimpleNamespace,
        schema_path: Path,
        prompt: str,
        iteration: int,
        output_dir: Path,
        summary_path: Path,
        thread_id: str | None,
    ) -> tuple[str, object]:
        """Return deterministic turn responses while recording prompts."""

        prompts.append(prompt)
        return "thread-xyz", responses[len(prompts) - 1]

    monkeypatch.setattr(codex_session_controller, "parse_args", lambda: _make_args(tmp_path))
    monkeypatch.setattr(codex_session_controller, "run_codex_turn", fake_run_codex_turn)
    monkeypatch.setattr(
        codex_session_controller,
        "capture_repo_code_snapshot",
        lambda: {},
    )
    monkeypatch.setattr(
        codex_session_controller,
        "run_training_command",
        lambda training_command, iteration, output_dir: training_calls.append((training_command, iteration))
        or codex_session_controller.TrainingCommandResult(
            training_command=training_command,
            exit_code=0,
            duration_seconds=2.5,
            stdout="ok stdout",
            stderr="",
            stdout_path=str(output_dir / "stdout.log"),
            stderr_path=str(output_dir / "stderr.log"),
        ),
    )

    codex_session_controller.main()
    captured = capsys.readouterr()

    assert len(prompts) == 2
    assert training_calls == [("python train.py", 1)]
    assert "Remember the goal:\nmaximize architecture quality" in prompts[1]
    assert "Controller-observed repo changes from your previous turn: none." in prompts[1]
    assert 'Training command summary: "python train.py"' in prompts[1]
    assert "Stdout log:" in prompts[1]
    assert "stdout.log" in prompts[1]
    assert "Stderr log:" in prompts[1]
    assert "stderr.log" in prompts[1]
    assert "attach to this Codex session with: codex resume thread-xyz" in captured.out
    assert (tmp_path / "session_resume_command.txt").read_text(encoding="utf-8") == "codex resume thread-xyz\n"
    summary_records = _read_jsonl(tmp_path / "summary.jsonl")
    training_events = [
        record for record in summary_records if record["event"] == "training_command"
    ]
    assert training_events == [
        {
            "duration_seconds": 2.5,
            "event": "training_command",
            "exit_code": 0,
            "iteration": 1,
            "succeeded": True,
            "training_command": "python train.py",
        }
    ]


def test_main_records_failed_training_command_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed training command should be logged and passed back into the next turn."""

    prompts: list[str] = []
    responses = [
        codex_session_controller.ControllerResponse(
            summary="launch training",
            training_command="python train.py",
            fatal=False,
            fatal_reason="",
        ),
        codex_session_controller.ControllerResponse(
            summary="inspect failure",
            training_command="",
            fatal=False,
            fatal_reason="",
        ),
    ]

    def fake_run_codex_turn(
        args: SimpleNamespace,
        schema_path: Path,
        prompt: str,
        iteration: int,
        output_dir: Path,
        summary_path: Path,
        thread_id: str | None,
    ) -> tuple[str, object]:
        """Return deterministic turn responses while recording the prompts."""

        prompts.append(prompt)
        return "thread-xyz", responses[len(prompts) - 1]

    monkeypatch.setattr(codex_session_controller, "parse_args", lambda: _make_args(tmp_path))
    monkeypatch.setattr(codex_session_controller, "run_codex_turn", fake_run_codex_turn)
    snapshots = iter(
        [
            {"world_model_v2/model.py": "old"},
            {"world_model_v2/model.py": "new"},
            {"world_model_v2/model.py": "new"},
            {"world_model_v2/model.py": "new"},
        ]
    )
    monkeypatch.setattr(codex_session_controller, "capture_repo_code_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(
        codex_session_controller,
        "run_training_command",
        lambda training_command, iteration, output_dir: codex_session_controller.TrainingCommandResult(
            training_command=training_command,
            exit_code=3,
            duration_seconds=1.25,
            stdout="bad stdout",
            stderr="bad stderr",
            stdout_path=str(output_dir / "stdout.log"),
            stderr_path=str(output_dir / "stderr.log"),
        ),
    )

    codex_session_controller.main()

    assert len(prompts) == 2
    assert "Remember the goal:\nmaximize architecture quality" in prompts[1]
    assert "failed" in prompts[1]
    assert "Exit code: 3" in prompts[1]
    assert 'Training command summary: "python train.py"' in prompts[1]
    assert "Stdout log:" in prompts[1]
    assert "stdout.log" in prompts[1]
    assert "Stderr log:" in prompts[1]
    assert "stderr.log" in prompts[1]
    assert "bad stdout" in prompts[1]
    assert "bad stderr" in prompts[1]
    assert "M world_model_v2/model.py" in prompts[1]

    summary_records = _read_jsonl(tmp_path / "summary.jsonl")
    training_events = [record for record in summary_records if record["event"] == "training_command"]
    assert training_events == [
        {
            "duration_seconds": 1.25,
            "event": "training_command",
            "exit_code": 3,
            "iteration": 1,
            "succeeded": False,
            "training_command": "python train.py",
        }
    ]
    codex_turn_events = [record for record in summary_records if record["event"] == "codex_turn"]
    assert codex_turn_events[0]["repo_changes"] == ["M world_model_v2/model.py"]
