"""Restart one active world-model training run after the next checkpoint save.

This helper attaches to an already-running `python -m world_model_v2.run`
process, waits for the next `last.pt` checkpoint update, stops the current
process tree, and relaunches the same command with `--resume <last.pt>`.

Examples
--------
python scripts/check/restart_training_after_checkpoint.py
python scripts/check/restart_training_after_checkpoint.py --pid 8040
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STABLE_FILE_POLLS = 2


@dataclass(frozen=True)
class RunProcess:
    """Describe one running `python -m world_model_v2.run` process."""

    pid: int
    parent_pid: int | None
    command_line: str
    argv: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the checkpoint-restart helper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser.parse_args()


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one subprocess and capture its text outputs."""

    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def windows_command_line_to_argv(command_line: str) -> list[str]:
    """Split one Windows command line into argv tokens."""

    if os.name != "nt":
        return shlex.split(command_line)
    command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
    local_free = ctypes.windll.kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p
    argc = ctypes.c_int()
    argv_ptr = command_line_to_argv(command_line, ctypes.byref(argc))
    if not argv_ptr:
        raise ValueError(f"Unable to parse Windows command line: {command_line!r}")
    try:
        return [argv_ptr[index] for index in range(argc.value)]
    finally:
        local_free(argv_ptr)


def is_world_model_run_argv(argv: list[str] | tuple[str, ...]) -> bool:
    """Return whether one argv sequence launches `world_model_v2.run`."""

    for index, token in enumerate(argv[:-1]):
        if token == "-m" and argv[index + 1] == "world_model_v2.run":
            return True
    return False


def parse_windows_process_payload(raw_output: str) -> list[RunProcess]:
    """Parse one PowerShell JSON payload into world-model run processes."""

    stripped = raw_output.strip()
    if not stripped:
        return []
    payload = json.loads(stripped)
    records = payload if isinstance(payload, list) else [payload]
    processes: list[RunProcess] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        command_line = record.get("CommandLine")
        if not isinstance(command_line, str) or not command_line.strip():
            continue
        argv = tuple(windows_command_line_to_argv(command_line))
        if not is_world_model_run_argv(argv):
            continue
        parent_pid = record.get("ParentProcessId")
        processes.append(
            RunProcess(
                pid=int(record["ProcessId"]),
                parent_pid=None if parent_pid is None else int(parent_pid),
                command_line=command_line,
                argv=argv,
            )
        )
    return processes


def list_world_model_run_processes() -> list[RunProcess]:
    """Return the currently running `python -m world_model_v2.run` processes."""

    if os.name != "nt":
        raise RuntimeError("This helper currently supports Windows only.")
    completed = run_subprocess(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "$ErrorActionPreference='Stop'; "
                "Get-CimInstance Win32_Process "
                "| Where-Object { $_.Name -in @('python.exe', 'python3.exe') } "
                "| Select-Object ProcessId,ParentProcessId,CommandLine "
                "| ConvertTo-Json -Compress"
            ),
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to enumerate running python processes:\n"
            f"{completed.stdout}\n{completed.stderr}".strip()
        )
    return parse_windows_process_payload(completed.stdout)


def choose_target_process(processes: list[RunProcess], pid: int | None = None) -> RunProcess:
    """Choose the active top-level training process to watch and restart."""

    if pid is not None:
        for process in processes:
            if process.pid == pid:
                return process
        raise ValueError(f"Could not find a running world-model process with pid={pid}.")
    if not processes:
        raise ValueError("No active `python -m world_model_v2.run` processes were found.")
    matching_pids = {process.pid for process in processes}
    roots = [process for process in processes if process.parent_pid not in matching_pids]
    if len(roots) != 1:
        root_ids = ", ".join(str(process.pid) for process in roots) or "none"
        raise ValueError(
            "Expected exactly one top-level active world-model run. "
            f"Found {len(roots)} candidates: {root_ids}. Pass --pid to disambiguate."
        )
    return roots[0]


def _flag_value(argv: tuple[str, ...], flag: str) -> str | None:
    """Return one CLI flag value from the target argv when present."""

    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def resolve_checkpoint_path(argv: tuple[str, ...], repo_root: Path) -> Path:
    """Resolve the active `last.pt` path for one running training command."""

    resume_value = _flag_value(argv, "--resume")
    if resume_value not in {None, ""}:
        return _resolve_against_repo_root(Path(str(resume_value)), repo_root)
    run_name = _flag_value(argv, "--run-name")
    if run_name in {None, ""}:
        raise ValueError(
            "Unable to infer the checkpoint path from the running command. "
            "The process must include either --resume or --run-name."
        )
    output_dir = _flag_value(argv, "--output-dir") or "outputs"
    return _resolve_against_repo_root(
        Path(output_dir) / str(run_name) / "checkpoints" / "last.pt",
        repo_root,
    )


def _resolve_against_repo_root(path: Path, repo_root: Path) -> Path:
    """Resolve one possibly-relative path against the repository root."""

    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def build_resumed_command(argv: tuple[str, ...], checkpoint_path: Path) -> list[str]:
    """Return the restart command with one fresh `--resume` checkpoint override."""

    stripped: list[str] = []
    skip_next = False
    remove_flags = {"--resume", "--load-encoder-decoder", "--load-dynamics"}
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in remove_flags:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in remove_flags):
            continue
        stripped.append(token)
    stripped.extend(["--resume", str(checkpoint_path)])
    return stripped


def file_signature(path: Path) -> tuple[int, int] | None:
    """Return one stable file signature based on size and mtime nanoseconds."""

    if not path.exists():
        return None
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def process_is_alive(pid: int) -> bool:
    """Return whether the requested process id is still running."""

    if os.name == "nt":
        completed = run_subprocess(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"$process = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
                    "if ($process) { exit 0 } else { exit 1 }"
                ),
            ]
        )
        return completed.returncode == 0
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_stable_checkpoint_update(path: Path, previous_signature: tuple[int, int] | None, poll_seconds: float) -> tuple[int, int]:
    """Wait until one changed checkpoint file stops changing across a few polls."""

    current_signature = previous_signature
    stable_polls = 0
    while stable_polls < STABLE_FILE_POLLS:
        time.sleep(poll_seconds)
        observed_signature = file_signature(path)
        if observed_signature is None:
            stable_polls = 0
            continue
        if observed_signature == current_signature:
            stable_polls += 1
            continue
        current_signature = observed_signature
        stable_polls = 0
    if current_signature is None:
        raise RuntimeError(f"Checkpoint {path} never materialized after the update was detected.")
    return current_signature


def wait_for_next_checkpoint(pid: int, checkpoint_path: Path, poll_seconds: float) -> tuple[int, int]:
    """Wait until the checkpoint file changes while the target process is still alive."""

    previous_signature = file_signature(checkpoint_path)
    while True:
        if not process_is_alive(pid):
            raise RuntimeError(
                f"Process {pid} exited before {checkpoint_path} changed to the next checkpoint."
            )
        current_signature = file_signature(checkpoint_path)
        if current_signature is not None and current_signature != previous_signature:
            return wait_for_stable_checkpoint_update(
                checkpoint_path,
                current_signature,
                poll_seconds,
            )
        time.sleep(poll_seconds)


def stop_process_tree(pid: int) -> None:
    """Stop the watched training process and any worker children."""

    if os.name != "nt":
        raise RuntimeError("This helper currently supports Windows only.")
    completed = run_subprocess(["taskkill", "/PID", str(pid), "/T", "/F"])
    if completed.returncode != 0 and process_is_alive(pid):
        raise RuntimeError(
            "Failed to stop the running training process:\n"
            f"{completed.stdout}\n{completed.stderr}".strip()
        )


def relaunch_command(command: list[str], repo_root: Path) -> subprocess.Popen[str]:
    """Launch the resumed training command from the repository root."""

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(  # noqa: S603
        command,
        cwd=repo_root,
        creationflags=creationflags,
        text=True,
    )


def main() -> None:
    """Watch the current training run and restart it after the next checkpoint save."""

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    target = choose_target_process(list_world_model_run_processes(), pid=args.pid)
    checkpoint_path = resolve_checkpoint_path(target.argv, repo_root)
    print(
        json.dumps(
            {
                "event": "watching_checkpoint_for_restart",
                "checkpoint_path": str(checkpoint_path),
                "pid": target.pid,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    checkpoint_signature = wait_for_next_checkpoint(
        target.pid,
        checkpoint_path=checkpoint_path,
        poll_seconds=float(args.poll_seconds),
    )
    print(
        json.dumps(
            {
                "event": "checkpoint_updated",
                "checkpoint_path": str(checkpoint_path),
                "pid": target.pid,
                "signature": list(checkpoint_signature),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stop_process_tree(target.pid)
    resumed_command = build_resumed_command(target.argv, checkpoint_path)
    relaunched = relaunch_command(resumed_command, repo_root=repo_root)
    print(
        json.dumps(
            {
                "event": "relaunched_training",
                "new_pid": relaunched.pid,
                "resume_checkpoint": str(checkpoint_path),
                "resumed_command": resumed_command,
                "stopped_pid": target.pid,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
