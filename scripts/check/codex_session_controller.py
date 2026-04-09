"""Drive a simple stop/resume Codex loop around long-running training commands.

This helper is intentionally lightweight: each Codex turn can inspect code,
edit files, and run small verification commands inside Codex. When a turn
needs a long-running training or rollout job, it stops and hands back a
`training_command`. The controller runs that command outside of Codex, waits
for it to finish, and resumes the same Codex session with the results. The
loop continues until the configured iteration limit unless Codex marks the
situation as fatal.

Example smoke test:

source .venv/bin/activate
python scripts/check/codex_session_controller.py \
  --prompt-file controller.md \
  --max-iterations 2
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "codex_session_controller"
CODEX_RETRY_BACKOFF_SECONDS = (15, 30, 60, 120, 240)
TRANSIENT_CODEX_FAILURE_MARKERS = (
    "Selected model is at capacity",
    "rate limit",
    "overloaded",
    "temporarily unavailable",
    "timed out",
)
CODE_SNAPSHOT_ROOTS = ("world_model_v2", "scripts", "tests")
CODE_SNAPSHOT_TOP_LEVEL_FILES = (
    "requirements.txt",
    "pytest.ini",
)
IGNORED_SNAPSHOT_DIR_NAMES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "__pycache__", "outputs"}
IGNORED_SNAPSHOT_SUFFIXES = {".pyc", ".pyo"}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "training_command": {"type": "string"},
        "fatal": {"type": "boolean"},
        "fatal_reason": {"type": "string"},
    },
    "required": ["summary", "training_command", "fatal", "fatal_reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ControllerResponse:
    """Represent one structured reply returned by a Codex turn."""

    summary: str
    training_command: str
    fatal: bool
    fatal_reason: str


@dataclass(frozen=True)
class TrainingCommandResult:
    """Record the outcome of one externally executed training command."""

    training_command: str
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str
    stdout_path: str
    stderr_path: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Codex session controller."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum number of Codex turns to execute before stopping.",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--sandbox",
        default="danger-full-access",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Sandbox mode passed to `codex exec` and `codex exec resume`.",
    )
    parser.add_argument(
        "--dangerously-bypass-approvals-and-sandbox",
        action="store_true",
        help="Pass the matching Codex CLI flag instead of `--sandbox`.",
    )
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for controller logs. Defaults to a timestamped output path.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=12000,
        help="Maximum stdout/stderr characters to include in each resume prompt.",
    )
    args = parser.parse_args()
    if args.max_iterations < 1:
        raise ValueError("`--max-iterations` must be at least 1.")
    return args


def read_goal(args: argparse.Namespace) -> str:
    """Load the user goal from CLI text or a file path."""

    if args.prompt and args.prompt_file:
        raise ValueError("Use either `--prompt` or `--prompt-file`, not both.")
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return str(args.prompt).strip()
    raise ValueError("Provide a goal with `--prompt` or `--prompt-file`.")


def timestamp_slug() -> str:
    """Return a filesystem-friendly local timestamp slug."""

    return time.strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory used for logs and summaries."""

    if args.output_dir:
        return Path(args.output_dir)
    return DEFAULT_OUTPUT_ROOT / timestamp_slug()


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON payload to disk with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL log file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    """Write plain text to disk using UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_controller_prompt(goal: str) -> str:
    """Compose the initial prompt sent to the first Codex turn."""

    return (
        "You are working under an external session controller.\n"
        "Inspect the repo, edit files, and run small checks inside the Codex turn.\n"
        "Use the external controller only for long-running training, rollout, or benchmark jobs.\n"
        "Maximize stable GPU and VRAM utilization when choosing training settings and search directions.\n"
        "Always reply using the provided JSON schema only.\n"
        "Field rules:\n"
        "- `summary`: short operator-facing status.\n"
        "- `fatal`: true only when the controller must stop immediately because the run is unrecoverable.\n"
        "- `fatal_reason`: empty string when `fatal` is false; otherwise explain the fatal stop briefly.\n"
        "- `training_command`: the next long-running training-style command to run from the repo root.\n"
        "- Leave `training_command` empty when the next step is only inspection, editing, or a small check "
        "that should run inside Codex itself.\n"
        "- Do not offload small tests, quick `pytest` invocations, file inspection, `rg`, or other short "
        "commands to `training_command`.\n"
        "- A failed `training_command` is not automatically fatal. Inspect it on the next Codex turn and only "
        "set `fatal: true` if the run is genuinely unrecoverable.\n"
        "The controller will continue looping until `--max-iterations` is reached unless you set `fatal: true`.\n"
        "User goal:\n"
        f"{goal}\n"
    )


def truncate_for_prompt(text: str, max_chars: int) -> str:
    """Return the tail of a long log so resume prompts stay compact."""

    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"[truncated {omitted} characters]\n{text[-max_chars:]}"


def truncate_middle(text: str, max_chars: int) -> str:
    """Return a compact middle-truncated string for prompt previews."""

    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    edge = (max_chars - 5) // 2
    return f"{text[:edge]} ... {text[-edge:]}"


def summarize_training_command(command: str, max_chars: int = 240) -> str:
    """Collapse one training command into a compact single-line preview."""

    collapsed = " ".join(command.split())
    return truncate_middle(collapsed, max_chars=max_chars)


def compact_json_line_for_prompt(line: str) -> str:
    """Drop noisy fields from one JSON line before embedding it in a prompt."""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return line
    if not isinstance(payload, dict):
        return line

    compact_payload = dict(payload)
    if "frame_start" in compact_payload and "frame_end" in compact_payload:
        compact_payload["span"] = f"{compact_payload['frame_start']}:{compact_payload['frame_end']}"
        compact_payload.pop("frame_start", None)
        compact_payload.pop("frame_end", None)
    for key in list(compact_payload):
        if key.endswith("_dir") or key.endswith("_path"):
            compact_payload.pop(key, None)
    return json.dumps(compact_payload, sort_keys=True)


def summarize_output_for_prompt(text: str, max_chars: int, max_lines: int = 12) -> str:
    """Return a compact output excerpt with JSON lines reduced when possible."""

    stripped = text.strip()
    if not stripped:
        return "[empty]"

    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines and all(line.lstrip().startswith("{") for line in lines):
        if len(lines) > max_lines:
            head_count = max_lines // 2
            tail_count = max_lines - head_count
            selected_lines = lines[:head_count] + [f"[truncated {len(lines) - max_lines} lines]"] + lines[-tail_count:]
        else:
            selected_lines = lines
        compacted = "\n".join(
            line if line.startswith("[truncated ") else compact_json_line_for_prompt(line)
            for line in selected_lines
        )
        return truncate_for_prompt(compacted, max_chars=max_chars)
    return truncate_for_prompt(stripped, max_chars=max_chars)


def build_resume_command(thread_id: str) -> str:
    """Return the user-facing Codex CLI command for reopening one session."""

    return f"codex resume {thread_id}"


def build_resume_prompt(
    result: TrainingCommandResult | None,
    repo_changes: list[str],
    max_output_chars: int,
    iteration: int,
) -> str:
    """Compose the prompt used to resume Codex after one training command completes."""

    if repo_changes:
        repo_change_status = "Controller-observed repo changes from your previous turn:\n" + "\n".join(
            f"- {change}" for change in repo_changes
        )
    else:
        repo_change_status = "Controller-observed repo changes from your previous turn: none."
    if result is None:
        return (
            f"Resume turn {iteration}.\n"
            "No external training command was run after the previous turn.\n"
            f"{repo_change_status}\n"
            "Continue inspecting or editing the repo and reply with JSON only.\n"
        )
    stdout_text = summarize_output_for_prompt(result.stdout, max_chars=max_output_chars, max_lines=12)
    stderr_text = summarize_output_for_prompt(result.stderr, max_chars=max(800, max_output_chars // 3), max_lines=8)
    if result.exit_code == 0:
        status_line = "The external training command finished successfully."
        failure_guidance = ""
    else:
        status_line = "The external training command failed."
        failure_guidance = (
            "This failure is not automatically fatal. Diagnose it and decide on the next step unless "
            "recovery is impossible.\n"
        )
    return (
        f"Resume turn {iteration}.\n"
        f"{status_line}\n"
        f"Training command summary: `{summarize_training_command(result.training_command)}`\n"
        f"Exit code: {result.exit_code}\n"
        f"Duration seconds: {result.duration_seconds:.2f}\n"
        f"Stdout log: {result.stdout_path}\n"
        f"Stderr log: {result.stderr_path}\n"
        f"{repo_change_status}\n"
        f"{failure_guidance}"
        "Stdout excerpt:\n"
        f"```\n{stdout_text}\n```\n\n"
        "Stderr excerpt:\n"
        f"```\n{stderr_text}\n```\n\n"
        "Validate the result, inspect or edit the repo again if needed, and reply with JSON only.\n"
    )


def build_codex_base_command(args: argparse.Namespace, schema_path: Path) -> list[str]:
    """Build the Codex CLI argument prefix shared by exec and resume calls."""

    command = [args.codex_bin]
    if args.dangerously_bypass_approvals_and_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["--sandbox", args.sandbox])
    return command + [
        "exec",
        "--json",
        "--output-schema",
        str(schema_path),
        "-C",
        str(REPO_ROOT),
    ]


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one subprocess and capture its outputs."""

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects from a mixed plain-text and JSONL output stream."""

    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def combine_process_output(completed: subprocess.CompletedProcess[str]) -> str:
    """Return the combined stdout and stderr text for one subprocess result."""

    return completed.stdout + completed.stderr


def classify_transient_codex_failure(text: str) -> str | None:
    """Return the matching transient Codex failure marker when one is present."""

    lowered = text.lower()
    for marker in TRANSIENT_CODEX_FAILURE_MARKERS:
        if marker.lower() in lowered:
            return marker
    return None


def hash_file(path: Path) -> str:
    """Return a stable hash for one file on disk."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def capture_repo_code_snapshot() -> dict[str, str]:
    """Capture a hash snapshot of code-oriented repo files."""

    snapshot: dict[str, str] = {}
    for root_name in CODE_SNAPSHOT_ROOTS:
        root_path = REPO_ROOT / root_name
        if not root_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [name for name in dirnames if name not in IGNORED_SNAPSHOT_DIR_NAMES]
            current_dir = Path(dirpath)
            for filename in filenames:
                file_path = current_dir / filename
                if file_path.suffix in IGNORED_SNAPSHOT_SUFFIXES:
                    continue
                relative_path = str(file_path.relative_to(REPO_ROOT))
                snapshot[relative_path] = hash_file(file_path)
    for file_name in CODE_SNAPSHOT_TOP_LEVEL_FILES:
        file_path = REPO_ROOT / file_name
        if file_path.exists() and file_path.is_file():
            snapshot[str(file_path.relative_to(REPO_ROOT))] = hash_file(file_path)
    return snapshot


def summarize_repo_code_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Return a sorted summary of added, modified, and deleted code files."""

    changes: list[str] = []
    all_paths = sorted(set(before) | set(after))
    for path in all_paths:
        if path not in before:
            changes.append(f"A {path}")
        elif path not in after:
            changes.append(f"D {path}")
        elif before[path] != after[path]:
            changes.append(f"M {path}")
    return changes


def parse_codex_turn_output(text: str) -> tuple[str | None, str]:
    """Extract the thread id and final agent message from one Codex JSONL stream."""

    thread_id: str | None = None
    last_message = ""
    for event in parse_json_lines(text):
        if event.get("type") == "thread.started":
            candidate = event.get("thread_id")
            if isinstance(candidate, str) and candidate:
                thread_id = candidate
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "agent_message":
            continue
        text_value = item.get("text")
        if isinstance(text_value, str):
            last_message = text_value
    return thread_id, last_message


def parse_controller_response(raw_message: str) -> ControllerResponse:
    """Parse and validate one structured JSON reply from Codex."""

    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("Codex response was not a JSON object.")
    summary = payload.get("summary")
    training_command = payload.get("training_command")
    fatal = payload.get("fatal")
    fatal_reason = payload.get("fatal_reason")
    if not isinstance(summary, str):
        raise ValueError("Codex response is missing string field `summary`.")
    if not isinstance(training_command, str):
        raise ValueError("Codex response is missing string field `training_command`.")
    if not isinstance(fatal, bool):
        raise ValueError("Codex response is missing boolean field `fatal`.")
    if not isinstance(fatal_reason, str):
        raise ValueError("Codex response is missing string field `fatal_reason`.")
    return ControllerResponse(
        summary=summary,
        training_command=training_command,
        fatal=fatal,
        fatal_reason=fatal_reason,
    )


def run_codex_turn(
    args: argparse.Namespace,
    schema_path: Path,
    prompt: str,
    iteration: int,
    output_dir: Path,
    summary_path: Path,
    thread_id: str | None,
) -> tuple[str, ControllerResponse]:
    """Start or resume a Codex session and return its structured reply."""

    base_command = build_codex_base_command(args, schema_path)
    if args.skip_git_repo_check:
        base_command.append("--skip-git-repo-check")
    if args.model:
        base_command.extend(["--model", args.model])
    turn_log_path = output_dir / f"turn_{iteration:03d}.log"
    raw_message = ""

    for attempt in range(1, len(CODEX_RETRY_BACKOFF_SECONDS) + 2):
        if thread_id is None:
            command = base_command + [prompt]
        else:
            command = base_command + ["resume", thread_id, prompt]
        attempt_log_path = output_dir / f"turn_{iteration:03d}_attempt_{attempt:02d}.log"
        completed = run_subprocess(command)
        combined_output = combine_process_output(completed)
        write_text(attempt_log_path, combined_output)
        write_text(turn_log_path, combined_output)
        parsed_thread_id, raw_message = parse_codex_turn_output(combined_output)
        if not thread_id and parsed_thread_id:
            thread_id = parsed_thread_id
        if completed.returncode == 0:
            break
        retry_reason = classify_transient_codex_failure(combined_output)
        if retry_reason is None or attempt > len(CODEX_RETRY_BACKOFF_SECONDS):
            raise RuntimeError(f"Codex turn {iteration} failed; see {turn_log_path}.")
        backoff_seconds = CODEX_RETRY_BACKOFF_SECONDS[attempt - 1]
        append_jsonl(
            summary_path,
            {
                "attempt": attempt,
                "backoff_seconds": backoff_seconds,
                "event": "codex_turn_retry",
                "iteration": iteration,
                "retry_reason": retry_reason,
                "thread_id": thread_id,
            },
        )
        print(
            "[controller] transient Codex turn failure "
            f"(attempt {attempt}, reason={retry_reason!r}); retrying in {backoff_seconds}s",
            flush=True,
        )
        time.sleep(backoff_seconds)
    else:
        raise RuntimeError(f"Codex turn {iteration} failed; see {turn_log_path}.")

    if not thread_id:
        raise RuntimeError(f"Codex turn {iteration} did not expose a thread id; see {turn_log_path}.")
    if not raw_message:
        raise RuntimeError(f"Codex turn {iteration} returned no agent message; see {turn_log_path}.")
    response = parse_controller_response(raw_message)
    write_json(output_dir / f"turn_{iteration:03d}_response.json", asdict(response))
    return thread_id, response


def run_training_command(
    training_command: str,
    iteration: int,
    output_dir: Path,
) -> TrainingCommandResult:
    """Execute one external training command and capture its outputs."""

    stdout_path = output_dir / f"training_command_{iteration:03d}.stdout.log"
    stderr_path = output_dir / f"training_command_{iteration:03d}.stderr.log"
    started_at = time.time()
    completed = subprocess.run(
        ["bash", "-lc", training_command],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    finished_at = time.time()
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return TrainingCommandResult(
        training_command=training_command,
        exit_code=int(completed.returncode),
        duration_seconds=finished_at - started_at,
        stdout=completed.stdout,
        stderr=completed.stderr,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def choose_training_command(response: ControllerResponse) -> str:
    """Choose the training command the controller should execute next."""

    return response.training_command.strip()


def main() -> None:
    """Run the Codex session controller until completion or iteration exhaustion."""

    args = parse_args()
    goal = read_goal(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "response_schema.json"
    write_json(schema_path, RESPONSE_SCHEMA)

    prompt = build_controller_prompt(goal=goal)
    summary_path = output_dir / "summary.jsonl"
    thread_id: str | None = None
    printed_resume_command = False
    fatal = False
    fatal_reason = ""
    iterations_attempted = 0

    for iteration in range(1, args.max_iterations + 1):
        iterations_attempted = iteration
        print(f"[controller] codex turn {iteration} starting", flush=True)
        pre_turn_snapshot = capture_repo_code_snapshot()
        thread_id, response = run_codex_turn(
            args=args,
            schema_path=schema_path,
            prompt=prompt,
            iteration=iteration,
            output_dir=output_dir,
            summary_path=summary_path,
            thread_id=thread_id,
        )
        if not printed_resume_command:
            resume_command = build_resume_command(thread_id)
            write_text(output_dir / "session_resume_command.txt", resume_command + "\n")
            print(f"[controller] attach to this Codex session with: {resume_command}", flush=True)
            printed_resume_command = True
        post_turn_snapshot = capture_repo_code_snapshot()
        repo_changes = summarize_repo_code_changes(pre_turn_snapshot, post_turn_snapshot)
        append_jsonl(
            summary_path,
            {
                "event": "codex_turn",
                "iteration": iteration,
                "repo_changes": repo_changes,
                "response": asdict(response),
                "thread_id": thread_id,
            },
        )
        print(f"[controller] codex summary: {response.summary}", flush=True)
        if repo_changes:
            print("[controller] repo changes this turn: " + ", ".join(repo_changes), flush=True)
        else:
            print("[controller] repo changes this turn: none detected", flush=True)
        if response.fatal:
            fatal = True
            fatal_reason = response.fatal_reason
            print(f"[controller] fatal stop requested: {fatal_reason}", flush=True)
            break

        training_command = choose_training_command(response=response)
        result: TrainingCommandResult | None = None
        if training_command:
            print(f"[controller] running training command: {training_command}", flush=True)
            result = run_training_command(
                training_command=training_command,
                iteration=iteration,
                output_dir=output_dir,
            )
            append_jsonl(
                summary_path,
                {
                    "training_command": training_command,
                    "duration_seconds": result.duration_seconds,
                    "event": "training_command",
                    "exit_code": result.exit_code,
                    "iteration": iteration,
                    "succeeded": result.exit_code == 0,
                },
            )
            print(
                "[controller] training command "
                f"{'finished' if result.exit_code == 0 else 'failed'} "
                f"(exit={result.exit_code}, duration={result.duration_seconds:.2f}s)",
                flush=True,
            )
        else:
            append_jsonl(
                summary_path,
                {
                    "event": "no_training_command",
                    "iteration": iteration,
                },
            )
            print("[controller] no training command requested; continuing", flush=True)
        prompt = build_resume_prompt(
            result=result,
            repo_changes=repo_changes,
            max_output_chars=args.max_output_chars,
            iteration=iteration + 1,
        )

    append_jsonl(
        summary_path,
        {
            "event": "controller_finished",
            "fatal": fatal,
            "fatal_reason": fatal_reason,
            "iterations_attempted": iterations_attempted,
            "max_iterations": args.max_iterations,
            "thread_id": thread_id,
        },
    )
    if fatal:
        raise SystemExit(
            f"Fatal stop requested after {iterations_attempted} iteration(s): {fatal_reason}. "
            f"Inspect logs in {output_dir}."
        )
    if iterations_attempted >= args.max_iterations:
        print(
            "[controller] reached max iterations normally; "
            f"logs in {output_dir}",
            flush=True,
        )
        return
    print(f"[controller] finished successfully; logs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
