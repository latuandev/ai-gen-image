from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import IO

from django.conf import settings

from apps.agent_workspace.ai_agent.cli_wrapper import (
    CodexCLIInvocation,
    CodexCLIWrapper,
)
from apps.agent_workspace.ai_agent.executor import (
    AgentCancellationCheck,
    AgentExecutionOutcome,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentExecutor,
)
from apps.agent_workspace.exceptions import (
    AgentExecutionError,
    AgentProcessSpawnError,
    InvalidAgentExecutorConfiguration,
)

DEFAULT_PARENT_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
)
FORBIDDEN_CHILD_ENVIRONMENT_KEYS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "CELERY_BROKER_URL",
        "DATABASE_PASSWORD",
        "DATABASE_URL",
        "DJANGO_SECRET_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "POSTGRES_PASSWORD",
        "REDIS_URL",
        "SECRET_KEY",
        "SENTRY_DSN",
    }
)
LOGS_DIRECTORY_NAME = "logs"
RUNTIME_DIRECTORY_NAME = "runtime"
RUNTIME_HOME_DIRECTORY_NAME = "home"
STDOUT_LOG_FILENAME = "stdout.log"
STDERR_LOG_FILENAME = "stderr.log"
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIRECTORY_MODE = 0o700
STDIN_WRITE_CHUNK_SIZE = 65536
OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
)
OPEN_LOG_FILE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
)
LINUX_PROC_ROOT = Path("/proc")
NON_LIVE_PROCESS_STATES = frozenset({"Z", "X", "x"})


@dataclass(frozen=True)
class ProcessLogFiles:
    """
    Hold opened stdout and stderr log file objects for a child process.
    """

    stdout: IO[bytes]
    stderr: IO[bytes]


@dataclass(frozen=True)
class ProcessStat:
    """
    Hold the minimal process metadata needed for lifecycle checks.
    """

    state: str
    process_group_id: int


class LocalSubprocessExecutor(AgentExecutor):
    """
    Execute an Agent request through a trusted local Codex subprocess.

    This executor runs with the host security context of the worker process. It
    can control its local process group for lifecycle correctness, but it is not
    a production sandbox, filesystem isolation boundary, process isolation
    boundary, or network isolation boundary.
    """

    def __init__(
        self,
        cli_wrapper: CodexCLIWrapper | None = None,
        parent_environment_allowlist: tuple[str, ...] = DEFAULT_PARENT_ENVIRONMENT_ALLOWLIST,
        timeout_seconds: float | None = None,
        termination_grace_seconds: float | None = None,
        cancellation_poll_interval_seconds: float | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """
        Store subprocess dependencies and runtime control configuration.
        """

        self.cli_wrapper = cli_wrapper or CodexCLIWrapper()
        self.parent_environment_allowlist = tuple(parent_environment_allowlist)
        self.timeout_seconds = _validate_positive_seconds(
            _setting_value(
                timeout_seconds,
                "AGENT_EXECUTION_TIMEOUT_SECONDS",
            ),
            "timeout_seconds",
        )
        self.termination_grace_seconds = _validate_non_negative_seconds(
            _setting_value(
                termination_grace_seconds,
                "AGENT_EXECUTION_TERMINATION_GRACE_SECONDS",
            ),
            "termination_grace_seconds",
        )
        self.cancellation_poll_interval_seconds = _validate_positive_seconds(
            _setting_value(
                cancellation_poll_interval_seconds,
                "AGENT_EXECUTION_CANCELLATION_POLL_SECONDS",
            ),
            "cancellation_poll_interval_seconds",
        )
        self.monotonic_clock = monotonic_clock
        self.sleep = sleep

    def execute(
        self,
        request: AgentExecutionRequest,
        *,
        is_cancel_requested: AgentCancellationCheck | None = None,
    ) -> AgentExecutionResult:
        """
        Run Codex through a local subprocess and return a normalized result.

        Natural process completion is checked before cancellation and timeout.
        If cancellation and timeout are both pending in the same polling
        iteration, cancellation wins because it is observed before the timeout
        decision.
        """

        invocation = self.cli_wrapper.build_invocation(request)

        try:
            if _is_cancellation_requested(is_cancel_requested):
                return AgentExecutionResult(outcome=AgentExecutionOutcome.CANCELLED)
        except Exception as exc:
            raise AgentExecutionError("Agent cancellation check failed.") from exc

        child_environment = self._build_child_environment(invocation)

        with _open_process_log_files(invocation.cwd) as log_files:
            process = self._spawn_process(invocation, child_environment, log_files)
            process_group_id = _capture_process_group_id(process)
            try:
                return self._monitor_process(
                    process,
                    process_group_id,
                    invocation.stdin_text,
                    is_cancel_requested,
                )
            except Exception as exc:
                self._terminate_for_executor_error(process, process_group_id, exc)
                raise

    def _build_child_environment(self, invocation: CodexCLIInvocation) -> dict[str, str]:
        """
        Build the child process environment from an explicit allowlist.
        """

        runtime_home = _prepare_runtime_home(invocation.cwd)
        child_environment = _copy_allowed_parent_environment(self.parent_environment_allowlist)
        child_environment["HOME"] = str(runtime_home)

        _merge_environment_overrides(
            child_environment,
            invocation.environment_overrides,
        )
        _assert_forbidden_environment_keys_absent(child_environment)

        return child_environment

    def _spawn_process(
        self,
        invocation: CodexCLIInvocation,
        child_environment: dict[str, str],
        log_files: ProcessLogFiles,
    ) -> subprocess.Popen:
        """
        Spawn the child process in its own process group without a shell.
        """

        stdin = subprocess.PIPE if invocation.stdin_text is not None else subprocess.DEVNULL

        try:
            return subprocess.Popen(
                invocation.argv,
                cwd=invocation.cwd,
                env=child_environment,
                stdin=stdin,
                stdout=log_files.stdout,
                stderr=log_files.stderr,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise AgentProcessSpawnError("Subprocess could not be started") from exc

    def _monitor_process(
        self,
        process: subprocess.Popen,
        process_group_id: int,
        stdin_text: str | None,
        is_cancel_requested: AgentCancellationCheck | None,
    ) -> AgentExecutionResult:
        """
        Poll the child process for completion, cancellation, and timeout.
        """

        stdin_buffer = _stdin_bytes(stdin_text)
        deadline = self.monotonic_clock() + self.timeout_seconds

        while True:
            natural_result = self._natural_completion_result(process, process_group_id)
            if natural_result is not None:
                _close_process_stdin(process)
                return natural_result

            stdin_buffer = self._write_stdin_chunk(process, process_group_id, stdin_buffer)

            natural_result = self._natural_completion_result(process, process_group_id)
            if natural_result is not None:
                _close_process_stdin(process)
                return natural_result

            if self._cancel_requested_or_fail_safely(
                process,
                process_group_id,
                is_cancel_requested,
            ):
                self._terminate_process_group(process, process_group_id)
                return AgentExecutionResult(outcome=AgentExecutionOutcome.CANCELLED)

            now = self.monotonic_clock()
            if now >= deadline:
                self._terminate_process_group(process, process_group_id)
                return AgentExecutionResult(outcome=AgentExecutionOutcome.TIMED_OUT)

            self.sleep(
                _next_poll_sleep_seconds(now, deadline, self.cancellation_poll_interval_seconds)
            )

    def _natural_completion_result(
        self,
        process: subprocess.Popen,
        process_group_id: int,
    ) -> AgentExecutionResult | None:
        """
        Return a natural result only after no live process-group members remain.
        """

        returncode = process.poll()

        if returncode is None:
            return None

        _reap_parent_process(process)

        if returncode == 0:
            result = AgentExecutionResult(
                outcome=AgentExecutionOutcome.SUCCEEDED,
                exit_code=0,
            )
        else:
            result = AgentExecutionResult(
                outcome=AgentExecutionOutcome.FAILED,
                exit_code=returncode,
            )

        if _process_group_has_live_members(process_group_id):
            self._terminate_process_group(process, process_group_id)

        return result

    def _write_stdin_chunk(
        self,
        process: subprocess.Popen,
        process_group_id: int,
        stdin_buffer: bytes | None,
    ) -> bytes | None:
        """
        Write one bounded stdin chunk without preventing lifecycle polling.
        """

        process_stdin = getattr(process, "stdin", None)

        if stdin_buffer is None or process_stdin is None:
            return stdin_buffer

        if not stdin_buffer:
            _close_process_stdin(process)
            return None

        try:
            os.set_blocking(process_stdin.fileno(), False)
            written_byte_count = os.write(
                process_stdin.fileno(),
                stdin_buffer[:STDIN_WRITE_CHUNK_SIZE],
            )
        except BlockingIOError:
            return stdin_buffer
        except BrokenPipeError:
            _close_process_stdin(process)
            return None
        except OSError as exc:
            self._terminate_process_group(process, process_group_id)
            raise AgentExecutionError("Agent process stdin write failed.") from exc

        remaining_buffer = stdin_buffer[written_byte_count:]

        if not remaining_buffer:
            _close_process_stdin(process)
            return None

        return remaining_buffer

    def _cancel_requested_or_fail_safely(
        self,
        process: subprocess.Popen,
        process_group_id: int,
        is_cancel_requested: AgentCancellationCheck | None,
    ) -> bool:
        """
        Check cancellation and terminate safely if the callback fails.
        """

        try:
            return _is_cancellation_requested(is_cancel_requested)
        except Exception as exc:
            self._terminate_process_group(process, process_group_id)
            raise AgentExecutionError("Agent cancellation check failed.") from exc

    def _terminate_for_executor_error(
        self,
        process: subprocess.Popen,
        process_group_id: int,
        original_error: BaseException,
    ) -> None:
        """
        Terminate and reap a process when an unexpected executor path raises.
        """

        _force_kill_process_group_after_lifecycle_error(
            process_group_id,
            original_error,
        )
        self._terminate_parent_after_lifecycle_error_preserving_original(
            process,
            original_error,
        )

    def _terminate_process_group(
        self,
        process: subprocess.Popen,
        process_group_id: int,
    ) -> None:
        """
        Terminate the executor-owned process group and reap the child process.
        """

        lifecycle_error = None

        try:
            if _process_group_has_live_members(process_group_id):
                try:
                    _signal_process_group(process_group_id, signal.SIGTERM)
                except AgentExecutionError as exc:
                    lifecycle_error = exc
                    _signal_parent_process(process, signal.SIGTERM)

            if not self._wait_for_live_process_group_members_to_exit(
                process,
                process_group_id,
                self.termination_grace_seconds,
            ):
                if _process_group_has_live_members(process_group_id):
                    try:
                        _signal_process_group(process_group_id, signal.SIGKILL)
                    except AgentExecutionError as exc:
                        lifecycle_error = exc
                        _signal_parent_process(process, signal.SIGKILL)

                if not self._wait_for_live_process_group_members_to_exit(
                    process,
                    process_group_id,
                    max(
                        self.termination_grace_seconds,
                        self.cancellation_poll_interval_seconds,
                    ),
                ):
                    raise AgentExecutionError(
                        "Agent process group still has live members after termination."
                    )
        except AgentExecutionError as exc:
            _force_kill_process_group_after_lifecycle_error(process_group_id, exc)
            self._terminate_parent_after_lifecycle_error_preserving_original(process, exc)
            raise
        finally:
            _reap_parent_if_exited(process)

        if lifecycle_error is not None:
            _reap_parent_process(process)
            raise lifecycle_error

        _reap_parent_process(process)

    def _wait_for_live_process_group_members_to_exit(
        self,
        process: subprocess.Popen,
        process_group_id: int,
        timeout_seconds: float,
    ) -> bool:
        """
        Wait until the process group no longer contains live members.
        """

        deadline = self.monotonic_clock() + timeout_seconds
        probe_error = None

        while True:
            _reap_parent_if_exited(process)

            try:
                if not _process_group_has_live_members(process_group_id):
                    return True
            except AgentExecutionError as exc:
                probe_error = exc
            else:
                probe_error = None

            if self.monotonic_clock() >= deadline:
                if probe_error is not None:
                    raise probe_error

                return False

            self.sleep(
                min(
                    self.cancellation_poll_interval_seconds,
                    max(0.0, deadline - self.monotonic_clock()),
                )
            )

    def _terminate_parent_after_lifecycle_error(self, process: subprocess.Popen) -> None:
        """
        Best-effort terminate and reap the direct parent before failing closed.
        """

        if process.poll() is None:
            _signal_parent_process(process, signal.SIGTERM)

        if process.poll() is None and not self._wait_for_parent_exit(
            process,
            self.termination_grace_seconds,
        ):
            _signal_parent_process(process, signal.SIGKILL)

            if not self._wait_for_parent_exit(
                process,
                max(
                    self.termination_grace_seconds,
                    self.cancellation_poll_interval_seconds,
                ),
            ):
                raise AgentExecutionError(
                    "Agent process could not be reaped after lifecycle failure."
                )

        _reap_parent_if_exited(process)

    def _terminate_parent_after_lifecycle_error_preserving_original(
        self,
        process: subprocess.Popen,
        original_error: BaseException,
    ) -> None:
        """
        Best-effort cleanup the direct parent without masking the original error.
        """

        try:
            self._terminate_parent_after_lifecycle_error(process)
        except AgentExecutionError:
            _add_exception_note(
                original_error,
                "Agent direct parent cleanup failed after lifecycle error.",
            )

    def _wait_for_parent_exit(
        self,
        process: subprocess.Popen,
        timeout_seconds: float,
    ) -> bool:
        """
        Wait for a bounded interval until the direct child exits.
        """

        deadline = self.monotonic_clock() + timeout_seconds

        while True:
            if process.poll() is not None:
                _reap_parent_process(process)
                return True

            if self.monotonic_clock() >= deadline:
                return False

            self.sleep(
                min(
                    self.cancellation_poll_interval_seconds,
                    max(0.0, deadline - self.monotonic_clock()),
                )
            )


def _setting_value(configured_value: float | None, setting_name: str) -> float:
    """
    Resolve executor configuration from explicit value or Django settings.
    """

    if configured_value is not None:
        return configured_value

    return getattr(settings, setting_name)


def _validate_positive_seconds(value: float, name: str) -> float:
    """
    Validate a positive finite duration.
    """

    value = _coerce_finite_seconds(value, name)

    if value <= 0:
        raise InvalidAgentExecutorConfiguration(f"{name} must be greater than 0")

    return value


def _validate_non_negative_seconds(value: float, name: str) -> float:
    """
    Validate a non-negative finite duration.
    """

    value = _coerce_finite_seconds(value, name)

    if value < 0:
        raise InvalidAgentExecutorConfiguration(f"{name} must be greater than or equal to 0")

    return value


def _coerce_finite_seconds(value: float, name: str) -> float:
    """
    Coerce a duration configuration to a finite float.
    """

    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidAgentExecutorConfiguration(f"{name} must be a finite number") from exc

    if not isfinite(float_value):
        raise InvalidAgentExecutorConfiguration(f"{name} must be a finite number")

    return float_value


def _stdin_bytes(stdin_text: str | None) -> bytes | None:
    """
    Encode stdin text for bounded non-blocking pipe writes.
    """

    if stdin_text is None:
        return None

    return stdin_text.encode("utf-8")


def _next_poll_sleep_seconds(now: float, deadline: float, poll_interval_seconds: float) -> float:
    """
    Return the next bounded polling sleep duration.
    """

    return min(poll_interval_seconds, max(0.0, deadline - now))


def _is_cancellation_requested(is_cancel_requested: AgentCancellationCheck | None) -> bool:
    """
    Return whether cancellation has been requested by the orchestration layer.
    """

    if is_cancel_requested is None:
        return False

    return bool(is_cancel_requested())


def _capture_process_group_id(process: subprocess.Popen) -> int:
    """
    Capture the process group id owned by a successful child spawn.
    """

    process_group_id = int(process.pid)

    if process_group_id <= 0:
        raise AgentExecutionError("Agent process group id is invalid.")

    return process_group_id


def _process_group_has_live_members(
    process_group_id: int,
    proc_root: Path = LINUX_PROC_ROOT,
) -> bool:
    """
    Return whether an executor-owned process group has non-zombie members.

    Linux keeps a process group visible while it contains unreaped zombies.
    Those zombies cannot execute more work, so lifecycle completion depends on
    live members rather than process-group existence alone.
    """

    if proc_root.is_dir():
        return _linux_proc_process_group_has_live_members(process_group_id, proc_root)

    return _process_group_exists(process_group_id)


def _linux_proc_process_group_has_live_members(
    process_group_id: int,
    proc_root: Path,
) -> bool:
    """
    Inspect Linux procfs for live members of one expected process group.
    """

    try:
        proc_entries = list(proc_root.iterdir())
    except OSError as exc:
        raise AgentExecutionError("Agent process metadata cannot be listed.") from exc

    for proc_entry in proc_entries:
        if not proc_entry.name.isdecimal():
            continue

        process_stat = _read_linux_process_stat(proc_entry / "stat")

        if process_stat is None:
            continue

        if process_stat.process_group_id != process_group_id:
            continue

        if process_stat.state not in NON_LIVE_PROCESS_STATES:
            return True

    return False


def _read_linux_process_stat(stat_path: Path) -> ProcessStat | None:
    """
    Read the minimal process stat metadata, treating disappearance as a race.
    """

    try:
        stat_content = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except ProcessLookupError:
        return None
    except OSError as exc:
        raise AgentExecutionError("Agent process metadata cannot be read.") from exc

    try:
        return _parse_linux_process_stat(stat_content)
    except ValueError as exc:
        raise AgentExecutionError("Agent process metadata is invalid.") from exc


def _parse_linux_process_stat(stat_content: str) -> ProcessStat:
    """
    Parse Linux /proc/<pid>/stat without depending on the comm field content.
    """

    closing_parenthesis_index = stat_content.rfind(")")

    if closing_parenthesis_index == -1:
        raise ValueError("Process stat comm field is not terminated")

    fields_after_comm = stat_content[closing_parenthesis_index + 1 :].strip().split()

    if len(fields_after_comm) < 3:
        raise ValueError("Process stat metadata is incomplete")

    return ProcessStat(
        state=fields_after_comm[0],
        process_group_id=int(fields_after_comm[2]),
    )


def _process_group_exists(process_group_id: int) -> bool:
    """
    Probe whether the executor-owned process group still exists.
    """

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise AgentExecutionError("Agent process group existence check failed.") from exc

    return True


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    """
    Signal the process group created for the child process.
    """

    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise AgentExecutionError("Agent process group signal failed.") from exc


def _force_kill_process_group_after_lifecycle_error(
    process_group_id: int,
    original_error: BaseException,
) -> None:
    """
    Best-effort kill the owned process group without consulting liveness probes.
    """

    try:
        _signal_process_group(process_group_id, signal.SIGKILL)
    except AgentExecutionError:
        _add_exception_note(
            original_error,
            "Agent process group force cleanup failed after lifecycle error.",
        )


def _signal_parent_process(process: subprocess.Popen, signal_number: int) -> None:
    """
    Fallback to signaling the direct child when process-group signaling fails.
    """

    try:
        process.send_signal(signal_number)
    except ProcessLookupError:
        return
    except OSError:
        return


def _reap_parent_if_exited(process: subprocess.Popen) -> None:
    """
    Reap the direct child when it has already exited.
    """

    if process.poll() is not None:
        _reap_parent_process(process)


def _reap_parent_process(process: subprocess.Popen) -> None:
    """
    Reap the direct child process.
    """

    process.wait()


def _add_exception_note(error: BaseException, note: str) -> None:
    """
    Attach a safe diagnostic note to an exception when supported.
    """

    error.add_note(note)


def _close_process_stdin(process: subprocess.Popen) -> None:
    """
    Close child stdin if it is still open.
    """

    process_stdin = getattr(process, "stdin", None)

    if process_stdin is None or process_stdin.closed:
        return

    try:
        process_stdin.close()
    except BrokenPipeError:
        return


def _copy_allowed_parent_environment(allowlist: tuple[str, ...]) -> dict[str, str]:
    """
    Copy only explicitly allowed variables from the parent process environment.
    """

    child_environment = {}

    for key in allowlist:
        if key in FORBIDDEN_CHILD_ENVIRONMENT_KEYS:
            raise AgentProcessSpawnError("Parent environment allowlist contains a forbidden key")

        if key in os.environ:
            child_environment[key] = os.environ[key]

    return child_environment


def _merge_environment_overrides(
    child_environment: dict[str, str],
    environment_overrides: Mapping[str, str],
) -> None:
    """
    Merge explicit invocation environment overrides into the child environment.
    """

    for key, value in environment_overrides.items():
        if key in FORBIDDEN_CHILD_ENVIRONMENT_KEYS:
            raise AgentProcessSpawnError(
                "Invocation environment overrides contain a forbidden key"
            )

        child_environment[key] = value


def _assert_forbidden_environment_keys_absent(child_environment: dict[str, str]) -> None:
    """
    Ensure known worker secret variables are absent from the child environment.
    """

    leaked_keys = FORBIDDEN_CHILD_ENVIRONMENT_KEYS.intersection(child_environment)

    if leaked_keys:
        raise AgentProcessSpawnError("Child environment contains a forbidden key")


def _prepare_runtime_home(workspace_path: Path) -> Path:
    """
    Create or validate a workspace-contained HOME directory for the child.
    """

    runtime_path = workspace_path / RUNTIME_DIRECTORY_NAME
    runtime_home_path = runtime_path / RUNTIME_HOME_DIRECTORY_NAME

    try:
        runtime_fd = os.open(runtime_path, OPEN_DIRECTORY_FLAGS)
    except OSError as exc:
        raise AgentProcessSpawnError("Runtime directory cannot be opened") from exc

    try:
        try:
            os.mkdir(
                RUNTIME_HOME_DIRECTORY_NAME,
                PRIVATE_DIRECTORY_MODE,
                dir_fd=runtime_fd,
            )
        except FileExistsError:
            pass

        home_fd = os.open(
            RUNTIME_HOME_DIRECTORY_NAME,
            OPEN_DIRECTORY_FLAGS,
            dir_fd=runtime_fd,
        )
        os.close(home_fd)
    except OSError as exc:
        raise AgentProcessSpawnError("Runtime HOME directory cannot be prepared") from exc
    finally:
        os.close(runtime_fd)

    return runtime_home_path


@contextmanager
def _open_process_log_files(workspace_path: Path):
    """
    Open stdout and stderr log files under the prepared workspace logs directory.
    """

    stack = ExitStack()

    try:
        logs_fd = os.open(workspace_path / LOGS_DIRECTORY_NAME, OPEN_DIRECTORY_FLAGS)
        stack.callback(os.close, logs_fd)
        stdout = stack.enter_context(
            os.fdopen(
                _open_log_file_at(logs_fd, STDOUT_LOG_FILENAME),
                "wb",
            )
        )
        stderr = stack.enter_context(
            os.fdopen(
                _open_log_file_at(logs_fd, STDERR_LOG_FILENAME),
                "wb",
            )
        )
    except OSError as exc:
        stack.close()
        raise AgentProcessSpawnError("Agent process log files cannot be opened") from exc

    try:
        yield ProcessLogFiles(stdout=stdout, stderr=stderr)
    finally:
        stack.close()


def _open_log_file_at(directory_fd: int, filename: str) -> int:
    """
    Open a new private log file relative to a trusted logs directory descriptor.
    """

    return os.open(
        filename,
        OPEN_LOG_FILE_FLAGS,
        PRIVATE_FILE_MODE,
        dir_fd=directory_fd,
    )
