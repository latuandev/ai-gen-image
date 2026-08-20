import hashlib
import json
import os
import signal
import stat
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
from unittest.mock import patch
from uuid import UUID, uuid4

from django.contrib.auth import get_user_model
from django.core.exceptions import ImproperlyConfigured
from django.db import close_old_connections, connection, transaction
from django.test import TestCase, TransactionTestCase, override_settings

from apps.agent_workspace.ai_agent import cli_wrapper as cli_wrapper_module
from apps.agent_workspace.ai_agent import executor as executor_module
from apps.agent_workspace.ai_agent import local_subprocess_executor as local_executor_module
from apps.agent_workspace.ai_agent import workspace_manager as workspace_manager_module
from apps.agent_workspace.ai_agent.cli_wrapper import (
    CODEX_HOME_ENV_KEY,
    CodexCLIInvocation,
    CodexCLIWrapper,
)
from apps.agent_workspace.ai_agent.context_manifest import load_context_manifest
from apps.agent_workspace.ai_agent.executor import (
    AgentExecutionOutcome,
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentExecutor,
)
from apps.agent_workspace.ai_agent.local_subprocess_executor import (
    STDERR_LOG_FILENAME,
    STDOUT_LOG_FILENAME,
    LocalSubprocessExecutor,
)
from apps.agent_workspace.ai_agent.workspace_manager import WorkspaceManager
from apps.agent_workspace.exceptions import (
    AgentExecutionError,
    AgentProcessSpawnError,
    AgentRunContextConflict,
    AgentWorkspaceError,
    InvalidAgentExecutorConfiguration,
    InvalidAgentRunContextState,
    InvalidAgentRunTransition,
    InvalidCodexCLIConfiguration,
    InvalidContextManifest,
    WorkspaceCleanupError,
    WorkspaceIdentityError,
)
from apps.agent_workspace.models import AgentArtifact, AgentRun
from apps.agent_workspace.services.agent_run_service import (
    claim_agent_run_for_execution,
    complete_agent_run_execution,
    is_agent_run_cancellation_requested,
    queue_agent_run_for_execution,
    record_agent_run_context,
    request_agent_run_cancellation,
    transition_agent_run,
)
from apps.agent_workspace.services.execution_service import execute_agent_run_lifecycle
from apps.agent_workspace.tasks import execute_agent_run
from common.constants import AgentRunStatus
from common.utils.helpers import agent_workspace_root_from_env, default_agent_workspace_root
from core.celery import app as celery_app


class AgentExecutorContractTests(TestCase):
    """
    Cover Agent execution request, result, outcome, and interface contracts.
    """

    def test_execution_request_normalizes_run_id_to_canonical_uuid(self):
        run_id = uuid4()

        request = AgentExecutionRequest(
            run_id=str(run_id).upper(),
            workspace_path="/tmp/workspaces/example-run",
            prompt="Generate an image.",
        )

        self.assertEqual(request.run_id, run_id)
        self.assertIsInstance(request.run_id, UUID)

    def test_execution_request_preserves_workspace_path_and_prompt(self):
        workspace_path = Path("/tmp/workspaces/prepared-run")
        prompt = "Create a minimal image."

        request = AgentExecutionRequest(
            run_id=uuid4(),
            workspace_path=workspace_path,
            prompt=prompt,
        )

        self.assertEqual(request.workspace_path, workspace_path)
        self.assertEqual(request.prompt, prompt)

    def test_execution_request_is_immutable(self):
        request = AgentExecutionRequest(
            run_id=uuid4(),
            workspace_path="/tmp/workspaces/immutable-run",
            prompt="Create an immutable request.",
        )

        with self.assertRaises(FrozenInstanceError):
            request.prompt = "Mutate the request."

    def test_execution_outcome_enum_contains_required_semantic_outcomes(self):
        self.assertEqual(
            set(AgentExecutionOutcome),
            {
                AgentExecutionOutcome.SUCCEEDED,
                AgentExecutionOutcome.FAILED,
                AgentExecutionOutcome.TIMED_OUT,
                AgentExecutionOutcome.CANCELLED,
            },
        )

    def test_execution_result_represents_success(self):
        result = AgentExecutionResult(
            outcome=AgentExecutionOutcome.SUCCEEDED,
            exit_code=0,
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)

    def test_execution_result_represents_failure_with_exit_code(self):
        result = AgentExecutionResult(
            outcome=AgentExecutionOutcome.FAILED,
            exit_code=1,
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.FAILED)
        self.assertEqual(result.exit_code, 1)

    def test_execution_result_represents_timeout_without_exit_code(self):
        result = AgentExecutionResult(outcome=AgentExecutionOutcome.TIMED_OUT)

        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)
        self.assertIsNone(result.exit_code)

    def test_execution_result_represents_cancellation_without_exit_code(self):
        result = AgentExecutionResult(outcome=AgentExecutionOutcome.CANCELLED)

        self.assertEqual(result.outcome, AgentExecutionOutcome.CANCELLED)
        self.assertIsNone(result.exit_code)

    def test_execution_result_is_immutable(self):
        result = AgentExecutionResult(
            outcome=AgentExecutionOutcome.FAILED,
            exit_code=2,
        )

        with self.assertRaises(FrozenInstanceError):
            result.exit_code = 3

    def test_execution_result_rejects_ambiguous_success_exit_code(self):
        with self.assertRaises(ValueError):
            AgentExecutionResult(
                outcome=AgentExecutionOutcome.SUCCEEDED,
                exit_code=1,
            )

    def test_execution_result_rejects_ambiguous_failure_exit_code(self):
        with self.assertRaises(ValueError):
            AgentExecutionResult(
                outcome=AgentExecutionOutcome.FAILED,
                exit_code=0,
            )

    def test_execution_result_rejects_exit_code_for_timeout_or_cancellation(self):
        for outcome in [
            AgentExecutionOutcome.TIMED_OUT,
            AgentExecutionOutcome.CANCELLED,
        ]:
            with self.subTest(outcome=outcome):
                with self.assertRaises(ValueError):
                    AgentExecutionResult(outcome=outcome, exit_code=124)

    def test_executor_implementation_can_receive_request_and_return_result(self):
        class FakeAgentExecutor(AgentExecutor):
            """
            Minimal test executor implementation.
            """

            def __init__(self):
                """
                Store the request passed to the fake executor.
                """

                self.received_request = None

            def execute(
                self,
                request: AgentExecutionRequest,
                *,
                is_cancel_requested=None,
            ) -> AgentExecutionResult:
                """
                Return a deterministic executor-level success result.
                """

                self.received_request = request
                return AgentExecutionResult(
                    outcome=AgentExecutionOutcome.SUCCEEDED,
                    exit_code=0,
                )

        request = AgentExecutionRequest(
            run_id=uuid4(),
            workspace_path="/tmp/workspaces/fake-executor-run",
            prompt="Run through the fake executor.",
        )
        executor = FakeAgentExecutor()

        result = executor.execute(request)

        self.assertIs(executor.received_request, request)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)

    def test_executor_contract_does_not_require_agent_run_orm_instance(self):
        request_fields = set(AgentExecutionRequest.__dataclass_fields__)
        result_fields = set(AgentExecutionResult.__dataclass_fields__)

        self.assertEqual(request_fields, {"run_id", "workspace_path", "prompt"})
        self.assertEqual(result_fields, {"outcome", "exit_code"})
        self.assertFalse(hasattr(executor_module, "AgentRun"))


class CodexCLIWrapperTests(TestCase):
    """
    Cover Codex-specific non-interactive invocation construction.
    """

    def create_request(
        self,
        prompt: str = "Generate an image.",
        workspace_path: Path | None = None,
    ) -> AgentExecutionRequest:
        """
        Create a reusable Agent execution request fixture.
        """

        return AgentExecutionRequest(
            run_id=uuid4(),
            workspace_path=workspace_path or Path("/tmp/agent-workspaces/prepared-run"),
            prompt=prompt,
        )

    def test_valid_request_creates_deterministic_invocation(self):
        request = self.create_request()
        wrapper = CodexCLIWrapper()

        first_invocation = wrapper.build_invocation(request)
        second_invocation = wrapper.build_invocation(request)

        self.assertEqual(first_invocation, second_invocation)
        self.assertIsInstance(first_invocation, CodexCLIInvocation)

    def test_executable_is_first_argv_element(self):
        invocation = CodexCLIWrapper().build_invocation(self.create_request())

        self.assertEqual(invocation.argv[0], "codex")

    def test_argv_is_sequence_not_shell_command_string(self):
        invocation = CodexCLIWrapper().build_invocation(self.create_request())

        self.assertIsInstance(invocation.argv, tuple)
        self.assertNotIsInstance(invocation.argv, str)
        self.assertFalse(hasattr(invocation, "shell"))

    def test_workspace_path_becomes_invocation_cwd_and_codex_cd(self):
        workspace_path = Path("/tmp/agent-workspaces/workspace-cwd")
        invocation = CodexCLIWrapper().build_invocation(
            self.create_request(workspace_path=workspace_path)
        )

        self.assertEqual(invocation.cwd, workspace_path)
        self.assertEqual(
            invocation.argv[invocation.argv.index("--cd") + 1],
            str(workspace_path),
        )

    def test_user_prompt_is_transported_through_stdin(self):
        prompt = "Create image from the user's prompt."
        invocation = CodexCLIWrapper().build_invocation(self.create_request(prompt=prompt))

        self.assertEqual(invocation.stdin_text, prompt)
        self.assertEqual(invocation.argv[-1], "-")
        self.assertNotIn(prompt, invocation.argv)

    def test_prompt_with_shell_metacharacters_does_not_change_argv_semantics(self):
        prompt = 'Hello $(cat /etc/passwd) `whoami`; echo unsafe && ls | cat\n"quoted"'
        request = self.create_request(prompt=prompt)
        wrapper = CodexCLIWrapper()

        safe_invocation = wrapper.build_invocation(self.create_request(prompt="safe prompt"))
        metacharacter_invocation = wrapper.build_invocation(request)

        self.assertEqual(
            metacharacter_invocation.argv,
            safe_invocation.argv,
        )
        self.assertEqual(metacharacter_invocation.stdin_text, prompt)
        self.assertNotIn(prompt, metacharacter_invocation.argv)

    def test_invocation_is_non_interactive(self):
        workspace_path = Path("/tmp/agent-workspaces/non-interactive")
        invocation = CodexCLIWrapper().build_invocation(
            self.create_request(workspace_path=workspace_path)
        )

        self.assertEqual(
            invocation.argv,
            (
                "codex",
                "--ask-for-approval",
                "never",
                "exec",
                "--cd",
                str(workspace_path),
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--color",
                "never",
                "--json",
                "-",
            ),
        )
        self.assertIn("exec", invocation.argv)
        exec_index = invocation.argv.index("exec")
        approval_index = invocation.argv.index("--ask-for-approval")

        self.assertLess(approval_index, exec_index)
        self.assertEqual(
            invocation.argv[approval_index + 1],
            "never",
        )
        for exec_option in (
            "--cd",
            "--skip-git-repo-check",
            "--sandbox",
            "--color",
            "--json",
        ):
            with self.subTest(exec_option=exec_option):
                self.assertLess(exec_index, invocation.argv.index(exec_option))

        self.assertIn("--json", invocation.argv)
        self.assertEqual(
            invocation.argv[invocation.argv.index("--color") + 1],
            "never",
        )
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", invocation.argv)

    def test_developer_repository_context_path_is_not_injected(self):
        workspace_path = Path("/tmp/agent-workspaces/no-developer-context")
        invocation = CodexCLIWrapper().build_invocation(
            self.create_request(workspace_path=workspace_path)
        )
        repository_root = Path(__file__).resolve().parents[2]
        invocation_values = [*invocation.argv, *invocation.environment_overrides.values()]

        self.assertNotIn(str(repository_root / "AGENTS.md"), invocation_values)
        self.assertNotIn(str(repository_root / ".codex"), invocation_values)

    def test_developer_home_codex_path_is_not_hard_coded(self):
        workspace_path = Path("/tmp/agent-workspaces/no-home-codex")
        invocation = CodexCLIWrapper().build_invocation(
            self.create_request(workspace_path=workspace_path)
        )
        developer_codex_home = str(Path.home() / ".codex")
        invocation_values = [*invocation.argv, *invocation.environment_overrides.values()]

        self.assertNotIn(developer_codex_home, invocation_values)

    def test_codex_environment_overrides_are_workspace_scoped(self):
        workspace_path = Path("/tmp/agent-workspaces/workspace-codex-home")
        invocation = CodexCLIWrapper().build_invocation(
            self.create_request(workspace_path=workspace_path)
        )

        self.assertEqual(
            dict(invocation.environment_overrides),
            {
                CODEX_HOME_ENV_KEY: str(workspace_path / ".codex"),
            },
        )

    def test_environment_overrides_do_not_contain_worker_secrets(self):
        invocation = CodexCLIWrapper().build_invocation(self.create_request())
        forbidden_keys = {
            "DATABASE_URL",
            "REDIS_URL",
            "DJANGO_SECRET_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        }

        self.assertTrue(forbidden_keys.isdisjoint(invocation.environment_overrides))

    def test_executable_configuration_override_is_used(self):
        invocation = CodexCLIWrapper(executable="/opt/codex/bin/codex").build_invocation(
            self.create_request()
        )

        self.assertEqual(invocation.argv[0], "/opt/codex/bin/codex")

    def test_blank_executable_configuration_is_rejected(self):
        for executable in ["", "  \t\n  "]:
            with self.subTest(executable=executable):
                with self.assertRaises(InvalidCodexCLIConfiguration):
                    CodexCLIWrapper(executable=executable)

    def test_null_byte_executable_configuration_is_rejected(self):
        with self.assertRaises(InvalidCodexCLIConfiguration):
            CodexCLIWrapper(executable="codex\x00")

    def test_wrapper_does_not_mutate_input_request(self):
        request = self.create_request(prompt="Preserve this prompt.")
        original_values = (request.run_id, request.workspace_path, request.prompt)

        CodexCLIWrapper().build_invocation(request)

        self.assertEqual(
            (request.run_id, request.workspace_path, request.prompt),
            original_values,
        )

    def test_wrapper_does_not_access_agent_run_orm_or_state(self):
        self.assertFalse(hasattr(cli_wrapper_module, "AgentRun"))
        self.assertFalse(hasattr(cli_wrapper_module, "AgentRunStatus"))

    def test_wrapper_runtime_implementation_does_not_import_subprocess(self):
        self.assertFalse(hasattr(cli_wrapper_module, "subprocess"))


class LocalSubprocessExecutorTests(TestCase):
    """
    Cover trusted local subprocess execution behavior.
    """

    def setUp(self):
        """
        Create isolated filesystem fixtures for subprocess executor tests.
        """

        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.temp_root = Path(self.temp_directory.name).resolve()
        self.fake_bin_path = self.temp_root / "bin"
        self.fake_bin_path.mkdir()

    def create_workspace(self, name: str = "prepared-run") -> Path:
        """
        Create a prepared-workspace-shaped test fixture.
        """

        workspace_path = self.temp_root / name
        workspace_path.mkdir()

        for directory_name in [
            "inputs",
            "outputs",
            "runtime",
            "logs",
            ".codex",
        ]:
            (workspace_path / directory_name).mkdir()

        (workspace_path / "AGENTS.md").write_text("Use the workspace only.\n", encoding="utf-8")

        return workspace_path

    def create_request(
        self,
        workspace_path: Path,
        prompt: str = "Generate an image.",
    ) -> AgentExecutionRequest:
        """
        Create a subprocess execution request fixture.
        """

        return AgentExecutionRequest(
            run_id=uuid4(),
            workspace_path=workspace_path,
            prompt=prompt,
        )

    def create_fake_codex_executable(
        self,
        exit_code: int = 0,
        executable_name: str = "fake-codex",
    ) -> Path:
        """
        Create a deterministic executable that behaves like a short-lived CLI.
        """

        executable_path = self.fake_bin_path / executable_name
        inspected_environment_keys = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "CELERY_BROKER_URL",
            "CODEX_HOME",
            "DATABASE_PASSWORD",
            "DATABASE_URL",
            "DJANGO_SECRET_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HOME",
            "LANG",
            "PATH",
            "POSTGRES_PASSWORD",
            "REDIS_URL",
            "SECRET_KEY",
            "SENTRY_DSN",
        ]
        executable_path.write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import json",
                    "import os",
                    "import sys",
                    f"INSPECTED_ENVIRONMENT_KEYS = {inspected_environment_keys!r}",
                    "stdin_text = sys.stdin.read()",
                    "payload = {",
                    "    'argv': sys.argv,",
                    "    'cwd': os.getcwd(),",
                    "    'stdin': stdin_text,",
                    "    'env': {",
                    "        key: os.environ[key]",
                    "        for key in INSPECTED_ENVIRONMENT_KEYS",
                    "        if key in os.environ",
                    "    },",
                    "}",
                    "print(json.dumps(payload, sort_keys=True))",
                    "print('stdout marker')",
                    "print('stderr marker', file=sys.stderr)",
                    f"sys.exit({exit_code})",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        executable_path.chmod(0o700)

        return executable_path

    def create_lifecycle_executable(
        self,
        *,
        executable_name: str,
        exit_code: int = 0,
        sleep_seconds: float = 0.0,
        ignore_sigterm: bool = False,
        spawn_child: bool = False,
        child_pid_path: Path | None = None,
        side_effect_path: Path | None = None,
    ) -> Path:
        """
        Create a fake executable for timeout and cancellation lifecycle tests.
        """

        executable_path = self.fake_bin_path / executable_name
        child_code = "\n".join(
            [
                "import signal",
                "import sys",
                "import time",
                "",
                "def handle_sigterm(signum, frame):",
                "    sys.exit(0)",
                "",
                "signal.signal(signal.SIGTERM, handle_sigterm)",
                "time.sleep(30)",
                "",
            ]
        )
        executable_path.write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import os",
                    "import signal",
                    "import subprocess",
                    "import sys",
                    "import time",
                    f"EXIT_CODE = {exit_code!r}",
                    f"SLEEP_SECONDS = {sleep_seconds!r}",
                    f"IGNORE_SIGTERM = {ignore_sigterm!r}",
                    f"SPAWN_CHILD = {spawn_child!r}",
                    f"CHILD_PID_PATH = {str(child_pid_path) if child_pid_path else None!r}",
                    f"SIDE_EFFECT_PATH = {str(side_effect_path) if side_effect_path else None!r}",
                    "child_process = None",
                    "",
                    "def handle_sigterm(signum, frame):",
                    "    if IGNORE_SIGTERM:",
                    "        return",
                    "    if child_process is not None:",
                    "        child_process.terminate()",
                    "        try:",
                    "            child_process.wait(timeout=2)",
                    "        except subprocess.TimeoutExpired:",
                    "            child_process.kill()",
                    "            child_process.wait()",
                    "    sys.exit(0)",
                    "",
                    "signal.signal(signal.SIGTERM, handle_sigterm)",
                    "if SPAWN_CHILD:",
                    f"    child_code = {child_code!r}",
                    "    child_process = subprocess.Popen([sys.executable, '-c', child_code])",
                    "    with open(CHILD_PID_PATH, 'w', encoding='utf-8') as child_pid_file:",
                    "        child_pid_file.write(str(child_process.pid))",
                    "stdin_text = sys.stdin.read()",
                    "if SIDE_EFFECT_PATH is not None:",
                    "    with open(SIDE_EFFECT_PATH, 'w', encoding='utf-8') as side_effect_file:",
                    "        side_effect_file.write('started\\n')",
                    "print('started', flush=True)",
                    "if SLEEP_SECONDS:",
                    "    time.sleep(SLEEP_SECONDS)",
                    "if child_process is not None:",
                    "    child_process.terminate()",
                    "    child_process.wait()",
                    "sys.exit(EXIT_CODE)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        executable_path.chmod(0o700)

        return executable_path

    def create_descendant_escape_executable(
        self,
        *,
        executable_name: str,
        process_metadata_path: Path,
        exit_code: int = 0,
        child_ignores_sigterm: bool = False,
        parent_exits_on_sigterm: bool = False,
    ) -> Path:
        """
        Create an executable whose descendant stays in the executor process group.
        """

        executable_path = self.fake_bin_path / executable_name
        child_ready_path = process_metadata_path.with_suffix(".ready")
        child_lines = [
            "import os",
            "import signal",
            f"READY_PATH = {str(child_ready_path)!r}",
            "",
        ]

        if child_ignores_sigterm:
            child_lines.extend(
                [
                    "def handle_sigterm(signum, frame):",
                    "    return",
                    "",
                    "signal.signal(signal.SIGTERM, handle_sigterm)",
                ]
            )

        child_lines.extend(
            [
                "with open(READY_PATH, 'w', encoding='utf-8') as ready_file:",
                "    ready_file.write('ready\\n')",
                "while True:",
                "    signal.pause()",
                "",
            ]
        )
        child_code = "\n".join(child_lines)
        executable_path.write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import json",
                    "import os",
                    "import signal",
                    "import subprocess",
                    "import sys",
                    "import time",
                    f"EXIT_CODE = {exit_code!r}",
                    f"CHILD_CODE = {child_code!r}",
                    f"CHILD_READY_PATH = {str(child_ready_path)!r}",
                    f"METADATA_PATH = {str(process_metadata_path)!r}",
                    f"PARENT_EXITS_ON_SIGTERM = {parent_exits_on_sigterm!r}",
                    "",
                    "def handle_sigterm(signum, frame):",
                    "    if PARENT_EXITS_ON_SIGTERM:",
                    "        sys.exit(0)",
                    "",
                    "signal.signal(signal.SIGTERM, handle_sigterm)",
                    "child_process = subprocess.Popen([sys.executable, '-c', CHILD_CODE])",
                    "for _ in range(500):",
                    "    if os.path.exists(CHILD_READY_PATH):",
                    "        break",
                    "    time.sleep(0.01)",
                    "else:",
                    "    sys.exit(98)",
                    "payload = {",
                    "    'child_pid': child_process.pid,",
                    "    'process_group_id': os.getpgrp(),",
                    "}",
                    "with open(METADATA_PATH, 'w', encoding='utf-8') as metadata_file:",
                    "    json.dump(payload, metadata_file, sort_keys=True)",
                    "stdin_text = sys.stdin.read()",
                    "if PARENT_EXITS_ON_SIGTERM:",
                    "    signal.pause()",
                    f"sys.exit({exit_code})",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        executable_path.chmod(0o700)

        return executable_path

    def create_detached_descendant_executable(
        self,
        *,
        executable_name: str,
        process_metadata_path: Path,
    ) -> Path:
        """
        Create an executable whose child leaves the executor process group.
        """

        executable_path = self.fake_bin_path / executable_name
        child_ready_path = process_metadata_path.with_suffix(".ready")
        child_code = "\n".join(
            [
                "import os",
                "import signal",
                f"READY_PATH = {str(child_ready_path)!r}",
                "with open(READY_PATH, 'w', encoding='utf-8') as ready_file:",
                "    ready_file.write('ready\\n')",
                "while True:",
                "    signal.pause()",
                "",
            ]
        )
        executable_path.write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import json",
                    "import os",
                    "import subprocess",
                    "import sys",
                    "import time",
                    f"CHILD_CODE = {child_code!r}",
                    f"CHILD_READY_PATH = {str(child_ready_path)!r}",
                    f"METADATA_PATH = {str(process_metadata_path)!r}",
                    "child_process = subprocess.Popen(",
                    "    [sys.executable, '-c', CHILD_CODE],",
                    "    start_new_session=True,",
                    ")",
                    "for _ in range(500):",
                    "    if os.path.exists(CHILD_READY_PATH):",
                    "        break",
                    "    time.sleep(0.01)",
                    "else:",
                    "    sys.exit(98)",
                    "payload = {",
                    "    'child_pid': child_process.pid,",
                    "    'child_process_group_id': os.getpgid(child_process.pid),",
                    "    'parent_process_group_id': os.getpgrp(),",
                    "}",
                    "with open(METADATA_PATH, 'w', encoding='utf-8') as metadata_file:",
                    "    json.dump(payload, metadata_file, sort_keys=True)",
                    "stdin_text = sys.stdin.read()",
                    "sys.exit(0)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        executable_path.chmod(0o700)

        return executable_path

    def create_executor(
        self,
        executable_path: Path | str,
        *,
        timeout_seconds: float = 5.0,
        termination_grace_seconds: float = 0.5,
        cancellation_poll_interval_seconds: float = 0.05,
    ) -> LocalSubprocessExecutor:
        """
        Create a local subprocess executor using a fake Codex executable.
        """

        return LocalSubprocessExecutor(
            cli_wrapper=CodexCLIWrapper(executable=str(executable_path)),
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
            cancellation_poll_interval_seconds=cancellation_poll_interval_seconds,
        )

    def read_stdout_payload(self, workspace_path: Path) -> dict:
        """
        Read the JSON payload written by the fake executable.
        """

        stdout_text = (workspace_path / "logs" / STDOUT_LOG_FILENAME).read_text(encoding="utf-8")

        return json.loads(stdout_text.splitlines()[0])

    def pid_exists(self, pid: int) -> bool:
        """
        Return whether a process id still exists.
        """

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

        return True

    def process_group_exists(self, process_group_id: int) -> bool:
        """
        Return whether a process group still exists.
        """

        try:
            os.kill(-process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

        return True

    def process_is_live(self, pid: int) -> bool:
        """
        Return whether a process exists and is not a zombie/dead process.
        """

        proc_root = Path("/proc")

        if proc_root.is_dir():
            process_stat = local_executor_module._read_linux_process_stat(
                proc_root / str(pid) / "stat"
            )

            return (
                process_stat is not None
                and process_stat.state not in local_executor_module.NON_LIVE_PROCESS_STATES
            )

        return self.pid_exists(pid)

    def process_group_has_live_members(self, process_group_id: int) -> bool:
        """
        Return whether a process group contains a non-zombie process.
        """

        proc_root = Path("/proc")

        if proc_root.is_dir():
            return local_executor_module._process_group_has_live_members(process_group_id)

        return self.process_group_exists(process_group_id)

    def cleanup_process_group_from_metadata(self, process_metadata_path: Path) -> None:
        """
        Kill a recorded process group if a regression assertion fails.
        """

        if not process_metadata_path.exists():
            return

        metadata = json.loads(process_metadata_path.read_text(encoding="utf-8"))

        try:
            os.killpg(metadata["process_group_id"], signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return

    def cleanup_detached_pid_from_metadata(self, process_metadata_path: Path) -> None:
        """
        Kill a recorded detached process if a limitation test fails.
        """

        if not process_metadata_path.exists():
            return

        metadata = json.loads(process_metadata_path.read_text(encoding="utf-8"))

        try:
            os.kill(metadata["child_pid"], signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return

    def read_process_metadata(self, process_metadata_path: Path) -> dict:
        """
        Read descendant process metadata written by a fake executable.
        """

        return json.loads(process_metadata_path.read_text(encoding="utf-8"))

    def write_fake_proc_stat(
        self,
        proc_root: Path,
        *,
        pid: int,
        process_group_id: int,
        state: str,
        comm: str = "fake process",
    ) -> None:
        """
        Write a minimal Linux proc stat fixture.
        """

        process_root = proc_root / str(pid)
        process_root.mkdir()
        stat_fields = [
            str(pid),
            f"({comm})",
            state,
            "1",
            str(process_group_id),
            "1",
            "0",
            "0",
        ]
        (process_root / "stat").write_text(" ".join(stat_fields), encoding="utf-8")

    def live_probe_side_effect(self, values):
        """
        Return a process-group live probe side effect with a stable terminal value.
        """

        remaining_values = list(values)
        observed_values = []

        def probe(process_group_id):
            if remaining_values:
                value = remaining_values.pop(0)
                observed_values.append(value)
                return value

            observed_values.append(False)
            return False

        probe.observed_values = observed_values

        return probe

    def test_linux_process_stat_parser_handles_comm_with_spaces_and_parentheses(self):
        process_stat = local_executor_module._parse_linux_process_stat(
            "123 (name with ) spaces) S 1 456 1 0 0"
        )

        self.assertEqual(process_stat.state, "S")
        self.assertEqual(process_stat.process_group_id, 456)

    def test_process_group_live_member_helper_ignores_zombie_only_group(self):
        proc_root = self.temp_root / "proc-zombie-only"
        proc_root.mkdir()
        self.write_fake_proc_stat(
            proc_root,
            pid=1001,
            process_group_id=9001,
            state="Z",
        )

        self.assertFalse(
            local_executor_module._process_group_has_live_members(
                9001,
                proc_root=proc_root,
            )
        )

    def test_process_group_live_member_helper_detects_non_zombie_states(self):
        for state in ["R", "S", "D", "T"]:
            with self.subTest(state=state):
                proc_root = self.temp_root / f"proc-live-{state}"
                proc_root.mkdir()
                self.write_fake_proc_stat(
                    proc_root,
                    pid=1001,
                    process_group_id=9001,
                    state=state,
                )

                self.assertTrue(
                    local_executor_module._process_group_has_live_members(
                        9001,
                        proc_root=proc_root,
                    )
                )

    def test_process_group_live_member_helper_ignores_other_process_groups(self):
        proc_root = self.temp_root / "proc-other-group"
        proc_root.mkdir()
        self.write_fake_proc_stat(
            proc_root,
            pid=1001,
            process_group_id=9002,
            state="S",
        )

        self.assertFalse(
            local_executor_module._process_group_has_live_members(
                9001,
                proc_root=proc_root,
            )
        )

    def test_process_group_live_member_helper_handles_stat_disappearance_race(self):
        proc_root = self.temp_root / "proc-disappeared"
        proc_root.mkdir()
        (proc_root / "1001").mkdir()

        self.assertFalse(
            local_executor_module._process_group_has_live_members(
                9001,
                proc_root=proc_root,
            )
        )

    def test_process_group_already_gone_before_signal_returns_natural_result(self):
        workspace_path = self.create_workspace("group-gone-before-signal")
        executable_path = self.create_fake_codex_executable(exit_code=0)

        with (
            patch(
                "apps.agent_workspace.ai_agent.local_subprocess_executor."
                "_process_group_has_live_members",
                return_value=False,
            ),
            patch(
                "apps.agent_workspace.ai_agent.local_subprocess_executor._signal_process_group",
            ) as signal_process_group,
        ):
            result = self.create_executor(executable_path).execute(
                self.create_request(workspace_path)
            )

        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)
        signal_process_group.assert_not_called()

    def test_local_subprocess_executor_conforms_agent_executor(self):
        executor = self.create_executor(self.create_fake_codex_executable())

        self.assertIsInstance(executor, AgentExecutor)

    def test_invalid_timeout_configuration_is_rejected(self):
        for timeout_seconds in [0, -1, float("inf"), "not-a-number"]:
            with self.subTest(timeout_seconds=timeout_seconds):
                with self.assertRaises(InvalidAgentExecutorConfiguration):
                    LocalSubprocessExecutor(timeout_seconds=timeout_seconds)

    def test_invalid_grace_period_configuration_is_rejected(self):
        for termination_grace_seconds in [-1, float("inf"), "not-a-number"]:
            with self.subTest(termination_grace_seconds=termination_grace_seconds):
                with self.assertRaises(InvalidAgentExecutorConfiguration):
                    LocalSubprocessExecutor(termination_grace_seconds=termination_grace_seconds)

    def test_invalid_cancellation_poll_configuration_is_rejected(self):
        for cancellation_poll_interval_seconds in [0, -1, float("inf"), "not-a-number"]:
            with self.subTest(
                cancellation_poll_interval_seconds=cancellation_poll_interval_seconds
            ):
                with self.assertRaises(InvalidAgentExecutorConfiguration):
                    LocalSubprocessExecutor(
                        cancellation_poll_interval_seconds=cancellation_poll_interval_seconds
                    )

    def test_executor_uses_codex_cli_wrapper_to_build_invocation(self):
        workspace_path = self.create_workspace("wrapper-used")
        executable_path = self.create_fake_codex_executable()
        request = self.create_request(workspace_path, prompt="Use injected wrapper.")

        class FakeCodexCLIWrapper:
            """
            Build a deterministic invocation and record the received request.
            """

            def __init__(self):
                """
                Store wrapper call state for assertions.
                """

                self.received_request = None

            def build_invocation(self, received_request):
                """
                Return a fake executable invocation.
                """

                self.received_request = received_request
                return CodexCLIInvocation(
                    argv=(str(executable_path),),
                    cwd=workspace_path,
                    stdin_text=received_request.prompt,
                    environment_overrides={
                        CODEX_HOME_ENV_KEY: str(workspace_path / ".codex"),
                    },
                )

        cli_wrapper = FakeCodexCLIWrapper()
        executor = LocalSubprocessExecutor(cli_wrapper=cli_wrapper)

        result = executor.execute(request)

        self.assertIs(cli_wrapper.received_request, request)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)

    def test_subprocess_receives_argv_sequence_and_shell_false(self):
        workspace_path = self.create_workspace("argv-sequence")
        executable_path = self.create_fake_codex_executable()
        request = self.create_request(workspace_path)
        executor = self.create_executor(executable_path)
        captured = {}

        class FakeProcess:
            """
            Minimal successful Popen replacement.
            """

            pid = 999999
            returncode = 0
            stdin = None

            def poll(self):
                """
                Return immediate successful completion.
                """

                return self.returncode

            def wait(self):
                """
                Return the already completed exit code.
                """

                return self.returncode

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

            return FakeProcess()

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor.subprocess.Popen",
            side_effect=fake_popen,
        ):
            result = executor.execute(request)

        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertIsInstance(captured["argv"], tuple)
        self.assertFalse(captured["kwargs"]["shell"])
        self.assertTrue(captured["kwargs"]["start_new_session"])

    def test_subprocess_cwd_is_prepared_workspace(self):
        workspace_path = self.create_workspace("cwd")
        executable_path = self.create_fake_codex_executable()

        self.create_executor(executable_path).execute(self.create_request(workspace_path))

        payload = self.read_stdout_payload(workspace_path)
        self.assertEqual(payload["cwd"], str(workspace_path))

    def test_stdin_text_is_transmitted_exactly_and_eof_is_delivered(self):
        prompt = "Line one\nline two with 'quotes' and \"double quotes\"."
        workspace_path = self.create_workspace("stdin")
        executable_path = self.create_fake_codex_executable()

        result = self.create_executor(executable_path).execute(
            self.create_request(workspace_path, prompt=prompt)
        )

        payload = self.read_stdout_payload(workspace_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(payload["stdin"], prompt)

    def test_missing_stdin_content_uses_non_interactive_stdin(self):
        workspace_path = self.create_workspace("stdin-none")
        executable_path = self.create_fake_codex_executable()
        invocation = CodexCLIInvocation(
            argv=(str(executable_path),),
            cwd=workspace_path,
            stdin_text=None,
            environment_overrides={
                CODEX_HOME_ENV_KEY: str(workspace_path / ".codex"),
            },
        )

        class NoStdinWrapper:
            """
            Return an invocation without stdin text.
            """

            def build_invocation(self, request):
                """
                Return the no-stdin invocation.
                """

                return invocation

        result = LocalSubprocessExecutor(cli_wrapper=NoStdinWrapper()).execute(
            self.create_request(workspace_path)
        )

        payload = self.read_stdout_payload(workspace_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(payload["stdin"], "")

    def test_stdout_and_stderr_are_written_to_workspace_logs(self):
        workspace_path = self.create_workspace("logs")
        executable_path = self.create_fake_codex_executable()

        self.create_executor(executable_path).execute(self.create_request(workspace_path))

        stdout_text = (workspace_path / "logs" / STDOUT_LOG_FILENAME).read_text(encoding="utf-8")
        stderr_text = (workspace_path / "logs" / STDERR_LOG_FILENAME).read_text(encoding="utf-8")

        self.assertIn("stdout marker", stdout_text)
        self.assertNotIn("stderr marker", stdout_text)
        self.assertIn("stderr marker", stderr_text)
        self.assertNotIn("stdout marker", stderr_text)

    def test_exit_zero_returns_succeeded_result(self):
        workspace_path = self.create_workspace("success")
        executable_path = self.create_fake_codex_executable(exit_code=0)

        result = self.create_executor(executable_path).execute(self.create_request(workspace_path))

        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)

    def test_non_zero_exit_returns_failed_result_with_exit_code(self):
        workspace_path = self.create_workspace("failure")
        executable_path = self.create_fake_codex_executable(exit_code=7)

        result = self.create_executor(executable_path).execute(self.create_request(workspace_path))

        self.assertEqual(result.outcome, AgentExecutionOutcome.FAILED)
        self.assertEqual(result.exit_code, 7)

    def test_natural_success_cleans_descendant_before_returning(self):
        workspace_path = self.create_workspace("natural-success-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="natural-success-descendant-codex",
            process_metadata_path=process_metadata_path,
            exit_code=0,
        )

        result = self.create_executor(
            executable_path,
            termination_grace_seconds=0.2,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(self.process_is_live(metadata["child_pid"]))
        self.assertFalse(self.process_group_has_live_members(metadata["process_group_id"]))

    def test_natural_success_returns_when_sigkill_leaves_zombie_only_group(self):
        workspace_path = self.create_workspace("natural-success-zombie-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="natural-success-zombie-descendant-codex",
            process_metadata_path=process_metadata_path,
            exit_code=0,
            child_ignores_sigterm=True,
        )

        live_probe = self.live_probe_side_effect([True, True, True, True, False])

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor."
            "_process_group_has_live_members",
            side_effect=live_probe,
        ):
            result = self.create_executor(
                executable_path,
                termination_grace_seconds=0.0,
                cancellation_poll_interval_seconds=0.02,
            ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("child_pid", metadata)
        self.assertIn(False, live_probe.observed_values)

    def test_liveness_probe_failure_force_kills_group_after_parent_exit(self):
        workspace_path = self.create_workspace("probe-failure-parent-exited")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="probe-failure-parent-exited-codex",
            process_metadata_path=process_metadata_path,
            exit_code=0,
        )
        injected_error = AgentExecutionError("Injected liveness probe failure.")

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor."
            "_process_group_has_live_members",
            side_effect=injected_error,
        ) as live_probe:
            with self.assertRaises(AgentExecutionError) as context:
                self.create_executor(
                    executable_path,
                    termination_grace_seconds=0.05,
                    cancellation_poll_interval_seconds=0.02,
                ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertIs(context.exception, injected_error)
        self.assertEqual(live_probe.call_count, 1)
        self.assertFalse(self.process_is_live(metadata["child_pid"]))
        self.assertFalse(self.process_group_has_live_members(metadata["process_group_id"]))

    def test_natural_failure_cleans_descendant_and_preserves_exit_code(self):
        workspace_path = self.create_workspace("natural-failure-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="natural-failure-descendant-codex",
            process_metadata_path=process_metadata_path,
            exit_code=7,
        )

        result = self.create_executor(
            executable_path,
            termination_grace_seconds=0.2,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.FAILED)
        self.assertEqual(result.exit_code, 7)
        self.assertFalse(self.process_is_live(metadata["child_pid"]))
        self.assertFalse(self.process_group_has_live_members(metadata["process_group_id"]))

    def test_natural_failure_returns_when_sigkill_leaves_zombie_only_group(self):
        workspace_path = self.create_workspace("natural-failure-zombie-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="natural-failure-zombie-descendant-codex",
            process_metadata_path=process_metadata_path,
            exit_code=7,
            child_ignores_sigterm=True,
        )

        live_probe = self.live_probe_side_effect([True, True, True, True, False])

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor."
            "_process_group_has_live_members",
            side_effect=live_probe,
        ):
            result = self.create_executor(
                executable_path,
                termination_grace_seconds=0.0,
                cancellation_poll_interval_seconds=0.02,
            ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.FAILED)
        self.assertEqual(result.exit_code, 7)
        self.assertIn("child_pid", metadata)
        self.assertIn(False, live_probe.observed_values)

    def test_detached_descendant_is_outside_local_process_group_guarantee(self):
        workspace_path = self.create_workspace("detached-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_detached_pid_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_detached_descendant_executable(
            executable_name="detached-descendant-codex",
            process_metadata_path=process_metadata_path,
        )

        result = self.create_executor(
            executable_path,
            termination_grace_seconds=0.05,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertNotEqual(
            metadata["child_process_group_id"],
            metadata["parent_process_group_id"],
        )
        self.assertTrue(self.process_is_live(metadata["child_pid"]))

    def test_executable_not_found_raises_spawn_exception(self):
        workspace_path = self.create_workspace("missing-executable")
        missing_executable_path = self.temp_root / "missing-codex"

        with self.assertRaises(AgentProcessSpawnError):
            self.create_executor(missing_executable_path).execute(
                self.create_request(workspace_path)
            )

    def test_process_is_reaped_by_communicate(self):
        workspace_path = self.create_workspace("reaped")
        executable_path = self.create_fake_codex_executable()
        request = self.create_request(workspace_path)
        executor = self.create_executor(executable_path)
        captured = {}

        class FakeProcess:
            """
            Record process communication for reaping semantics.
            """

            pid = 999999
            returncode = 0
            stdin = None

            def __init__(self):
                """
                Track whether wait was called.
                """

                self.waited = False

            def poll(self):
                """
                Return immediate successful completion.
                """

                return self.returncode

            def wait(self):
                """
                Mark the process as reaped.
                """

                self.waited = True
                return self.returncode

        def fake_popen(*args, **kwargs):
            process = FakeProcess()
            captured["process"] = process

            return process

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor.subprocess.Popen",
            side_effect=fake_popen,
        ):
            executor.execute(request)

        self.assertTrue(captured["process"].waited)

    def test_timeout_terminates_process_group_and_returns_timed_out(self):
        workspace_path = self.create_workspace("timeout")
        executable_path = self.create_lifecycle_executable(
            executable_name="timeout-codex",
            sleep_seconds=10,
        )

        result = self.create_executor(
            executable_path,
            timeout_seconds=0.2,
            termination_grace_seconds=0.2,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)
        self.assertIsNone(result.exit_code)

    def test_timeout_force_kills_process_that_ignores_sigterm(self):
        workspace_path = self.create_workspace("timeout-force-kill")
        executable_path = self.create_lifecycle_executable(
            executable_name="timeout-force-codex",
            sleep_seconds=10,
            ignore_sigterm=True,
        )

        result = self.create_executor(
            executable_path,
            timeout_seconds=0.2,
            termination_grace_seconds=0.05,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)
        self.assertIsNone(result.exit_code)

    def test_timeout_kills_descendant_after_parent_exits_on_sigterm(self):
        workspace_path = self.create_workspace("timeout-descendant-ignores-sigterm")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="timeout-descendant-codex",
            process_metadata_path=process_metadata_path,
            child_ignores_sigterm=True,
            parent_exits_on_sigterm=True,
        )

        result = self.create_executor(
            executable_path,
            timeout_seconds=1.0,
            termination_grace_seconds=0.05,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)
        self.assertIsNone(result.exit_code)
        self.assertFalse(self.process_is_live(metadata["child_pid"]))
        self.assertFalse(self.process_group_has_live_members(metadata["process_group_id"]))

    def test_timeout_returns_when_sigkill_leaves_zombie_only_group(self):
        workspace_path = self.create_workspace("timeout-zombie-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="timeout-zombie-descendant-codex",
            process_metadata_path=process_metadata_path,
            child_ignores_sigterm=True,
            parent_exits_on_sigterm=True,
        )

        live_probe = self.live_probe_side_effect([True, True, True, False])

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor."
            "_process_group_has_live_members",
            side_effect=live_probe,
        ):
            result = self.create_executor(
                executable_path,
                timeout_seconds=1.0,
                termination_grace_seconds=0.0,
                cancellation_poll_interval_seconds=0.02,
            ).execute(self.create_request(workspace_path))

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)
        self.assertIsNone(result.exit_code)
        self.assertIn("child_pid", metadata)
        self.assertIn(False, live_probe.observed_values)

    def test_timeout_terminates_spawned_child_process(self):
        workspace_path = self.create_workspace("timeout-child")
        child_pid_path = workspace_path / "runtime" / "child.pid"
        executable_path = self.create_lifecycle_executable(
            executable_name="timeout-child-codex",
            sleep_seconds=10,
            spawn_child=True,
            child_pid_path=child_pid_path,
        )

        result = self.create_executor(
            executable_path,
            timeout_seconds=1.0,
            termination_grace_seconds=0.3,
            cancellation_poll_interval_seconds=0.02,
        ).execute(self.create_request(workspace_path))

        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)
        self.assertFalse(self.process_is_live(child_pid))

    def test_cancellation_callback_terminates_process_and_returns_cancelled(self):
        workspace_path = self.create_workspace("cancel")
        executable_path = self.create_lifecycle_executable(
            executable_name="cancel-codex",
            sleep_seconds=10,
        )
        cancellation_checks = iter([False, False, True])

        result = self.create_executor(
            executable_path,
            timeout_seconds=5,
            termination_grace_seconds=0.2,
            cancellation_poll_interval_seconds=0.02,
        ).execute(
            self.create_request(workspace_path),
            is_cancel_requested=lambda: next(cancellation_checks, True),
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.CANCELLED)
        self.assertIsNone(result.exit_code)

    def test_cancellation_kills_descendant_after_parent_exits_on_sigterm(self):
        workspace_path = self.create_workspace("cancel-descendant-ignores-sigterm")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="cancel-descendant-codex",
            process_metadata_path=process_metadata_path,
            child_ignores_sigterm=True,
            parent_exits_on_sigterm=True,
        )

        result = self.create_executor(
            executable_path,
            timeout_seconds=5,
            termination_grace_seconds=0.05,
            cancellation_poll_interval_seconds=0.02,
        ).execute(
            self.create_request(workspace_path),
            is_cancel_requested=lambda: process_metadata_path.exists(),
        )

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.CANCELLED)
        self.assertIsNone(result.exit_code)
        self.assertFalse(self.process_is_live(metadata["child_pid"]))
        self.assertFalse(self.process_group_has_live_members(metadata["process_group_id"]))

    def test_cancellation_returns_when_sigkill_leaves_zombie_only_group(self):
        workspace_path = self.create_workspace("cancel-zombie-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="cancel-zombie-descendant-codex",
            process_metadata_path=process_metadata_path,
            child_ignores_sigterm=True,
            parent_exits_on_sigterm=True,
        )

        live_probe = self.live_probe_side_effect([True, True, True, False])

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor."
            "_process_group_has_live_members",
            side_effect=live_probe,
        ):
            result = self.create_executor(
                executable_path,
                timeout_seconds=5.0,
                termination_grace_seconds=0.0,
                cancellation_poll_interval_seconds=0.02,
            ).execute(
                self.create_request(workspace_path),
                is_cancel_requested=lambda: process_metadata_path.exists(),
            )

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertEqual(result.outcome, AgentExecutionOutcome.CANCELLED)
        self.assertIsNone(result.exit_code)
        self.assertIn("child_pid", metadata)
        self.assertIn(False, live_probe.observed_values)

    def test_pre_spawn_cancellation_does_not_launch_process(self):
        workspace_path = self.create_workspace("pre-spawn-cancel")
        side_effect_path = workspace_path / "runtime" / "started.txt"
        executable_path = self.create_lifecycle_executable(
            executable_name="pre-spawn-cancel-codex",
            side_effect_path=side_effect_path,
        )

        result = self.create_executor(executable_path).execute(
            self.create_request(workspace_path),
            is_cancel_requested=lambda: True,
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.CANCELLED)
        self.assertFalse(side_effect_path.exists())
        self.assertFalse((workspace_path / "logs" / STDOUT_LOG_FILENAME).exists())

    def test_pre_spawn_cancellation_callback_failure_raises_execution_error(self):
        workspace_path = self.create_workspace("pre-spawn-cancel-failure")
        side_effect_path = workspace_path / "runtime" / "started.txt"
        executable_path = self.create_lifecycle_executable(
            executable_name="pre-spawn-cancel-failure-codex",
            side_effect_path=side_effect_path,
        )

        with self.assertRaises(AgentExecutionError):
            self.create_executor(executable_path).execute(
                self.create_request(workspace_path),
                is_cancel_requested=lambda: (_ for _ in ()).throw(RuntimeError("Read failed.")),
            )

        self.assertFalse(side_effect_path.exists())

    def test_natural_success_observed_before_cancellation_wins(self):
        workspace_path = self.create_workspace("success-before-cancel")
        executable_path = self.create_fake_codex_executable(exit_code=0)

        result = self.create_executor(executable_path).execute(
            self.create_request(workspace_path),
            is_cancel_requested=lambda: False,
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.SUCCEEDED)
        self.assertEqual(result.exit_code, 0)

    def test_natural_failure_observed_before_timeout_wins(self):
        workspace_path = self.create_workspace("failure-before-timeout")
        executable_path = self.create_fake_codex_executable(exit_code=9)

        result = self.create_executor(
            executable_path,
            timeout_seconds=5,
        ).execute(self.create_request(workspace_path))

        self.assertEqual(result.outcome, AgentExecutionOutcome.FAILED)
        self.assertEqual(result.exit_code, 9)

    def test_cancellation_observed_before_timeout_decision_wins(self):
        workspace_path = self.create_workspace("cancel-timeout-precedence")
        executable_path = self.create_lifecycle_executable(
            executable_name="cancel-timeout-codex",
            sleep_seconds=10,
        )
        cancellation_checks = iter([False, True])

        result = self.create_executor(
            executable_path,
            timeout_seconds=0.001,
            termination_grace_seconds=0.1,
            cancellation_poll_interval_seconds=0.02,
        ).execute(
            self.create_request(workspace_path),
            is_cancel_requested=lambda: next(cancellation_checks, True),
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.CANCELLED)

    def test_timeout_established_before_cancellation_observed_wins(self):
        workspace_path = self.create_workspace("timeout-cancel-precedence")
        executable_path = self.create_lifecycle_executable(
            executable_name="timeout-cancel-codex",
            sleep_seconds=10,
        )

        result = self.create_executor(
            executable_path,
            timeout_seconds=0.001,
            termination_grace_seconds=0.1,
            cancellation_poll_interval_seconds=0.02,
        ).execute(
            self.create_request(workspace_path),
            is_cancel_requested=lambda: False,
        )

        self.assertEqual(result.outcome, AgentExecutionOutcome.TIMED_OUT)

    def test_cancellation_callback_failure_terminates_process_and_raises(self):
        workspace_path = self.create_workspace("cancel-callback-failure")
        executable_path = self.create_lifecycle_executable(
            executable_name="cancel-callback-failure-codex",
            sleep_seconds=10,
        )
        cancellation_checks = iter([False, RuntimeError("Database read failed.")])

        def failing_cancellation_check():
            cancellation_check = next(cancellation_checks)

            if isinstance(cancellation_check, Exception):
                raise cancellation_check

            return cancellation_check

        with self.assertRaises(AgentExecutionError):
            self.create_executor(
                executable_path,
                timeout_seconds=5,
                termination_grace_seconds=0.2,
                cancellation_poll_interval_seconds=0.02,
            ).execute(
                self.create_request(workspace_path),
                is_cancel_requested=failing_cancellation_check,
            )

    def test_cancellation_callback_failure_cleans_live_descendant_before_raising(self):
        workspace_path = self.create_workspace("cancel-callback-failure-descendant")
        process_metadata_path = workspace_path / "runtime" / "process.json"
        self.addCleanup(
            self.cleanup_process_group_from_metadata,
            process_metadata_path,
        )
        executable_path = self.create_descendant_escape_executable(
            executable_name="cancel-callback-failure-descendant-codex",
            process_metadata_path=process_metadata_path,
            child_ignores_sigterm=True,
            parent_exits_on_sigterm=True,
        )

        def failing_cancellation_check():
            if process_metadata_path.exists():
                raise RuntimeError("Database read failed.")

            return False

        with self.assertRaises(AgentExecutionError):
            self.create_executor(
                executable_path,
                timeout_seconds=5,
                termination_grace_seconds=0.05,
                cancellation_poll_interval_seconds=0.02,
            ).execute(
                self.create_request(workspace_path),
                is_cancel_requested=failing_cancellation_check,
            )

        metadata = self.read_process_metadata(process_metadata_path)
        self.assertFalse(self.process_is_live(metadata["child_pid"]))
        self.assertFalse(self.process_group_has_live_members(metadata["process_group_id"]))

    def test_process_group_signal_failure_raises_after_parent_reap(self):
        workspace_path = self.create_workspace("signal-failure")
        executable_path = self.create_lifecycle_executable(
            executable_name="signal-failure-codex",
            sleep_seconds=10,
        )

        with patch(
            "apps.agent_workspace.ai_agent.local_subprocess_executor.os.killpg",
            side_effect=OSError("Signal failed."),
        ):
            with self.assertRaises(AgentExecutionError):
                self.create_executor(
                    executable_path,
                    timeout_seconds=0.2,
                    termination_grace_seconds=0.1,
                    cancellation_poll_interval_seconds=0.02,
                ).execute(self.create_request(workspace_path))

    def test_process_group_still_live_after_sigkill_raises_execution_error(self):
        executor = self.create_executor(
            self.create_fake_codex_executable(),
            termination_grace_seconds=0.0,
            cancellation_poll_interval_seconds=0.02,
        )

        class FakeProcess:
            """
            Minimal exited parent process for fail-closed lifecycle tests.
            """

            pid = 999999

            def __init__(self):
                """
                Track direct parent reaping.
                """

                self.wait_count = 0

            def poll(self):
                """
                Return exited parent state.
                """

                return 0

            def wait(self):
                """
                Record parent reap attempts.
                """

                self.wait_count += 1
                return 0

        process = FakeProcess()

        with (
            patch(
                "apps.agent_workspace.ai_agent.local_subprocess_executor."
                "_process_group_has_live_members",
                return_value=True,
            ),
            patch(
                "apps.agent_workspace.ai_agent.local_subprocess_executor._signal_process_group",
            ),
            self.assertRaises(AgentExecutionError),
        ):
            executor._terminate_process_group(process, process.pid)

        self.assertGreater(process.wait_count, 0)

    def test_prompt_shell_metacharacters_are_not_interpreted(self):
        prompts = [
            "hello; touch SHOULD_NOT_EXIST",
            "$(touch SHOULD_NOT_EXIST)",
            "`touch SHOULD_NOT_EXIST`",
            "hello && touch SHOULD_NOT_EXIST",
            "hello | touch SHOULD_NOT_EXIST",
        ]

        for index, prompt in enumerate(prompts):
            with self.subTest(prompt=prompt):
                workspace_path = self.create_workspace(f"shell-injection-{index}")
                executable_path = self.create_fake_codex_executable(
                    executable_name=f"fake-codex-{index}"
                )

                self.create_executor(executable_path).execute(
                    self.create_request(workspace_path, prompt=prompt)
                )

                payload = self.read_stdout_payload(workspace_path)
                self.assertEqual(payload["stdin"], prompt)
                self.assertFalse((workspace_path / "SHOULD_NOT_EXIST").exists())

    def test_child_environment_does_not_inherit_worker_secrets(self):
        workspace_path = self.create_workspace("env-secrets")
        executable_path = self.create_fake_codex_executable()
        parent_environment = {
            "AWS_ACCESS_KEY_ID": "SHOULD_NOT_LEAK",
            "AWS_SECRET_ACCESS_KEY": "SHOULD_NOT_LEAK",
            "AWS_SESSION_TOKEN": "SHOULD_NOT_LEAK",
            "CELERY_BROKER_URL": "SHOULD_NOT_LEAK",
            "DATABASE_PASSWORD": "SHOULD_NOT_LEAK",
            "DATABASE_URL": "SHOULD_NOT_LEAK",
            "DJANGO_SECRET_KEY": "SHOULD_NOT_LEAK",
            "GOOGLE_APPLICATION_CREDENTIALS": "SHOULD_NOT_LEAK",
            "PATH": os.environ.get("PATH", ""),
            "POSTGRES_PASSWORD": "SHOULD_NOT_LEAK",
            "REDIS_URL": "SHOULD_NOT_LEAK",
            "SECRET_KEY": "SHOULD_NOT_LEAK",
            "SENTRY_DSN": "SHOULD_NOT_LEAK",
        }

        with patch.dict(os.environ, parent_environment, clear=True):
            self.create_executor(executable_path).execute(self.create_request(workspace_path))

        payload = self.read_stdout_payload(workspace_path)
        leaked_environment = {
            key: value for key, value in payload["env"].items() if value == "SHOULD_NOT_LEAK"
        }

        self.assertEqual(leaked_environment, {})

    def test_allowed_parent_environment_values_are_propagated(self):
        workspace_path = self.create_workspace("env-allowlist")
        executable_path = self.create_fake_codex_executable()
        parent_environment = {
            "LANG": "C.UTF-8",
            "PATH": str(self.fake_bin_path),
        }

        with patch.dict(os.environ, parent_environment, clear=True):
            LocalSubprocessExecutor(
                cli_wrapper=CodexCLIWrapper(executable=executable_path.name)
            ).execute(self.create_request(workspace_path))

        payload = self.read_stdout_payload(workspace_path)
        self.assertEqual(payload["env"]["LANG"], "C.UTF-8")
        self.assertEqual(payload["env"]["PATH"], str(self.fake_bin_path))

    def test_child_home_is_workspace_runtime_home_not_parent_home(self):
        workspace_path = self.create_workspace("runtime-home")
        executable_path = self.create_fake_codex_executable()

        with patch.dict(
            os.environ,
            {
                "HOME": "/fake/developer/home",
                "PATH": os.environ.get("PATH", ""),
            },
            clear=True,
        ):
            self.create_executor(executable_path).execute(self.create_request(workspace_path))

        payload = self.read_stdout_payload(workspace_path)
        self.assertEqual(payload["env"]["HOME"], str(workspace_path / "runtime" / "home"))
        self.assertNotEqual(payload["env"]["HOME"], "/fake/developer/home")

    def test_codex_invocation_environment_overrides_are_merged(self):
        workspace_path = self.create_workspace("codex-home")
        executable_path = self.create_fake_codex_executable()

        self.create_executor(executable_path).execute(self.create_request(workspace_path))

        payload = self.read_stdout_payload(workspace_path)
        self.assertEqual(payload["env"][CODEX_HOME_ENV_KEY], str(workspace_path / ".codex"))

    def test_forbidden_invocation_environment_override_is_rejected(self):
        workspace_path = self.create_workspace("forbidden-override")
        executable_path = self.create_fake_codex_executable()
        invocation = CodexCLIInvocation(
            argv=(str(executable_path),),
            cwd=workspace_path,
            stdin_text="Prompt.",
            environment_overrides={
                "DJANGO_SECRET_KEY": "SHOULD_NOT_LEAK",
            },
        )

        class ForbiddenEnvironmentWrapper:
            """
            Return an invocation with a forbidden environment override.
            """

            def build_invocation(self, request):
                """
                Return the forbidden invocation.
                """

                return invocation

        with self.assertRaises(AgentProcessSpawnError):
            LocalSubprocessExecutor(cli_wrapper=ForbiddenEnvironmentWrapper()).execute(
                self.create_request(workspace_path)
            )

    def test_existing_stdout_log_file_fails_safely(self):
        workspace_path = self.create_workspace("existing-stdout")
        executable_path = self.create_fake_codex_executable()
        stdout_path = workspace_path / "logs" / STDOUT_LOG_FILENAME
        stdout_path.write_text("Existing log.\n", encoding="utf-8")

        with self.assertRaises(AgentProcessSpawnError):
            self.create_executor(executable_path).execute(self.create_request(workspace_path))

        self.assertEqual(stdout_path.read_text(encoding="utf-8"), "Existing log.\n")

    def test_log_symlink_does_not_write_outside_workspace(self):
        workspace_path = self.create_workspace("log-symlink")
        executable_path = self.create_fake_codex_executable()
        outside_log_target = self.temp_root / "outside.log"
        outside_log_target.write_text("Outside content.\n", encoding="utf-8")
        (workspace_path / "logs" / STDOUT_LOG_FILENAME).symlink_to(outside_log_target)

        with self.assertRaises(AgentProcessSpawnError):
            self.create_executor(executable_path).execute(self.create_request(workspace_path))

        self.assertEqual(
            outside_log_target.read_text(encoding="utf-8"),
            "Outside content.\n",
        )

    def test_logs_directory_symlink_is_rejected(self):
        workspace_path = self.create_workspace("logs-dir-symlink")
        executable_path = self.create_fake_codex_executable()
        outside_logs_path = self.temp_root / "outside-logs"
        outside_logs_path.mkdir()
        logs_path = workspace_path / "logs"
        logs_path.rmdir()
        logs_path.symlink_to(outside_logs_path, target_is_directory=True)

        with self.assertRaises(AgentProcessSpawnError):
            self.create_executor(executable_path).execute(self.create_request(workspace_path))

        self.assertFalse((outside_logs_path / STDOUT_LOG_FILENAME).exists())

    def test_executor_does_not_access_agent_run_orm_or_state(self):
        self.assertFalse(hasattr(local_executor_module, "AgentRun"))
        self.assertFalse(hasattr(local_executor_module, "AgentRunStatus"))


class AgentRunModelTests(TestCase):
    """
    Cover AgentRun and AgentArtifact persistence behavior.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for model persistence tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="model-user",
            password="test-password",
        )

    def create_agent_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)

        return AgentRun.objects.create(**defaults)

    def test_agent_run_defaults_to_created_status(self):
        agent_run = self.create_agent_run()

        self.assertEqual(agent_run.status, AgentRunStatus.CREATED.value)

    def test_agent_run_generates_uuid(self):
        agent_run = self.create_agent_run()

        self.assertIsInstance(agent_run.id, UUID)

    def test_agent_run_links_to_user(self):
        agent_run = self.create_agent_run()

        self.assertEqual(agent_run.user, self.user)
        self.assertEqual(list(self.user.agent_runs.all()), [agent_run])

    def test_agent_artifact_links_to_agent_run(self):
        agent_run = self.create_agent_run()

        artifact = AgentArtifact.objects.create(
            agent_run=agent_run,
            kind="image",
            filename="result.png",
            storage_key="agent-runs/result.png",
            mime_type="image/png",
            size=1024,
            sha256="a" * 64,
        )

        self.assertEqual(artifact.agent_run, agent_run)
        self.assertEqual(list(agent_run.artifacts.all()), [artifact])

    def test_agent_run_metadata_default_is_not_shared_between_instances(self):
        first_run = self.create_agent_run()
        second_run = self.create_agent_run(prompt="Generate another image.")

        first_run.metadata["key"] = "value"

        self.assertEqual(second_run.metadata, {})
        self.assertIsNot(first_run.metadata, second_run.metadata)


class AgentWorkspaceRootConfigurationTests(TestCase):
    """
    Cover Agent workspace root default and explicit override behavior.
    """

    def setUp(self):
        """
        Create an isolated filesystem fixture for workspace-root tests.
        """

        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.temp_root = Path(self.temp_directory.name).resolve()

    def test_default_workspace_root_uses_canonical_system_temp_directory(self):
        temp_directory = self.temp_root / "system-temp"
        temp_directory.mkdir()

        with patch("common.utils.helpers.tempfile.gettempdir", return_value=str(temp_directory)):
            workspace_root = default_agent_workspace_root()

        self.assertEqual(workspace_root, temp_directory / "ai-gen-image" / "workspaces")

    def test_workspace_root_from_env_uses_default_when_override_is_absent(self):
        temp_directory = self.temp_root / "system-temp-for-env"
        temp_directory.mkdir()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("common.utils.helpers.tempfile.gettempdir", return_value=str(temp_directory)),
        ):
            workspace_root = agent_workspace_root_from_env()

        self.assertEqual(workspace_root, temp_directory / "ai-gen-image" / "workspaces")

    def test_workspace_root_from_env_uses_default_when_override_is_empty(self):
        temp_directory = self.temp_root / "system-temp-for-empty-env"
        temp_directory.mkdir()

        with (
            patch.dict(os.environ, {"AGENT_WORKSPACE_ROOT": ""}, clear=True),
            patch("common.utils.helpers.tempfile.gettempdir", return_value=str(temp_directory)),
        ):
            workspace_root = agent_workspace_root_from_env()

        self.assertEqual(workspace_root, temp_directory / "ai-gen-image" / "workspaces")

    def test_workspace_root_from_env_uses_default_when_override_is_whitespace(self):
        temp_directory = self.temp_root / "system-temp-for-whitespace-env"
        temp_directory.mkdir()

        with (
            patch.dict(os.environ, {"AGENT_WORKSPACE_ROOT": "  \t\n  "}, clear=True),
            patch("common.utils.helpers.tempfile.gettempdir", return_value=str(temp_directory)),
        ):
            workspace_root = agent_workspace_root_from_env()

        self.assertEqual(workspace_root, temp_directory / "ai-gen-image" / "workspaces")

    def test_default_workspace_root_resolves_symlink_temp_alias(self):
        real_temp_directory = self.temp_root / "real-system-temp"
        real_temp_directory.mkdir()
        temp_alias = self.temp_root / "system-temp-alias"
        temp_alias.symlink_to(real_temp_directory, target_is_directory=True)

        with patch("common.utils.helpers.tempfile.gettempdir", return_value=str(temp_alias)):
            workspace_root = default_agent_workspace_root()

        self.assertEqual(
            workspace_root,
            real_temp_directory / "ai-gen-image" / "workspaces",
        )
        self.assertNotEqual(workspace_root, temp_alias / "ai-gen-image" / "workspaces")

    def test_explicit_workspace_root_override_is_preserved(self):
        explicit_workspace_root = self.temp_root / "explicit-workspaces"

        with patch.dict(
            os.environ,
            {"AGENT_WORKSPACE_ROOT": str(explicit_workspace_root)},
        ):
            workspace_root = agent_workspace_root_from_env()

        self.assertEqual(workspace_root, explicit_workspace_root)

    def test_explicit_relative_workspace_root_override_is_rejected(self):
        for configured_workspace_root in [
            "workspaces",
            "./workspaces",
            "../workspaces",
            ".",
        ]:
            with self.subTest(configured_workspace_root=configured_workspace_root):
                with patch.dict(
                    os.environ,
                    {"AGENT_WORKSPACE_ROOT": configured_workspace_root},
                ):
                    with self.assertRaises(ImproperlyConfigured):
                        agent_workspace_root_from_env()

    def test_explicit_symlink_workspace_root_override_is_preserved(self):
        symlink_target = self.temp_root / "explicit-symlink-preserve-target"
        symlink_target.mkdir()
        explicit_workspace_root = self.temp_root / "explicit-symlink-workspaces"
        explicit_workspace_root.symlink_to(symlink_target, target_is_directory=True)

        with patch.dict(
            os.environ,
            {"AGENT_WORKSPACE_ROOT": str(explicit_workspace_root)},
        ):
            workspace_root = agent_workspace_root_from_env()

        self.assertEqual(workspace_root, explicit_workspace_root)
        self.assertTrue(workspace_root.is_symlink())
        self.assertNotEqual(workspace_root, symlink_target)

    def test_explicit_unsafe_workspace_root_is_rejected(self):
        run_id = uuid4()
        symlink_target = self.temp_root / "explicit-symlink-target"
        symlink_target.mkdir()
        explicit_workspace_root = self.temp_root / "explicit-workspaces"
        explicit_workspace_root.symlink_to(symlink_target, target_is_directory=True)

        with patch.dict(
            os.environ,
            {"AGENT_WORKSPACE_ROOT": str(explicit_workspace_root)},
        ):
            workspace_root = agent_workspace_root_from_env()

        workspace_manager = WorkspaceManager(workspace_root=workspace_root)

        with self.assertRaises(AgentWorkspaceError):
            workspace_manager.prepare_workspace(run_id)

        self.assertFalse((symlink_target / str(run_id)).exists())


class ContextManifestLoaderTests(TestCase):
    """
    Cover end-user context manifest validation and fingerprinting.
    """

    def setUp(self):
        """
        Create an isolated context fixture for manifest loader tests.
        """

        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.context_root = Path(self.temp_directory.name) / "end_user_context"
        self.codex_directory = self.context_root / ".codex"
        self.codex_directory.mkdir(parents=True)
        self.user_agents_path = self.context_root / "USER_AGENTS.md"
        self.config_path = self.codex_directory / "config.toml"
        self.manifest_path = self.context_root / "context_manifest.json"

        self.user_agents_path.write_bytes(b"Use inputs and outputs.\n")
        self.config_path.write_bytes(b"# Config\n")
        self.write_manifest()

    def write_manifest(self, manifest_data=None) -> None:
        """
        Write a manifest fixture to the temporary context root.
        """

        manifest = self.valid_manifest() if manifest_data is None else manifest_data

        self.manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def valid_manifest(self) -> dict:
        """
        Return a valid end-user context manifest fixture.
        """

        return {
            "context_version": "test-context-v1",
            "files": [
                {
                    "source": "USER_AGENTS.md",
                    "target": "AGENTS.md",
                },
                {
                    "source": ".codex/config.toml",
                    "target": ".codex/config.toml",
                },
            ],
        }

    def load_manifest(self):
        """
        Load the temporary context manifest fixture.
        """

        return load_context_manifest(
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )

    def assert_invalid_manifest(self, manifest_data) -> None:
        """
        Assert that a manifest fixture is rejected.
        """

        self.write_manifest(manifest_data)

        with self.assertRaises(InvalidContextManifest):
            self.load_manifest()

    def test_loads_valid_manifest(self):
        context_manifest = self.load_manifest()

        self.assertEqual(context_manifest.context_version, "test-context-v1")
        self.assertEqual(len(context_manifest.files), 2)
        self.assertEqual(context_manifest.files[0].source, "USER_AGENTS.md")
        self.assertEqual(context_manifest.files[0].destination, "AGENTS.md")
        self.assertTrue(context_manifest.files[0].source_path.is_absolute())
        self.assertEqual(context_manifest.files[0].content, b"Use inputs and outputs.\n")
        self.assertEqual(len(context_manifest.context_hash), 64)

    def test_malformed_manifest_is_rejected(self):
        self.manifest_path.write_text("{", encoding="utf-8")

        with self.assertRaises(InvalidContextManifest):
            self.load_manifest()

    def test_missing_version_is_rejected(self):
        manifest = self.valid_manifest()
        del manifest["context_version"]

        self.assert_invalid_manifest(manifest)

    def test_invalid_files_list_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"] = {}

        self.assert_invalid_manifest(manifest)

    def test_malformed_file_entry_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"] = ["USER_AGENTS.md"]

        self.assert_invalid_manifest(manifest)

    def test_missing_source_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0] = {
            "target": "AGENTS.md",
        }

        self.assert_invalid_manifest(manifest)

    def test_missing_destination_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0] = {
            "source": "USER_AGENTS.md",
        }

        self.assert_invalid_manifest(manifest)

    def test_traversal_source_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["source"] = "../AGENTS.md"

        self.assert_invalid_manifest(manifest)

    def test_nested_traversal_source_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["source"] = "nested/../../AGENTS.md"

        self.assert_invalid_manifest(manifest)

    def test_absolute_source_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["source"] = "/tmp/foo"

        self.assert_invalid_manifest(manifest)

    def test_traversal_destination_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["target"] = "../foo"

        self.assert_invalid_manifest(manifest)

    def test_nested_traversal_destination_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["target"] = "outputs/../../AGENTS.md"

        self.assert_invalid_manifest(manifest)

    def test_absolute_destination_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["target"] = "/etc/passwd"

        self.assert_invalid_manifest(manifest)

    def test_symlink_source_is_rejected(self):
        symlink_path = self.context_root / "linked.md"
        symlink_path.symlink_to(self.user_agents_path)
        manifest = self.valid_manifest()
        manifest["files"][0]["source"] = "linked.md"

        self.assert_invalid_manifest(manifest)

    def test_directory_source_is_rejected(self):
        directory_path = self.context_root / "directory-source"
        directory_path.mkdir()
        manifest = self.valid_manifest()
        manifest["files"][0]["source"] = "directory-source"

        self.assert_invalid_manifest(manifest)

    def test_duplicate_destination_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][1]["target"] = "./AGENTS.md"

        self.assert_invalid_manifest(manifest)

    def test_nonexistent_source_is_rejected(self):
        manifest = self.valid_manifest()
        manifest["files"][0]["source"] = "missing.md"

        self.assert_invalid_manifest(manifest)

    def test_context_fingerprint_is_deterministic(self):
        first_manifest = self.load_manifest()
        second_manifest = self.load_manifest()

        self.assertEqual(first_manifest.context_hash, second_manifest.context_hash)

    def test_context_fingerprint_is_independent_of_manifest_entry_order(self):
        first_hash = self.load_manifest().context_hash
        manifest = self.valid_manifest()
        manifest["files"] = list(reversed(manifest["files"]))
        self.write_manifest(manifest)

        second_hash = self.load_manifest().context_hash

        self.assertEqual(first_hash, second_hash)

    def test_context_fingerprint_is_independent_of_absolute_context_root(self):
        first_hash = self.load_manifest().context_hash
        second_temp_directory = TemporaryDirectory()
        self.addCleanup(second_temp_directory.cleanup)
        second_context_root = Path(second_temp_directory.name) / "end_user_context"
        second_codex_directory = second_context_root / ".codex"
        second_codex_directory.mkdir(parents=True)
        second_manifest_path = second_context_root / "context_manifest.json"

        (second_context_root / "USER_AGENTS.md").write_bytes(self.user_agents_path.read_bytes())
        (second_codex_directory / "config.toml").write_bytes(self.config_path.read_bytes())
        second_manifest_path.write_text(json.dumps(self.valid_manifest()), encoding="utf-8")

        second_manifest = load_context_manifest(
            manifest_path=second_manifest_path,
            context_root=second_context_root,
        )

        self.assertEqual(first_hash, second_manifest.context_hash)

    def test_context_fingerprint_is_independent_of_file_mtime(self):
        first_hash = self.load_manifest().context_hash

        self.user_agents_path.touch()
        self.config_path.touch()
        second_hash = self.load_manifest().context_hash

        self.assertEqual(first_hash, second_hash)

    def test_context_fingerprint_changes_when_version_changes(self):
        first_hash = self.load_manifest().context_hash
        manifest = self.valid_manifest()
        manifest["context_version"] = "test-context-v2"
        self.write_manifest(manifest)

        second_hash = self.load_manifest().context_hash

        self.assertNotEqual(first_hash, second_hash)

    def test_context_fingerprint_changes_when_destination_changes(self):
        first_hash = self.load_manifest().context_hash
        manifest = self.valid_manifest()
        manifest["files"][0]["target"] = "RULES.md"
        self.write_manifest(manifest)

        second_hash = self.load_manifest().context_hash

        self.assertNotEqual(first_hash, second_hash)

    def test_context_fingerprint_changes_when_content_changes(self):
        first_hash = self.load_manifest().context_hash
        self.user_agents_path.write_bytes(b"Updated end-user instructions.\n")

        second_hash = self.load_manifest().context_hash

        self.assertNotEqual(first_hash, second_hash)


class WorkspaceManagerTests(TestCase):
    """
    Cover per-AgentRun workspace bootstrap and cleanup behavior.
    """

    def setUp(self):
        """
        Create isolated workspace, context, and developer-context fixtures.
        """

        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.temp_root = Path(self.temp_directory.name).resolve()
        self.workspace_root = self.temp_root / "workspaces"
        self.context_root = self.temp_root / "end_user_context"
        self.codex_directory = self.context_root / ".codex"
        self.codex_directory.mkdir(parents=True)
        self.user_agents_path = self.context_root / "USER_AGENTS.md"
        self.config_path = self.codex_directory / "config.toml"
        self.manifest_path = self.context_root / "context_manifest.json"
        self.developer_codex_directory = self.temp_root / ".codex"
        self.developer_codex_directory.mkdir()
        self.developer_agents_path = self.temp_root / "AGENTS.md"
        self.developer_config_path = self.developer_codex_directory / "config.toml"
        self.developer_only_config_path = self.developer_codex_directory / "developer-only.toml"

        self.user_agents_path.write_text("End-user instructions.\n", encoding="utf-8")
        self.config_path.write_text('sandbox_mode = "workspace-write"\n', encoding="utf-8")
        self.developer_agents_path.write_text("Developer instructions.\n", encoding="utf-8")
        self.developer_config_path.write_text("developer = true\n", encoding="utf-8")
        self.developer_only_config_path.write_text("developer_only = true\n", encoding="utf-8")
        (self.context_root / "EXTRA.md").write_text("Extra context.\n", encoding="utf-8")
        (self.codex_directory / "extra.toml").write_text("extra = true\n", encoding="utf-8")
        self.write_manifest()

        self.workspace_manager = WorkspaceManager(
            workspace_root=self.workspace_root,
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )

    def write_manifest(self, manifest_data=None) -> None:
        """
        Write a workspace manifest fixture to the temporary context root.
        """

        manifest = self.valid_manifest() if manifest_data is None else manifest_data

        self.manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    def valid_manifest(self) -> dict:
        """
        Return a valid workspace bootstrap manifest fixture.
        """

        return {
            "context_version": "workspace-context-v1",
            "files": [
                {
                    "source": "USER_AGENTS.md",
                    "target": "AGENTS.md",
                },
                {
                    "source": ".codex/config.toml",
                    "target": ".codex/config.toml",
                },
            ],
        }

    def test_get_workspace_path_uses_canonical_uuid(self):
        run_id = uuid4()
        uppercase_run_id = str(run_id).upper()

        workspace_path = self.workspace_manager.get_workspace_path(uppercase_run_id)

        self.assertEqual(workspace_path, self.workspace_root / str(run_id))

    def test_prepare_workspace_creates_expected_layout(self):
        run_id = uuid4()

        prepared_workspace = self.workspace_manager.prepare_workspace(run_id)
        workspace_path = prepared_workspace.workspace_path

        self.assertEqual(workspace_path, self.workspace_root / str(run_id))
        self.assertTrue((workspace_path / "AGENTS.md").is_file())
        self.assertTrue((workspace_path / ".codex" / "config.toml").is_file())
        self.assertTrue((workspace_path / "inputs").is_dir())
        self.assertTrue((workspace_path / "outputs").is_dir())
        self.assertTrue((workspace_path / "runtime").is_dir())
        self.assertTrue((workspace_path / "logs").is_dir())
        self.assertEqual(stat.S_IMODE(self.workspace_root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(workspace_path.stat().st_mode), 0o700)
        self.assertEqual(prepared_workspace.context_version, "workspace-context-v1")
        self.assertEqual(len(prepared_workspace.context_hash), 64)

    def test_prepare_workspace_returns_canonical_real_workspace_path(self):
        run_id = uuid4()

        prepared_workspace = self.workspace_manager.prepare_workspace(run_id)

        self.assertEqual(prepared_workspace.workspace_path, self.workspace_root / str(run_id))
        self.assertTrue(prepared_workspace.workspace_path.is_dir())
        self.assertFalse(prepared_workspace.workspace_path.is_symlink())

    def test_prepare_workspace_does_not_leak_file_descriptors_after_success(self):
        before_prepare_fd_count = self.open_fd_count()

        self.workspace_manager.prepare_workspace(uuid4())

        self.assertEqual(self.open_fd_count(), before_prepare_fd_count)

    def test_prepare_workspace_materializes_user_agents_as_agents(self):
        workspace_path = self.workspace_manager.prepare_workspace(uuid4()).workspace_path

        self.assertEqual(
            (workspace_path / "AGENTS.md").read_text(encoding="utf-8"),
            self.user_agents_path.read_text(encoding="utf-8"),
        )

    def test_prepare_workspace_materializes_codex_config(self):
        workspace_path = self.workspace_manager.prepare_workspace(uuid4()).workspace_path

        self.assertEqual(
            (workspace_path / ".codex" / "config.toml").read_text(encoding="utf-8"),
            self.config_path.read_text(encoding="utf-8"),
        )

    def test_prepare_workspace_context_hash_reflects_materialized_bytes(self):
        prepared_workspace = self.workspace_manager.prepare_workspace(uuid4())
        workspace_path = prepared_workspace.workspace_path

        expected_hash = self.compute_context_hash(
            prepared_workspace.context_version,
            {
                "AGENTS.md": (workspace_path / "AGENTS.md").read_bytes(),
                ".codex/config.toml": (workspace_path / ".codex" / "config.toml").read_bytes(),
            },
        )

        self.assertEqual(prepared_workspace.context_hash, expected_hash)

    def test_prepare_workspace_uses_validated_snapshot_when_source_content_changes(self):
        original_user_agents_content = self.user_agents_path.read_bytes()
        loaded_context_hashes = []

        def load_then_mutate_source(*args, **kwargs):
            """
            Mutate source after validation to reproduce the old TOCTOU window.
            """

            context_manifest = load_context_manifest(*args, **kwargs)
            loaded_context_hashes.append(context_manifest.context_hash)
            self.user_agents_path.write_bytes(b"Mutated after validation.\n")

            return context_manifest

        with patch(
            "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
            side_effect=load_then_mutate_source,
        ):
            prepared_workspace = self.workspace_manager.prepare_workspace(uuid4())

        workspace_path = prepared_workspace.workspace_path

        self.assertEqual((workspace_path / "AGENTS.md").read_bytes(), original_user_agents_content)
        self.assertNotEqual(
            (workspace_path / "AGENTS.md").read_bytes(), self.user_agents_path.read_bytes()
        )
        self.assertEqual(prepared_workspace.context_hash, loaded_context_hashes[0])
        self.assertEqual(
            prepared_workspace.context_hash,
            self.compute_context_hash(
                prepared_workspace.context_version,
                {
                    "AGENTS.md": (workspace_path / "AGENTS.md").read_bytes(),
                    ".codex/config.toml": (workspace_path / ".codex" / "config.toml").read_bytes(),
                },
            ),
        )

    def test_prepare_workspace_does_not_materialize_symlink_when_source_is_swapped(self):
        original_user_agents_content = self.user_agents_path.read_bytes()
        replacement_path = self.temp_root / "replacement-agents.md"
        replacement_path.write_bytes(b"Replacement context.\n")

        def load_then_swap_source_to_symlink(*args, **kwargs):
            """
            Replace source with a symlink after validation to reproduce TOCTOU.
            """

            context_manifest = load_context_manifest(*args, **kwargs)
            self.user_agents_path.unlink()
            self.user_agents_path.symlink_to(replacement_path)

            return context_manifest

        with patch(
            "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
            side_effect=load_then_swap_source_to_symlink,
        ):
            prepared_workspace = self.workspace_manager.prepare_workspace(uuid4())

        agents_path = prepared_workspace.workspace_path / "AGENTS.md"

        self.assertFalse(agents_path.is_symlink())
        self.assertTrue(agents_path.is_file())
        self.assertEqual(agents_path.read_bytes(), original_user_agents_content)

    def test_prepare_workspace_materializes_runtime_context_as_regular_files(self):
        workspace_path = self.workspace_manager.prepare_workspace(uuid4()).workspace_path
        agents_path = workspace_path / "AGENTS.md"
        config_path = workspace_path / ".codex" / "config.toml"

        self.assertTrue(agents_path.is_file())
        self.assertFalse(agents_path.is_symlink())
        self.assertTrue(config_path.is_file())
        self.assertFalse(config_path.is_symlink())

    def test_prepare_workspace_rejects_swapped_codex_parent_symlink(self):
        run_id = uuid4()
        workspace_path = self.workspace_root / str(run_id)
        symlink_target = self.temp_root / "codex-parent-symlink-target"
        symlink_target.mkdir()
        original_open = os.open
        swapped_codex_parent = False

        def open_then_swap_codex_parent(path, flags, mode=0o777, *, dir_fd=None):
            """
            Replace .codex with a symlink before opening it for config materialization.
            """

            nonlocal swapped_codex_parent

            codex_parent = workspace_path / ".codex"
            if path == ".codex" and not swapped_codex_parent and codex_parent.exists():
                codex_parent.rmdir()
                codex_parent.symlink_to(symlink_target, target_is_directory=True)
                swapped_codex_parent = True

            return original_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.open",
                side_effect=open_then_swap_codex_parent,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertFalse((symlink_target / "config.toml").exists())
        self.assertTrue((workspace_path / "AGENTS.md").is_file())
        self.assertFalse((workspace_path / "AGENTS.md").is_symlink())

    def test_prepare_workspace_does_not_leak_file_descriptors_after_failure(self):
        before_prepare_fd_count = self.open_fd_count()

        def load_then_fail(*args, **kwargs):
            """
            Fail after the workspace and root descriptors have been opened.
            """

            raise AgentWorkspaceError("Injected workspace failure")

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
                side_effect=load_then_fail,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(uuid4())

        self.assertEqual(self.open_fd_count(), before_prepare_fd_count)

    def test_prepare_workspace_fails_when_canonical_workspace_is_renamed(self):
        run_id = uuid4()
        workspace_path = self.workspace_root / str(run_id)
        pinned_workspace_path = self.temp_root / "pinned-workspace-after-rename"

        def load_then_rename_workspace_path(*args, **kwargs):
            """
            Rename the workspace path after the workspace directory is pinned.
            """

            context_manifest = load_context_manifest(*args, **kwargs)
            workspace_path.rename(pinned_workspace_path)

            return context_manifest

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
                side_effect=load_then_rename_workspace_path,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertFalse(workspace_path.exists())
        self.assertTrue((pinned_workspace_path / "AGENTS.md").is_file())
        self.assertTrue((pinned_workspace_path / ".codex" / "config.toml").is_file())

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())
        self.assertTrue((pinned_workspace_path / "AGENTS.md").is_file())

    def test_prepare_workspace_fails_when_workspace_path_is_replaced_by_symlink(self):
        run_id = uuid4()
        workspace_path = self.workspace_root / str(run_id)
        pinned_workspace_path = self.temp_root / "pinned-workspace-after-swap"
        symlink_target = self.temp_root / "workspace-path-symlink-target"
        symlink_target.mkdir()
        symlink_target_file = symlink_target / "sentinel.txt"
        symlink_target_file.write_text("Do not delete.\n", encoding="utf-8")

        def load_then_replace_workspace_path(*args, **kwargs):
            """
            Replace the workspace path after the workspace directory is pinned.
            """

            context_manifest = load_context_manifest(*args, **kwargs)
            workspace_path.rename(pinned_workspace_path)
            workspace_path.symlink_to(symlink_target, target_is_directory=True)

            return context_manifest

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
                side_effect=load_then_replace_workspace_path,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertTrue(workspace_path.is_symlink())
        self.assertTrue((pinned_workspace_path / "AGENTS.md").is_file())
        self.assertTrue((pinned_workspace_path / ".codex" / "config.toml").is_file())
        self.assertFalse((symlink_target / "AGENTS.md").exists())
        self.assertFalse((symlink_target / ".codex" / "config.toml").exists())
        self.assertEqual(symlink_target_file.read_text(encoding="utf-8"), "Do not delete.\n")

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_symlink())
        self.assertEqual(symlink_target_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_prepare_workspace_fails_when_workspace_path_is_replaced_by_directory(self):
        run_id = uuid4()
        workspace_path = self.workspace_root / str(run_id)
        pinned_workspace_path = self.temp_root / "pinned-workspace-after-directory-swap"
        replacement_workspace_path = self.temp_root / "replacement-workspace-directory"
        replacement_workspace_file = workspace_path / "sentinel.txt"

        def load_then_replace_workspace_path(*args, **kwargs):
            """
            Replace the workspace path with a different directory after pinning it.
            """

            context_manifest = load_context_manifest(*args, **kwargs)
            workspace_path.rename(pinned_workspace_path)
            replacement_workspace_path.mkdir()
            (replacement_workspace_path / "sentinel.txt").write_text(
                "Do not delete.\n",
                encoding="utf-8",
            )
            replacement_workspace_path.rename(workspace_path)

            return context_manifest

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
                side_effect=load_then_replace_workspace_path,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertTrue((pinned_workspace_path / "AGENTS.md").is_file())
        self.assertTrue((pinned_workspace_path / ".codex" / "config.toml").is_file())
        self.assertFalse((workspace_path / "AGENTS.md").exists())
        self.assertFalse((workspace_path / ".codex" / "config.toml").exists())
        self.assertEqual(
            replacement_workspace_file.read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(
            replacement_workspace_file.read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

    def test_prepare_workspace_rejects_pre_pin_workspace_replacement(self):
        run_id = uuid4()
        workspace_name = str(run_id)
        workspace_path = self.workspace_root / workspace_name
        pinned_workspace_path = self.temp_root / "pre-pin-original-workspace"
        replacement_workspace_path = self.temp_root / "pre-pin-replacement-workspace"
        replacement_workspace_file = workspace_path / "sentinel.txt"
        original_create_workspace_directory = workspace_manager_module._create_workspace_directory

        def create_then_replace_workspace(root_fd, created_workspace_name):
            """
            Replace the canonical workspace after identity capture but before pinning.
            """

            created_workspace_identity = original_create_workspace_directory(
                root_fd,
                created_workspace_name,
            )
            workspace_path.rename(pinned_workspace_path)
            replacement_workspace_path.mkdir()
            (replacement_workspace_path / "sentinel.txt").write_text(
                "Do not delete.\n",
                encoding="utf-8",
            )
            replacement_workspace_path.rename(workspace_path)

            return created_workspace_identity

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._create_workspace_directory",
                side_effect=create_then_replace_workspace,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertFalse((workspace_path / "AGENTS.md").exists())
        self.assertFalse((workspace_path / ".codex" / "config.toml").exists())
        self.assertEqual(
            replacement_workspace_file.read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(
            replacement_workspace_file.read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

    def test_prepare_workspace_copies_only_manifest_allowlist(self):
        workspace_path = self.workspace_manager.prepare_workspace(uuid4()).workspace_path

        self.assertFalse((workspace_path / "EXTRA.md").exists())
        self.assertFalse((workspace_path / ".codex" / "extra.toml").exists())

    def test_prepare_workspace_does_not_leak_developer_context(self):
        workspace_path = self.workspace_manager.prepare_workspace(uuid4()).workspace_path

        self.assertNotEqual(
            (workspace_path / "AGENTS.md").read_text(encoding="utf-8"),
            self.developer_agents_path.read_text(encoding="utf-8"),
        )
        self.assertNotEqual(
            (workspace_path / ".codex" / "config.toml").read_text(encoding="utf-8"),
            self.developer_config_path.read_text(encoding="utf-8"),
        )
        self.assertFalse((workspace_path / ".codex" / "developer-only.toml").exists())

    def test_prepare_workspace_uses_distinct_workspace_per_run(self):
        first_workspace = self.workspace_manager.prepare_workspace(uuid4())
        second_workspace = self.workspace_manager.prepare_workspace(uuid4())

        self.assertNotEqual(first_workspace.workspace_path, second_workspace.workspace_path)
        self.assertTrue(first_workspace.workspace_path.exists())
        self.assertTrue(second_workspace.workspace_path.exists())

    def test_prepare_workspace_rejects_existing_workspace(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.get_workspace_path(run_id)
        workspace_path.mkdir(parents=True)

        with self.assertRaises(AgentWorkspaceError):
            self.workspace_manager.prepare_workspace(run_id)

    def test_prepare_workspace_rejects_unsafe_workspace_root_permissions(self):
        run_id = uuid4()
        unsafe_workspace_root = self.temp_root / "unsafe-permissions-workspaces"
        unsafe_workspace_root.mkdir()
        unsafe_workspace_root.chmod(0o777)
        workspace_manager = WorkspaceManager(
            workspace_root=unsafe_workspace_root,
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )

        with self.assertRaises(AgentWorkspaceError):
            workspace_manager.prepare_workspace(run_id)

        self.assertFalse((unsafe_workspace_root / str(run_id)).exists())

    def test_prepare_workspace_normalizes_workspace_creation_filesystem_failure(self):
        run_id = uuid4()
        workspace_name = str(run_id)
        self.workspace_root.mkdir()
        original_mkdir = os.mkdir
        expected_error = PermissionError("Denied")

        def mkdir_with_workspace_failure(path, mode=0o777, *, dir_fd=None):
            """
            Fail the per-run workspace directory creation operation.
            """

            if path == workspace_name:
                raise expected_error

            return original_mkdir(path, mode, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.mkdir",
                side_effect=mkdir_with_workspace_failure,
            ),
            self.assertRaises(AgentWorkspaceError) as raised_error,
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertIs(raised_error.exception.__cause__, expected_error)

    def test_prepare_workspace_normalizes_context_write_filesystem_failure(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.get_workspace_path(run_id)
        original_open = os.open
        expected_error = PermissionError("Denied")

        def open_with_context_write_failure(path, flags, mode=0o777, *, dir_fd=None):
            """
            Fail the context destination file creation operation.
            """

            if path == "AGENTS.md" and flags & os.O_CREAT:
                raise expected_error

            return original_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.open",
                side_effect=open_with_context_write_failure,
            ),
            self.assertRaises(AgentWorkspaceError) as raised_error,
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertIs(raised_error.exception.__cause__, expected_error)

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_prepare_workspace_marks_unsafe_when_ordinary_error_hides_workspace_replacement(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.get_workspace_path(run_id)
        moved_workspace_path = self.temp_root / "combined-failure-original-workspace"
        replacement_workspace_path = self.temp_root / "combined-failure-replacement-workspace"
        replacement_file = workspace_path / "sentinel.txt"
        original_write_context_file = workspace_manager_module._write_context_file_at
        replaced_workspace = False

        def replace_workspace_then_fail(workspace_fd, destination, content):
            """
            Replace the canonical workspace before raising an ordinary bootstrap error.
            """

            nonlocal replaced_workspace

            if destination == "AGENTS.md" and not replaced_workspace:
                workspace_path.rename(moved_workspace_path)
                replacement_workspace_path.mkdir()
                (replacement_workspace_path / "sentinel.txt").write_text(
                    "Do not delete.\n",
                    encoding="utf-8",
                )
                replacement_workspace_path.rename(workspace_path)
                replaced_workspace = True

                raise AgentWorkspaceError("Injected ordinary bootstrap failure")

            return original_write_context_file(workspace_fd, destination, content)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._write_context_file_at",
                side_effect=replace_workspace_then_fail,
            ),
            self.assertRaises(WorkspaceIdentityError) as raised_error,
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertIsInstance(raised_error.exception.__cause__, AgentWorkspaceError)
        self.assertTrue(moved_workspace_path.is_dir())
        self.assertTrue((moved_workspace_path / "inputs").is_dir())
        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(replacement_file.read_text(encoding="utf-8"), "Do not delete.\n")
        self.assertFalse((workspace_path / "AGENTS.md").exists())

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(replacement_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_prepare_workspace_marks_unsafe_when_ordinary_error_hides_root_replacement(self):
        run_id = uuid4()
        workspace_name = str(run_id)
        moved_workspace_root = self.temp_root / "combined-failure-original-root"
        replacement_workspace_root = self.temp_root / "workspaces-replacement-root"
        replacement_workspace_path = self.workspace_root / workspace_name
        replacement_file = replacement_workspace_path / "sentinel.txt"
        replaced_workspace_root = False

        def replace_root_then_fail(*args, **kwargs):
            """
            Replace the configured workspace root before an ordinary manifest error.
            """

            nonlocal replaced_workspace_root

            if not replaced_workspace_root:
                self.workspace_root.rename(moved_workspace_root)
                replacement_workspace_root.mkdir(mode=0o700)
                (replacement_workspace_root / workspace_name).mkdir()
                (replacement_workspace_root / workspace_name / "sentinel.txt").write_text(
                    "Do not delete.\n",
                    encoding="utf-8",
                )
                replacement_workspace_root.rename(self.workspace_root)
                replaced_workspace_root = True

            raise InvalidContextManifest("Injected manifest failure")

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
                side_effect=replace_root_then_fail,
            ),
            self.assertRaises(WorkspaceIdentityError) as raised_error,
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.assertIsInstance(raised_error.exception.__cause__, InvalidContextManifest)
        self.assertTrue((moved_workspace_root / workspace_name / "inputs").is_dir())
        self.assertTrue(replacement_workspace_path.is_dir())
        self.assertEqual(replacement_file.read_text(encoding="utf-8"), "Do not delete.\n")

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(replacement_workspace_path.is_dir())
        self.assertEqual(replacement_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_prepare_workspace_context_destination_conflict_can_cleanup_partial_workspace(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.get_workspace_path(run_id)

        def load_then_create_destination_conflict(*args, **kwargs):
            """
            Create a destination conflict after workspace identity remains valid.
            """

            context_manifest = load_context_manifest(*args, **kwargs)
            (workspace_path / "AGENTS.md").write_text("Already exists.\n", encoding="utf-8")

            return context_manifest

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.load_context_manifest",
                side_effect=load_then_create_destination_conflict,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_prepare_workspace_subdirectory_failure_can_cleanup_partial_workspace(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.get_workspace_path(run_id)
        original_mkdir = os.mkdir
        expected_error = PermissionError("Denied")

        def mkdir_with_subdirectory_failure(path, mode=0o777, *, dir_fd=None):
            """
            Fail a bootstrap subdirectory creation without changing workspace identity.
            """

            if path == "runtime":
                raise expected_error

            return original_mkdir(path, mode, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.mkdir",
                side_effect=mkdir_with_subdirectory_failure,
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            self.workspace_manager.prepare_workspace(run_id)

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_prepare_workspace_rejects_symlinked_workspace_root(self):
        run_id = uuid4()
        symlink_target = self.temp_root / "symlink-root-target"
        symlink_target.mkdir()
        symlinked_workspace_root = self.temp_root / "symlink-workspaces"
        symlinked_workspace_root.symlink_to(symlink_target, target_is_directory=True)
        workspace_manager = WorkspaceManager(
            workspace_root=symlinked_workspace_root,
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )

        with self.assertRaises(AgentWorkspaceError):
            workspace_manager.prepare_workspace(run_id)

        self.assertFalse((symlink_target / str(run_id)).exists())
        self.assertTrue(symlinked_workspace_root.is_symlink())

    def test_prepare_workspace_rejects_symlinked_workspace_root_ancestor(self):
        run_id = uuid4()
        symlink_target = self.temp_root / "ancestor-target"
        symlink_target.mkdir()
        symlinked_parent = self.temp_root / "symlink-parent"
        symlinked_parent.symlink_to(symlink_target, target_is_directory=True)
        configured_workspace_root = symlinked_parent / "workspaces"
        workspace_manager = WorkspaceManager(
            workspace_root=configured_workspace_root,
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )

        with self.assertRaises(AgentWorkspaceError):
            workspace_manager.prepare_workspace(run_id)

        self.assertFalse((symlink_target / "workspaces").exists())

    def test_prepare_workspace_fails_when_workspace_root_path_is_replaced_after_pin(self):
        run_id = uuid4()
        configured_parent = self.temp_root / "race-configured-parent"
        configured_workspace_root = configured_parent / "workspaces"
        replacement_parent_target = self.temp_root / "race-replacement-parent-target"
        replacement_parent_target.mkdir()
        pinned_parent = self.temp_root / "race-pinned-parent"
        workspace_manager = WorkspaceManager(
            workspace_root=configured_workspace_root,
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )
        original_acquire = workspace_manager._acquire_workspace_root

        def acquire_then_replace_ancestor(*, create):
            """
            Replace the configured ancestor after root acquisition.
            """

            trusted_root = original_acquire(create=create)
            configured_parent.rename(pinned_parent)
            configured_parent.symlink_to(replacement_parent_target, target_is_directory=True)

            return trusted_root

        with patch.object(
            workspace_manager,
            "_acquire_workspace_root",
            side_effect=acquire_then_replace_ancestor,
        ):
            with self.assertRaises(AgentWorkspaceError):
                workspace_manager.prepare_workspace(run_id)

        self.assertTrue((pinned_parent / "workspaces" / str(run_id)).is_dir())
        self.assertTrue((pinned_parent / "workspaces" / str(run_id) / "AGENTS.md").is_file())
        self.assertFalse((replacement_parent_target / "workspaces" / str(run_id)).exists())

        workspace_manager.cleanup_workspace(run_id)

        self.assertTrue((pinned_parent / "workspaces" / str(run_id) / "AGENTS.md").is_file())
        self.assertFalse((replacement_parent_target / "workspaces" / str(run_id)).exists())

    def test_get_workspace_path_rejects_invalid_run_id(self):
        with self.assertRaises(AgentWorkspaceError):
            self.workspace_manager.get_workspace_path("../not-a-uuid")

    def test_cleanup_workspace_removes_workspace(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_cleanup_workspace_is_idempotent(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path

        self.workspace_manager.cleanup_workspace(run_id)
        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_cleanup_workspace_normalizes_filesystem_failure(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        workspace_name = str(run_id)
        original_rmdir = os.rmdir
        expected_error = PermissionError("Denied")

        def rmdir_with_workspace_failure(path, *, dir_fd=None):
            """
            Fail the final per-run workspace directory removal operation.
            """

            if path == workspace_name:
                raise expected_error

            return original_rmdir(path, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.rmdir",
                side_effect=rmdir_with_workspace_failure,
            ),
            self.assertRaises(WorkspaceCleanupError) as raised_error,
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertIs(raised_error.exception.__cause__, expected_error)
        self.assertTrue(workspace_path.exists())

    def test_cleanup_workspace_recursively_removes_nested_contents(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        nested_directory = workspace_path / "outputs" / "nested" / "deeper"
        nested_directory.mkdir(parents=True)
        (nested_directory / "result.txt").write_text("Delete this.\n", encoding="utf-8")

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_cleanup_workspace_does_not_follow_nested_symlink(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        outside_directory = self.temp_root / "nested-symlink-target"
        outside_directory.mkdir()
        outside_file = outside_directory / "sentinel.txt"
        outside_file.write_text("Do not delete.\n", encoding="utf-8")
        symlink_path = workspace_path / "outputs" / "external-link"
        symlink_path.symlink_to(outside_directory, target_is_directory=True)

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())
        self.assertEqual(outside_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_cleanup_workspace_fails_when_nested_directory_is_replaced_by_symlink(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        nested_directory = workspace_path / "outputs" / "nested"
        nested_directory.mkdir()
        (nested_directory / "original.txt").write_text("Original.\n", encoding="utf-8")
        moved_nested_directory = self.temp_root / "moved-nested-directory"
        symlink_target = self.temp_root / "nested-replacement-symlink-target"
        symlink_target.mkdir()
        symlink_target_file = symlink_target / "sentinel.txt"
        symlink_target_file.write_text("Do not delete.\n", encoding="utf-8")
        original_open = os.open
        replaced_nested_directory = False

        def open_then_replace_nested_directory(path, flags, mode=0o777, *, dir_fd=None):
            """
            Replace a nested directory with a symlink before cleanup can pin it.
            """

            nonlocal replaced_nested_directory

            if path == "nested" and not replaced_nested_directory and nested_directory.exists():
                nested_directory.rename(moved_nested_directory)
                nested_directory.symlink_to(symlink_target, target_is_directory=True)
                replaced_nested_directory = True

            return original_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.open",
                side_effect=open_then_replace_nested_directory,
            ),
            self.assertRaises(WorkspaceCleanupError),
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(nested_directory.is_symlink())
        self.assertEqual(symlink_target_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_cleanup_workspace_fails_when_nested_directory_is_replaced_by_directory(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        nested_directory = workspace_path / "outputs" / "nested"
        nested_directory.mkdir()
        (nested_directory / "original.txt").write_text("Original.\n", encoding="utf-8")
        moved_nested_directory = self.temp_root / "moved-nested-directory-for-replacement"
        replacement_directory = self.temp_root / "nested-replacement-directory"
        replacement_directory.mkdir()
        replacement_file = replacement_directory / "sentinel.txt"
        replacement_file.write_text("Do not delete.\n", encoding="utf-8")
        original_open = os.open
        replaced_nested_directory = False

        def open_then_replace_nested_directory(path, flags, mode=0o777, *, dir_fd=None):
            """
            Replace a nested directory with another directory before cleanup can pin it.
            """

            nonlocal replaced_nested_directory

            if path == "nested" and not replaced_nested_directory and nested_directory.exists():
                nested_directory.rename(moved_nested_directory)
                replacement_directory.rename(nested_directory)
                replaced_nested_directory = True

            return original_open(path, flags, mode, dir_fd=dir_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager.os.open",
                side_effect=open_then_replace_nested_directory,
            ),
            self.assertRaises(WorkspaceCleanupError),
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(nested_directory.is_dir())
        self.assertEqual(
            (nested_directory / "sentinel.txt").read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

    def test_cleanup_workspace_fails_when_canonical_workspace_replaced_before_deletion(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        (workspace_path / "original.txt").write_text("Original.\n", encoding="utf-8")
        moved_workspace = self.temp_root / "moved-workspace-before-cleanup-delete"
        replacement_workspace = self.temp_root / "replacement-before-cleanup-delete"
        replacement_workspace.mkdir()
        replacement_file = replacement_workspace / "sentinel.txt"
        replacement_file.write_text("Do not delete.\n", encoding="utf-8")
        original_remove_contents = workspace_manager_module._remove_directory_contents
        replaced_workspace = False

        def remove_contents_after_replacing_workspace(directory_fd):
            """
            Replace the canonical workspace after cleanup has pinned it.
            """

            nonlocal replaced_workspace

            if not replaced_workspace:
                workspace_path.rename(moved_workspace)
                replacement_workspace.rename(workspace_path)
                replaced_workspace = True

            return original_remove_contents(directory_fd)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._remove_directory_contents",
                side_effect=remove_contents_after_replacing_workspace,
            ),
            self.assertRaises(WorkspaceCleanupError),
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(
            (workspace_path / replacement_file.name).read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

    def test_cleanup_workspace_fails_when_canonical_workspace_replaced_during_deletion(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        (workspace_path / "extra.txt").write_text("Original.\n", encoding="utf-8")
        moved_workspace = self.temp_root / "moved-workspace-during-cleanup-delete"
        replacement_workspace = self.temp_root / "replacement-during-cleanup-delete"
        replacement_workspace.mkdir()
        replacement_file = replacement_workspace / "sentinel.txt"
        replacement_file.write_text("Do not delete.\n", encoding="utf-8")
        original_unlink_entry = workspace_manager_module._unlink_cleanup_entry
        replaced_workspace = False

        def unlink_then_replace_workspace(parent_fd, entry_name, *, missing_ok):
            """
            Replace the canonical workspace while recursive deletion is in progress.
            """

            nonlocal replaced_workspace

            original_unlink_entry(parent_fd, entry_name, missing_ok=missing_ok)

            if entry_name == "AGENTS.md" and not replaced_workspace:
                workspace_path.rename(moved_workspace)
                replacement_workspace.rename(workspace_path)
                replaced_workspace = True

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._unlink_cleanup_entry",
                side_effect=unlink_then_replace_workspace,
            ),
            self.assertRaises(WorkspaceCleanupError),
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(
            (workspace_path / replacement_file.name).read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

    def test_cleanup_workspace_fails_when_canonical_workspace_replaced_before_final_rmdir(self):
        run_id = uuid4()
        workspace_path = self.workspace_manager.prepare_workspace(run_id).workspace_path
        moved_workspace = self.temp_root / "moved-workspace-before-final-rmdir"
        replacement_workspace = self.temp_root / "replacement-before-final-rmdir"
        replacement_workspace.mkdir()
        replacement_file = replacement_workspace / "sentinel.txt"
        replacement_file.write_text("Do not delete.\n", encoding="utf-8")
        original_remove_contents = workspace_manager_module._remove_directory_contents
        replaced_workspace = False

        def remove_contents_then_replace_workspace(directory_fd):
            """
            Replace the canonical workspace after contents are removed.
            """

            nonlocal replaced_workspace

            original_remove_contents(directory_fd)

            if not replaced_workspace:
                workspace_path.rename(moved_workspace)
                replacement_workspace.rename(workspace_path)
                replaced_workspace = True

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._remove_directory_contents",
                side_effect=remove_contents_then_replace_workspace,
            ),
            self.assertRaises(WorkspaceCleanupError),
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(workspace_path.is_dir())
        self.assertEqual(
            (workspace_path / replacement_file.name).read_text(encoding="utf-8"),
            "Do not delete.\n",
        )

    def test_cleanup_workspace_does_not_leak_file_descriptors_after_success(self):
        run_id = uuid4()
        self.workspace_manager.prepare_workspace(run_id)
        before_cleanup_fd_count = self.open_fd_count()

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertEqual(self.open_fd_count(), before_cleanup_fd_count)

    def test_cleanup_workspace_does_not_leak_file_descriptors_after_failure(self):
        run_id = uuid4()
        self.workspace_manager.prepare_workspace(run_id)
        before_cleanup_fd_count = self.open_fd_count()

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._remove_directory_contents",
                side_effect=WorkspaceCleanupError("Injected cleanup failure"),
            ),
            self.assertRaises(WorkspaceCleanupError),
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertEqual(self.open_fd_count(), before_cleanup_fd_count)

    def test_cleanup_workspace_does_not_follow_workspace_symlink(self):
        run_id = uuid4()
        self.workspace_root.mkdir()
        outside_directory = self.temp_root / "outside"
        outside_directory.mkdir()
        outside_file = outside_directory / "sentinel.txt"
        outside_file.write_text("Do not delete.\n", encoding="utf-8")
        workspace_path = self.workspace_manager.get_workspace_path(run_id)
        workspace_path.symlink_to(outside_directory, target_is_directory=True)

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())
        self.assertTrue(outside_file.exists())

    def test_cleanup_workspace_rejects_symlinked_workspace_root(self):
        run_id = uuid4()
        symlink_target = self.temp_root / "cleanup-symlink-root-target"
        target_workspace = symlink_target / str(run_id)
        target_workspace.mkdir(parents=True)
        target_file = target_workspace / "sentinel.txt"
        target_file.write_text("Do not delete.\n", encoding="utf-8")
        symlinked_workspace_root = self.temp_root / "cleanup-symlink-workspaces"
        symlinked_workspace_root.symlink_to(symlink_target, target_is_directory=True)
        workspace_manager = WorkspaceManager(workspace_root=symlinked_workspace_root)

        with self.assertRaises(AgentWorkspaceError):
            workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(symlinked_workspace_root.is_symlink())
        self.assertTrue(target_workspace.exists())
        self.assertEqual(target_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_cleanup_workspace_rejects_symlinked_workspace_root_ancestor(self):
        run_id = uuid4()
        symlink_target = self.temp_root / "cleanup-ancestor-target"
        target_workspace_root = symlink_target / "workspaces"
        target_workspace = target_workspace_root / str(run_id)
        target_workspace.mkdir(parents=True)
        target_file = target_workspace / "sentinel.txt"
        target_file.write_text("Do not delete.\n", encoding="utf-8")
        symlinked_parent = self.temp_root / "cleanup-symlink-parent"
        symlinked_parent.symlink_to(symlink_target, target_is_directory=True)
        workspace_manager = WorkspaceManager(workspace_root=symlinked_parent / "workspaces")

        with self.assertRaises(AgentWorkspaceError):
            workspace_manager.cleanup_workspace(run_id)

        self.assertTrue(target_workspace.exists())
        self.assertEqual(target_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_cleanup_workspace_uses_pinned_root_when_root_is_replaced(self):
        run_id = uuid4()
        workspace_name = str(run_id)
        self.workspace_root.mkdir()
        real_workspace = self.workspace_root / workspace_name
        real_workspace.mkdir()
        (real_workspace / "sentinel.txt").write_text("Delete this workspace.\n", encoding="utf-8")
        replacement_target = self.temp_root / "cleanup-race-replacement-target"
        replacement_workspace = replacement_target / workspace_name
        replacement_workspace.mkdir(parents=True)
        replacement_file = replacement_workspace / "sentinel.txt"
        replacement_file.write_text("Do not delete.\n", encoding="utf-8")
        pinned_workspace_root = self.temp_root / "cleanup-race-pinned-workspaces"
        original_acquire = self.workspace_manager._acquire_workspace_root

        def acquire_then_replace_root(*, create):
            """
            Replace the configured root after root acquisition.
            """

            trusted_root = original_acquire(create=create)
            self.workspace_root.rename(pinned_workspace_root)
            self.workspace_root.symlink_to(replacement_target, target_is_directory=True)

            return trusted_root

        with patch.object(
            self.workspace_manager,
            "_acquire_workspace_root",
            side_effect=acquire_then_replace_root,
        ):
            self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse((pinned_workspace_root / workspace_name).exists())
        self.assertTrue(replacement_workspace.exists())
        self.assertEqual(replacement_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_cleanup_workspace_uses_pinned_root_when_ancestor_is_replaced(self):
        run_id = uuid4()
        workspace_name = str(run_id)
        configured_parent = self.temp_root / "cleanup-race-configured-parent"
        configured_workspace_root = configured_parent / "workspaces"
        configured_workspace = configured_workspace_root / workspace_name
        configured_workspace.mkdir(parents=True)
        (configured_workspace / "sentinel.txt").write_text(
            "Delete this workspace.\n",
            encoding="utf-8",
        )
        replacement_parent_target = self.temp_root / "cleanup-race-replacement-parent-target"
        replacement_workspace = replacement_parent_target / "workspaces" / workspace_name
        replacement_workspace.mkdir(parents=True)
        replacement_file = replacement_workspace / "sentinel.txt"
        replacement_file.write_text("Do not delete.\n", encoding="utf-8")
        pinned_parent = self.temp_root / "cleanup-race-pinned-parent"
        workspace_manager = WorkspaceManager(workspace_root=configured_workspace_root)
        original_acquire = workspace_manager._acquire_workspace_root

        def acquire_then_replace_ancestor(*, create):
            """
            Replace the configured ancestor after root acquisition.
            """

            trusted_root = original_acquire(create=create)
            configured_parent.rename(pinned_parent)
            configured_parent.symlink_to(replacement_parent_target, target_is_directory=True)

            return trusted_root

        with patch.object(
            workspace_manager,
            "_acquire_workspace_root",
            side_effect=acquire_then_replace_ancestor,
        ):
            workspace_manager.cleanup_workspace(run_id)

        self.assertFalse((pinned_parent / "workspaces" / workspace_name).exists())
        self.assertTrue(replacement_workspace.exists())
        self.assertEqual(replacement_file.read_text(encoding="utf-8"), "Do not delete.\n")

    def test_partial_bootstrap_can_be_cleaned_up(self):
        run_id = uuid4()
        manifest = self.valid_manifest()
        manifest["files"][1]["source"] = "missing-config.toml"
        self.write_manifest(manifest)

        with self.assertRaises(InvalidContextManifest):
            self.workspace_manager.prepare_workspace(run_id)

        workspace_path = self.workspace_manager.get_workspace_path(run_id)
        self.assertTrue(workspace_path.exists())

        self.workspace_manager.cleanup_workspace(run_id)

        self.assertFalse(workspace_path.exists())

    def test_prepare_workspace_does_not_mutate_source_context(self):
        user_agents_content = self.user_agents_path.read_bytes()
        config_content = self.config_path.read_bytes()

        self.workspace_manager.prepare_workspace(uuid4())

        self.assertEqual(self.user_agents_path.read_bytes(), user_agents_content)
        self.assertEqual(self.config_path.read_bytes(), config_content)

    def compute_context_hash(self, context_version: str, files: dict[str, bytes]) -> str:
        """
        Compute the expected context hash contract from materialized bytes.
        """

        digest = hashlib.sha256()
        self.update_length_prefixed(digest, b"agent-context-manifest-v1")
        self.update_length_prefixed(digest, context_version.encode("utf-8"))

        for destination, content in sorted(files.items()):
            self.update_length_prefixed(digest, destination.encode("utf-8"))
            self.update_length_prefixed(digest, content)

        return digest.hexdigest()

    def update_length_prefixed(self, digest, value: bytes) -> None:
        """
        Add unambiguous length-prefixed bytes to a hash digest.
        """

        digest.update(str(len(value)).encode("ascii"))
        digest.update(b":")
        digest.update(value)
        digest.update(b"\n")

    def open_fd_count(self) -> int:
        """
        Return the current process file descriptor count when the platform exposes it.
        """

        for descriptor_directory in (Path("/proc/self/fd"), Path("/dev/fd")):
            if descriptor_directory.exists():
                try:
                    return len(os.listdir(descriptor_directory))
                except OSError:
                    continue

        self.skipTest("Open file descriptor count is not available on this platform.")


class AgentRunLifecycleServiceTests(TestCase):
    """
    Cover AgentRun lifecycle transitions and cancellation semantics.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for lifecycle service tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="lifecycle-user",
            password="test-password",
        )

    def create_agent_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)

        return AgentRun.objects.create(**defaults)

    def create_queued_run(self) -> AgentRun:
        """
        Create an AgentRun and transition it to QUEUED through the service.
        """

        agent_run = self.create_agent_run()

        return transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def create_running_run(self) -> AgentRun:
        """
        Create an AgentRun and transition it to RUNNING through the service.
        """

        agent_run = self.create_queued_run()

        return transition_agent_run(agent_run.id, AgentRunStatus.RUNNING)

    def create_terminal_run(self, target_status: AgentRunStatus) -> AgentRun:
        """
        Create a RUNNING AgentRun and transition it to a terminal status.
        """

        agent_run = self.create_running_run()

        return transition_agent_run(agent_run.id, target_status)

    def assert_invalid_transition(
        self, source_status: AgentRunStatus, target_status: AgentRunStatus
    ):
        """
        Assert a transition from a prepared source state is rejected.
        """

        agent_run = self.create_terminal_run(source_status)

        with self.assertRaises(InvalidAgentRunTransition):
            transition_agent_run(agent_run.id, target_status)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, source_status.value)

    def test_created_to_queued_transition(self):
        agent_run = self.create_agent_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.QUEUED.value)

    def test_queued_to_running_transition(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.RUNNING)

        self.assertEqual(transitioned_run.status, AgentRunStatus.RUNNING.value)

    def test_queued_to_cancelled_transition(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.CANCELLED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.CANCELLED.value)

    def test_running_to_succeeded_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.SUCCEEDED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.SUCCEEDED.value)

    def test_running_to_failed_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.FAILED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.FAILED.value)

    def test_running_to_timed_out_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.TIMED_OUT)

        self.assertEqual(transitioned_run.status, AgentRunStatus.TIMED_OUT.value)

    def test_running_to_cancelled_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.CANCELLED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.CANCELLED.value)

    def test_succeeded_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.SUCCEEDED, AgentRunStatus.RUNNING)

    def test_failed_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.FAILED, AgentRunStatus.RUNNING)

    def test_timed_out_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.TIMED_OUT, AgentRunStatus.RUNNING)

    def test_cancelled_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.CANCELLED, AgentRunStatus.RUNNING)

    def test_succeeded_to_failed_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.SUCCEEDED, AgentRunStatus.FAILED)

    def test_invalid_transition_raises_deterministic_exception(self):
        agent_run = self.create_terminal_run(AgentRunStatus.SUCCEEDED)

        with self.assertRaises(InvalidAgentRunTransition) as context:
            transition_agent_run(agent_run.id, AgentRunStatus.FAILED)

        exception = context.exception
        self.assertEqual(exception.source_status, AgentRunStatus.SUCCEEDED.value)
        self.assertEqual(exception.target_status, AgentRunStatus.FAILED.value)
        self.assertEqual(exception.run_id, agent_run.id)
        self.assertEqual(
            str(exception),
            (f"Invalid AgentRun transition for {agent_run.id}: SUCCEEDED -> FAILED."),
        )

    def test_created_to_queued_sets_queued_at(self):
        agent_run = self.create_agent_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

        self.assertIsNotNone(transitioned_run.queued_at)

    def test_queued_to_running_sets_started_at(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.RUNNING)

        self.assertIsNotNone(transitioned_run.started_at)

    def test_running_to_terminal_sets_finished_at(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.SUCCEEDED)

        self.assertIsNotNone(transitioned_run.finished_at)

    def test_queued_to_cancelled_sets_finished_at(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.CANCELLED)

        self.assertIsNotNone(transitioned_run.finished_at)

    def test_request_cancellation_when_queued_transitions_to_cancelled(self):
        agent_run = self.create_queued_run()

        cancelled_run = request_agent_run_cancellation(agent_run.id)

        self.assertEqual(cancelled_run.status, AgentRunStatus.CANCELLED.value)
        self.assertIsNotNone(cancelled_run.finished_at)

    def test_request_cancellation_when_running_keeps_running_and_sets_cancel_requested_at(self):
        agent_run = self.create_running_run()

        cancellation_requested_run = request_agent_run_cancellation(agent_run.id)

        self.assertEqual(cancellation_requested_run.status, AgentRunStatus.RUNNING.value)
        self.assertIsNotNone(cancellation_requested_run.cancel_requested_at)
        self.assertIsNone(cancellation_requested_run.finished_at)

    def test_cancellation_requested_helper_reads_authoritative_state(self):
        agent_run = self.create_running_run()

        self.assertFalse(is_agent_run_cancellation_requested(agent_run.id))

        request_agent_run_cancellation(agent_run.id)

        self.assertTrue(is_agent_run_cancellation_requested(agent_run.id))

    def test_repeated_running_cancellation_does_not_overwrite_cancel_requested_at(self):
        agent_run = self.create_running_run()
        first_request_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        second_request_time = datetime(2026, 1, 1, 12, 5, tzinfo=UTC)

        with patch(
            "apps.agent_workspace.services.agent_run_service.timezone.now",
            return_value=first_request_time,
        ):
            request_agent_run_cancellation(agent_run.id)

        with patch(
            "apps.agent_workspace.services.agent_run_service.timezone.now",
            return_value=second_request_time,
        ):
            cancellation_requested_run = request_agent_run_cancellation(agent_run.id)

        self.assertEqual(cancellation_requested_run.status, AgentRunStatus.RUNNING.value)
        self.assertEqual(cancellation_requested_run.cancel_requested_at, first_request_time)

    def test_running_run_can_record_context_metadata(self):
        agent_run = self.create_running_run()

        recorded_run = record_agent_run_context(
            agent_run.id,
            "context-v1",
            "a" * 64,
        )

        self.assertEqual(recorded_run.context_version, "context-v1")
        self.assertEqual(recorded_run.context_hash, "a" * 64)

    def test_context_metadata_is_persisted(self):
        agent_run = self.create_running_run()

        record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.context_version, "context-v1")
        self.assertEqual(agent_run.context_hash, "a" * 64)

    def test_same_context_metadata_is_idempotent(self):
        agent_run = self.create_running_run()
        first_recorded_run = record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        second_recorded_run = record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        self.assertEqual(second_recorded_run.id, first_recorded_run.id)
        self.assertEqual(second_recorded_run.context_version, "context-v1")
        self.assertEqual(second_recorded_run.context_hash, "a" * 64)

    def test_conflicting_context_metadata_is_rejected(self):
        agent_run = self.create_running_run()
        record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        with self.assertRaises(AgentRunContextConflict):
            record_agent_run_context(agent_run.id, "context-v2", "b" * 64)

    def test_conflicting_context_metadata_does_not_mutate_existing_values(self):
        agent_run = self.create_running_run()
        record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        with self.assertRaises(AgentRunContextConflict):
            record_agent_run_context(agent_run.id, "context-v2", "b" * 64)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.context_version, "context-v1")
        self.assertEqual(agent_run.context_hash, "a" * 64)

    def test_non_running_run_context_metadata_is_rejected(self):
        agent_run = self.create_queued_run()

        with self.assertRaises(InvalidAgentRunContextState):
            record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        agent_run.refresh_from_db()
        self.assertIsNone(agent_run.context_version)
        self.assertIsNone(agent_run.context_hash)

    def test_terminal_run_context_metadata_is_rejected_without_mutation(self):
        agent_run = self.create_terminal_run(AgentRunStatus.SUCCEEDED)

        with self.assertRaises(InvalidAgentRunContextState):
            record_agent_run_context(agent_run.id, "context-v1", "a" * 64)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNone(agent_run.context_version)
        self.assertIsNone(agent_run.context_hash)


class AgentRunExecutionClaimTests(TestCase):
    """
    Cover atomic AgentRun execution claim behavior.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for execution claim tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="claim-user",
            password="test-password",
        )

    def create_agent_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)

        return AgentRun.objects.create(**defaults)

    def create_queued_run(self) -> AgentRun:
        """
        Create an AgentRun and transition it to QUEUED through the service.
        """

        agent_run = self.create_agent_run()

        return transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def test_queued_agent_run_can_be_claimed_for_execution(self):
        agent_run = self.create_queued_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNotNone(claimed_run)

    def test_successful_claim_transitions_queued_to_running(self):
        agent_run = self.create_queued_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertEqual(claimed_run.status, AgentRunStatus.RUNNING.value)

    def test_successful_claim_sets_started_at(self):
        agent_run = self.create_queued_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNotNone(claimed_run.started_at)

    def test_non_queued_agent_run_is_not_claimed(self):
        agent_run = self.create_agent_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNone(claimed_run)

    def test_second_claim_attempt_does_not_create_second_claim(self):
        agent_run = self.create_queued_run()

        first_claim = claim_agent_run_for_execution(agent_run.id)
        second_claim = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNotNone(first_claim)
        self.assertIsNone(second_claim)

    def test_second_claim_attempt_does_not_destroy_current_state(self):
        agent_run = self.create_queued_run()
        first_claim = claim_agent_run_for_execution(agent_run.id)

        claim_agent_run_for_execution(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.RUNNING.value)
        self.assertEqual(agent_run.started_at, first_claim.started_at)


class AgentRunTaskContractTests(TestCase):
    """
    Cover the Celery task boundary for AgentRun execution.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for task contract tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="task-user",
            password="test-password",
        )

    def create_agent_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)

        return AgentRun.objects.create(**defaults)

    def test_execute_agent_run_task_accepts_only_run_id_business_input(self):
        self.assertEqual(execute_agent_run.run.__code__.co_argcount, 1)
        self.assertEqual(execute_agent_run.run.__code__.co_varnames[:1], ("run_id",))

    def test_execute_agent_run_task_delegates_run_id_to_lifecycle_service(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run_lifecycle") as lifecycle:
            execute_agent_run.run(str(agent_run.id))

        lifecycle.assert_called_once_with(str(agent_run.id))

    def test_execute_agent_run_task_delegates_missing_run_to_service_boundary(self):
        missing_run_id = str(uuid4())

        with patch("apps.agent_workspace.tasks.execute_agent_run_lifecycle") as lifecycle:
            execute_agent_run.run(missing_run_id)

        lifecycle.assert_called_once_with(missing_run_id)

    def test_execute_agent_run_task_delegates_lifecycle_workflow(self):
        agent_run = self.create_agent_run()
        transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

        with patch("apps.agent_workspace.tasks.execute_agent_run_lifecycle") as lifecycle:
            execute_agent_run.run(str(agent_run.id))

        lifecycle.assert_called_once_with(str(agent_run.id))
        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.QUEUED.value)

    def test_execute_agent_run_lifecycle_raises_for_missing_run(self):
        missing_run_id = str(uuid4())

        with self.assertRaises(AgentRun.DoesNotExist):
            execute_agent_run_lifecycle(missing_run_id)

    def test_execute_agent_run_task_does_not_use_result_backend_for_business_state(self):
        self.assertTrue(execute_agent_run.ignore_result)
        self.assertIsNone(celery_app.conf.result_backend)


class AgentRunDummyExecutionTests(TestCase):
    """
    Cover dummy execution lifecycle outcomes.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for dummy execution tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="dummy-execution-user",
            password="test-password",
        )

    def setUp(self):
        """
        Use an isolated WorkspaceManager for execution lifecycle tests.
        """

        self.temp_directory = TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.temp_root = Path(self.temp_directory.name).resolve()
        self.workspace_root = self.temp_root / "workspaces"
        self.fake_bin_path = self.temp_root / "bin"
        self.context_root = self.temp_root / "end_user_context"
        self.codex_directory = self.context_root / ".codex"
        self.fake_bin_path.mkdir()
        self.codex_directory.mkdir(parents=True)
        self.manifest_path = self.context_root / "context_manifest.json"
        self.user_agents_path = self.context_root / "USER_AGENTS.md"
        self.config_path = self.codex_directory / "config.toml"
        self.user_agents_path.write_text("End-user execution context.\n", encoding="utf-8")
        self.config_path.write_text('sandbox_mode = "workspace-write"\n', encoding="utf-8")
        self.write_manifest()
        self.workspace_manager = WorkspaceManager(
            workspace_root=self.workspace_root,
            manifest_path=self.manifest_path,
            context_root=self.context_root,
        )
        self.workspace_manager_patcher = patch(
            "apps.agent_workspace.services.execution_service.WorkspaceManager",
            return_value=self.workspace_manager,
        )
        self.workspace_manager_patcher.start()
        self.addCleanup(self.workspace_manager_patcher.stop)

    def write_manifest(self, manifest_data=None) -> None:
        """
        Write an execution context manifest fixture.
        """

        manifest = self.valid_manifest() if manifest_data is None else manifest_data
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def valid_manifest(self) -> dict:
        """
        Return a valid execution context manifest fixture.
        """

        return {
            "context_version": "execution-context-v1",
            "files": [
                {
                    "source": "USER_AGENTS.md",
                    "target": "AGENTS.md",
                },
                {
                    "source": ".codex/config.toml",
                    "target": ".codex/config.toml",
                },
            ],
        }

    def create_queued_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun and transition it to QUEUED through the service.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)
        agent_run = AgentRun.objects.create(**defaults)

        return transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def create_agent_lifecycle_executable(
        self,
        *,
        executable_name: str = "fake-agent-codex",
        exit_code: int = 0,
        sleep_seconds: float = 0.0,
        side_effect_path: Path | None = None,
    ) -> Path:
        """
        Create a fake Codex executable for ExecutionService integration tests.
        """

        executable_path = self.fake_bin_path / executable_name
        executable_path.write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import json",
                    "import os",
                    "import sys",
                    "import time",
                    f"EXIT_CODE = {exit_code!r}",
                    f"SLEEP_SECONDS = {sleep_seconds!r}",
                    f"SIDE_EFFECT_PATH = {str(side_effect_path) if side_effect_path else None!r}",
                    "stdin_text = sys.stdin.read()",
                    "payload = {",
                    "    'argv': sys.argv,",
                    "    'cwd': os.getcwd(),",
                    "    'stdin': stdin_text,",
                    "    'has_agents': os.path.exists('AGENTS.md'),",
                    "    'has_codex_config': os.path.exists('.codex/config.toml'),",
                    "}",
                    "if SIDE_EFFECT_PATH is not None:",
                    "    with open(SIDE_EFFECT_PATH, 'a', encoding='utf-8') as marker_file:",
                    "        marker_file.write(json.dumps(payload, sort_keys=True) + '\\n')",
                    "print('stdout marker')",
                    "print('stderr marker', file=sys.stderr)",
                    "if SLEEP_SECONDS:",
                    "    time.sleep(SLEEP_SECONDS)",
                    "sys.exit(EXIT_CODE)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        executable_path.chmod(0o700)

        return executable_path

    def test_successful_dummy_execution_reaches_succeeded(self):
        agent_run = self.create_queued_run()

        executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(executed_run.started_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_dummy_execution_failure_reaches_failed_before_raising(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "failure"})

        with self.assertRaises(AgentExecutionError):
            execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertIsNotNone(agent_run.started_at)
        self.assertIsNotNone(agent_run.finished_at)

    def test_dummy_execution_timeout_reaches_timed_out(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        self.assertIsNotNone(executed_run.started_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_duplicate_delivery_does_not_execute_dummy_work_twice(self):
        agent_run = self.create_queued_run()

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            return_value=AgentRunStatus.SUCCEEDED,
        ) as dummy_execution:
            first_result = execute_agent_run_lifecycle(agent_run.id)
            second_result = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(first_result.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNone(second_result)
        dummy_execution.assert_called_once()

    def test_duplicate_delivery_after_first_claim_is_execution_boundary_no_op(self):
        agent_run = self.create_queued_run()

        execute_agent_run_lifecycle(agent_run.id)
        second_result = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertIsNone(second_result)
        self.assertEqual(agent_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(agent_run.finished_at)

    def test_agent_run_database_status_is_authoritative_execution_outcome(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.TIMED_OUT.value)

    def test_successful_execution_prepares_workspace(self):
        agent_run = self.create_queued_run()

        with patch.object(
            self.workspace_manager,
            "prepare_workspace",
            wraps=self.workspace_manager.prepare_workspace,
        ) as prepare_workspace:
            execute_agent_run_lifecycle(agent_run.id)

        prepare_workspace.assert_called_once_with(agent_run.id)

    def test_context_metadata_exists_before_dummy_execution(self):
        agent_run = self.create_queued_run()

        def assert_context_exists_before_work(claimed_run):
            """
            Verify context audit metadata is persisted before dummy work starts.
            """

            persisted_run = AgentRun.objects.get(id=claimed_run.id)
            self.assertEqual(persisted_run.context_version, "execution-context-v1")
            self.assertIsNotNone(persisted_run.context_hash)
            self.assertTrue(self.workspace_manager.get_workspace_path(claimed_run.id).exists())

            return AgentRunStatus.SUCCEEDED

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=assert_context_exists_before_work,
        ):
            execute_agent_run_lifecycle(agent_run.id)

    def test_successful_execution_cleans_workspace(self):
        agent_run = self.create_queued_run()
        workspace_path = self.workspace_manager.get_workspace_path(agent_run.id)

        execute_agent_run_lifecycle(agent_run.id)

        self.assertFalse(workspace_path.exists())

    def test_failed_execution_cleans_workspace(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "failure"})
        workspace_path = self.workspace_manager.get_workspace_path(agent_run.id)

        with self.assertRaises(AgentExecutionError):
            execute_agent_run_lifecycle(agent_run.id)

        self.assertFalse(workspace_path.exists())

    def test_timeout_execution_cleans_workspace(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})
        workspace_path = self.workspace_manager.get_workspace_path(agent_run.id)

        execute_agent_run_lifecycle(agent_run.id)

        self.assertFalse(workspace_path.exists())

    def test_cancelled_execution_cleans_workspace(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "cancelled"})
        workspace_path = self.workspace_manager.get_workspace_path(agent_run.id)

        execute_agent_run_lifecycle(agent_run.id)

        self.assertFalse(workspace_path.exists())

    def test_bootstrap_failure_reaches_failed(self):
        agent_run = self.create_queued_run()
        manifest = self.valid_manifest()
        manifest["files"][1]["source"] = "missing-config.toml"
        self.write_manifest(manifest)

        with self.assertRaises(InvalidContextManifest):
            execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertIsNotNone(agent_run.finished_at)

    def test_bootstrap_failure_attempts_partial_workspace_cleanup(self):
        agent_run = self.create_queued_run()
        manifest = self.valid_manifest()
        manifest["files"][1]["source"] = "missing-config.toml"
        self.write_manifest(manifest)
        workspace_path = self.workspace_manager.get_workspace_path(agent_run.id)

        with self.assertRaises(InvalidContextManifest):
            execute_agent_run_lifecycle(agent_run.id)

        self.assertFalse(workspace_path.exists())

    def test_workspace_bootstrap_error_attempts_partial_workspace_cleanup(self):
        agent_run = self.create_queued_run()
        workspace_path = self.workspace_manager.get_workspace_path(agent_run.id)

        with (
            patch(
                "apps.agent_workspace.ai_agent.workspace_manager._write_context_file_at",
                side_effect=AgentWorkspaceError("Injected workspace bootstrap failure"),
            ),
            self.assertRaises(AgentWorkspaceError),
        ):
            execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertFalse(workspace_path.exists())

    def test_terminal_state_is_persisted_before_cleanup(self):
        agent_run = self.create_queued_run()

        def assert_terminal_state_persisted(run_id):
            """
            Verify terminal business state is persisted before cleanup.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.SUCCEEDED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch.object(
            self.workspace_manager,
            "cleanup_workspace",
            side_effect=assert_terminal_state_persisted,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)

    def test_cleanup_failure_does_not_rewrite_succeeded(self):
        agent_run = self.create_queued_run()

        with (
            patch.object(
                self.workspace_manager,
                "cleanup_workspace",
                side_effect=RuntimeError("Cleanup failed"),
            ),
            self.assertLogs(
                "apps.agent_workspace.services.execution_service",
                level="ERROR",
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertEqual(agent_run.status, AgentRunStatus.SUCCEEDED.value)

    def test_cleanup_failure_does_not_rewrite_failed(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "failure"})

        with (
            patch.object(
                self.workspace_manager,
                "cleanup_workspace",
                side_effect=RuntimeError("Cleanup failed"),
            ),
            self.assertLogs(
                "apps.agent_workspace.services.execution_service",
                level="ERROR",
            ),
        ):
            with self.assertRaises(AgentExecutionError):
                execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)

    def test_cleanup_failure_does_not_rewrite_timed_out(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        with (
            patch.object(
                self.workspace_manager,
                "cleanup_workspace",
                side_effect=RuntimeError("Cleanup failed"),
            ),
            self.assertLogs(
                "apps.agent_workspace.services.execution_service",
                level="ERROR",
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        self.assertEqual(agent_run.status, AgentRunStatus.TIMED_OUT.value)

    def test_cleanup_failure_does_not_rewrite_cancelled(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "cancelled"})

        with (
            patch.object(
                self.workspace_manager,
                "cleanup_workspace",
                side_effect=RuntimeError("Cleanup failed"),
            ),
            self.assertLogs(
                "apps.agent_workspace.services.execution_service",
                level="ERROR",
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(executed_run.status, AgentRunStatus.CANCELLED.value)
        self.assertEqual(agent_run.status, AgentRunStatus.CANCELLED.value)

    def test_duplicate_delivery_does_not_prepare_second_workspace(self):
        agent_run = self.create_queued_run()

        with patch.object(
            self.workspace_manager,
            "prepare_workspace",
            wraps=self.workspace_manager.prepare_workspace,
        ) as prepare_workspace:
            execute_agent_run_lifecycle(agent_run.id)
            second_result = execute_agent_run_lifecycle(agent_run.id)

        self.assertIsNone(second_result)
        prepare_workspace.assert_called_once_with(agent_run.id)

    def test_duplicate_delivery_does_not_mutate_context_metadata(self):
        agent_run = self.create_queued_run()

        execute_agent_run_lifecycle(agent_run.id)
        first_persisted_run = AgentRun.objects.get(id=agent_run.id)
        first_context_version = first_persisted_run.context_version
        first_context_hash = first_persisted_run.context_hash

        with patch(
            "apps.agent_workspace.services.execution_service.record_agent_run_context",
        ) as record_context:
            second_result = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertIsNone(second_result)
        self.assertEqual(agent_run.context_version, first_context_version)
        self.assertEqual(agent_run.context_hash, first_context_hash)
        record_context.assert_not_called()

    def test_successful_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run()

        def assert_succeeded_persisted(workspace_manager, run_id):
            """
            Verify cleanup is attempted after terminal state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.SUCCEEDED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service._attempt_workspace_cleanup",
            side_effect=assert_succeeded_persisted,
        ) as cleanup_workspace:
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        cleanup_workspace.assert_called_once_with(self.workspace_manager, executed_run.id)

    def test_failed_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "failure"})

        def assert_failed_persisted(workspace_manager, run_id):
            """
            Verify cleanup is attempted after failure state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.FAILED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service._attempt_workspace_cleanup",
            side_effect=assert_failed_persisted,
        ) as cleanup_workspace:
            with self.assertRaises(AgentExecutionError):
                execute_agent_run_lifecycle(agent_run.id)

        cleanup_workspace.assert_called_once_with(self.workspace_manager, agent_run.id)

    def test_timed_out_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        def assert_timed_out_persisted(workspace_manager, run_id):
            """
            Verify cleanup is attempted after timeout state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.TIMED_OUT.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service._attempt_workspace_cleanup",
            side_effect=assert_timed_out_persisted,
        ) as cleanup_workspace:
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        cleanup_workspace.assert_called_once_with(self.workspace_manager, executed_run.id)

    def test_confirmed_cancelled_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "cancelled"})

        def assert_cancelled_persisted(workspace_manager, run_id):
            """
            Verify cleanup is attempted after cancellation state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.CANCELLED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service._attempt_workspace_cleanup",
            side_effect=assert_cancelled_persisted,
        ) as cleanup_workspace:
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.CANCELLED.value)
        cleanup_workspace.assert_called_once_with(self.workspace_manager, executed_run.id)

    def test_terminal_completion_uses_explicit_outcome_after_cancellation_request(self):
        agent_run = self.create_queued_run()
        claim_agent_run_for_execution(agent_run.id)
        request_agent_run_cancellation(agent_run.id)

        completed_run = complete_agent_run_execution(agent_run.id, AgentRunStatus.SUCCEEDED)

        self.assertEqual(completed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(completed_run.cancel_requested_at)
        self.assertIsNotNone(completed_run.finished_at)

    def test_cancellation_requested_during_successful_dummy_execution_reaches_succeeded(self):
        agent_run = self.create_queued_run()

        def request_cancellation_and_succeed(claimed_run):
            """
            Request cancellation while dummy work is running.
            """

            request_agent_run_cancellation(claimed_run.id)

            return AgentRunStatus.SUCCEEDED

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=request_cancellation_and_succeed,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(executed_run.cancel_requested_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_cancellation_requested_during_timed_out_dummy_execution_reaches_timed_out(self):
        agent_run = self.create_queued_run()

        def request_cancellation_and_timeout(claimed_run):
            """
            Request cancellation while dummy work is running.
            """

            request_agent_run_cancellation(claimed_run.id)

            return AgentRunStatus.TIMED_OUT

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=request_cancellation_and_timeout,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        self.assertIsNotNone(executed_run.cancel_requested_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_cancellation_requested_during_failed_dummy_execution_reaches_failed(self):
        agent_run = self.create_queued_run()

        def request_cancellation_and_fail(claimed_run):
            """
            Request cancellation while dummy work is running.
            """

            request_agent_run_cancellation(claimed_run.id)
            raise AgentExecutionError("Dummy failure after cancellation.")

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=request_cancellation_and_fail,
        ):
            with self.assertRaises(AgentExecutionError):
                execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertIsNotNone(agent_run.cancel_requested_at)
        self.assertIsNotNone(agent_run.finished_at)

    def test_local_subprocess_success_reaches_succeeded(self):
        marker_path = self.temp_root / "local-success.jsonl"
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-success",
            side_effect_path=marker_path,
        )
        agent_run = self.create_queued_run(prompt="Run local subprocess.")

        with self.settings(
            AGENT_EXECUTOR_BACKEND="local_subprocess",
            AGENT_CODEX_EXECUTABLE=str(executable_path),
            AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
            AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.1,
            AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        payload = json.loads(marker_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertEqual(payload["stdin"], "Run local subprocess.")
        self.assertEqual(
            payload["cwd"], str(self.workspace_manager.get_workspace_path(agent_run.id))
        )
        self.assertTrue(payload["has_agents"])
        self.assertTrue(payload["has_codex_config"])
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())

    def test_local_subprocess_non_zero_exit_reaches_failed(self):
        marker_path = self.temp_root / "local-failed.jsonl"
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-failed",
            exit_code=7,
            side_effect_path=marker_path,
        )
        agent_run = self.create_queued_run()

        with self.settings(
            AGENT_EXECUTOR_BACKEND="local_subprocess",
            AGENT_CODEX_EXECUTABLE=str(executable_path),
            AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
            AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.1,
            AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.FAILED.value)
        self.assertTrue(marker_path.exists())
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())

    def test_local_subprocess_timeout_reaches_timed_out(self):
        marker_path = self.temp_root / "local-timeout.jsonl"
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-timeout",
            sleep_seconds=5.0,
            side_effect_path=marker_path,
        )
        agent_run = self.create_queued_run()

        with self.settings(
            AGENT_EXECUTOR_BACKEND="local_subprocess",
            AGENT_CODEX_EXECUTABLE=str(executable_path),
            AGENT_EXECUTION_TIMEOUT_SECONDS=1.0,
            AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
            AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        self.assertTrue(marker_path.exists())
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())

    def test_local_subprocess_running_cancellation_reaches_cancelled(self):
        marker_path = self.temp_root / "local-cancelled.jsonl"
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-cancelled",
            sleep_seconds=5.0,
            side_effect_path=marker_path,
        )
        agent_run = self.create_queued_run()

        def request_cancellation_after_process_start(run_id):
            """
            Request cancellation through authoritative AgentRun state while running.
            """

            if marker_path.exists():
                request_agent_run_cancellation(run_id)

            return is_agent_run_cancellation_requested(run_id)

        with (
            self.settings(
                AGENT_EXECUTOR_BACKEND="local_subprocess",
                AGENT_CODEX_EXECUTABLE=str(executable_path),
                AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
                AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
                AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
            ),
            patch(
                "apps.agent_workspace.services.execution_service."
                "is_agent_run_cancellation_requested",
                side_effect=request_cancellation_after_process_start,
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.CANCELLED.value)
        self.assertIsNotNone(executed_run.cancel_requested_at)
        self.assertTrue(marker_path.exists())
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())

    def test_local_subprocess_pre_spawn_cancellation_does_not_start_process(self):
        marker_path = self.temp_root / "local-pre-spawn-cancelled.jsonl"
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-pre-spawn-cancelled",
            side_effect_path=marker_path,
        )
        agent_run = self.create_queued_run()

        def request_cancellation_before_spawn(run_id):
            """
            Request cancellation during the executor's pre-spawn cancellation check.
            """

            request_agent_run_cancellation(run_id)

            return True

        with (
            self.settings(
                AGENT_EXECUTOR_BACKEND="local_subprocess",
                AGENT_CODEX_EXECUTABLE=str(executable_path),
                AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
                AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
                AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
            ),
            patch(
                "apps.agent_workspace.services.execution_service."
                "is_agent_run_cancellation_requested",
                side_effect=request_cancellation_before_spawn,
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.CANCELLED.value)
        self.assertFalse(marker_path.exists())
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())

    def test_local_subprocess_spawn_failure_reaches_failed_and_cleans_workspace(self):
        missing_executable_path = self.fake_bin_path / "missing-codex"
        agent_run = self.create_queued_run()

        with self.settings(
            AGENT_EXECUTOR_BACKEND="local_subprocess",
            AGENT_CODEX_EXECUTABLE=str(missing_executable_path),
            AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
            AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
            AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
        ):
            with self.assertRaises(AgentProcessSpawnError):
                execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertIsNotNone(agent_run.finished_at)
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())

    def test_local_subprocess_duplicate_delivery_does_not_start_process_twice(self):
        marker_path = self.temp_root / "local-duplicate.jsonl"
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-duplicate",
            side_effect_path=marker_path,
        )
        agent_run = self.create_queued_run()

        with self.settings(
            AGENT_EXECUTOR_BACKEND="local_subprocess",
            AGENT_CODEX_EXECUTABLE=str(executable_path),
            AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
            AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
            AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
        ):
            first_result = execute_agent_run_lifecycle(agent_run.id)
            second_result = execute_agent_run_lifecycle(agent_run.id)

        marker_lines = marker_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(first_result.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNone(second_result)
        self.assertEqual(len(marker_lines), 1)

    def test_local_backend_persists_context_before_executor_starts(self):
        agent_run = self.create_queued_run()

        def assert_context_before_execute(executor, request, *, is_cancel_requested=None):
            """
            Verify context audit metadata is persisted before AgentExecutor starts.
            """

            persisted_run = AgentRun.objects.get(id=request.run_id)
            self.assertEqual(persisted_run.context_version, "execution-context-v1")
            self.assertIsNotNone(persisted_run.context_hash)
            self.assertTrue(request.workspace_path.exists())
            self.assertIsNotNone(is_cancel_requested)

            return AgentExecutionResult(
                outcome=AgentExecutionOutcome.SUCCEEDED,
                exit_code=0,
            )

        with (
            self.settings(
                AGENT_EXECUTOR_BACKEND="local_subprocess",
                AGENT_CODEX_EXECUTABLE="unused-codex",
            ),
            patch(
                "apps.agent_workspace.services.execution_service.LocalSubprocessExecutor.execute",
                autospec=True,
                side_effect=assert_context_before_execute,
            ) as execute,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        execute.assert_called_once()

    def test_local_subprocess_terminal_state_is_persisted_before_cleanup(self):
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-terminal-before-cleanup",
        )
        agent_run = self.create_queued_run()

        def assert_succeeded_before_cleanup(run_id):
            """
            Verify local subprocess terminal state is persisted before cleanup.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.SUCCEEDED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with (
            self.settings(
                AGENT_EXECUTOR_BACKEND="local_subprocess",
                AGENT_CODEX_EXECUTABLE=str(executable_path),
                AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
                AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
                AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
            ),
            patch.object(
                self.workspace_manager,
                "cleanup_workspace",
                side_effect=assert_succeeded_before_cleanup,
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)

    def test_local_subprocess_cleanup_failure_does_not_rewrite_succeeded(self):
        executable_path = self.create_agent_lifecycle_executable(
            executable_name="fake-local-cleanup-failure",
        )
        agent_run = self.create_queued_run()

        with (
            self.settings(
                AGENT_EXECUTOR_BACKEND="local_subprocess",
                AGENT_CODEX_EXECUTABLE=str(executable_path),
                AGENT_EXECUTION_TIMEOUT_SECONDS=2.0,
                AGENT_EXECUTION_TERMINATION_GRACE_SECONDS=0.05,
                AGENT_EXECUTION_CANCELLATION_POLL_SECONDS=0.02,
            ),
            patch.object(
                self.workspace_manager,
                "cleanup_workspace",
                side_effect=RuntimeError("Cleanup failed"),
            ),
            self.assertLogs(
                "apps.agent_workspace.services.execution_service",
                level="ERROR",
            ),
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertEqual(agent_run.status, AgentRunStatus.SUCCEEDED.value)

    @override_settings(AGENT_EXECUTOR_BACKEND="unexpected")
    def test_invalid_executor_backend_fails_clearly_after_claim(self):
        agent_run = self.create_queued_run()

        with self.assertRaises(AgentExecutionError):
            execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertFalse(self.workspace_manager.get_workspace_path(agent_run.id).exists())


class AgentRunPostCommitDispatchTests(TransactionTestCase):
    """
    Cover post-commit Celery publication behavior.
    """

    reset_sequences = True

    def setUp(self):
        """
        Create a reusable user for transaction dispatch tests.
        """

        self.user = get_user_model().objects.create_user(
            username="dispatch-user",
            password="test-password",
        )

    def create_agent_run(self) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        return AgentRun.objects.create(
            user=self.user,
            prompt="Generate an image.",
        )

    def test_queue_inside_transaction_does_not_publish_before_commit(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run.delay") as delay:
            with transaction.atomic():
                queue_agent_run_for_execution(agent_run.id)
                delay.assert_not_called()

        delay.assert_called_once_with(str(agent_run.id))

    def test_queue_after_successful_commit_publishes_only_run_id(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run.delay") as delay:
            with transaction.atomic():
                queue_agent_run_for_execution(agent_run.id)

        delay.assert_called_once_with(str(agent_run.id))
        self.assertEqual(delay.call_args.args, (str(agent_run.id),))
        self.assertEqual(delay.call_args.kwargs, {})

    def test_queue_rollback_does_not_publish_task(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run.delay") as delay:
            with self.assertRaises(RuntimeError):
                with transaction.atomic():
                    queue_agent_run_for_execution(agent_run.id)
                    raise RuntimeError("rollback")

        delay.assert_not_called()
        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.CREATED.value)


class AgentRunConcurrentClaimTests(TransactionTestCase):
    """
    Cover PostgreSQL row-lock semantics for concurrent claim attempts.
    """

    reset_sequences = True

    def setUp(self):
        """
        Create a queued AgentRun for concurrent claim tests.
        """

        self.user = get_user_model().objects.create_user(
            username="concurrent-claim-user",
            password="test-password",
        )
        agent_run = AgentRun.objects.create(
            user=self.user,
            prompt="Generate an image.",
        )
        self.agent_run = transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def test_two_concurrent_claim_attempts_allow_only_one_success(self):
        if connection.vendor != "postgresql":
            self.skipTest("Concurrent claim verification requires PostgreSQL.")

        barrier = Barrier(2)
        claim_results = []
        errors = []

        def attempt_claim():
            """
            Attempt to claim the same AgentRun from an independent DB connection.
            """

            close_old_connections()

            try:
                barrier.wait()
                claim_results.append(claim_agent_run_for_execution(self.agent_run.id))
            except Exception as error:
                errors.append(error)
            finally:
                close_old_connections()

        threads = [
            Thread(target=attempt_claim),
            Thread(target=attempt_claim),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(sum(result is not None for result in claim_results), 1)
        self.assertEqual(sum(result is None for result in claim_results), 1)

        self.agent_run.refresh_from_db()
        self.assertEqual(self.agent_run.status, AgentRunStatus.RUNNING.value)
