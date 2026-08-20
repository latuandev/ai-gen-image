from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from apps.agent_workspace.ai_agent.executor import AgentExecutionRequest
from apps.agent_workspace.exceptions import InvalidCodexCLIConfiguration

DEFAULT_CODEX_EXECUTABLE = "codex"
CODEX_HOME_ENV_KEY = "CODEX_HOME"
CODEX_WORKSPACE_CONFIG_DIRECTORY = ".codex"
SECRET_ENVIRONMENT_KEYS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_URL",
        "DJANGO_SECRET_KEY",
        "REDIS_URL",
    }
)


@dataclass(frozen=True)
class CodexCLIInvocation:
    """
    Describe a deterministic non-interactive Codex CLI invocation.

    The invocation is data only. It contains no process handles, subprocess
    state, timeout state, cancellation state, or collected output streams.
    """

    argv: tuple[str, ...]
    cwd: Path
    stdin_text: str | None
    environment_overrides: Mapping[str, str]

    def __post_init__(self) -> None:
        """
        Normalize invocation fields while keeping the value object immutable.
        """

        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "cwd", Path(self.cwd))
        object.__setattr__(
            self,
            "environment_overrides",
            MappingProxyType(dict(self.environment_overrides)),
        )


class CodexCLIWrapper:
    """
    Build Codex-specific CLI invocation data for a prepared Agent execution.

    The wrapper does not spawn Codex, forward worker environment variables,
    manage workspace lifecycle, or interpret AgentRun database state.
    """

    def __init__(self, executable: str = DEFAULT_CODEX_EXECUTABLE):
        """
        Store validated Codex executable configuration.
        """

        self.executable = _validate_executable(executable)

    def build_invocation(self, request: AgentExecutionRequest) -> CodexCLIInvocation:
        """
        Build a non-interactive Codex exec invocation for an execution request.

        User prompt content is transported through stdin using the `-` prompt
        marker supported by `codex exec`; it is never interpolated into shell
        syntax or included as a prompt argv element.
        """

        workspace_path = request.workspace_path
        codex_home = workspace_path / CODEX_WORKSPACE_CONFIG_DIRECTORY
        argv = (
            self.executable,
            "exec",
            "--cd",
            str(workspace_path),
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--ask-for-approval",
            "never",
            "--color",
            "never",
            "--json",
            "-",
        )
        environment_overrides = {
            CODEX_HOME_ENV_KEY: str(codex_home),
        }

        _assert_environment_overrides_do_not_include_worker_secrets(environment_overrides)

        return CodexCLIInvocation(
            argv=argv,
            cwd=workspace_path,
            stdin_text=request.prompt,
            environment_overrides=environment_overrides,
        )


def _validate_executable(executable: str) -> str:
    """
    Validate wrapper-owned Codex executable configuration.
    """

    if not isinstance(executable, str):
        raise InvalidCodexCLIConfiguration("Executable must be a string")

    executable = executable.strip()

    if not executable:
        raise InvalidCodexCLIConfiguration("Executable cannot be blank")

    if "\x00" in executable:
        raise InvalidCodexCLIConfiguration("Executable cannot contain null bytes")

    return executable


def _assert_environment_overrides_do_not_include_worker_secrets(
    environment_overrides: Mapping[str, str],
) -> None:
    """
    Ensure Codex-specific overrides do not declare worker secret variables.
    """

    leaked_keys = SECRET_ENVIRONMENT_KEYS.intersection(environment_overrides)

    if leaked_keys:
        raise InvalidCodexCLIConfiguration(
            "Environment overrides cannot include worker secret variables"
        )
