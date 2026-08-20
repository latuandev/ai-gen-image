import logging
from uuid import UUID

from django.conf import settings

from apps.agent_workspace.ai_agent.cli_wrapper import CodexCLIWrapper
from apps.agent_workspace.ai_agent.executor import (
    AgentExecutionOutcome,
    AgentExecutionRequest,
    AgentExecutionResult,
)
from apps.agent_workspace.ai_agent.local_subprocess_executor import LocalSubprocessExecutor
from apps.agent_workspace.ai_agent.workspace_manager import WorkspaceManager
from apps.agent_workspace.exceptions import AgentExecutionError
from apps.agent_workspace.models import AgentRun
from apps.agent_workspace.services.agent_run_service import (
    claim_agent_run_for_execution,
    complete_agent_run_execution,
    is_agent_run_cancellation_requested,
    record_agent_run_context,
)
from common.constants import AgentRunStatus

logger = logging.getLogger(__name__)

AGENT_EXECUTOR_BACKEND_DUMMY = "dummy"
AGENT_EXECUTOR_BACKEND_LOCAL_SUBPROCESS = "local_subprocess"


def execute_agent_run_lifecycle(run_id: UUID | str) -> AgentRun | None:
    """
    Orchestrate a claimed AgentRun through workspace and execution lifecycle.

    Duplicate task delivery is handled by the atomic claim operation. If the
    run is no longer QUEUED, no workspace or execution work is created.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        AgentExecutionError: If execution reports an execution failure.
    """

    agent_run = claim_agent_run_for_execution(run_id)

    if agent_run is None:
        return None

    workspace_manager = WorkspaceManager()

    try:
        prepared_workspace = workspace_manager.prepare_workspace(agent_run.id)
        record_agent_run_context(
            agent_run.id,
            prepared_workspace.context_version,
            prepared_workspace.context_hash,
        )
        terminal_status = _run_agent_execution(agent_run, prepared_workspace.workspace_path)
    except AgentExecutionError:
        _complete_execution(agent_run.id, AgentRunStatus.FAILED, workspace_manager)
        raise
    except Exception:
        _complete_execution(agent_run.id, AgentRunStatus.FAILED, workspace_manager)
        raise

    return _complete_execution(agent_run.id, terminal_status, workspace_manager)


def _run_agent_execution(agent_run: AgentRun, workspace_path) -> AgentRunStatus:
    """
    Run the configured executor backend for a prepared AgentRun workspace.

    The default dummy backend intentionally preserves the Phase 5 deterministic
    behavior. The local subprocess backend is explicit opt-in trusted execution.
    """

    backend = settings.AGENT_EXECUTOR_BACKEND

    if backend == AGENT_EXECUTOR_BACKEND_DUMMY:
        return _run_dummy_execution(agent_run)

    if backend != AGENT_EXECUTOR_BACKEND_LOCAL_SUBPROCESS:
        raise AgentExecutionError("Unsupported Agent executor backend.")

    request = AgentExecutionRequest(
        run_id=agent_run.id,
        workspace_path=workspace_path,
        prompt=agent_run.prompt,
    )
    executor = LocalSubprocessExecutor(
        cli_wrapper=CodexCLIWrapper(executable=settings.AGENT_CODEX_EXECUTABLE),
    )
    result = executor.execute(
        request,
        is_cancel_requested=lambda: is_agent_run_cancellation_requested(agent_run.id),
    )

    return _agent_run_status_from_execution_result(result)


def _complete_execution(
    run_id: UUID | str,
    terminal_status: AgentRunStatus,
    workspace_manager: WorkspaceManager,
) -> AgentRun:
    """
    Persist the terminal AgentRun state and attempt workspace cleanup.

    Terminal state is written first because required execution data must be
    persisted before cleanup can remove workspace data.
    """

    completed_run = complete_agent_run_execution(run_id, terminal_status)
    _attempt_workspace_cleanup(workspace_manager, completed_run.id)

    return completed_run


def _attempt_workspace_cleanup(workspace_manager: WorkspaceManager, run_id: UUID | str) -> None:
    """
    Attempt workspace cleanup and log cleanup failures as operational errors.
    """

    try:
        workspace_manager.cleanup_workspace(run_id)
    except Exception:
        logger.exception("Agent workspace cleanup failed for AgentRun %s.", run_id)


def _run_dummy_execution(agent_run: AgentRun) -> AgentRunStatus:
    """
    Run deterministic dummy work and return the resulting terminal status.

    The temporary metadata key `dummy_execution_outcome` may be set to
    `success`, `failure`, `timeout`, or `cancelled` to exercise lifecycle paths
    before a real executor exists.
    """

    outcome = agent_run.metadata.get("dummy_execution_outcome", "success")

    if outcome == "success":
        return AgentRunStatus.SUCCEEDED

    if outcome == "timeout":
        return AgentRunStatus.TIMED_OUT

    if outcome == "cancelled":
        return AgentRunStatus.CANCELLED

    raise AgentExecutionError("Dummy AgentRun execution failed.")


def _agent_run_status_from_execution_result(
    result: AgentExecutionResult,
) -> AgentRunStatus:
    """
    Map executor-level outcomes to explicit AgentRun terminal states.
    """

    outcome_map = {
        AgentExecutionOutcome.SUCCEEDED: AgentRunStatus.SUCCEEDED,
        AgentExecutionOutcome.FAILED: AgentRunStatus.FAILED,
        AgentExecutionOutcome.TIMED_OUT: AgentRunStatus.TIMED_OUT,
        AgentExecutionOutcome.CANCELLED: AgentRunStatus.CANCELLED,
    }

    try:
        return outcome_map[result.outcome]
    except KeyError as exc:
        raise AgentExecutionError("Unsupported Agent execution outcome.") from exc
