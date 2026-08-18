import logging
from uuid import UUID

from apps.agent_workspace.ai_agent.workspace_manager import WorkspaceManager
from apps.agent_workspace.exceptions import AgentExecutionError
from apps.agent_workspace.models import AgentRun
from apps.agent_workspace.services.agent_run_service import (
    claim_agent_run_for_execution,
    complete_agent_run_execution,
    record_agent_run_context,
)
from common.constants import AgentRunStatus

logger = logging.getLogger(__name__)


def execute_agent_run_lifecycle(run_id: UUID | str) -> AgentRun | None:
    """
    Orchestrate a claimed AgentRun through workspace and dummy execution lifecycle.

    Duplicate task delivery is handled by the atomic claim operation. If the
    run is no longer QUEUED, no workspace or dummy work is created.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        AgentExecutionError: If dummy execution reports an execution failure.
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
        terminal_status = _run_dummy_execution(agent_run)
    except AgentExecutionError:
        _complete_execution(agent_run.id, AgentRunStatus.FAILED, workspace_manager)
        raise
    except Exception:
        _complete_execution(agent_run.id, AgentRunStatus.FAILED, workspace_manager)
        raise

    return _complete_execution(agent_run.id, terminal_status, workspace_manager)


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
