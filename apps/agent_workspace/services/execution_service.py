from uuid import UUID

from apps.agent_workspace.ai_agent.workspace_manager import cleanup_agent_run_workspace
from apps.agent_workspace.exceptions import AgentExecutionError
from apps.agent_workspace.models import AgentRun
from apps.agent_workspace.services.agent_run_service import (
    claim_agent_run_for_execution,
    complete_agent_run_execution,
)
from common.constants import AgentRunStatus


def execute_agent_run_lifecycle(run_id: UUID | str) -> AgentRun | None:
    """
    Orchestrate a claimed AgentRun through the dummy execution lifecycle.

    Duplicate task delivery is handled by the atomic claim operation. If the
    run is no longer QUEUED, no dummy work is executed.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        AgentExecutionError: If dummy execution reports an execution failure.
    """

    agent_run = claim_agent_run_for_execution(run_id)

    if agent_run is None:
        return None

    try:
        terminal_status = _run_dummy_execution(agent_run)
    except AgentExecutionError:
        _complete_execution(agent_run.id, AgentRunStatus.FAILED)
        raise
    except Exception:
        _complete_execution(agent_run.id, AgentRunStatus.FAILED)
        raise

    return _complete_execution(agent_run.id, terminal_status)


def _complete_execution(run_id: UUID | str, terminal_status: AgentRunStatus) -> AgentRun:
    """
    Persist the terminal AgentRun state and attempt workspace cleanup.

    Terminal state is written first because required execution data must be
    persisted before cleanup can remove workspace data.
    """

    completed_run = complete_agent_run_execution(run_id, terminal_status)
    cleanup_agent_run_workspace(completed_run.id)

    return completed_run


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
