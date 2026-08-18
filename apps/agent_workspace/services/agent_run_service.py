from uuid import UUID

from django.db import transaction
from django.utils import timezone

from apps.agent_workspace.exceptions import (
    AgentRunContextConflict,
    InvalidAgentRunContextState,
    InvalidAgentRunTransition,
)
from apps.agent_workspace.models import AgentRun
from common.constants import AgentRunStatus


def transition_agent_run(run_id: UUID | str, target_status: AgentRunStatus | str) -> AgentRun:
    """
    Transition an AgentRun to an explicitly allowed target status.

    The run is locked within a database transaction before validating and
    applying the status update and lifecycle timestamp changes.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        InvalidAgentRunTransition: If the requested transition is not allowed.
    """

    target_status_value = _status_value(target_status)

    with transaction.atomic():
        agent_run = _lock_agent_run(run_id)
        now = timezone.now()

        return _apply_transition(agent_run, target_status_value, now)


def queue_agent_run_for_execution(run_id: UUID | str) -> AgentRun:
    """
    Transition an AgentRun to QUEUED and publish execution after commit.

    The Celery task receives only the stable AgentRun identifier. If broker
    publication fails after commit, the run may remain QUEUED for later
    operational reconciliation.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        InvalidAgentRunTransition: If the run cannot be queued.
    """

    with transaction.atomic():
        agent_run = _lock_agent_run(run_id)
        queued_run = _apply_transition(agent_run, AgentRunStatus.QUEUED.value, timezone.now())
        queued_run_id = str(queued_run.id)

        transaction.on_commit(lambda: _publish_agent_run_execution(queued_run_id))

        return queued_run


def claim_agent_run_for_execution(run_id: UUID | str) -> AgentRun | None:
    """
    Atomically claim a queued AgentRun for execution.

    A queued run is locked and transitioned to RUNNING. Non-queued runs are
    left unchanged and return None so duplicate task delivery is idempotent.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
    """

    with transaction.atomic():
        agent_run = _lock_agent_run(run_id)

        if agent_run.status != AgentRunStatus.QUEUED.value:
            return None

        return _apply_transition(agent_run, AgentRunStatus.RUNNING.value, timezone.now())


def complete_agent_run_execution(
    run_id: UUID | str,
    terminal_status: AgentRunStatus | str,
) -> AgentRun:
    """
    Complete a running AgentRun with an explicit terminal status.

    The run is locked before validating and writing the final terminal status.
    Cancellation metadata is not interpreted as the terminal outcome; callers
    must provide CANCELLED only after execution cancellation is confirmed.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        InvalidAgentRunTransition: If the final transition is not allowed.
    """

    terminal_status_value = _status_value(terminal_status)

    with transaction.atomic():
        agent_run = _lock_agent_run(run_id)

        return _apply_transition(agent_run, terminal_status_value, timezone.now())


def record_agent_run_context(
    run_id: UUID | str,
    context_version: str,
    context_hash: str,
) -> AgentRun:
    """
    Persist execution context audit metadata for a running AgentRun.

    Existing matching metadata is treated as an idempotent retry. Existing
    different metadata is rejected because context metadata identifies the
    execution attempt that was actually bootstrapped.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        InvalidAgentRunContextState: If the run is not currently RUNNING.
        AgentRunContextConflict: If different context metadata already exists.
    """

    with transaction.atomic():
        agent_run = _lock_agent_run(run_id)

        if agent_run.status != AgentRunStatus.RUNNING.value:
            raise InvalidAgentRunContextState(agent_run.status, agent_run.id)

        has_existing_context = (
            agent_run.context_version is not None or agent_run.context_hash is not None
        )
        has_matching_context = (
            agent_run.context_version == context_version and agent_run.context_hash == context_hash
        )

        if has_existing_context:
            if has_matching_context:
                return agent_run

            raise AgentRunContextConflict(agent_run.id)

        agent_run.context_version = context_version
        agent_run.context_hash = context_hash
        agent_run.save(update_fields=["context_version", "context_hash"])

        return agent_run


def request_agent_run_cancellation(run_id: UUID | str) -> AgentRun:
    """
    Request cancellation for an AgentRun according to lifecycle semantics.

    Queued runs move directly to CANCELLED. Running runs keep RUNNING status
    and record the first cancellation request timestamp without overwriting it.

    Raises:
        AgentRun.DoesNotExist: If no run exists for the provided identifier.
        InvalidAgentRunTransition: If the run cannot accept cancellation.
    """

    with transaction.atomic():
        agent_run = _lock_agent_run(run_id)
        now = timezone.now()

        if agent_run.status == AgentRunStatus.QUEUED.value:
            return _apply_transition(agent_run, AgentRunStatus.CANCELLED.value, now)

        if agent_run.status == AgentRunStatus.RUNNING.value:
            if agent_run.cancel_requested_at is None:
                agent_run.cancel_requested_at = now
                agent_run.save(update_fields=["cancel_requested_at"])

            return agent_run

        raise InvalidAgentRunTransition(
            agent_run.status,
            AgentRunStatus.CANCELLED.value,
            agent_run.id,
        )


def _lock_agent_run(run_id: UUID | str) -> AgentRun:
    """
    Return an AgentRun locked for update in the current transaction.
    """

    return AgentRun.objects.select_for_update().get(id=run_id)


def _apply_transition(agent_run: AgentRun, target_status: str, transition_time) -> AgentRun:
    """
    Apply a validated status transition and matching lifecycle timestamp.
    """

    _validate_transition(agent_run.status, target_status, agent_run.id)

    update_fields = ["status"]
    agent_run.status = target_status

    timestamp_field = _timestamp_field_for_transition(target_status)
    if timestamp_field is not None:
        setattr(agent_run, timestamp_field, transition_time)
        update_fields.append(timestamp_field)

    agent_run.save(update_fields=update_fields)

    return agent_run


def _validate_transition(source_status: str, target_status: str, run_id: UUID) -> None:
    """
    Validate a status transition against the explicit transition map.
    """

    transition_map = {
        AgentRunStatus.CREATED.value: frozenset({AgentRunStatus.QUEUED.value}),
        AgentRunStatus.QUEUED.value: frozenset(
            {
                AgentRunStatus.RUNNING.value,
                AgentRunStatus.CANCELLED.value,
            }
        ),
        AgentRunStatus.RUNNING.value: frozenset(
            {
                AgentRunStatus.SUCCEEDED.value,
                AgentRunStatus.FAILED.value,
                AgentRunStatus.TIMED_OUT.value,
                AgentRunStatus.CANCELLED.value,
            }
        ),
        AgentRunStatus.SUCCEEDED.value: frozenset(),
        AgentRunStatus.FAILED.value: frozenset(),
        AgentRunStatus.TIMED_OUT.value: frozenset(),
        AgentRunStatus.CANCELLED.value: frozenset(),
    }

    allowed_targets = transition_map.get(source_status, frozenset())

    if target_status not in allowed_targets:
        raise InvalidAgentRunTransition(source_status, target_status, run_id)


def _timestamp_field_for_transition(target_status: str) -> str | None:
    """
    Return the lifecycle timestamp field updated by the target status.
    """

    timestamp_fields = {
        AgentRunStatus.QUEUED.value: "queued_at",
        AgentRunStatus.RUNNING.value: "started_at",
        AgentRunStatus.SUCCEEDED.value: "finished_at",
        AgentRunStatus.FAILED.value: "finished_at",
        AgentRunStatus.TIMED_OUT.value: "finished_at",
        AgentRunStatus.CANCELLED.value: "finished_at",
    }

    return timestamp_fields.get(target_status)


def _status_value(status: AgentRunStatus | str) -> str:
    """
    Return the string value for an AgentRun status enum or raw status.
    """

    if isinstance(status, AgentRunStatus):
        return status.value

    return status


def _publish_agent_run_execution(run_id: str) -> None:
    """
    Publish an AgentRun execution task with only the run identifier payload.
    """

    from apps.agent_workspace.tasks import execute_agent_run

    execute_agent_run.delay(run_id)
