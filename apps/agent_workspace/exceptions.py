from uuid import UUID


class InvalidAgentRunTransition(Exception):
    """
    Signal that an AgentRun status transition is not allowed.
    """

    def __init__(self, source_status: str, target_status: str, run_id: UUID | None = None):
        """
        Store transition details and build a deterministic exception message.
        """

        self.source_status = source_status
        self.target_status = target_status
        self.run_id = run_id

        message = f"Invalid AgentRun transition: {source_status} -> {target_status}."
        if run_id is not None:
            message = (
                f"Invalid AgentRun transition for {run_id}: {source_status} -> {target_status}."
            )

        super().__init__(message)


class AgentExecutionError(Exception):
    """
    Signal that an AgentRun execution failed inside the orchestration boundary.
    """


class InvalidContextManifest(Exception):
    """
    Signal that the end-user context manifest is invalid or unsafe.
    """

    def __init__(self, reason: str):
        """
        Store the validation failure reason in a deterministic message.
        """

        self.reason = reason

        super().__init__(f"Invalid context manifest: {reason}.")


class AgentWorkspaceError(Exception):
    """
    Signal that an AgentRun workspace operation failed validation.
    """

    def __init__(self, reason: str):
        """
        Store the workspace failure reason in a deterministic message.
        """

        self.reason = reason

        super().__init__(f"Agent workspace error: {reason}.")


class WorkspaceCleanupError(AgentWorkspaceError):
    """
    Signal that an AgentRun workspace cleanup operation failed.
    """


class WorkspaceIdentityError(AgentWorkspaceError):
    """
    Signal that a workspace canonical path no longer matches its pinned identity.
    """


class InvalidAgentRunContextState(Exception):
    """
    Signal that context metadata cannot be recorded for the current run state.
    """

    def __init__(self, status: str, run_id: UUID | None = None):
        """
        Store the invalid status and build a deterministic exception message.
        """

        self.status = status
        self.run_id = run_id

        message = f"AgentRun context metadata cannot be recorded while {status}."
        if run_id is not None:
            message = f"AgentRun context metadata cannot be recorded for {run_id} while {status}."

        super().__init__(message)


class AgentRunContextConflict(Exception):
    """
    Signal that AgentRun context audit metadata conflicts with existing values.
    """

    def __init__(self, run_id: UUID | None = None):
        """
        Store the run identifier and build a deterministic exception message.
        """

        self.run_id = run_id

        message = "AgentRun context metadata already exists with a different fingerprint."
        if run_id is not None:
            message = (
                f"AgentRun context metadata for {run_id} already exists with a "
                "different fingerprint."
            )

        super().__init__(message)
