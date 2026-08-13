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
