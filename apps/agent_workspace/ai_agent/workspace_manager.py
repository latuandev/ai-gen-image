from uuid import UUID


def cleanup_agent_run_workspace(run_id: UUID | str) -> None:
    """
    Attempt cleanup for a terminal AgentRun workspace.

    Phase 4 dummy execution does not create a real workspace yet, so the
    cleanup operation is currently an intentional no-op at the WorkspaceManager
    boundary. Real workspace deletion belongs here when workspace lifecycle is
    introduced.
    """
