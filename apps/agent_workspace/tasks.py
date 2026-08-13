from celery import shared_task

from apps.agent_workspace.services.execution_service import execute_agent_run_lifecycle


@shared_task(name="agent_workspace.execute_agent_run", ignore_result=True)
def execute_agent_run(run_id: str) -> None:
    """
    Execute an AgentRun from its stable database identifier.

    The task does not accept prompts, credentials, workspaces, files, or
    serialized AgentRun objects through the broker.
    """

    execute_agent_run_lifecycle(run_id)
