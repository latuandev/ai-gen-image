import uuid

from django.conf import settings
from django.db import models

from common.constants import AgentRunStatus


class AgentRun(models.Model):
    """
    Represent a persisted end-user agent execution lifecycle.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="agent_runs",
    )

    status = models.CharField(
        max_length=20,
        choices=AgentRunStatus.choices(),
        default=AgentRunStatus.CREATED.value,
    )
    prompt = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)
    queued_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    cancel_requested_at = models.DateTimeField(null=True, blank=True)

    celery_task_id = models.CharField(max_length=255, null=True, blank=True)
    context_version = models.CharField(max_length=128, null=True, blank=True)
    context_hash = models.CharField(max_length=128, null=True, blank=True)
    model_identifier = models.CharField(max_length=255, null=True, blank=True)
    codex_version = models.CharField(max_length=128, null=True, blank=True)
    runtime_version = models.CharField(max_length=128, null=True, blank=True)
    error_code = models.CharField(max_length=100, null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)
    result_summary = models.TextField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)


class AgentArtifact(models.Model):
    """
    Represent a persisted artifact produced by an agent execution.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    agent_run = models.ForeignKey(
        AgentRun,
        on_delete=models.CASCADE,
        related_name="artifacts",
    )

    kind = models.CharField(max_length=100)
    filename = models.CharField(max_length=255)
    storage_key = models.CharField(max_length=1024)
    mime_type = models.CharField(max_length=255, null=True, blank=True)
    size = models.PositiveBigIntegerField()
    sha256 = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
