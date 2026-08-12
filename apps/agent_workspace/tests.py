from datetime import UTC, datetime
from unittest.mock import patch
from uuid import UUID

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.agent_workspace.exceptions import InvalidAgentRunTransition
from apps.agent_workspace.models import AgentArtifact, AgentRun
from apps.agent_workspace.services.agent_run_service import (
    request_agent_run_cancellation,
    transition_agent_run,
)
from common.constants import AgentRunStatus


class AgentRunModelTests(TestCase):
    """
    Cover AgentRun and AgentArtifact persistence behavior.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for model persistence tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="model-user",
            password="test-password",
        )

    def create_agent_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)

        return AgentRun.objects.create(**defaults)

    def test_agent_run_defaults_to_created_status(self):
        agent_run = self.create_agent_run()

        self.assertEqual(agent_run.status, AgentRunStatus.CREATED.value)

    def test_agent_run_generates_uuid(self):
        agent_run = self.create_agent_run()

        self.assertIsInstance(agent_run.id, UUID)

    def test_agent_run_links_to_user(self):
        agent_run = self.create_agent_run()

        self.assertEqual(agent_run.user, self.user)
        self.assertEqual(list(self.user.agent_runs.all()), [agent_run])

    def test_agent_artifact_links_to_agent_run(self):
        agent_run = self.create_agent_run()

        artifact = AgentArtifact.objects.create(
            agent_run=agent_run,
            kind="image",
            filename="result.png",
            storage_key="agent-runs/result.png",
            mime_type="image/png",
            size=1024,
            sha256="a" * 64,
        )

        self.assertEqual(artifact.agent_run, agent_run)
        self.assertEqual(list(agent_run.artifacts.all()), [artifact])

    def test_agent_run_metadata_default_is_not_shared_between_instances(self):
        first_run = self.create_agent_run()
        second_run = self.create_agent_run(prompt="Generate another image.")

        first_run.metadata["key"] = "value"

        self.assertEqual(second_run.metadata, {})
        self.assertIsNot(first_run.metadata, second_run.metadata)


class AgentRunLifecycleServiceTests(TestCase):
    """
    Cover AgentRun lifecycle transitions and cancellation semantics.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for lifecycle service tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="lifecycle-user",
            password="test-password",
        )

    def create_agent_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)

        return AgentRun.objects.create(**defaults)

    def create_queued_run(self) -> AgentRun:
        """
        Create an AgentRun and transition it to QUEUED through the service.
        """

        agent_run = self.create_agent_run()

        return transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def create_running_run(self) -> AgentRun:
        """
        Create an AgentRun and transition it to RUNNING through the service.
        """

        agent_run = self.create_queued_run()

        return transition_agent_run(agent_run.id, AgentRunStatus.RUNNING)

    def create_terminal_run(self, target_status: AgentRunStatus) -> AgentRun:
        """
        Create a RUNNING AgentRun and transition it to a terminal status.
        """

        agent_run = self.create_running_run()

        return transition_agent_run(agent_run.id, target_status)

    def assert_invalid_transition(
        self, source_status: AgentRunStatus, target_status: AgentRunStatus
    ):
        """
        Assert a transition from a prepared source state is rejected.
        """

        agent_run = self.create_terminal_run(source_status)

        with self.assertRaises(InvalidAgentRunTransition):
            transition_agent_run(agent_run.id, target_status)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, source_status.value)

    def test_created_to_queued_transition(self):
        agent_run = self.create_agent_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.QUEUED.value)

    def test_queued_to_running_transition(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.RUNNING)

        self.assertEqual(transitioned_run.status, AgentRunStatus.RUNNING.value)

    def test_queued_to_cancelled_transition(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.CANCELLED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.CANCELLED.value)

    def test_running_to_succeeded_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.SUCCEEDED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.SUCCEEDED.value)

    def test_running_to_failed_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.FAILED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.FAILED.value)

    def test_running_to_timed_out_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.TIMED_OUT)

        self.assertEqual(transitioned_run.status, AgentRunStatus.TIMED_OUT.value)

    def test_running_to_cancelled_transition(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.CANCELLED)

        self.assertEqual(transitioned_run.status, AgentRunStatus.CANCELLED.value)

    def test_succeeded_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.SUCCEEDED, AgentRunStatus.RUNNING)

    def test_failed_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.FAILED, AgentRunStatus.RUNNING)

    def test_timed_out_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.TIMED_OUT, AgentRunStatus.RUNNING)

    def test_cancelled_to_running_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.CANCELLED, AgentRunStatus.RUNNING)

    def test_succeeded_to_failed_transition_is_rejected(self):
        self.assert_invalid_transition(AgentRunStatus.SUCCEEDED, AgentRunStatus.FAILED)

    def test_invalid_transition_raises_deterministic_exception(self):
        agent_run = self.create_terminal_run(AgentRunStatus.SUCCEEDED)

        with self.assertRaises(InvalidAgentRunTransition) as context:
            transition_agent_run(agent_run.id, AgentRunStatus.FAILED)

        exception = context.exception
        self.assertEqual(exception.source_status, AgentRunStatus.SUCCEEDED.value)
        self.assertEqual(exception.target_status, AgentRunStatus.FAILED.value)
        self.assertEqual(exception.run_id, agent_run.id)
        self.assertEqual(
            str(exception),
            (f"Invalid AgentRun transition for {agent_run.id}: SUCCEEDED -> FAILED."),
        )

    def test_created_to_queued_sets_queued_at(self):
        agent_run = self.create_agent_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

        self.assertIsNotNone(transitioned_run.queued_at)

    def test_queued_to_running_sets_started_at(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.RUNNING)

        self.assertIsNotNone(transitioned_run.started_at)

    def test_running_to_terminal_sets_finished_at(self):
        agent_run = self.create_running_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.SUCCEEDED)

        self.assertIsNotNone(transitioned_run.finished_at)

    def test_queued_to_cancelled_sets_finished_at(self):
        agent_run = self.create_queued_run()

        transitioned_run = transition_agent_run(agent_run.id, AgentRunStatus.CANCELLED)

        self.assertIsNotNone(transitioned_run.finished_at)

    def test_request_cancellation_when_queued_transitions_to_cancelled(self):
        agent_run = self.create_queued_run()

        cancelled_run = request_agent_run_cancellation(agent_run.id)

        self.assertEqual(cancelled_run.status, AgentRunStatus.CANCELLED.value)
        self.assertIsNotNone(cancelled_run.finished_at)

    def test_request_cancellation_when_running_keeps_running_and_sets_cancel_requested_at(self):
        agent_run = self.create_running_run()

        cancellation_requested_run = request_agent_run_cancellation(agent_run.id)

        self.assertEqual(cancellation_requested_run.status, AgentRunStatus.RUNNING.value)
        self.assertIsNotNone(cancellation_requested_run.cancel_requested_at)
        self.assertIsNone(cancellation_requested_run.finished_at)

    def test_repeated_running_cancellation_does_not_overwrite_cancel_requested_at(self):
        agent_run = self.create_running_run()
        first_request_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        second_request_time = datetime(2026, 1, 1, 12, 5, tzinfo=UTC)

        with patch(
            "apps.agent_workspace.services.agent_run_service.timezone.now",
            return_value=first_request_time,
        ):
            request_agent_run_cancellation(agent_run.id)

        with patch(
            "apps.agent_workspace.services.agent_run_service.timezone.now",
            return_value=second_request_time,
        ):
            cancellation_requested_run = request_agent_run_cancellation(agent_run.id)

        self.assertEqual(cancellation_requested_run.status, AgentRunStatus.RUNNING.value)
        self.assertEqual(cancellation_requested_run.cancel_requested_at, first_request_time)
