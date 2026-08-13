from datetime import UTC, datetime
from threading import Barrier, Thread
from unittest.mock import patch
from uuid import UUID, uuid4

from django.contrib.auth import get_user_model
from django.db import close_old_connections, connection, transaction
from django.test import TestCase, TransactionTestCase

from apps.agent_workspace.exceptions import AgentExecutionError, InvalidAgentRunTransition
from apps.agent_workspace.models import AgentArtifact, AgentRun
from apps.agent_workspace.services.agent_run_service import (
    claim_agent_run_for_execution,
    complete_agent_run_execution,
    queue_agent_run_for_execution,
    request_agent_run_cancellation,
    transition_agent_run,
)
from apps.agent_workspace.services.execution_service import execute_agent_run_lifecycle
from apps.agent_workspace.tasks import execute_agent_run
from common.constants import AgentRunStatus
from core.celery import app as celery_app


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


class AgentRunExecutionClaimTests(TestCase):
    """
    Cover atomic AgentRun execution claim behavior.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for execution claim tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="claim-user",
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

    def test_queued_agent_run_can_be_claimed_for_execution(self):
        agent_run = self.create_queued_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNotNone(claimed_run)

    def test_successful_claim_transitions_queued_to_running(self):
        agent_run = self.create_queued_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertEqual(claimed_run.status, AgentRunStatus.RUNNING.value)

    def test_successful_claim_sets_started_at(self):
        agent_run = self.create_queued_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNotNone(claimed_run.started_at)

    def test_non_queued_agent_run_is_not_claimed(self):
        agent_run = self.create_agent_run()

        claimed_run = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNone(claimed_run)

    def test_second_claim_attempt_does_not_create_second_claim(self):
        agent_run = self.create_queued_run()

        first_claim = claim_agent_run_for_execution(agent_run.id)
        second_claim = claim_agent_run_for_execution(agent_run.id)

        self.assertIsNotNone(first_claim)
        self.assertIsNone(second_claim)

    def test_second_claim_attempt_does_not_destroy_current_state(self):
        agent_run = self.create_queued_run()
        first_claim = claim_agent_run_for_execution(agent_run.id)

        claim_agent_run_for_execution(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.RUNNING.value)
        self.assertEqual(agent_run.started_at, first_claim.started_at)


class AgentRunTaskContractTests(TestCase):
    """
    Cover the Celery task boundary for AgentRun execution.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for task contract tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="task-user",
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

    def test_execute_agent_run_task_accepts_only_run_id_business_input(self):
        self.assertEqual(execute_agent_run.run.__code__.co_argcount, 1)
        self.assertEqual(execute_agent_run.run.__code__.co_varnames[:1], ("run_id",))

    def test_execute_agent_run_task_delegates_run_id_to_lifecycle_service(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run_lifecycle") as lifecycle:
            execute_agent_run.run(str(agent_run.id))

        lifecycle.assert_called_once_with(str(agent_run.id))

    def test_execute_agent_run_task_delegates_missing_run_to_service_boundary(self):
        missing_run_id = str(uuid4())

        with patch("apps.agent_workspace.tasks.execute_agent_run_lifecycle") as lifecycle:
            execute_agent_run.run(missing_run_id)

        lifecycle.assert_called_once_with(missing_run_id)

    def test_execute_agent_run_task_delegates_lifecycle_workflow(self):
        agent_run = self.create_agent_run()
        transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

        with patch("apps.agent_workspace.tasks.execute_agent_run_lifecycle") as lifecycle:
            execute_agent_run.run(str(agent_run.id))

        lifecycle.assert_called_once_with(str(agent_run.id))
        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.QUEUED.value)

    def test_execute_agent_run_lifecycle_raises_for_missing_run(self):
        missing_run_id = str(uuid4())

        with self.assertRaises(AgentRun.DoesNotExist):
            execute_agent_run_lifecycle(missing_run_id)

    def test_execute_agent_run_task_does_not_use_result_backend_for_business_state(self):
        self.assertTrue(execute_agent_run.ignore_result)
        self.assertIsNone(celery_app.conf.result_backend)


class AgentRunDummyExecutionTests(TestCase):
    """
    Cover dummy execution lifecycle outcomes.
    """

    @classmethod
    def setUpTestData(cls):
        """
        Create a reusable user for dummy execution tests.
        """

        cls.user = get_user_model().objects.create_user(
            username="dummy-execution-user",
            password="test-password",
        )

    def create_queued_run(self, **kwargs) -> AgentRun:
        """
        Create an AgentRun and transition it to QUEUED through the service.
        """

        defaults = {
            "user": self.user,
            "prompt": "Generate an image.",
        }
        defaults.update(kwargs)
        agent_run = AgentRun.objects.create(**defaults)

        return transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def test_successful_dummy_execution_reaches_succeeded(self):
        agent_run = self.create_queued_run()

        executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(executed_run.started_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_dummy_execution_failure_reaches_failed_before_raising(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "failure"})

        with self.assertRaises(AgentExecutionError):
            execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertIsNotNone(agent_run.started_at)
        self.assertIsNotNone(agent_run.finished_at)

    def test_dummy_execution_timeout_reaches_timed_out(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        self.assertIsNotNone(executed_run.started_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_duplicate_delivery_does_not_execute_dummy_work_twice(self):
        agent_run = self.create_queued_run()

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            return_value=AgentRunStatus.SUCCEEDED,
        ) as dummy_execution:
            first_result = execute_agent_run_lifecycle(agent_run.id)
            second_result = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(first_result.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNone(second_result)
        dummy_execution.assert_called_once()

    def test_duplicate_delivery_after_first_claim_is_execution_boundary_no_op(self):
        agent_run = self.create_queued_run()

        execute_agent_run_lifecycle(agent_run.id)
        second_result = execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertIsNone(second_result)
        self.assertEqual(agent_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(agent_run.finished_at)

    def test_agent_run_database_status_is_authoritative_execution_outcome(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.TIMED_OUT.value)

    def test_successful_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run()

        def assert_succeeded_persisted(run_id):
            """
            Verify cleanup is attempted after terminal state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.SUCCEEDED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service.cleanup_agent_run_workspace",
            side_effect=assert_succeeded_persisted,
        ) as cleanup_workspace:
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        cleanup_workspace.assert_called_once_with(executed_run.id)

    def test_failed_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "failure"})

        def assert_failed_persisted(run_id):
            """
            Verify cleanup is attempted after failure state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.FAILED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service.cleanup_agent_run_workspace",
            side_effect=assert_failed_persisted,
        ) as cleanup_workspace:
            with self.assertRaises(AgentExecutionError):
                execute_agent_run_lifecycle(agent_run.id)

        cleanup_workspace.assert_called_once_with(agent_run.id)

    def test_timed_out_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "timeout"})

        def assert_timed_out_persisted(run_id):
            """
            Verify cleanup is attempted after timeout state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.TIMED_OUT.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service.cleanup_agent_run_workspace",
            side_effect=assert_timed_out_persisted,
        ) as cleanup_workspace:
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        cleanup_workspace.assert_called_once_with(executed_run.id)

    def test_confirmed_cancelled_execution_attempts_cleanup_after_terminal_persist(self):
        agent_run = self.create_queued_run(metadata={"dummy_execution_outcome": "cancelled"})

        def assert_cancelled_persisted(run_id):
            """
            Verify cleanup is attempted after cancellation state is persisted.
            """

            persisted_run = AgentRun.objects.get(id=run_id)
            self.assertEqual(persisted_run.status, AgentRunStatus.CANCELLED.value)
            self.assertIsNotNone(persisted_run.finished_at)

        with patch(
            "apps.agent_workspace.services.execution_service.cleanup_agent_run_workspace",
            side_effect=assert_cancelled_persisted,
        ) as cleanup_workspace:
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.CANCELLED.value)
        cleanup_workspace.assert_called_once_with(executed_run.id)

    def test_terminal_completion_uses_explicit_outcome_after_cancellation_request(self):
        agent_run = self.create_queued_run()
        claim_agent_run_for_execution(agent_run.id)
        request_agent_run_cancellation(agent_run.id)

        completed_run = complete_agent_run_execution(agent_run.id, AgentRunStatus.SUCCEEDED)

        self.assertEqual(completed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(completed_run.cancel_requested_at)
        self.assertIsNotNone(completed_run.finished_at)

    def test_cancellation_requested_during_successful_dummy_execution_reaches_succeeded(self):
        agent_run = self.create_queued_run()

        def request_cancellation_and_succeed(claimed_run):
            """
            Request cancellation while dummy work is running.
            """

            request_agent_run_cancellation(claimed_run.id)

            return AgentRunStatus.SUCCEEDED

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=request_cancellation_and_succeed,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.SUCCEEDED.value)
        self.assertIsNotNone(executed_run.cancel_requested_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_cancellation_requested_during_timed_out_dummy_execution_reaches_timed_out(self):
        agent_run = self.create_queued_run()

        def request_cancellation_and_timeout(claimed_run):
            """
            Request cancellation while dummy work is running.
            """

            request_agent_run_cancellation(claimed_run.id)

            return AgentRunStatus.TIMED_OUT

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=request_cancellation_and_timeout,
        ):
            executed_run = execute_agent_run_lifecycle(agent_run.id)

        self.assertEqual(executed_run.status, AgentRunStatus.TIMED_OUT.value)
        self.assertIsNotNone(executed_run.cancel_requested_at)
        self.assertIsNotNone(executed_run.finished_at)

    def test_cancellation_requested_during_failed_dummy_execution_reaches_failed(self):
        agent_run = self.create_queued_run()

        def request_cancellation_and_fail(claimed_run):
            """
            Request cancellation while dummy work is running.
            """

            request_agent_run_cancellation(claimed_run.id)
            raise AgentExecutionError("Dummy failure after cancellation.")

        with patch(
            "apps.agent_workspace.services.execution_service._run_dummy_execution",
            side_effect=request_cancellation_and_fail,
        ):
            with self.assertRaises(AgentExecutionError):
                execute_agent_run_lifecycle(agent_run.id)

        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.FAILED.value)
        self.assertIsNotNone(agent_run.cancel_requested_at)
        self.assertIsNotNone(agent_run.finished_at)


class AgentRunPostCommitDispatchTests(TransactionTestCase):
    """
    Cover post-commit Celery publication behavior.
    """

    reset_sequences = True

    def setUp(self):
        """
        Create a reusable user for transaction dispatch tests.
        """

        self.user = get_user_model().objects.create_user(
            username="dispatch-user",
            password="test-password",
        )

    def create_agent_run(self) -> AgentRun:
        """
        Create an AgentRun owned by the reusable test user.
        """

        return AgentRun.objects.create(
            user=self.user,
            prompt="Generate an image.",
        )

    def test_queue_inside_transaction_does_not_publish_before_commit(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run.delay") as delay:
            with transaction.atomic():
                queue_agent_run_for_execution(agent_run.id)
                delay.assert_not_called()

        delay.assert_called_once_with(str(agent_run.id))

    def test_queue_after_successful_commit_publishes_only_run_id(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run.delay") as delay:
            with transaction.atomic():
                queue_agent_run_for_execution(agent_run.id)

        delay.assert_called_once_with(str(agent_run.id))
        self.assertEqual(delay.call_args.args, (str(agent_run.id),))
        self.assertEqual(delay.call_args.kwargs, {})

    def test_queue_rollback_does_not_publish_task(self):
        agent_run = self.create_agent_run()

        with patch("apps.agent_workspace.tasks.execute_agent_run.delay") as delay:
            with self.assertRaises(RuntimeError):
                with transaction.atomic():
                    queue_agent_run_for_execution(agent_run.id)
                    raise RuntimeError("rollback")

        delay.assert_not_called()
        agent_run.refresh_from_db()
        self.assertEqual(agent_run.status, AgentRunStatus.CREATED.value)


class AgentRunConcurrentClaimTests(TransactionTestCase):
    """
    Cover PostgreSQL row-lock semantics for concurrent claim attempts.
    """

    reset_sequences = True

    def setUp(self):
        """
        Create a queued AgentRun for concurrent claim tests.
        """

        self.user = get_user_model().objects.create_user(
            username="concurrent-claim-user",
            password="test-password",
        )
        agent_run = AgentRun.objects.create(
            user=self.user,
            prompt="Generate an image.",
        )
        self.agent_run = transition_agent_run(agent_run.id, AgentRunStatus.QUEUED)

    def test_two_concurrent_claim_attempts_allow_only_one_success(self):
        if connection.vendor != "postgresql":
            self.skipTest("Concurrent claim verification requires PostgreSQL.")

        barrier = Barrier(2)
        claim_results = []
        errors = []

        def attempt_claim():
            """
            Attempt to claim the same AgentRun from an independent DB connection.
            """

            close_old_connections()

            try:
                barrier.wait()
                claim_results.append(claim_agent_run_for_execution(self.agent_run.id))
            except Exception as error:
                errors.append(error)
            finally:
                close_old_connections()

        threads = [
            Thread(target=attempt_claim),
            Thread(target=attempt_claim),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(sum(result is not None for result in claim_results), 1)
        self.assertEqual(sum(result is None for result in claim_results), 1)

        self.agent_run.refresh_from_db()
        self.assertEqual(self.agent_run.status, AgentRunStatus.RUNNING.value)
