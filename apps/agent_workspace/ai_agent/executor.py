from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import UUID

AgentCancellationCheck = Callable[[], bool]


class AgentExecutionOutcome(str, Enum):
    """
    Represent executor-level completion semantics for one AgentRun execution.

    These outcomes are intentionally separate from persisted AgentRun statuses.
    The orchestration layer owns any mapping from execution outcome to database
    state transitions.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class AgentExecutionRequest:
    """
    Describe one AgentRun execution inside an already prepared workspace.

    The workspace path must refer to a workspace prepared by WorkspaceManager
    before this request is passed to an executor. The executor must not create,
    materialize, hash, persist, or clean up workspace context.
    """

    run_id: UUID | str
    workspace_path: Path | str
    prompt: str

    def __post_init__(self) -> None:
        """
        Normalize stable execution identifiers and filesystem paths.
        """

        object.__setattr__(self, "run_id", _normalize_run_id(self.run_id))
        object.__setattr__(self, "workspace_path", Path(self.workspace_path))


@dataclass(frozen=True)
class AgentExecutionResult:
    """
    Describe an executor-level outcome without mutating AgentRun persistence.

    A successful result always has exit code 0. A failed result may include a
    non-zero exit code when one exists. Timeout and cancellation outcomes do
    not expose an exit code because callers should decide from the normalized
    outcome rather than inspecting process-specific failure details.
    """

    outcome: AgentExecutionOutcome
    exit_code: int | None = None

    def __post_init__(self) -> None:
        """
        Enforce result invariants that keep outcome interpretation explicit.
        """

        if self.outcome == AgentExecutionOutcome.SUCCEEDED and self.exit_code != 0:
            raise ValueError("Successful Agent execution must have exit code 0")

        if self.outcome == AgentExecutionOutcome.FAILED and self.exit_code == 0:
            raise ValueError("Failed Agent execution cannot have exit code 0")

        if (
            self.outcome
            in {
                AgentExecutionOutcome.TIMED_OUT,
                AgentExecutionOutcome.CANCELLED,
            }
            and self.exit_code is not None
        ):
            raise ValueError("Timed out or cancelled Agent execution must not expose an exit code")


class AgentExecutor(ABC):
    """
    Execute one AgentRun inside an already prepared per-run workspace.

    Implementations must not depend on Celery, mutate AgentRun persistence,
    manage workspace lifecycle, or perform application state transitions.
    """

    @abstractmethod
    def execute(
        self,
        request: AgentExecutionRequest,
        *,
        is_cancel_requested: AgentCancellationCheck | None = None,
    ) -> AgentExecutionResult:
        """
        Execute a prepared AgentRun request and return a normalized outcome.
        """


def _normalize_run_id(run_id: UUID | str) -> UUID:
    """
    Normalize an AgentRun identifier into a canonical UUID value.
    """

    return run_id if isinstance(run_id, UUID) else UUID(str(run_id))
