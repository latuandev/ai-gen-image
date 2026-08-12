# ADR-003: AgentRun Database State Is the Source of Truth

* **Status:** Accepted
* **Date:** 2026-08-11
* **Decision Owners:** Project Maintainers
* **Scope:** Agent task lifecycle, Celery orchestration, retries, cancellation, and observability

## Context

End-user AI executions are asynchronous and may involve:

* Django API requests.
* PostgreSQL persistence.
* Celery tasks.
* An isolated workspace.
* A Codex process.
* Generated artifacts.
* Retries.
* Timeouts.
* Cancellation.

Celery already maintains task execution metadata, but Celery task state is infrastructure state rather than application/domain state.

Relying on Celery alone would make it difficult to provide:

* Stable API semantics.
* Ownership validation.
* Audit history.
* Retries.
* Artifact association.
* Lifecycle invariants.
* Implementation independence from Celery.

## Decision

The `AgentRun` database entity SHALL be the authoritative source of truth for the end-user execution lifecycle.

Celery is an execution transport and orchestration mechanism.

It is not the authoritative business state.

Conceptually:

```text
Database AgentRun
      │
      └── Authoritative lifecycle state

Celery
      │
      └── Schedules and executes work
```

## AgentRun Identity

Each execution MUST have a server-generated immutable identifier.

UUID is recommended.

The identifier MUST NOT be derived from user-controlled filenames, prompts, usernames, or paths.

Example:

```text
run_id = UUID
workspace = /workspaces/<run_id>/
```

## Core States

The initial state machine SHALL support:

```text
CREATED
QUEUED
RUNNING
SUCCEEDED
FAILED
TIMED_OUT
CANCELLED
```

## Allowed State Transitions

Normal execution:

```text
CREATED
   │
   ▼
QUEUED
   │
   ▼
RUNNING
   │
   ├────────────► SUCCEEDED
   │
   ├────────────► FAILED
   │
   ├────────────► TIMED_OUT
   │
   └────────────► CANCELLED
```

Cancellation before execution:

```text
QUEUED ─────────► CANCELLED
```

The following transitions are invalid:

```text
SUCCEEDED → RUNNING
FAILED → RUNNING
TIMED_OUT → RUNNING
CANCELLED → RUNNING
SUCCEEDED → FAILED
```

A retry that represents a new user-visible execution SHOULD normally create a new execution attempt or explicitly modeled attempt record rather than silently mutating a terminal run back to `RUNNING`.

## Recommended Fields

`AgentRun` SHOULD contain enough information to reconstruct the business lifecycle.

Conceptual fields:

```text
id
user_id
status

prompt

created_at
queued_at
started_at
finished_at
cancel_requested_at

celery_task_id

context_version
context_hash

codex_version
runtime_version
model_identifier

error_code
error_message

result_summary
metadata
```

Exact schema may evolve without changing this ADR.

## Artifact Ownership

Generated files SHALL be associated with the `AgentRun`.

A separate `AgentArtifact` model is recommended.

Conceptually:

```text
AgentArtifact
─────────────
id
agent_run_id
kind
filename
storage_key
mime_type
size
sha256
created_at
```

Celery messages MUST NOT contain generated binary artifacts.

Artifacts must be persisted to the appropriate storage system and referenced by stable identifiers.

## Celery Message Contract

A Celery execution task SHOULD receive only the minimum stable identifier required to load authoritative state.

Preferred:

```text
execute_agent_run(run_id)
```

Avoid sending an entire serialized domain object or credential bundle through the broker.

The worker loads the latest `AgentRun` state from the database.

## Transaction Boundary

The system MUST NOT enqueue a Celery task before the corresponding `AgentRun` is safely committed.

Preferred conceptual sequence:

```text
Database transaction
    │
    ├── Create AgentRun
    └── Commit
             │
             ▼
        Enqueue run_id
```

Where appropriate, queue dispatch SHOULD occur through an `on_commit` mechanism.

## Idempotency

Celery tasks may be retried or delivered more than once.

The execution service MUST therefore tolerate duplicate invocation.

Before starting an execution, the worker MUST atomically validate the current run state.

Conceptually:

```text
Lock AgentRun
      │
      ▼
Is state QUEUED?
   │
   ├── Yes → Transition RUNNING
   │
   └── No  → Do not start duplicate process
```

Only one active execution for a specific execution attempt may exist at a time.

## Failure Semantics

Execution failures SHALL be normalized into application-level outcomes.

Examples:

```text
Codex returned non-zero exit
    → FAILED

Workspace bootstrap failed
    → FAILED

Artifact validation failed
    → FAILED

Execution exceeded permitted runtime
    → TIMED_OUT

User requested cancellation
    → CANCELLED
```

Internal exceptions SHOULD NOT leak raw stack traces or secrets through the public API.

## Cleanup Semantics

Workspace cleanup is independent from the terminal business state.

For every terminal outcome:

```text
SUCCEEDED
FAILED
TIMED_OUT
CANCELLED
```

the system must eventually attempt workspace cleanup.

Required data must be persisted before deletion.

## Cancellation

Cancellation MUST be modeled as a business action.

A cancellation request MAY occur while:

```text
QUEUED
RUNNING
```

When a run is `RUNNING`, cancellation requires cooperation from the executor to terminate the entire execution process tree or sandbox.

The system MUST distinguish:

```text
Cancellation requested
```

from:

```text
Execution confirmed terminated
```

if the infrastructure requires asynchronous termination.

A cancellation request does not require a dedicated `CANCEL_REQUESTED` execution state. While a running execution is being terminated, the AgentRun may remain `RUNNING` with cancel_requested_at populated. The terminal state becomes `CANCELLED` only after the execution has actually stopped or has been safely invalidated. Both user-initiated and system-initiated intentional cancellation may result in `CANCELLED`; their origin should be represented as metadata when required.

## Timeouts

The architecture SHOULD use multiple timeout layers:

```text
Application execution timeout
Codex process timeout
Celery soft time limit
Celery hard time limit
Runtime/container limit
```

These layers are defense in depth.

A timeout caused by infrastructure should still be reflected in the authoritative `AgentRun` state where operationally possible.

## Public API Semantics

Creating an agent execution is asynchronous.

The API SHOULD create a run and return an identifier rather than holding the HTTP request until Codex finishes.

Conceptual response:

```text
HTTP 202 Accepted

{
  "id": "<uuid>",
  "status": "QUEUED"
}
```

Clients query the run resource for authoritative state.

## Ownership

Every user-facing `AgentRun` lookup MUST enforce ownership or the project's authorization policy.

Knowledge of a UUID alone MUST NOT grant access to another user's run.

The same applies to associated artifacts.

## Auditability

For reproducibility and incident investigation, an execution SHOULD record:

```text
run_id
user_id
context_version
context_hash
Codex version
Runtime version/Image digest
Model identifier
Input hashes
Start/end timestamps
Terminal status
Exit metadata
Artifact hashes
```

Secrets must not be recorded as audit metadata.

## Security Invariants

### INV-RUN-001

Database state is authoritative; Celery state is not.

### INV-RUN-002

Only explicitly allowed state transitions may occur.

### INV-RUN-003

Duplicate Celery delivery MUST NOT launch duplicate concurrent executions for the same execution attempt.

### INV-RUN-004

A terminal execution MUST NOT silently become active again.

### INV-RUN-005

Every run and artifact lookup MUST enforce authorization.

### INV-RUN-006

Workspace cleanup MUST be attempted for every terminal outcome.

### INV-RUN-007

Celery messages MUST NOT contain unnecessary secrets or large execution payloads.

### INV-RUN-008

Run identifiers and workspace names MUST be generated by the server.

## Rejected Alternatives

### Use only Celery task status

Rejected because Celery state is infrastructure-specific and insufficient for domain authorization, audit, artifacts, and stable API semantics.

### Store the entire execution state inside Celery arguments

Rejected because broker messages are not the correct authoritative data store and may unnecessarily duplicate sensitive information.

### Restart failed runs by changing `FAILED` back to `RUNNING`

Rejected because it destroys execution history and complicates auditability.

### Allow concurrent retries for the same run

Rejected because they may write to the same workspace, duplicate model costs, or produce inconsistent artifacts.

## Required Tests

At minimum, tests SHALL verify:

```text
Valid state transitions succeed
Invalid state transitions fail
Duplicate task delivery does not create duplicate execution
Terminal runs do not restart
Queued cancellation works
Running cancellation reaches terminal state
Timeout becomes TIMED_OUT
Failure becomes FAILED
Success becomes SUCCEEDED
Artifacts belong to the correct user/run
Workspace cleanup occurs for every terminal outcome
```

## Implementation Gate

Celery integration MUST first prove the complete `AgentRun` lifecycle with a dummy executor before Codex CLI is introduced.

The expected validation sequence is:

```text
API/service creates AgentRun
        ↓
Celery receives run_id
        ↓
Dummy executor runs
        ↓
AgentRun becomes terminal
        ↓
Cleanup occurs
```

Only after this lifecycle is reliable should real Codex execution be connected.
