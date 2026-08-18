# ADR-002: End-user Agent Execution Must Use an Explicit Runtime Security Boundary

* **Status:** Accepted
* **Date:** 2026-08-11
* **Decision Owners:** Project Maintainers
* **Scope:** Codex CLI execution, operating-system isolation, filesystem access, credentials, networking, and resource limits

## Context

End-user requests are untrusted input.

The AI Agent may interpret those requests and execute tools or commands as part of its work.

An isolated working directory such as:

```text
/tmp/user_task_<uuid>/
```

provides organizational separation, but it does not by itself prevent a process from accessing files elsewhere on the host or container.

For example, the following execution:

```text
subprocess.run(
    ["codex", "..."],
    cwd=f"{workspace_path}",
)
```

changes the process working directory but does not create a filesystem sandbox.

If the process Unix user can read:

```text
/app/
/etc/
/proc/
$HOME/
```

the process may still be able to access those locations unless a separate runtime restriction prevents it.

Therefore, workspace isolation and runtime isolation must be treated as separate concerns.

## Decision

End-user Codex execution SHALL use defense in depth.

The architecture distinguishes the following boundaries:

1. Context isolation
2. Workspace isolation
3. Filesystem isolation
4. Process isolation
5. Credential isolation
6. Network isolation
7. Resource isolation
8. Lifecycle isolation

No single layer is considered sufficient by itself.

## Layer 1: Context Isolation

The end-user workspace receives only the approved end-user context described in ADR-001.

Behavioral instructions are considered policy guidance, not a security control.

## Layer 2: Workspace Isolation

Every `AgentRun` receives a unique server-generated workspace.

Example:

```text
/workspaces/<agent_run_uuid>/
```

The workspace SHALL NOT be derived directly from user-controlled path input.

The workspace MUST have a bounded lifecycle.

The workspace root itself MUST be a real, orchestration-owned directory. It
MUST NOT be a symlink, MUST NOT be writable by group or other users, and MUST
be treated as a dedicated boundary whose direct children are managed only by
the trusted orchestration process.

Expected structure:

```text
/workspaces/<uuid>/
├── AGENTS.md
├── .codex/
├── inputs/
├── outputs/
├── runtime/
└── logs/
```

## Layer 3: Filesystem Isolation

Production end-user execution MUST NOT depend only on `cwd`.

The runtime SHOULD expose only the files required for the execution.

The Django source repository SHOULD NOT be mounted into the end-user execution runtime.

The runtime MUST NOT intentionally expose:

```text
Django source
Deployment secrets
Host home directories
Docker socket
Other users' workspaces
Database data directories
```

Where supported, filesystem access SHOULD be deny-by-default.

## Layer 4: Process Isolation

End-user execution MUST have a bounded process tree.

The system MUST be able to terminate the full execution tree when:

* A timeout occurs.
* The user cancels the run.
* The worker shuts down.
* An unrecoverable error occurs.

Killing only the direct Codex parent process is not sufficient if child processes can remain alive.

Production execution SHOULD use an isolated runtime or sandbox capable of constraining spawned processes.

## Layer 5: Credential Isolation

The Codex execution environment MUST NOT inherit the complete Django or Celery environment.

This pattern is forbidden:

```text
env = os.environ.copy()
```

followed by forwarding that environment unchanged to the untrusted agent runtime.

The runtime environment MUST be built from an allowlist.

Example conceptual allowlist:

```text
PATH
HOME
LANG
LC_ALL
CODEX_HOME
Required Codex authentication
Runtime-specific non-secret variables
```

Credentials such as the following MUST NOT be present unless explicitly required:

```text
DJANGO_SECRET_KEY
DATABASE_URL
POSTGRES_PASSWORD
REDIS_URL
AWS_SECRET_ACCESS_KEY
Cloud infrastructure credentials
Deployment credentials
Internal service tokens
```

Agent credentials and Django application credentials SHALL be treated as separate credential domains.

## Layer 6: Network Isolation

The agent runtime MUST NOT automatically inherit unrestricted access to the application's internal network.

At minimum, production policy SHOULD prevent unintended access to:

```text
PostgreSQL
Redis
Docker daemon
Cloud metadata endpoints
Private application services
Internal administration endpoints
Host-only services
```

Network access required by the agent platform itself should be separated conceptually from arbitrary network access initiated by generated commands.

Network policy SHOULD follow deny-by-default where practical.

## Layer 7: Resource Isolation

An end-user request MUST NOT be capable of consuming unbounded host resources.

The production runtime SHOULD enforce limits for:

* Wall-clock execution time.
* CPU.
* Memory.
* Process/PID count.
* Workspace disk usage.
* Number of generated artifacts.
* Maximum artifact size.
* Total output size.

The system MUST define explicit maximum values before public launch.

## Layer 8: Lifecycle Isolation

Every workspace MUST have a defined lifecycle:

```text
Create
  ↓
Bootstrap
  ↓
Stage inputs
  ↓
Execute
  ↓
Collect approved outputs
  ↓
Persist required information
  ↓
Cleanup
```

Cleanup MUST occur on:

* Success.
* Execution failure.
* Timeout.
* Cancellation.
* Parsing failure.
* Artifact validation failure.

A periodic sweeper SHOULD exist as a secondary defense against orphaned workspaces.

## Runtime Architecture

The preferred production architecture separates orchestration from untrusted agent execution:

```text
Django API
    │
    ▼
Celery orchestration
    │
    ▼
AgentExecutionService
    │
    ▼
AgentExecutor
    │
    ▼
Isolated agent runtime
    │
    ├── Codex CLI
    └── workspace
```

The end-user runtime SHOULD NOT require the complete Django repository.

## Executor Abstraction

The application SHALL depend on an abstraction similar to:

```text
AgentExecutor
```

rather than directly depending on `subprocess.run`.

Possible implementations:

```text
LocalSubprocessExecutor
SandboxedExecutor
```

`LocalSubprocessExecutor` may be used for trusted development and early integration testing.

It MUST NOT automatically be considered safe for untrusted public workloads.

`SandboxedExecutor` is the intended production implementation.

## Shell Command Construction

User-controlled text MUST NOT be concatenated into a shell command.

This pattern is forbidden:

```text
subprocess.run(
    f"codex exec {user_prompt}",
    shell=True,
)
```

Execution SHOULD use explicit argument arrays and `shell=False`.

User prompt content SHOULD be passed through a dedicated input channel such as stdin whenever practical.

## Filesystem Input Policy

Input handling MUST prevent path traversal and unexpected link behavior.

The implementation MUST reject or safely handle:

```text
../
Absolute paths
Symlink escape
Hardlink escape
Special device files
Unexpected filesystem entries
```

The real resolved path of every staged input MUST remain inside the designated input directory.

## Output Policy

Only designated output locations may be collected.

Example:

```text
<workspace>/outputs/
```

Output collection MUST NOT recursively trust arbitrary paths returned by the model.

Each artifact SHOULD be validated for:

* Real resolved path.
* Symlink status.
* File type.
* MIME type where applicable.
* Size.
* File count.
* Allowed extension or artifact category.

An output path that resolves outside the approved output directory MUST be rejected.

## Approval Policy

Server-side executions MUST NOT depend on interactive human approval.

An end-user task must not remain indefinitely blocked waiting for a terminal confirmation.

The intended execution model is:

```text
Action allowed by runtime policy
    → Execute

Action outside runtime policy
    → Deny or fail deterministically
```

not:

```text
Action requires human approval
    → Block Celery worker indefinitely
```

## Security Invariants

### INV-EXEC-001

`cwd` MUST NOT be treated as the filesystem security boundary.

### INV-EXEC-002

End-user Codex execution MUST NOT receive Django application secrets.

### INV-EXEC-003

Production agent execution MUST NOT receive the Docker daemon socket.

### INV-EXEC-004

One user's workspace MUST NOT be accessible from another user's execution.

### INV-EXEC-005

Django source MUST NOT be intentionally mounted into the production end-user agent runtime.

### INV-EXEC-006

The complete execution process tree MUST be terminable.

### INV-EXEC-007

Resource consumption MUST be bounded.

### INV-EXEC-008

Outputs MUST be collected from an explicit allowlisted location.

### INV-EXEC-009

User-controlled strings MUST NOT become shell syntax through string concatenation.

### INV-EXEC-010

Security MUST remain effective even if the AI attempts to violate `USER_AGENTS.md`.

### INV-EXEC-011

The workspace root MUST be a dedicated orchestration-owned filesystem boundary.

## Rejected Alternatives

### Use only an isolated `/tmp` directory

Rejected because filesystem accessibility is determined by operating-system permissions and sandboxing, not the current working directory.

### Trust `USER_AGENTS.md` to prevent dangerous behavior

Rejected because model instructions are not an access-control mechanism.

### Run Codex with the same environment as Celery

Rejected because it unnecessarily exposes application credentials.

### Mount the entire project repository read-only

Rejected as the default production design because confidentiality matters in addition to integrity.

Read-only access still permits data disclosure.

### Mount `/var/run/docker.sock` into the worker

Rejected as the preferred production architecture because Docker daemon access significantly expands the worker's effective privileges.

## Consequences

### Positive

* Reduced blast radius for malicious prompts.
* Prevents accidental Django secret propagation.
* Makes sandbox assumptions explicit.
* Local development remains simple through an executor abstraction.
* Production hardening can evolve independently of business logic.

### Negative

* Production infrastructure becomes more complex.
* Dedicated runtime management is required.
* Security regression tests and operational monitoring are necessary.
* Execution may have stricter limitations than local development.

## Required Security Tests

Before public exposure, automated or controlled integration tests MUST verify attempts to:

```text
Read Django source
Read environment secrets
Read another workspace
Follow output symlinks
Escape using ../ paths
Access internal PostgreSQL
Access internal Redis
Access Docker socket
Read process environment
Create excessive processes
Consume excessive disk
Run beyond timeout
Leave child processes after cancellation
```

The expected result is denial or safe termination at the runtime layer.

A polite refusal from the language model alone does not satisfy this requirement.

## Implementation Gate

End-user Codex execution may be considered production-ready only when:

1. Context isolation passes.
2. Credential allowlisting passes.
3. Filesystem isolation passes.
4. Cross-workspace isolation passes.
5. Timeout/cancellation passes.
6. Resource constraints are enforced.
7. Restricted-network tests pass.
