# ADR-001: Separate Developer and End-user Agent Contexts

* **Status:** Accepted
* **Date:** 2026-08-11
* **Decision Owners:** Project Maintainers
* **Scope:** Codex CLI context discovery, configuration, and agent execution

## Context

The `ai_gen_image` project uses Codex CLI in two different trust domains:

1. **Developer workflow**

   * A developer runs Codex locally from the project repository.
   * Codex is intentionally allowed to inspect the Django/DRF codebase.
   * Codex may use repository architecture, implementation details, tests, development commands, and developer-specific instructions to assist with software development.

2. **End-user workflow**

   * An end user submits a request through the Django/DRF API.
   * The request is processed asynchronously by an AI Agent execution layer.
   * Codex operates inside a workspace created specifically for that execution.
   * End-user executions must not receive developer-only context or unnecessary access to the Django application source code.

Although both workflows use Codex CLI, they do not have the same trust level.

Sharing the same `AGENTS.md`, `.codex/`, filesystem context, or credentials between these workflows could expose internal application details or secrets to untrusted end-user executions.

## Decision

Developer and end-user Codex contexts MUST be explicitly separated.

The repository SHALL contain two independent context roots.

### Developer context

Developer-only context is located at the repository root:

```text
ai-gen-image/
├── AGENTS.md
├── .codex/
├── core/
├── apps/
└── ...
```

The root `AGENTS.md` and root `.codex/` are intended exclusively for trusted development workflows.

Developer Codex executions may inspect the Django repository.

### End-user context

End-user context is located under the dedicated agent application:

```text
apps/
└── agent_workspace/
    └── end_user_context/
        ├── USER_AGENTS.md
        ├── context_manifest.json
        └── .codex/
            └── config.toml
```

`USER_AGENTS.md` is stored under this name in the source repository to make its trust scope explicit.

When an end-user workspace is initialized, `USER_AGENTS.md` SHALL be copied into the workspace as:

```text
<workspace>/AGENTS.md
```

The repository root `AGENTS.md` SHALL NOT be copied.

The repository root `.codex/` SHALL NOT be copied.

## Context Bootstrap Policy

Workspace initialization MUST use an explicit allowlist.

The implementation MUST NOT recursively copy arbitrary repository directories into an end-user workspace.

The following conceptual flow is required:

```text
end_user_context/context_manifest.json
                │
                ▼
        Explicit allowlist
                │
                ▼
        WorkspaceManager
                │
                ▼
       Isolated workspace
```

Only files explicitly authorized by the context manifest may enter the workspace.

Example:

```text
Source repository:

apps/agent_workspace/end_user_context/
├── USER_AGENTS.md
├── context_manifest.json
└── .codex/
    └── config.toml


Runtime workspace:

/workspaces/<run_uuid>/
├── AGENTS.md
├── .codex/
│   └── config.toml
├── inputs/
└── outputs/
```

## Security Invariants

The following invariants are mandatory:

### INV-CTX-001

The repository root `AGENTS.md` MUST NEVER be copied, mounted, symlinked, or otherwise exposed as context to an end-user Codex execution.

### INV-CTX-002

The repository root `.codex/` MUST NEVER be copied, mounted, symlinked, or otherwise exposed to an end-user Codex execution.

### INV-CTX-003

End-user context MUST be bootstrapped only from an explicit allowlist.

### INV-CTX-004

Developer-only architecture notes, source code instructions, internal URLs, deployment information, database details, and credentials MUST NOT be present in `USER_AGENTS.md`.

### INV-CTX-005

Each end-user run MUST have its own context instance and workspace.

Context files MUST NOT be modified in-place inside the source repository during an execution.

### INV-CTX-006

The exact end-user context version used by an execution MUST be auditable.

Each `AgentRun` SHOULD record at least:

* Context version.
* Context content hash or fingerprint.

### INV-CTX-007

Context separation is not considered a filesystem security boundary.

The presence of a restricted `AGENTS.md` does not imply that the process is unable to access the rest of the machine.

Filesystem and process isolation are addressed separately by ADR-002.

## Single Source of Truth

"Single Source of Truth" does NOT mean that developer and end-user instructions must be merged into one file.

The project SHALL use a single authoritative execution architecture while maintaining multiple explicitly scoped contexts.

Shared concerns may include:

* Execution contracts.
* State definitions.
* Output schemas.
* Runtime interfaces.
* Versioning conventions.
* Security invariants.

Trust-specific context MUST remain separate.

Therefore:

```text
Shared execution architecture
          │
    ┌─────┴─────┐
    │           │
Developer     End-user
context       context
```

is preferred over:

```text
One shared AGENTS.md
        │
Developer + End-user
```

## Rejected Alternatives

### Reuse the root `AGENTS.md`

Rejected because it may expose internal project architecture and developer-specific instructions to end-user executions.

### Copy the entire repository into every workspace

Rejected because the end-user agent does not require Django application source code and this unnecessarily expands the attack surface.

### Store developer and user rules in one instruction file

Rejected because instructions belonging to different trust domains would be difficult to audit and easy to accidentally expose.

### Rely only on prompts telling the model not to inspect internal files

Rejected because behavioral instructions are not a security boundary.

## Consequences

### Positive

* Clear trust-domain separation.
* Reduced accidental context leakage.
* Easier security auditing.
* End-user policies can evolve independently.
* Developer Codex remains powerful for local development.
* Context versions can be reproduced for historical executions.

### Negative

* Two context configurations must be maintained.
* Shared rules may require deliberate synchronization.
* Additional tests are required to prevent regression.

## Required Tests

At minimum, automated tests SHALL prove:

```text
Root AGENTS.md is not present in workspace
Root .codex is not present in workspace
Only manifest-authorized context files are copied
USER_AGENTS.md becomes <workspace>/AGENTS.md
Workspace context cannot be shared across AgentRuns
Context hash/version is deterministic
```

## Implementation Gate

No end-user Codex execution may be exposed through the public API until the context-isolation tests defined by this ADR pass.
