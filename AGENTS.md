# AGENTS.md

## Purpose

This file defines repository-level instructions for AI coding agents working on `ai_gen_image`.

This is **developer-only context**.

It MUST NOT be copied or exposed to end-user Agent workspaces.

---

## Project

`ai_gen_image` is a Django/DRF application that uses Codex CLI in two separate trust domains:

* **Developer workflow:** Codex runs against the trusted source repository.
* **End-user workflow:** Codex runs through the server-side Agent Layer using curated context and an isolated execution environment.

Developer context and end-user context MUST remain separate.

---

## Architecture References

Architecture documentation lives under:

```text
docs/architecture/
```

For AI Agent Layer changes, always read:

```text
docs/architecture/security-invariants.md
```

Then read the relevant ADRs:

```text
adr-001-agent-context-boundary.md
adr-002-agent-execution-boundary.md
adr-003-agent-run-state-machine.md
```

For security-sensitive Agent changes, also read:

```text
agent-threat-model.md
```

Accepted ADRs and security invariants are architectural constraints.

Do not silently implement behavior that contradicts them.

---

## Core Agent Rules

Never:

* Copy root `AGENTS.md` into an end-user workspace.
* Copy root `.codex/` into an end-user workspace.
* Treat `cwd` as a security sandbox.
* Expose Django source or application secrets to an end-user Agent runtime.
* Forward the complete Django/Celery environment to Codex.
* Interpolate user input into shell commands.
* Invoke Codex directly from DRF views.
* Treat Celery state as authoritative `AgentRun` state.
* Share mutable workspaces between Agent runs.
* Trust model-provided filesystem paths without validation.
* Treat model refusal as proof of security isolation.

Detailed requirements are defined in:

```text
docs/architecture/security-invariants.md
```

---

## Dependency Direction

Preserve this direction:

```text
DRF API
    ↓
Application Services
    ↓
Celery
    ↓
AgentExecutionService
    ↓
AgentExecutor
    ↓
Codex Runtime
```

Keep responsibilities separated:

```text
WorkspaceManager
    → Workspace lifecycle

CodexCLIWrapper
    → Codex CLI invocation

AgentExecutor
    → Runtime execution boundary

AgentExecutionService
    → Application orchestration
```

Do not put Codex subprocess logic directly in views or serializers.

---

## Django Conventions

* Keep views thin.
* Keep serializers focused on validation and representation.
* Put application workflows in services.
* Use explicit transactions for multi-step state changes.
* Avoid signals for core workflows when explicit service orchestration is clearer.
* Enforce ownership for user-scoped resources.
* Do not expose internal exceptions, paths, commands, or secrets through APIs.

---

## Celery Conventions

Celery tasks should be thin entry points.

Prefer task contracts such as:

```text
execute_agent_run(run_id)
```

Workers must reload authoritative state from persistent storage.

Assume tasks may be retried or delivered more than once.

Agent execution must therefore be idempotent at the domain level.

---

## Testing

Changes must include tests appropriate to their architectural layer.

For Agent Layer changes, consider:

* State transition tests.
* Duplicate execution tests.
* Workspace isolation tests.
* Path traversal tests.
* Symlink escape tests.
* Timeout and cancellation tests.
* Cleanup tests.
* Credential isolation tests.

Do not weaken security tests to make an implementation pass.

---

## File Naming

Use:

* Python modules/packages: `lowercase_snake_case`
* Project-owned docs: `lowercase-kebab-case.md`
* ADRs: `adr-NNN-short-description.md`

Preserve ecosystem/tool-defined names such as:

```text
AGENTS.md
README.md
Dockerfile
.gitignore
.dockerignore
```

---

## Development Workflow

Before modifying code:

1. Inspect the relevant implementation.
2. Read applicable architecture documents.
3. Search for existing abstractions.
4. Identify required tests.

During implementation:

1. Keep changes scoped.
2. Preserve architectural boundaries.
3. Avoid unrelated refactors.
4. Add or update tests.

After implementation:

1. Run relevant tests.
2. Review migrations and configuration changes.
3. Verify security invariants still hold.

---

## Scope Discipline

Do not perform unrelated refactors unless they are required for correctness.

If you discover an unrelated issue, report it instead of silently expanding the task.

If a requested implementation conflicts with an accepted ADR or security invariant, identify the conflict before proceeding.

---

## Guiding Principle

Use:

```text
Developer Agent
    → Trusted repository context

End-user Agent
    → Explicitly granted capabilities only
```

Behavioral instructions define what the Agent should do.

Runtime security defines what the Agent can do.

Never confuse the two.
