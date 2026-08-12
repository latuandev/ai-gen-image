# AGENTS.md

## Purpose

This file defines repository-level instructions for AI coding agents working on `ai_gen_image`.

This is **developer-only context**.

It MUST NOT be copied or exposed to end-user Agent workspaces.

## Project

`ai_gen_image` is a Django/DRF application that uses Codex CLI in two separate trust domains:

* **Developer workflow:** Codex runs against the trusted source repository.
* **End-user workflow:** Codex runs through the server-side Agent Layer using curated context and an isolated execution environment.

Developer context and end-user context MUST remain separate.

## Required References

Before implementing or modifying application code, read and comply with:

`docs/development/coding-conventions.md`

For AI Agent Layer changes, also read and comply with:

`docs/architecture/security-invariants.md`

Then read the ADRs relevant to the change under:

`docs/architecture/`

For security-sensitive Agent changes, also read:

`docs/architecture/agent-threat-model.md`

Accepted ADRs, active security invariants, and repository coding conventions are implementation constraints.

If a requested change conflicts with one of them, identify the conflict before implementation rather than silently bypassing it.

## Agent Layer Architecture

Preserve the intended dependency direction:

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

Do not put Codex subprocess or workspace-management logic directly in views or serializers.

## Development Workflow

Before modifying code:

1. Inspect the relevant existing implementation.
2. Read the applicable conventions, invariants, and ADRs.
3. Search for existing abstractions before creating new ones.
4. Identify the tests and verification checks required by the change.

During implementation:

1. Keep changes scoped to the requested task.
2. Preserve architectural and domain boundaries.
3. Avoid unrelated refactors.
4. Add or update appropriate tests.

After implementation:

1. Run the repository-defined verification checks.
2. Review migrations and configuration changes when applicable.
3. Verify relevant architecture and security constraints still hold.
4. Report any check that could not be executed and explain why.

## Guiding Principle

```text
Developer Agent
    → Trusted repository context

End-user Agent
    → Explicitly granted capabilities only
```

Behavioral instructions define what the end-user Agent should do.

Runtime security defines what the end-user Agent can do.

Never confuse the two.

## Communication

Report task results to the developer in Vietnamese.

Keep source code, identifiers, comments, configuration, and project documentation in English unless the existing file uses another convention.
