# AI Agent Security Invariants

This document is the consolidated registry of active architectural invariants for the `ai_gen_image` AI Agent Layer.

Invariant IDs are stable references. Detailed context, rationale, implementation requirements, and tests are defined by the source ADR.

| ID             | Invariant                                                                                               | Source                                         |
| -------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `INV-CTX-001`  | Root `AGENTS.md` MUST NOT be exposed to end-user Codex execution.                                       | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-CTX-002`  | Root `.codex/` MUST NOT be exposed to end-user Codex execution.                                         | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-CTX-003`  | End-user context MUST be bootstrapped only from an explicit allowlist.                                  | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-CTX-004`  | Developer-only information and credentials MUST NOT be present in end-user Agent context.               | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-CTX-005`  | Each end-user execution MUST have its own context instance and workspace.                               | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-CTX-006`  | The end-user context version used by each execution MUST be auditable.                                  | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-CTX-007`  | Context separation MUST NOT be treated as a filesystem security boundary.                               | [ADR-001](adr-001-agent-context-boundary.md)   |
| `INV-EXEC-001` | `cwd` MUST NOT be treated as a filesystem security boundary.                                            | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-002` | End-user Codex execution MUST NOT receive Django application secrets.                                   | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-003` | Production Agent execution MUST NOT receive Docker daemon access.                                       | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-004` | One user's workspace MUST NOT be accessible from another user's execution.                              | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-005` | Django source MUST NOT be intentionally exposed to the production end-user Agent runtime.               | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-006` | The complete Agent execution process tree MUST be terminable.                                           | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-007` | Agent execution resource consumption MUST be bounded.                                                   | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-008` | Outputs MUST be collected only from explicitly approved locations.                                      | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-009` | User-controlled input MUST NOT become shell syntax through command construction.                        | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-010` | Security MUST remain effective even if the AI attempts to violate behavioral instructions.              | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-EXEC-011` | The workspace root MUST be a dedicated orchestration-owned filesystem boundary.                         | [ADR-002](adr-002-agent-execution-boundary.md) |
| `INV-RUN-001`  | Database `AgentRun` state MUST be authoritative over Celery task state.                                 | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-002`  | Only explicitly allowed `AgentRun` state transitions may occur.                                         | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-003`  | Duplicate task delivery MUST NOT create duplicate concurrent executions for the same execution attempt. | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-004`  | A terminal `AgentRun` MUST NOT silently become active again.                                            | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-005`  | Every user-facing `AgentRun` and artifact lookup MUST enforce authorization.                            | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-006`  | Workspace cleanup MUST be attempted for every terminal execution outcome.                               | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-007`  | Celery messages MUST NOT contain unnecessary secrets or large execution payloads.                       | [ADR-003](adr-003-agent-run-state-machine.md)  |
| `INV-RUN-008`  | Run identifiers and workspace names MUST be generated by the server.                                    | [ADR-003](adr-003-agent-run-state-machine.md)  |

`INV-EXEC-006` is a production runtime containment requirement. The trusted
`LocalSubprocessExecutor` satisfies only the narrower local guarantee described
in ADR-002: it reaps the direct child and terminates live members that remain in
its executor-owned process group before returning.

If an invariant must be changed, superseded, or removed, update the corresponding ADR first and then update this registry.
