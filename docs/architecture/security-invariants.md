# AI Agent Security Invariants

These rules are architecture constraints, not recommendations.

1. Root AGENTS.md is developer-only.
2. Root .codex/ is developer-only.
3. End-user execution receives only allowlisted end_user_context files.
4. USER_AGENTS.md is not a security boundary.
5. cwd is not a filesystem sandbox.
6. End-user Codex must not receive Django source.
7. End-user Codex must not inherit Django/Celery environment wholesale.
8. Agent credentials and application credentials are separate.
9. Every AgentRun has a unique workspace.
10. Cross-workspace access is forbidden.
11. Input paths must remain inside inputs/.
12. Collected artifacts must resolve inside outputs/.
13. Symlink escapes are forbidden.
14. User prompt must never be interpolated into a shell command.
15. API views must never invoke Codex directly.
16. Celery tasks carry run_id, not authoritative execution state.
17. AgentRun database state is authoritative.
18. Duplicate task delivery must not create duplicate execution.
19. Every execution must have timeout and resource limits.
20. Every terminal path must eventually clean its workspace.
21. Production execution must use a real runtime security boundary.
22. A model refusal does not prove security isolation.
