# Coding Conventions

These conventions apply to application code across the `ai_gen_image` repository unless a more specific accepted architecture decision requires otherwise.

## Django Domain Apps

New Django domain apps under `apps/` MUST use `apps.common` as the canonical base layout.

The default structure is:

```text
apps/<domain>/
├── migrations/
├── services/
├── __init__.py
├── apps.py
├── models.py
├── serializers.py
├── tasks.py
├── tests.py
├── urls.py
└── views.py
```

Domain-specific subpackages MAY be added when the domain has a clear responsibility that justifies them.

Do not introduce a different Django app layout without a clear architectural reason.

## Domain Boundaries

Domain-specific code MUST remain inside its owning `apps/<domain>/` package unless it is genuinely reusable across multiple domains.

Examples of domain-specific code include:

* models
* services
* serializers
* tasks
* views
* Domain-specific validation
* Domain-specific infrastructure

Do not move code into the top-level `common/` package merely because it is used by multiple modules inside the same domain.

Shared code belongs in `common/` only when it is genuinely reusable across multiple domains or represents project-wide infrastructure.

## Django Conventions

* Keep views thin.
* Keep serializers focused on request validation and response representation.
* Put application workflows and business orchestration in `services/`.
* Keep Celery tasks thin and delegate application logic to services.
* Use explicit database transactions for multi-step state changes when atomicity is required.
* Prefer explicit service orchestration over Django signals for core workflows.
* Enforce ownership or the appropriate permission policy for user-scoped resources.
* Do not expose internal exceptions, filesystem paths, commands, credentials, or other sensitive implementation details through API responses.
* Use `settings.AUTH_USER_MODEL` instead of importing Django's default `User` model directly.
* Generate and inspect Django migrations for model schema changes.

## Model Choices

Do NOT use Django `TextChoices` or `IntegerChoices` for project-defined model choices.

Use:

```text
from common.utils.enum_choices import EnumChoices
```

Example:

```text
class Status(str, EnumChoices):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
```

Django model fields SHOULD consume the enum through:

```text
choices=Status.choices()
```

Choice enums MUST be placed within the `common/constants.py` file under the following section:
```text
# --------------------------------|
# Session for Class choice enums. |
# --------------------------------|
```

Do NOT use domain-level constants. Constants that are not choice enums should be placed after the following section:
```text
# -------------------------|
# Session for constants. |
# -------------------------|
```

## Shared Utilities

Reusable pure Python utilities shared across multiple domains SHOULD live under:

```text
common/utils/
```

Small, general-purpose helpers MAY be added to:

```text
common/utils/helpers.py
```

When a reusable concern has its own clear responsibility, create a dedicated module instead of continuously growing `helpers.py`.

For example:

```text
common/utils/helpers.py
common/utils/enum_choices.py
common/utils/path_utils.py
common/utils/hash_utils.py
```

Do not place domain-specific business logic in `common/utils/`.

Shared utilities SHOULD remain independent of Django domain models whenever practical.

## Python Style

Python code MUST follow PEP 8 and the repository's formatter and linter configuration.

Use:

* 4 spaces for indentation
* `snake_case` for variables, functions, methods, modules, and packages
* `PascalCase` for classes
* `UPPER_SNAKE_CASE` for constants
* English for source-code identifiers, comments, docstrings, configuration, and project documentation

Organize imports in this order:

1. Python standard library
2. Third-party packages
3. Local project imports

Prefer:

* Explicit names over ambiguous abbreviations
* Readable code over clever code
* Small focused functions over large multipurpose functions
* Existing abstractions over duplicate implementations
* Type hints where they improve clarity or correctness

Avoid:

* Unnecessary abstractions
* Generic utility classes without a clear responsibility
* Mutable default arguments
* Broad exception handling that silently suppresses failures
* Unrelated refactoring during a scoped implementation task

## Services

Application workflows SHOULD be implemented in the owning domain's `services/` package.

Services may coordinate:

* Domain models
* Transactions
* External infrastructure
* Task dispatch
* State transitions

Services SHOULD NOT depend on HTTP request or response objects.

Infrastructure-specific implementation details SHOULD remain behind appropriate boundaries instead of leaking into views, serializers, or unrelated domain code.

## Celery

Celery tasks SHOULD act as thin execution entry points.

Prefer task contracts based on stable identifiers, for example:

```text
execute_agent_run(run_id)
```

Do not pass unnecessary credentials, large serialized domain objects, files, or authoritative business state through Celery task arguments.

Assume Celery tasks may be retried or delivered more than once.

Domain execution logic MUST therefore be designed for idempotency where required.

## Tests

New behavior MUST include appropriate tests.

Test:

* Normal behavior
* Invalid input
* Important failure paths
* Domain state transitions
* Security-sensitive boundaries where applicable

Do not weaken or remove existing tests merely to make a new implementation pass.

Keep tests consistent with the existing Django app structure.

A `tests.py` module MAY be promoted to a `tests/` package when the test suite becomes large enough to justify the additional structure.

## File Naming

Use:

```text
Python modules/packages
→ lowercase_snake_case

Project-owned Markdown files
→ lowercase-kebab-case.md

ADR files
→ adr-NNN-short-description.md
```

Preserve ecosystem-defined or tool-recognized filenames exactly, including:

```text
AGENTS.md
README.md
Dockerfile
.gitignore
.dockerignore
```

## Verification

Before completing Python or Django changes, run the repository-defined verification checks.

At minimum, once the corresponding tooling is configured, verify:

```bash
python -m compileall apps common core
ruff check .
ruff format --check .
python manage.py check
python manage.py test
```

Do not report an implementation task as complete when required checks fail.

If a required check cannot be executed, report that explicitly together with the reason.

## Scope Discipline

Keep implementation changes scoped to the requested task.

Do not perform unrelated refactors unless they are required for correctness.

If an existing convention prevents a correct implementation, identify the conflict before intentionally deviating from it.
