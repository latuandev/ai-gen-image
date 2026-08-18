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

Domain-specific code MUST remain inside its owning `apps/<domain>/` package unless it is genuinely reusable across multiple domains or explicitly governed by a repository-wide convention.

Examples of domain-specific code include:

* Models
* Services
* Serializers
* Tasks
* Views
* Domain-specific validation
* Domain-specific infrastructure

Do not move code into the top-level `common/` package merely because it is used by multiple modules inside the same domain.

Shared code belongs in `common/` only when it is genuinely reusable across multiple domains, represents project-wide infrastructure, or is explicitly required there by another repository-wide convention.

Project-defined choice enums and constants are intentional repository-wide exceptions and MUST follow the rules defined in the `Model Choices` section.

## Domain Exceptions

Domain-specific exceptions MUST be defined in the owning app's `exceptions.py`
module when they represent a domain contract, service failure, validation
outcome, or deterministic error condition that callers may catch.

For example:

```text
apps/<domain>/exceptions.py
```

Do not place domain-specific exceptions in `common/exceptions.py`.

Shared exceptions belong in `common/exceptions.py` only when they are genuinely
reusable across multiple domains or represent project-wide infrastructure.

Services SHOULD import and raise domain exceptions from their owning app's
`exceptions.py` instead of defining catchable domain exceptions inline.

## User-Facing Messages

Human-readable messages intended to be returned through APIs and displayed to end users MUST be centralized in:

```text
common/messages.py
```

Do not hard-code end-user-facing error or validation messages directly in views, serializers, services, tasks, or domain exceptions.

Messages MUST:

* Be grouped first by language code
* Be grouped second by owning app namespace
* Use stable `lowercase_snake_case` message keys
* Provide the same message key for every supported language
* Contain only end-user-safe information
* Never expose internal exception details, filesystem paths, commands, credentials, stack traces, or implementation details

Use the following structure:

```python
from django.conf import settings


# Message lookup structure: LANGUAGE_CODE -> app namespace -> stable message key.
_MESSAGES = {
    "en-us": {
        "agent_workspace": {
            "duplicate_name": "You already have an agent definition with this name.",
        },
    },
    "vi": {
        "agent_workspace": {
            "duplicate_name": "Bạn đã có một định nghĩa agent với tên này.",
        },
    },
}

MESSAGES = _MESSAGES[settings.LANGUAGE_CODE]
```

Consume messages through the exported `MESSAGES` mapping:

```python
from common.messages import MESSAGES


message = MESSAGES["agent_workspace"]["duplicate_name"]
```

Do not use Python built-in names such as `str`, `list`, or `dict` as local variable names.

Domain exceptions SHOULD represent deterministic error conditions independently from their user-facing presentation text.

For example:

```text
raise DuplicateAgentNameError()
```

The API-facing layer or application boundary responsible for translating that error into a response SHOULD use the corresponding message from `common.messages`.

Internal logging and diagnostic messages that are not exposed to end users do NOT belong in `common/messages.py`.

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
# Section for Class choice enums. |
# --------------------------------|
```

Do NOT use domain-level constants. Constants that are not choice enums should be placed after the following section:
```text
# -------------------------|
# Section for constants.   |
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

## Comments and Human-Readable Messages

Sentence-like comments and human-readable message strings MUST begin with an uppercase letter.

This rule applies to:

* Code comments
* Exception messages
* Validation messages
* Log messages
* User-facing messages
* Other sentence-like human-readable text embedded in source code

Use:

```text
# Load and validate the context manifest.
raise InvalidContextManifest("Manifest file cannot be read")
logger.warning("Agent workspace cleanup failed")
```

Do not use:

```text
# load and validate the context manifest.
raise InvalidContextManifest("manifest file cannot be read")
logger.warning("agent workspace cleanup failed")
```

Stable identifiers, message keys, field names, paths, commands, and other non-sentence values are excluded from this rule.

For example:

```text
MESSAGES["agent_workspace"]["duplicate_name"]
error_code = "invalid_context_manifest"
```

## Docstrings

Project-owned classes, functions, and methods SHOULD include docstrings when they define a meaningful responsibility, behavior, contract, or reusable interface.

Public classes, service functions, reusable utilities, and non-trivial methods MUST include docstrings.

Docstrings MUST:

* Be written in English
* Follow PEP 257 conventions unless overridden by this repository style
* Use multi-line triple-quoted formatting, even for short docstrings
* Place the opening and closing triple quotes on their own lines
* Describe the purpose and behavior of the class, function, or method
* Document important arguments, return values, raised exceptions, side effects, or transactional behavior when they are not obvious from the signature
* Remain concise and avoid repeating information already clear from names and type hints

Use this format:

```text
class AgentRun(models.Model):
    """
    Represent a persisted end-user agent execution lifecycle.
    """
```

For functions and methods:

```text
def generate_context_hash(files: list[Path]) -> str:
    """
    Generate a deterministic SHA-256 hash for the provided context files.
    """
```

Use additional paragraphs when the behavior or contract requires more explanation.

Example:

```text
def transition_agent_run(run_id: UUID, target_status: str) -> AgentRun:
    """
    Transition an AgentRun to an allowed target status.

    The run is locked within a database transaction before validating and
    applying the state transition.

    Raises:
        InvalidAgentRunTransition: If the requested transition is not allowed.
    """
```

Do not use single-line docstrings such as:

```text
def generate_context_hash(files: list[Path]) -> str:
    """Generate a deterministic SHA-256 hash for the provided context files."""
```

Generated migrations, and standard framework boilerplate MAY omit docstrings when the behavior is already self-explanatory.

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
python manage.py makemigrations --check
python manage.py test
```

Do not report an implementation task as complete when required checks fail.

If a required check cannot be executed, report that explicitly together with the reason.

## Scope Discipline

Keep implementation changes scoped to the requested task.

Do not perform unrelated refactors unless they are required for correctness.

If an existing convention prevents a correct implementation, identify the conflict before intentionally deviating from it.
