# AGENT.md

## Purpose

This repository is Python-first. Use the public Google Python Style Guide as the main coding style reference:

- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
- Google Style Guides index: https://google.github.io/styleguide/

Other Google style guides may be used for non-Python files, but Python conventions in this file should be treated as the default for day-to-day work.

## Priority order

When style rules appear to conflict, use this order:

1. Explicit user or maintainer instructions.
2. Local repository configuration, such as `pyproject.toml`, `ruff.toml`, `.pylintrc`, `mypy.ini`, `setup.cfg`, `.editorconfig`, pre-commit, CI, or formatter settings.
3. Existing style in the files being edited.
4. This `AGENT.md`.
5. The Google Python Style Guide.
6. General readability, maintainability, and consistency.

Do not reformat or rewrite unrelated code only to make it match this guide. Prefer minimal, focused changes.

## Core principles

Write Python that is:

- Clear: the reader can quickly understand what the code does and why.
- Simple: avoid unnecessary abstraction and cleverness.
- Consistent: match nearby code and project conventions.
- Maintainable: future changes should be straightforward.
- Testable: important behavior should be covered by tests.

Clarity is more important than cleverness. Prefer boring, idiomatic Python over surprising constructs.

## Python version and tooling

Follow the Python versions supported by the project. Do not introduce syntax that is unsupported by the configured runtime.

Before changing style manually, check for project tooling. Common tools may include:

- `ruff`
- `pylint`
- `black`
- `pyink`
- `isort`
- `mypy`
- `pyright`
- `pytest`
- `tox`
- `nox`

Use the repository’s configured formatter and linter when available. Do not fight automated formatting.

## Formatting

Follow local formatter settings first. In the absence of local settings, follow Google Python style.

General expectations:

- Use spaces, not tabs.
- Do not use semicolons to put multiple statements on one line.
- Avoid unnecessary parentheses.
- Keep blank lines purposeful and consistent with surrounding code.
- Keep lines readable; prefer wrapping expressions cleanly instead of making dense one-liners.
- Avoid unrelated whitespace-only changes.
- Prefer trailing commas in multi-line literals or calls when they reduce diff noise and are accepted by the formatter.

## Imports

Use imports that make the origin of names clear.

Prefer:

```python
import os
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any
```

Avoid importing individual functions, classes, or constants from ordinary modules when importing the module is clearer.

Prefer module imports:

```python
import datetime

timestamp = datetime.datetime.now(tz=datetime.UTC)
```

Avoid this for ordinary modules unless there is a good reason:

```python
from datetime import datetime
```

Google-style import guidance for this repository:

- Import packages and modules, not individual symbols, except for typing-related imports and established local conventions. Importing individual names is still fine when those names are central to the module's purpose or part of a well-defined public API.
- Prefer absolute imports (full package paths) for modules outside the current package; they make each name's origin clear and are resilient to refactoring. Within a package, explicit relative imports such as `from . import helpers` are idiomatic. Avoid implicit relative imports.
- Use `from package import module` when it improves readability.
- Use aliases only when they are standard or resolve a real conflict, such as `import numpy as np`.
- Do not use wildcard imports (`from module import *`); they obscure where names come from, hinder static analysis, and can silently shadow existing names.
- Group imports into standard library, third-party, and local application sections separated by blank lines, and sort alphabetically within each group. Let `isort`/`ruff` apply this automatically when configured.
- Keep imports at the top of the file unless a local import is needed to avoid an expensive dependency, optional dependency, or circular import.
- Avoid circular imports. If two modules import each other, move the shared code into a separate module or reduce coupling, rather than papering over it with local imports.
- Remove unused imports.
- Keep type-checking-only imports inside `if TYPE_CHECKING:` only when necessary.

Recommended import grouping:

```python
from __future__ import annotations

import dataclasses
import pathlib

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import requests

from my_project import config
from my_project.storage import records

if TYPE_CHECKING:
    from my_project.clients import ApiClient
```

Follow the project’s import sorter if it uses one.

Avoid mixed styles, wildcard imports, and unrelated imports on one line:

```python
# Avoid
from app.utils.helpers import format_date, slugify
import requests, os, sys
from .models import *
```

```python
# Favor
import os
import sys

import requests

from app.utils import helpers
from . import models
```

Repository note: `vlm_mpc` task modules intentionally keep only light imports (e.g. `import os`) at the top and defer heavy imports (`torch`, `isaacsim`, `sim_free_mpc`) into `run()`, because `main.py` imports the selected task module before the Isaac app boots. This is the "expensive dependency" exception above, not a violation of the top-of-file rule.

## Naming

Use descriptive names that communicate intent.

Common Python naming:

- Modules and packages: `lower_snake_case`
- Functions and methods: `lower_snake_case`
- Variables and parameters: `lower_snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Classes and exceptions: `CapWords`
- Protected members intended for internal use: `_leading_underscore`
- Private module-level constants: `_UPPER_SNAKE_CASE`

Avoid:

- Single-letter names except for conventional loop indices, small local scopes, or mathematical code where the notation is established.
- Ambiguous names such as `data`, `obj`, `value`, `result`, or `temp` when a more precise name is available.
- Names that shadow built-ins, such as `list`, `dict`, `id`, `file`, `type`, or `input`.
- Abbreviations that are not already common in the project.

Use `self` for instance methods and `cls` for class methods.

## Type annotations

Use type annotations for new or changed public functions and for internal functions where annotations improve readability or help tooling.

Prefer modern Python typing when supported by the project:

```python
def normalize_names(names: Sequence[str]) -> list[str]:
    return [name.strip().lower() for name in names]
```

Guidelines:

- Annotate parameters and return types for public APIs.
- Prefer `str`, `bytes`, `list[str]`, `dict[str, int]`, and `tuple[str, ...]` over older `typing.List` or `typing.Dict` forms when supported.
- Prefer abstract collection types for inputs, such as `Sequence`, `Mapping`, and `Iterable`, when the function does not require a concrete type.
- Use concrete return types when the function returns a concrete object.
- Use `X | None` for optional values on Python 3.10+ when supported.
- Avoid `Any` unless the value is truly dynamic or constrained by external APIs.
- Do not add old-style `# type:` comments in new code.
- Keep annotations simple. If a type is too complex to read, consider a type alias or a simpler design.
- Avoid circular imports caused only by typing. Prefer refactoring; use `TYPE_CHECKING` only when needed.

Example:

```python
from collections.abc import Mapping, Sequence

def build_lookup(items: Sequence[str]) -> Mapping[str, int]:
    return {item: index for index, item in enumerate(items)}
```

## Docstrings

Use docstrings for public modules, classes, functions, and methods. Use triple double quotes.

A docstring should explain what the object does, not merely repeat its name or signature.

Function docstrings should usually include:

- A short summary.
- `Args:` when parameters need explanation.
- `Returns:` when the return value is not obvious.
- `Raises:` for meaningful exceptions callers should know about.
- `Yields:` for generator functions.

Example:

```python
def load_user(user_id: str) -> User:
    """Loads a user by ID.

    Args:
        user_id: Stable user identifier.

    Returns:
        The matching user.

    Raises:
        UserNotFoundError: If no user exists for `user_id`.
    """
    ...
```

Class docstrings should start with a one-line summary describing what instances represent.

```python
class RetryPolicy:
    """Configuration for retrying transient operations."""
```

Do not write noisy docstrings for obvious private helpers unless they clarify intent, constraints, or non-obvious behavior.

## Comments

Write comments for why, not what.

Good comments explain:

- Non-obvious business rules.
- Surprising edge cases.
- Security or correctness constraints.
- Workarounds for external bugs.
- Performance-sensitive decisions.

Avoid comments that restate the code.

```python
# Good: The upstream API rejects timezone-aware midnight values.
start_time = start_time.replace(tzinfo=None)
```

TODO comments should include enough context to be actionable. Prefer this form:

```python
# TODO(username): Replace polling after webhook support is available.
```

If the project uses issue IDs instead of usernames, follow that convention.

## Functions

Keep functions focused and reasonably short. A function should usually do one thing at one level of abstraction.

Guidelines:

- Prefer early returns when they reduce nesting.
- Keep parameters manageable. If a function takes many related parameters, consider a dataclass or configuration object.
- Avoid boolean flags that make one function perform multiple unrelated behaviors.
- Avoid hidden side effects.
- Avoid mutable default arguments.

Use this:

```python
def append_item(item: str, values: list[str] | None = None) -> list[str]:
    if values is None:
        values = []
    values.append(item)
    return values
```

Do not use this:

```python
def append_item(item: str, values: list[str] = []) -> list[str]:
    values.append(item)
    return values
```

## Classes

Use classes when they make state and behavior clearer. Do not introduce a class just to group unrelated functions.

Guidelines:

- Prefer small classes with clear responsibility.
- Keep initialization simple.
- Use dataclasses for plain data containers when appropriate.
- Use properties only when attribute-like access is natural and cheap.
- Avoid surprising work in property getters.
- Avoid unnecessary inheritance.
- Prefer composition over deep class hierarchies.
- Make public methods intentional; keep helpers private when they are implementation details.

## Exceptions and error handling

Exceptions are fine when used for exceptional conditions and clear error handling.

Guidelines:

- Raise specific exception types.
- Create custom exceptions when callers need to distinguish project-specific failures.
- Do not catch broad `Exception` unless you are adding context, cleaning up, or guarding a process boundary.
- Never use a bare `except:`.
- Preserve useful context when re-raising.
- Keep error messages precise and actionable.
- Do not use exceptions for normal control flow when a clearer conditional would do.

Good:

```python
try:
    payload = json.loads(raw_payload)
except json.JSONDecodeError as exc:
    raise InvalidPayloadError("Response body is not valid JSON.") from exc
```

Avoid:

```python
try:
    payload = json.loads(raw_payload)
except Exception:
    return {}
```

## Boolean expressions and `None`

Use Python truthiness where it is clear.

Prefer:

```python
if users:
    ...
```

Use explicit `is None` checks for `None`:

```python
if timeout is None:
    ...
```

Do not compare booleans to `True` or `False` with `==`.

Prefer:

```python
if is_enabled:
    ...
```

Avoid:

```python
if is_enabled == True:
    ...
```

## Comprehensions and generators

Use comprehensions when they are simple and readable.

Good:

```python
active_users = [user for user in users if user.is_active]
```

Avoid deeply nested or multi-condition comprehensions that are harder to read than a loop.

Use generators for streaming or large data when a full list is unnecessary.

```python
def iter_active_users(users: Iterable[User]) -> Iterator[User]:
    for user in users:
        if user.is_active:
            yield user
```

## Lambda functions

Use lambdas sparingly. Prefer named functions when logic is non-trivial or reused.

Good:

```python
users.sort(key=lambda user: user.created_at)
```

Better for complex logic:

```python
def sort_key(user: User) -> tuple[datetime.datetime, str]:
    return user.created_at, user.email

users.sort(key=sort_key)
```

## File and resource handling

Use context managers for files, locks, sockets, temporary directories, and similar resources.

Good:

```python
with pathlib.Path(path).open(encoding="utf-8") as file:
    return file.read()
```

Avoid manually opening and closing resources when a context manager is available.

## Logging

Use logging for diagnostic information, not `print`, except in command-line user output or quick scripts where `print` is intentional.

Guidelines:

- Use module-level loggers.
- Keep log messages useful and concise.
- Do not log secrets, credentials, tokens, or personal data.
- Prefer lazy `%` formatting for logging calls.

```python
import logging

_LOGGER = logging.getLogger(__name__)

_LOGGER.info("Processed %d records", record_count)
```

## Strings

Use f-strings for local string formatting when they improve readability.

For logging, prefer lazy logger formatting:

```python
_LOGGER.warning("Retrying request for user_id=%s", user_id)
```

Avoid building strings in loops with repeated concatenation when a list plus `"".join(...)` is clearer and more efficient.

## Main entry points

Python files intended to be run as scripts should use a main function and guard.

```python
def main() -> None:
    ...


if __name__ == "__main__":
    main()
```

Keep import-time side effects minimal. Importing a module should not start network calls, parse command-line arguments, mutate global state, or run expensive work unless that is the module’s explicit purpose.

## Testing

Use the project’s existing test framework and conventions. If none exist, prefer `pytest` unless instructed otherwise.

Tests should:

- Verify observable behavior.
- Cover success, edge, and failure cases.
- Use clear test names.
- Avoid brittle implementation details.
- Avoid real network calls, real external services, and time-dependent behavior unless explicitly part of an integration test.
- Use fixtures or helpers to reduce duplication when they improve clarity.
- Keep tests deterministic.

Prefer descriptive test names:

```python
def test_load_user_raises_when_user_is_missing() -> None:
    ...
```

## Dependencies

Do not add new dependencies unless they are clearly needed.

Before adding a dependency:

- Check whether the standard library is sufficient.
- Check whether the repository already has a suitable dependency.
- Consider maintenance, security, licensing, and package size.
- Update dependency files and lockfiles consistently.

## Security and privacy

When writing Python code:

- Validate and sanitize external input.
- Avoid shelling out with `shell=True` unless absolutely necessary.
- Do not log secrets or personal data.
- Do not commit credentials, tokens, private keys, or local environment files.
- Use secure defaults for file permissions, network calls, serialization, and cryptography.
- Avoid unsafe deserialization formats and functions.

## Code review behavior

When reviewing Python code, prioritize:

1. Correctness.
2. Security and data safety.
3. Public API clarity.
4. Error handling.
5. Test coverage.
6. Type correctness.
7. Readability and maintainability.
8. Style consistency.

Distinguish required changes from optional polish. Do not block progress on subjective style preferences when the code is otherwise clear and consistent.

## Agent workflow

When acting as a coding agent in this repository:

1. Inspect relevant files and project configuration before editing.
2. Prefer Python-specific Google style unless local tooling says otherwise.
3. Make the smallest coherent change that satisfies the request.
4. Add or update tests for behavior changes.
5. Run the most relevant checks available.
6. Report exactly what changed and what checks were run.

Recommended checks, depending on the project:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pylint path/to/changed_files.py
python -m mypy .
```

Use only the checks that are configured and relevant. If a check cannot be run, say so clearly.

## Do not

- Do not rewrite unrelated code for style only.
- Do not introduce broad architecture changes unless requested.
- Do not add new dependencies casually.
- Do not ignore local formatter or linter settings.
- Do not use wildcard imports.
- Do not use mutable default arguments.
- Do not hide errors with broad exception handlers.
- Do not add comments or docstrings that simply repeat obvious code.
- Do not claim checks passed unless they were actually run.
- Do not copy large sections of Google’s guide into the repository; summarize and link instead.

## Final response expectations

When completing a coding task, report:

- Files changed.
- Main behavior or style changes made.
- Tests, linters, formatters, or type checks run.
- Any checks not run.
- Any assumptions or follow-up risks.
