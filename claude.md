# Claude Code Project Guidelines

## Issue Tracking with Chainlink

**Always use chainlink for all issue and task management.** Before starting any work:

1. **Check existing issues**: `chainlink list` or `chainlink search <term>`
2. **Review blocked issues**: `chainlink blocked` and `chainlink ready`
3. **Get next priority**: `chainlink next`
4. **View issue details**: `chainlink show <id>`

### Workflow

- **Starting work**: Create or find the issue, then `chainlink start <id>` to track time
- **During work**: Add comments with `chainlink comment <id> "progress update"`
- **Completing work**: `chainlink tested <id>` after tests pass, then `chainlink close <id>`
- **Complex tasks**: Break down with `chainlink subissue <parent_id> "subtask title"`
- **Dependencies**: Use `chainlink block <id> <blocker_id>` to track blockers

### Issue Organization

- Use labels: `chainlink label <id> <label>` (e.g., `bug`, `feature`, `security`, `refactor`)
- Link related issues: `chainlink relate <id1> <id2>`
- Use milestones for releases: `chainlink milestone create "v1.0"`

---

## Python Best Practices

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Keep functions small and focused (single responsibility)
- Use descriptive variable and function names
- Prefer composition over inheritance
- Use dataclasses or Pydantic models for structured data

### Project Structure

```
src/
  package_name/
    __init__.py
    module.py
tests/
  __init__.py
  test_module.py
pyproject.toml
```

### Imports

- Use absolute imports
- Group imports: stdlib, third-party, local (separated by blank lines)
- Avoid wildcard imports (`from x import *`)

### Error Handling

- Use specific exception types, not bare `except:`
- Create custom exceptions for domain-specific errors
- Always include meaningful error messages
- Use context managers for resource management

---

## Security-Conscious Practices

### Input Validation

- **Never trust user input** — validate and sanitize all external data
- Use allowlists over denylists for validation
- Validate types, lengths, formats, and ranges

### Secrets Management

- **Never hardcode secrets** in source code
- Use environment variables or secret management tools
- Add `.env` files to `.gitignore`
- Rotate secrets regularly

### Injection Prevention

- Use parameterized queries for SQL (never string concatenation)
- Escape or sanitize data for shell commands — prefer `subprocess` with lists over `shell=True`
- Sanitize HTML output to prevent XSS

### Dependencies

- Pin dependency versions in `pyproject.toml`
- Regularly audit dependencies: `pip-audit` or `safety check`
- Use minimal dependencies — evaluate necessity before adding

### File Operations

- Validate file paths to prevent path traversal
- Use `pathlib` for safe path manipulation
- Set restrictive file permissions

### Authentication & Authorization

- Use established libraries (e.g., `passlib`, `python-jose`)
- Implement proper session management
- Apply principle of least privilege

---

## Verification and Testing

### Test Requirements

- Write tests for all new functionality
- Maintain test coverage above 80%
- Run tests before closing any issue: `chainlink tested <id>`

### Test Structure

```python
# tests/test_module.py
import pytest
from package_name.module import function_to_test

class TestFunctionName:
    def test_normal_case(self):
        """Test expected behavior with valid input."""
        result = function_to_test(valid_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test boundary conditions."""
        ...

    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ExpectedError):
            function_to_test(invalid_input)
```

### Test Types

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions
- **Security tests**: Test for common vulnerabilities

### Running Tests

```bash
pytest                        # Run all tests
pytest -v                     # Verbose output
pytest --cov=src              # With coverage
pytest -x                     # Stop on first failure
pytest -k "test_name"         # Run specific tests
```

### Pre-Commit Verification

Before committing, always verify:
1. All tests pass
2. Linting passes
3. Type checking passes (if using mypy)
4. No security issues detected

---

## Linting and Code Quality

### Required Tools

Run these before any commit:

```bash
# Formatting
black .                       # Auto-format code
isort .                       # Sort imports

# Linting
ruff check .                  # Fast linter (preferred)
# or
flake8 .                      # Traditional linter

# Type checking
mypy src/                     # Static type analysis

# Security
bandit -r src/                # Security linter
```

### Configuration

Maintain consistent configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "S", "B", "C4"]

[tool.mypy]
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.bandit]
exclude_dirs = ["tests"]
```

### CI Integration

Ensure all linting and tests run in CI before merging any changes.

---

## Workflow Summary

1. `chainlink next` — Find the next issue to work on
2. `chainlink start <id>` — Start the timer
3. Write code following best practices above
4. Write tests for new functionality
5. Run linting: `black . && isort . && ruff check .`
6. Run tests: `pytest`
7. Run security check: `bandit -r src/`
8. `chainlink tested <id>` — Mark tests as run
9. `chainlink comment <id> "summary of changes"`
10. `chainlink close <id>` — Close the issue
