# Contributing to devol

Thank you for considering a contribution! The project welcomes bug reports, feature requests, and
pull requests. This document outlines the preferred workflow.

## Getting Started

1. **Fork and clone** the repository.
2. Install the development environment:
   ```bash
   uv pip install -e ".[dev]"
   ```
3. Run the quality gates before making changes:
   ```bash
   ruff check . && ruff format --check .
   mypy src tests
   pyright
   pytest -q
   ```

## Development Workflow

1. Create a feature branch describing the change, e.g. `feat/new-schedule`.
2. Write tests for new functionality or to cover regressions.
3. Keep commits focused and include clear messages.
4. Ensure `CHANGELOG.md` documents user-facing changes.
5. Open a pull request and fill in the template (if provided). Include:
   - Motivation and summary of the change.
   - Testing performed.
   - Any follow-up work.

## Coding Guidelines

- Follow the project’s Ruff, MyPy, and Pyright settings.
- Prefer functional patterns and explicit typing.
- Use `pydantic` models for configuration surfaces.
- Add concise docstrings for public APIs.

## Issue Reporting

When filing an issue, please include:

- Environment details (`python --version`, OS).
- Steps to reproduce.
- Expected vs actual behaviour.
- Logs or stack traces when available.

## Code of Conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report any
unacceptable behaviour to the maintainers.

## License

Contributions are made under the MIT License. By submitting a pull request you confirm that you have
the right to license your contribution under the project’s terms.
