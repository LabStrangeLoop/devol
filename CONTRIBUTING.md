# Contributing

Thanks for your interest in devol. This is a small project, so the process is light.

## Before you start

For anything non-trivial, open an issue first to discuss the change. Bug fixes, typo fixes, and small quality-of-life improvements can go straight to a PR.

## Development setup

```bash
git clone https://github.com/LabStrangeLoop/devol.git
cd devol
uv sync --extra dev
```

## Before you open a PR

Please make sure these all pass locally:

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy src/
```

## Pull requests

- One logical change per PR.
- Include a short description of the "why" — what problem does this solve?
- If you add behavior, add tests.
- Add a line to `CHANGELOG.md` under `[Unreleased]` describing user-visible changes.

## Questions

Open an issue with the `question` label, or start a discussion. No wrong questions.
