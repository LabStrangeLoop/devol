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

## Releasing (maintainers only)

Releases are driven by git tags and published automatically via GitHub Actions using PyPI Trusted Publishers (OIDC). No API tokens are stored anywhere.

Tag shape drives the destination:

| Tag example | Destination |
|-------------|-------------|
| `v0.1.0`    | **PyPI** (production) |
| `v0.1.0rc1`, `v0.1.0a1`, `v0.1.0b1` | **TestPyPI** (dry run) |

Each destination has its own GitHub Environment (`pypi` / `testpypi`) with a required reviewer, so the publish step pauses for manual approval before uploading.

### Standard release flow

1. Bump `version` in `pyproject.toml`.
2. In `CHANGELOG.md`, move items from `[Unreleased]` into a new `[x.y.z]` section dated today.
3. Commit and push to `main`.
4. Tag and push:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. Approve the publish step in the GitHub Actions run when prompted. The workflow builds, validates, publishes, and creates a GitHub Release with the built artifacts attached.

### Dry-running against TestPyPI first

For a never-published version or when the release workflow itself has changed, tag a prerelease first:

```bash
git tag v0.2.0rc1
git push origin v0.2.0rc1
```

This runs the full pipeline against TestPyPI. Verify the upload at `https://test.pypi.org/project/devol/` before cutting the real tag.

### If a release fails mid-flight

- Pre-publish (validate/build): just fix the problem, delete the tag, retag.
- Post-publish: PyPI versions are immutable. Bump to the next patch and release again — do not attempt to overwrite.
