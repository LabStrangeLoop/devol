# Release Checklist

Use this checklist when cutting a new `devol` release.

1. **Bump Version**
   - Update the version in `pyproject.toml` and `src/devol/__init__.py`.
   - Note the changes in `CHANGELOG.md` under a new heading.
2. **Dependencies**
   - Regenerate `uv.lock` with `uv lock`.
   - Audit dependency changes and ensure extras remain optional.
3. **Quality Gates**
   - Run `uv pip install -e ".[dev]"`.
   - Execute `ruff check . && ruff format --check .`.
   - Execute `mypy src tests` and `pyright`.
   - Execute `pytest -q --cov`.
   - Ensure coverage meets the configured threshold.
4. **Packaging**
   - Run `uv build`.
   - Inspect `dist/` contents for unexpected files.
   - Run `python -m twine check dist/*`.
5. **Documentation**
   - Verify README Quickstart in a fresh virtual environment.
   - Confirm examples referenced in the README run with documented extras.
6. **Version Control**
   - Ensure working tree is clean.
   - Tag the release (`git tag vx.y.z`) and push tags.
7. **Publish**
   - Upload artifacts with `twine upload dist/*`.
   - Create a GitHub release linking to the changelog entry.
8. **Post-Release**
   - Bump to the next development version.
   - Announce the release (social, mailing list, etc.) if applicable.
