# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While on `0.x`, minor releases may include breaking changes. The public API is
the set of symbols exported from `devol.__all__`; anything else is internal.

## [Unreleased]

### Added

- Project metadata for PyPI publication (license, authors, classifiers, URLs).
- `LICENSE` file (MIT).
- `py.typed` marker so downstream type checkers pick up `devol`'s inline hints.
- `CONTRIBUTING.md` with minimum expectations for patches.
- This changelog.
- CI matrix covering Python 3.11, 3.12, and 3.13 with separate lint, typecheck, test, and build jobs.
- `examples`, `benchmark`, `dev`, and `all` optional dependency groups.
- `src/devol` is now strictly typed end-to-end (`mypy --strict` clean).
- Installation section in the README explaining the new extras.

### Changed

- Moved `pytest` from required dependencies to the `dev` extra.
- Moved `torch`, `torchvision`, `gymnasium`, and `matplotlib` out of required dependencies. Installing `devol` now pulls only `numpy`, `pydantic`, and `pydantic-yaml`; the heavy deps live under the `examples` extra.
- Dropped unused `pydantic-settings` from dependencies.
- Wheel contents narrowed to `src/devol` only — `examples/` and `benchmark/` stay in the repo but are no longer shipped.

### Fixed

- `DiffusionEvolution.run()` no longer requires `initial_population`; calling `algo.run()` now initializes a population automatically, matching the documented Quick Start usage and unblocking the `devol` CLI entry point.

## [0.1.0]

Initial development version. Not yet released to PyPI.
