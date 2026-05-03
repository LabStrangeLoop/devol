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

### Changed

- Moved `pytest` from required dependencies to the `dev` extra.

## [0.1.0]

Initial development version. Not yet released to PyPI.
