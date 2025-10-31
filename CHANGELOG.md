# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-01

### Added
- Initial release with the core Diffusion Evolution Algorithm implementation.
- Pydantic-backed configuration models and `DiffusionSettings`.
- Distance, schedule, and fitness strategy factories.
- Command-line interface for YAML-driven runs.
- Foundational test suite and integration scenario.
- Optional extras for examples and MNIST experiments.

### Packaging
- uv-based build backend with strict linting and typing configuration.
- GitHub Actions workflow proposal covering lint, type, test, and build gates.
