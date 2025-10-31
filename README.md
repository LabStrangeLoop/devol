# devol

Diffusion Evolution Algorithm — evolution through iterative denoising.

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart (60 seconds)](#quickstart-60-seconds)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

`devol` is a pure-Python implementation of the Diffusion Evolution Algorithm (DEA). DEA combines
diffusion-style denoising with evolutionary search to maintain population diversity while converging
on high-quality solutions. The library embraces a functional style with Pydantic-powered
configuration and ships with a small but expressive set of schedules, fitness mappings, and distance
metrics.

## Installation

```bash
uv pip install devol
```

Optional extras:

- `.[examples]` — plotting and Gym-based demos.
- `.[mnist]` — deeper Torch-based MNIST experiment.
- `.[dev]` — linting, typing, and testing stack used by the project.

## Quickstart (60 seconds)

```python
import numpy as np

from devol import DiffusionConfig, DiffusionEvolution


def two_peaks(x: np.ndarray) -> float:
    peak1 = np.exp(-np.sum((x - np.array([1.0, 1.0])) ** 2) / 0.05)
    peak2 = np.exp(-np.sum((x - np.array([-1.0, -1.0])) ** 2) / 0.05)
    return float((peak1 + peak2) / 2)


config = DiffusionConfig(param_dim=2, population_size=128, num_steps=60, seed=321)
algo = DiffusionEvolution(config, two_peaks)
final_population = algo.run()
best_individual, best_fitness = algo.get_best_individual()

print("Best individual:", best_individual)
print("Best fitness:", best_fitness)
```

## API Reference

- `DiffusionEvolution` — orchestrates the DEA loop. Key methods: `run`, `step`, `get_best_individual`.
- `DiffusionConfig` — validated configuration object with nested models:
  - `ScheduleConfig` (`ScheduleType` enum).
  - `FitnessConfig` (`FitnessMapping` enum).
  - `DistanceConfig` (`DistanceType` enum).
- `DiffusionSettings` — Pydantic Settings wrapper for `.env`/environment driven overrides.
- Exceptions exported from `devol.exceptions`:
  - `DevolError`
  - `ConfigurationError`
  - `EvolutionError`
  - `FitnessComputationError`

See `src/devol` docstrings for detailed signatures.

## Configuration

Create reusable settings with `.env` support using `DiffusionSettings`:

```python
from devol import DiffusionConfig
from devol.config import DiffusionSettings

settings = DiffusionSettings()
config = settings.to_config(param_dim=10, seed=42)
```

Environment variable schema (prefix `DEVOL_`, nested fields separated with `__`):

- `DEVOL_POPULATION_SIZE`
- `DEVOL_NUM_STEPS`
- `DEVOL_PARAM_DIM`
- `DEVOL_SIGMA_M`
- `DEVOL_SEED`
- `DEVOL_SCHEDULE__TYPE`
- `DEVOL_SCHEDULE__EPSILON`
- `DEVOL_FITNESS__MAPPING`
- `DEVOL_FITNESS__TEMPERATURE`
- `DEVOL_FITNESS__NORMALIZE`
- `DEVOL_FITNESS__SHIFT_NEGATIVE`
- `DEVOL_DISTANCE__TYPE`
- `DEVOL_DISTANCE__LATENT_DIM`

## Examples

| Path | Description | Extras |
| --- | --- | --- |
| `examples/two_peaks.py` | Basic 2D optimisation demo with optional contour plot. | `examples` |
| `examples/benchmark_schedules.py` | Compare linear, cosine, and DDPM schedules. | `examples` |
| `examples/cartpole.py` | Gymnasium CartPole controller search (headless). | `examples` |
| `examples/live_cartpole.py` | Live PCA visualisation of CartPole evolution. | `examples` |
| `examples/mnist/` | Torch-based MNIST embedding and optimisation. | `mnist` |

Each example can be executed with `uv run python <script.py>` once the matching extra is installed.

## FAQ

- **Why pure NumPy instead of PyTorch?** The core algorithm is CPU-friendly and benefits from
  minimal dependencies; PyTorch is kept as an optional extra for heavyweight demos.
- **How do I control randomness?** Supply `seed` on `DiffusionConfig` or via
  `DEVOL_SEED` environment variable.
- **What if my fitness function raises an error?** It will be wrapped in a `FitnessComputationError`
  with the original exception chained for debugging.
- **Can I pause/resume runs?** Not yet. The roadmap includes checkpointing hooks.

## Roadmap

- Documentation site with API reference and notebooks.
- Pluggable checkpointing and resumable runs.
- Additional distance metrics (e.g. learnable embeddings).
- Hyperparameter sweeps and CLI-friendly configuration presets.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines and the development workflow.

## License

MIT © Devol Maintainers
