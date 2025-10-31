"""Command-line interface for Devol."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError
from pydantic_yaml import parse_yaml_file_as

from devol.algorithm import DiffusionEvolution
from devol.config import DiffusionConfig
from devol.exceptions import ConfigurationError, EvolutionError, FitnessComputationError


def load_config(config_path: str) -> DiffusionConfig:
    """Load configuration from YAML file."""
    try:
        return parse_yaml_file_as(DiffusionConfig, Path(config_path))
    except (OSError, ValidationError, ValueError) as exc:
        msg = f"Failed to load configuration from {config_path!r}"
        raise ConfigurationError(msg) from exc


def sphere_function(x: np.ndarray) -> float:
    """Sphere function: maximize -(x^2)."""
    return -np.sum(x**2)


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function (minimization converted to maximization)."""
    result = 0.0
    for i in range(len(x) - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return -result


BUILTIN_FUNCTIONS: dict[str, Any] = {
    "sphere": sphere_function,
    "rosenbrock": rosenbrock_function,
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion Evolution Algorithm")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--function",
        type=str,
        choices=list(BUILTIN_FUNCTIONS.keys()),
        default="sphere",
        help="Built-in fitness function to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigurationError as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    fitness_fn = BUILTIN_FUNCTIONS[args.function]

    if args.verbose:
        print(f"Running Devol with {config.population_size} individuals for {config.num_steps} steps")

    algo = DiffusionEvolution(config, fitness_fn)
    try:
        final_population = algo.run()
        best_individual, best_fitness = algo.get_best_individual()
    except FitnessComputationError as exc:
        print(f"Fitness computation failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except EvolutionError as exc:
        print(f"Evolution failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nBest individual: {best_individual}")
    print(f"Best fitness: {best_fitness:.6f}")

    if args.verbose:
        fitness_values = np.array([fitness_fn(ind) for ind in final_population])
        print("\nPopulation statistics:")
        print(f"  Mean fitness: {np.mean(fitness_values):.6f}")
        print(f"  Std fitness: {np.std(fitness_values):.6f}")
        print(f"  Min fitness: {np.min(fitness_values):.6f}")
        print(f"  Max fitness: {np.max(fitness_values):.6f}")


if __name__ == "__main__":
    main()
