"""Command-line interface for Devol."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic_yaml import parse_yaml_file_as

from devol.algorithm import DiffusionEvolution
from devol.config import DiffusionConfig
from devol.distance import FloatArray


def load_config(config_path: str) -> DiffusionConfig:
    """Load configuration from YAML file."""
    return parse_yaml_file_as(DiffusionConfig, Path(config_path))


def sphere_function(x: FloatArray) -> float:
    """Sphere function: maximize -(x^2)."""
    return float(-np.sum(x**2))


def rosenbrock_function(x: FloatArray) -> float:
    """Rosenbrock function (minimization converted to maximization)."""
    result = 0.0
    for i in range(len(x) - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return -result


BUILTIN_FUNCTIONS: dict[str, Callable[[FloatArray], float]] = {
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
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    fitness_fn = BUILTIN_FUNCTIONS[args.function]

    if args.verbose:
        print(f"Running Devol with {config.population_size} individuals for {config.num_steps} steps")

    algo = DiffusionEvolution(config, fitness_fn)
    final_population = algo.run()

    best_individual, best_fitness = algo.get_best_individual()

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
