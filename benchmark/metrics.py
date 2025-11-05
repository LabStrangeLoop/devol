"""Metrics calculation for benchmark results."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class FitnessFunction(Protocol):
    """Protocol for fitness functions."""

    def __call__(self, x: NDArray) -> float:
        """Evaluate fitness for individual x."""
        ...


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    schedule_type: str
    population_size: int
    num_steps: int
    param_dim: int
    sigma_m: float
    seed: int

    # Performance metrics
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    distance_from_origin: float
    final_diversity: float
    runtime_seconds: float

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "schedule_type": self.schedule_type,
            "population_size": self.population_size,
            "num_steps": self.num_steps,
            "param_dim": self.param_dim,
            "sigma_m": self.sigma_m,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "distance_from_origin": self.distance_from_origin,
            "final_diversity": self.final_diversity,
            "runtime_seconds": self.runtime_seconds,
        }


def calculate_population_diversity(fitness: NDArray) -> float:
    """Calculate diversity as standard deviation of fitness values.

    Args:
        fitness: Array of fitness values

    Returns:
        Standard deviation of fitness
    """
    return float(np.std(fitness))


def evaluate_population_fitness(
    population: NDArray, fitness_fn: FitnessFunction
) -> NDArray:
    """Evaluate fitness for entire population.

    Args:
        population: Population array of shape (pop_size, param_dim)
        fitness_fn: Fitness function to evaluate

    Returns:
        Array of fitness values
    """
    return np.array([fitness_fn(ind) for ind in population])
