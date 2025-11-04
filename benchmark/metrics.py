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
    convergence_step: int | None
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
            "convergence_step": self.convergence_step,
            "final_diversity": self.final_diversity,
            "runtime_seconds": self.runtime_seconds,
        }


def calculate_convergence_step(
    fitness_history: list[NDArray], target_fraction: float = 0.9
) -> int | None:
    """Calculate step when fitness reached target fraction of final best fitness.

    Args:
        fitness_history: List of fitness arrays for each step
        target_fraction: Fraction of final fitness to consider converged

    Returns:
        Step number when converged, or None if never converged
    """
    if not fitness_history:
        return None

    final_best = np.max(fitness_history[-1])
    target = target_fraction * final_best

    for step, fitness in enumerate(reversed(fitness_history)):
        if np.max(fitness) >= target:
            return len(fitness_history) - step

    return None


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
