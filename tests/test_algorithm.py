"""Tests for the DiffusionEvolution high-level workflow."""

import numpy as np

from devol.algorithm import DiffusionEvolution
from devol.config import DiffusionConfig


def sphere_fitness(individual: np.ndarray) -> float:
    """Simple smooth fitness surface with maximum at the origin."""
    return float(-np.sum(individual ** 2))


def test_diffusion_evolution_runs_and_records_history() -> None:
    config = DiffusionConfig(
        population_size=16,
        num_steps=5,
        param_dim=3,
        sigma_m=0.8,
        seed=7,
    )

    algo = DiffusionEvolution(config, sphere_fitness)
    final_population = algo.run()

    assert final_population.shape == (config.population_size, config.param_dim)
    assert len(algo.fitness_history) == config.num_steps

    best_individual, best_fitness = algo.get_best_individual()
    assert best_individual.shape == (config.param_dim,)
    assert best_fitness <= 0  # sphere fitness is non-positive
