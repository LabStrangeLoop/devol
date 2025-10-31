"""Integration-style test covering the full Diffusion Evolution loop."""

import numpy as np

from devol.algorithm import DiffusionEvolution
from devol.config import DiffusionConfig


def two_peaks(individual: np.ndarray) -> float:
    """Two symmetric peaks around ±1 in each dimension."""
    peak1 = np.exp(-np.sum((individual - np.array([1.0, 1.0])) ** 2) / 0.05)
    peak2 = np.exp(-np.sum((individual - np.array([-1.0, -1.0])) ** 2) / 0.05)
    return float((peak1 + peak2) / 2)


def test_two_peaks_discovers_both_modes() -> None:
    config = DiffusionConfig(
        population_size=128,
        num_steps=60,
        param_dim=2,
        sigma_m=0.8,
        distance={"type": "latent", "latent_dim": 2},
        seed=321,
    )

    algo = DiffusionEvolution(config, two_peaks)
    final_population = algo.run()

    raw_fitness = np.array([two_peaks(individual) for individual in final_population])
    assert raw_fitness.max() > 0.45

    top_indices = np.argsort(raw_fitness)[-20:]
    top_solutions = final_population[top_indices]
    distances_to_peak_one = np.linalg.norm(top_solutions - np.array([1.0, 1.0]), axis=1)
    distances_to_peak_two = np.linalg.norm(top_solutions - np.array([-1.0, -1.0]), axis=1)

    assert np.any(distances_to_peak_one < 0.4)
    assert np.any(distances_to_peak_two < 0.4)
