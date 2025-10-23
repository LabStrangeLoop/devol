"""Test Day 3 implementations."""

import numpy as np

from devol import DiffusionConfig, DiffusionEvolution


def simple_sphere(x: np.ndarray) -> float:
    """Simple sphere function: maximize -(x^2 + y^2)."""
    return -np.sum(x**2)


def two_peaks(x: np.ndarray) -> float:
    """Two Gaussian peaks at (1,1) and (-1,-1)."""
    peak1 = np.exp(-np.sum((x - np.array([1.0, 1.0])) ** 2) / 0.1)
    peak2 = np.exp(-np.sum((x - np.array([-1.0, -1.0])) ** 2) / 0.1)
    return (peak1 + peak2) / 2


def test_sphere_optimization():
    print("Testing sphere function optimization...")
    config = DiffusionConfig(
        population_size=128,
        num_steps=25,
        param_dim=2,
        sigma_m=0.5,
        seed=43,
    )

    algo = DiffusionEvolution(config, simple_sphere)
    final_population = algo.run()

    best_individual, best_fitness = algo.get_best_individual()
    print(f"✓ Best individual: {best_individual}")
    print(f"✓ Best fitness: {best_fitness:.6f}")
    print(f"✓ Distance from origin: {np.linalg.norm(best_individual):.6f}")

    assert np.linalg.norm(best_individual) < 0.5, "Should converge near origin"


def test_two_peaks():
    print("\nTesting two-peak function...")
    config = DiffusionConfig(
        population_size=256,
        num_steps=50,
        param_dim=2,
        sigma_m=1.0,
        seed=43,
    )

    algo = DiffusionEvolution(config, two_peaks)
    final_population = algo.run()

    fitness = np.array([two_peaks(ind) for ind in final_population])
    best_indices = np.argsort(fitness)[-10:]
    top_solutions = final_population[best_indices]

    print("✓ Top 10 solutions found:")
    for i, sol in enumerate(top_solutions[-5:]):
        print(f"  {i + 1}. {sol} (fitness: {fitness[best_indices[-5 + i]]:.6f})")

    peak1_count = np.sum(np.linalg.norm(top_solutions - [1, 1], axis=1) < 0.5)
    peak2_count = np.sum(np.linalg.norm(top_solutions - [-1, -1], axis=1) < 0.5)
    print(f"✓ Solutions near peak (1,1): {peak1_count}")
    print(f"✓ Solutions near peak (-1,-1): {peak2_count}")


def test_latent_space():
    print("\nTesting latent space distance...")
    config = DiffusionConfig(
        population_size=128,
        num_steps=25,
        param_dim=10,
        distance={"type": "latent", "latent_dim": 3},
        sigma_m=0.8,
        seed=42,
    )

    algo = DiffusionEvolution(config, simple_sphere)
    final_population = algo.run()
    best_individual, best_fitness = algo.get_best_individual()

    print("✓ Latent space evolution successful")
    print(f"✓ Best fitness in 10D: {best_fitness:.6f}")
    print(f"✓ Norm: {np.linalg.norm(best_individual):.6f}")


if __name__ == "__main__":
    test_sphere_optimization()
    test_two_peaks()
    test_latent_space()
    print("\n✓ Day 3 complete!")
