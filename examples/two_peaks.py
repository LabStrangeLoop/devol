"""Two-peak optimization example with visualization."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from devol import DiffusionConfig, DiffusionEvolution


def two_peaks_function(x: NDArray) -> float:
    """Two Gaussian peaks at (1,1) and (-1,-1)."""
    peak1 = np.exp(-np.sum((x - np.array([1.0, 1.0])) ** 2) / 0.1)
    peak2 = np.exp(-np.sum((x - np.array([-1.0, -1.0])) ** 2) / 0.1)
    return (peak1 + peak2) / 2


def run_two_peaks() -> None:
    config = DiffusionConfig(
        population_size=512,
        num_steps=50,
        param_dim=2,
        sigma_m=1.0,
        seed=42,
    )

    algo = DiffusionEvolution(config, two_peaks_function)
    final_population = algo.run()

    fitness_values = np.array([two_peaks_function(ind) for ind in final_population])
    top_indices = np.argsort(fitness_values)[-20:]
    top_solutions = final_population[top_indices]

    print("Top 20 solutions:")
    for i, (sol, fit) in enumerate(zip(top_solutions, fitness_values[top_indices])):
        print(f"{i + 1:2d}. x={sol[0]:6.3f}, y={sol[1]:6.3f}, fitness={fit:.6f}")

    peak1 = np.array([1.0, 1.0])
    peak2 = np.array([-1.0, -1.0])
    near_peak1 = np.sum(np.linalg.norm(top_solutions - peak1, axis=1) < 0.5)
    near_peak2 = np.sum(np.linalg.norm(top_solutions - peak2, axis=1) < 0.5)

    print("\nDiversity analysis:")
    print(f"  Solutions near peak (1,1): {near_peak1}")
    print(f"  Solutions near peak (-1,-1): {near_peak2}")

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = two_peaks_function(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
    plt.colorbar(label="Fitness")
    plt.scatter(final_population[:, 0], final_population[:, 1], c="red", s=10, alpha=0.5)
    plt.scatter(top_solutions[:, 0], top_solutions[:, 1], c="white", s=50, edgecolors="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Two Peaks: Final Population Distribution")
    plt.savefig("two_peaks_result.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to two_peaks_result.png")


if __name__ == "__main__":
    run_two_peaks()
