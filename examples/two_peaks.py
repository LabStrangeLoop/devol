"""Two-peak optimization example with optional visualization."""

from __future__ import annotations

import argparse
import numpy as np

from devol import DiffusionConfig, DiffusionEvolution
from devol.types import FloatArray


def two_peaks_function(x: FloatArray) -> float:
    """Two Gaussian peaks at (1,1) and (-1,-1)."""
    peak1 = np.exp(-np.sum((x - np.array([1.0, 1.0])) ** 2) / 0.1)
    peak2 = np.exp(-np.sum((x - np.array([-1.0, -1.0])) ** 2) / 0.1)
    return (peak1 + peak2) / 2


def render_contour(
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    grid: np.ndarray,
    population: np.ndarray,
    top_solutions: np.ndarray,
    show: bool,
) -> None:
    """Render and optionally show the contour plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the examples extra: uv pip install 'devol[examples]'") from exc

    plt.figure(figsize=(10, 8))
    plt.contourf(mesh_x, mesh_y, grid, levels=20, cmap="viridis", alpha=0.6)
    plt.colorbar(label="Fitness")
    plt.scatter(population[:, 0], population[:, 1], c="red", s=10, alpha=0.5)
    plt.scatter(top_solutions[:, 0], top_solutions[:, 1], c="white", s=50, edgecolors="black")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Two Peaks: Final Population Distribution")
    plt.savefig("two_peaks_result.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to two_peaks_result.png")

    if show:
        plt.show()
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion Evolution two-peaks demo")
    parser.add_argument("--population-size", type=int, default=512, help="Population size")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of evolution steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib visualization")
    return parser.parse_args(argv)


def run_two_peaks(args: argparse.Namespace | None = None) -> None:
    parsed = args or parse_args()
    config = DiffusionConfig(
        population_size=parsed.population_size,
        num_steps=parsed.num_steps,
        param_dim=2,
        sigma_m=1.0,
        seed=parsed.seed,
    )

    algo = DiffusionEvolution(config, two_peaks_function)
    final_population = algo.run()

    fitness_values = np.array([two_peaks_function(ind) for ind in final_population])
    top_indices = np.argsort(fitness_values)[-20:]
    top_solutions = final_population[top_indices]

    print("Top 20 solutions:")
    for i, (sol, fit) in enumerate(zip(top_solutions, fitness_values[top_indices], strict=False)):
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
    mesh_x, mesh_y = np.meshgrid(x, y)
    grid = np.zeros_like(mesh_x)

    for i, row in enumerate(mesh_x):
        for j, _ in enumerate(row):
            grid[i, j] = two_peaks_function(np.array([mesh_x[i, j], mesh_y[i, j]]))

    if not parsed.no_plot:
        render_contour(mesh_x, mesh_y, grid, final_population, top_solutions, show=True)


if __name__ == "__main__":
    run_two_peaks()
