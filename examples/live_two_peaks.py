"""Two-peak optimization example with live visualization."""

import numpy as np
from numpy.typing import NDArray

from devol import DiffusionConfig, DiffusionEvolution


def two_peaks_function(x: NDArray) -> float:
    """Two Gaussian peaks at (1,1) and (-1,-1)."""
    peak1 = np.exp(-np.sum((x - np.array([1.0, 1.0])) ** 2) / 0.1)
    peak2 = np.exp(-np.sum((x - np.array([-1.0, -1.0])) ** 2) / 0.1)
    return (peak1 + peak2) / 2


class LiveEvolutionVisualizer:
    """Visualizer that shows evolution in real-time."""

    def __init__(self, fitness_fn, bounds=(-2, 2)):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except ImportError:
            raise ImportError("Install matplotlib: uv pip install matplotlib")

        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.populations = []
        self.fitness_histories = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.fitness_fn(np.array([X[i, j], Y[i, j]]))

        self.ax1.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("y")
        self.ax1.set_title("Population Evolution")

        self.scatter = self.ax1.scatter([], [], c="red", s=20, alpha=0.6)
        self.top_scatter = self.ax1.scatter([], [], c="white", s=80, edgecolors="black", linewidths=2)

        self.ax2.set_xlabel("Generation")
        self.ax2.set_ylabel("Fitness")
        self.ax2.set_title("Fitness Progress")
        self.ax2.grid(True, alpha=0.3)
        (self.line_mean,) = self.ax2.plot([], [], "b-", label="Mean", linewidth=2)
        (self.line_max,) = self.ax2.plot([], [], "g-", label="Max", linewidth=2)
        (self.line_min,) = self.ax2.plot([], [], "r-", label="Min", linewidth=2)
        self.ax2.legend()

        plt.tight_layout()

    def record_step(self, population: NDArray, fitness: NDArray) -> None:
        """Record population state at current step."""
        self.populations.append(population.copy())
        self.fitness_histories.append(
            {
                "mean": np.mean(fitness),
                "max": np.max(fitness),
                "min": np.min(fitness),
            }
        )

    def animate(self, frame: int):
        """Animation update function."""
        if frame >= len(self.populations):
            return self.scatter, self.top_scatter, self.line_mean, self.line_max, self.line_min

        population = self.populations[frame]
        fitness = np.array([self.fitness_fn(ind) for ind in population])
        top_indices = np.argsort(fitness)[-20:]

        self.scatter.set_offsets(population)
        self.top_scatter.set_offsets(population[top_indices])

        self.ax1.set_title(f"Population Evolution (Step {frame + 1}/{len(self.populations)})")

        generations = list(range(frame + 1))
        means = [h["mean"] for h in self.fitness_histories[: frame + 1]]
        maxs = [h["max"] for h in self.fitness_histories[: frame + 1]]
        mins = [h["min"] for h in self.fitness_histories[: frame + 1]]

        self.line_mean.set_data(generations, means)
        self.line_max.set_data(generations, maxs)
        self.line_min.set_data(generations, mins)

        self.ax2.relim()
        self.ax2.autoscale_view()

        return self.scatter, self.top_scatter, self.line_mean, self.line_max, self.line_min

    def show(self, interval: int = 200, save_path: str | None = None) -> None:
        """Show the animation."""
        from matplotlib.animation import FuncAnimation

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.populations),
            interval=interval,
            blit=True,
            repeat=True,
        )

        if save_path:
            anim.save(save_path, writer="pillow", fps=5)
            print(f"Animation saved to {save_path}")

        import matplotlib.pyplot as plt

        plt.show()


def run_two_peaks_live() -> None:
    """Run two-peak optimization with live visualization."""
    config = DiffusionConfig(
        population_size=512,
        num_steps=50,
        param_dim=2,
        sigma_m=1.0,
        seed=42,
    )

    try:
        visualizer = LiveEvolutionVisualizer(two_peaks_function)
    except ImportError as e:
        print(f"Visualization not available: {e}")
        print("Running without visualization...")
        algo = DiffusionEvolution(config, two_peaks_function)
        algo.run()
        return

    class MonitoredEvolution(DiffusionEvolution):
        def __init__(self, config, fitness_fn, visualizer):
            super().__init__(config, fitness_fn)
            self.visualizer = visualizer

        def step(self, t: int, population: NDArray) -> NDArray:
            result = super().step(t, population)
            fitness = self.evaluate_fitness(population)
            self.visualizer.record_step(population, fitness)
            return result

    print("Running evolution with live tracking...")
    algo = MonitoredEvolution(config, two_peaks_function, visualizer)
    final_population = algo.run()

    final_fitness = np.array([two_peaks_function(ind) for ind in final_population])
    visualizer.record_step(final_population, final_fitness)

    best_individual, best_fitness = algo.get_best_individual()
    top_indices = np.argsort(final_fitness)[-20:]
    top_solutions = final_population[top_indices]

    peak1 = np.array([1.0, 1.0])
    peak2 = np.array([-1.0, -1.0])
    near_peak1 = np.sum(np.linalg.norm(top_solutions - peak1, axis=1) < 0.5)
    near_peak2 = np.sum(np.linalg.norm(top_solutions - peak2, axis=1) < 0.5)

    print("\nEvolution complete!")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Solutions near peak (1,1): {near_peak1}")
    print(f"Solutions near peak (-1,-1): {near_peak2}")
    print("\nShowing animation... (close window to exit)")

    visualizer.show(interval=200, save_path="two_peaks_evolution.gif")


if __name__ == "__main__":
    run_two_peaks_live()
