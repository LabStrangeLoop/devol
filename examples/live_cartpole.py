"""CartPole reinforcement learning with live visualization."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from devol import DiffusionConfig, DiffusionEvolution
from devol.evolution import compute_epsilon_hat, estimate_x0, evolution_step
from devol.fitness import preprocess_fitness

from examples.cartpole import evaluate_cartpole


class LiveCartPoleVisualizer:
    """Live visualizer showing fitness trends and 2D population embedding."""

    def __init__(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
        except ImportError as exc:  # pragma: no cover - visualization dependency
            raise ImportError("Install matplotlib: uv pip install matplotlib") from exc

        self.plt = plt
        self.Normalize = Normalize
        self.ScalarMappable = ScalarMappable

        self.populations: list[NDArray] = []
        self.embedded_populations: list[NDArray] = []
        self.fitness_arrays: list[NDArray] = []
        self.fitness_stats: list[dict[str, float]] = []

        self.fig, (self.ax_embed, self.ax_fitness) = plt.subplots(1, 2, figsize=(15, 6))

        self.ax_embed.set_title("Population PCA Projection")
        self.ax_embed.set_xlabel("PC 1")
        self.ax_embed.set_ylabel("PC 2")
        self.ax_embed.grid(True, alpha=0.3)

        self.norm = self.Normalize(vmin=0.0, vmax=1.0)
        self.color_mappable = self.ScalarMappable(norm=self.norm, cmap="viridis")
        self.scatter = self.ax_embed.scatter([], [], c=[], cmap="viridis", norm=self.norm, s=30, alpha=0.7)
        self.best_scatter = self.ax_embed.scatter(
            [], [], c="red", s=80, edgecolors="black", linewidths=1.5, label="Best Individual"
        )
        self.colorbar = self.fig.colorbar(self.color_mappable, ax=self.ax_embed)
        self.colorbar.set_label("Average Episode Reward")
        self.ax_embed.legend(loc="upper right")

        self.ax_fitness.set_title("Fitness Progress")
        self.ax_fitness.set_xlabel("Generation")
        self.ax_fitness.set_ylabel("Average Episode Reward")
        self.ax_fitness.grid(True, alpha=0.3)
        (self.line_mean,) = self.ax_fitness.plot([], [], "b-", label="Mean", linewidth=2)
        (self.line_max,) = self.ax_fitness.plot([], [], "g-", label="Max", linewidth=2)
        (self.line_min,) = self.ax_fitness.plot([], [], "r-", label="Min", linewidth=2)
        self.ax_fitness.legend()

        plt.tight_layout()

    def record_step(self, population: NDArray, fitness: NDArray) -> None:
        """Store population snapshot and fitness values for animation."""
        self.populations.append(population.copy())
        self.fitness_arrays.append(fitness.copy())
        self.fitness_stats.append(
            {
                "mean": float(np.mean(fitness)),
                "max": float(np.max(fitness)),
                "min": float(np.min(fitness)),
            }
        )

    def prepare_embeddings(self) -> None:
        """Compute a consistent 2D projection for all recorded populations."""
        if not self.populations:
            self.embedded_populations = []
            return

        try:
            all_data = np.vstack(self.populations)
        except ValueError:
            # Unequal shapes should never happen, but fall back to zeros if it does.
            self.embedded_populations = [np.zeros((pop.shape[0], 2)) for pop in self.populations]
            return

        mean = np.mean(all_data, axis=0, keepdims=True)
        centered = all_data - mean

        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            self.embedded_populations = [np.zeros((pop.shape[0], 2)) for pop in self.populations]
            return

        components = vt[:2].T
        if components.shape[1] < 2:
            components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])), mode="constant")

        embeddings: list[NDArray] = []
        for pop in self.populations:
            coords = (pop - mean) @ components
            if coords.shape[1] < 2:
                coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
            embeddings.append(coords)

        self.embedded_populations = embeddings

    def animate(self, frame: int):
        """Matplotlib animation callback."""
        if frame >= len(self.populations):
            return self.scatter, self.best_scatter, self.line_mean, self.line_max, self.line_min

        population = self.populations[frame]
        embedding = self.embedded_populations[frame] if self.embedded_populations else np.zeros((len(population), 2))
        fitness = self.fitness_arrays[frame]

        if embedding.size:
            self.scatter.set_offsets(embedding)
            self.scatter.set_array(fitness)

            vmin = float(np.min(fitness))
            vmax = float(np.max(fitness))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-3
            self.norm.vmin = vmin
            self.norm.vmax = vmax
            self.color_mappable.set_clim(vmin, vmax)
            self.colorbar.update_normal(self.color_mappable)

            best_idx = int(np.argmax(fitness))
            self.best_scatter.set_offsets(embedding[best_idx : best_idx + 1])

            x_min, x_max = float(np.min(embedding[:, 0])), float(np.max(embedding[:, 0]))
            y_min, y_max = float(np.min(embedding[:, 1])), float(np.max(embedding[:, 1]))
            if np.isclose(x_min, x_max):
                x_min -= 1.0
                x_max += 1.0
            if np.isclose(y_min, y_max):
                y_min -= 1.0
                y_max += 1.0
            margin_x = 0.05 * (x_max - x_min)
            margin_y = 0.05 * (y_max - y_min)
            self.ax_embed.set_xlim(x_min - margin_x, x_max + margin_x)
            self.ax_embed.set_ylim(y_min - margin_y, y_max + margin_y)

        self.ax_embed.set_title(f"Population PCA Projection (Gen {frame + 1}/{len(self.populations)})")

        generations = list(range(frame + 1))
        means = [stats["mean"] for stats in self.fitness_stats[: frame + 1]]
        maxs = [stats["max"] for stats in self.fitness_stats[: frame + 1]]
        mins = [stats["min"] for stats in self.fitness_stats[: frame + 1]]

        self.line_mean.set_data(generations, means)
        self.line_max.set_data(generations, maxs)
        self.line_min.set_data(generations, mins)

        self.ax_fitness.relim()
        self.ax_fitness.autoscale_view()

        return self.scatter, self.best_scatter, self.line_mean, self.line_max, self.line_min

    def show(self, interval: int = 200, save_path: str | None = None) -> None:
        """Render the animation."""
        from matplotlib.animation import FuncAnimation

        self.prepare_embeddings()

        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.populations),
            interval=interval,
            blit=True,
            repeat=True,
        )

        target_path = save_path or "cartpole_evolution.gif"
        anim.save(target_path, writer="pillow", fps=5)
        print(f"Animation saved to {target_path}")

        backend = (self.plt.get_backend() or "").lower()
        interactive = not backend.endswith("agg") and not backend.endswith("pdf") and not backend.endswith("svg")
        if interactive:
            self.plt.show()
        else:
            print("Matplotlib backend is non-interactive; skipping live window.")
        self.plt.close(self.fig)


class MonitoredCartPoleEvolution(DiffusionEvolution):
    """Diffusion evolution variant that records progress for visualization."""

    def __init__(self, config: DiffusionConfig, fitness_fn: Callable[[NDArray], float], visualizer: LiveCartPoleVisualizer):
        super().__init__(config, fitness_fn)
        self.visualizer = visualizer

    def step(self, t: int, population: NDArray) -> NDArray:
        raw_fitness = np.array([self.fitness_fn(individual) for individual in population])
        processed_fitness = preprocess_fitness(raw_fitness, self.shift_negative, self.normalize)
        self.fitness_history.append(processed_fitness)
        fitness_weights = self.fitness_mapper(processed_fitness)

        alpha_t = self.alpha[t]
        alpha_t_minus_1 = self.alpha[t - 1]
        sigma_t = self.sigma[t]

        new_population = np.zeros_like(population)
        for i, x_t in enumerate(population):
            x_hat_0 = estimate_x0(x_t, population, fitness_weights, alpha_t, self.distance_computer)
            epsilon_hat = compute_epsilon_hat(x_t, x_hat_0, alpha_t)
            new_population[i] = evolution_step(
                x_t,
                x_hat_0,
                epsilon_hat,
                alpha_t,
                alpha_t_minus_1,
                sigma_t,
                self.rng,
            )

        self.visualizer.record_step(population, raw_fitness)
        return new_population


def run_cartpole_live() -> None:
    """Execute CartPole optimization with live monitoring."""
    input_size, hidden_size, output_size = 4, 8, 2
    param_dim = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size

    print(f"CartPole neural network: {param_dim} parameters")

    config = DiffusionConfig(
        population_size=128,
        num_steps=25,
        param_dim=param_dim,
        sigma_m=1.0,
        distance={"type": "latent", "latent_dim": 5},
        seed=42,
    )

    # Fewer episodes per evaluation keeps the live demo responsive.
    fitness_fn: Callable[[NDArray], float] = lambda params: evaluate_cartpole(params, num_episodes=1)

    try:
        visualizer = LiveCartPoleVisualizer()
    except ImportError as exc:  # pragma: no cover - visualization dependency
        print(f"Visualization not available: {exc}")
        print("Running without visualization...")
        algo = DiffusionEvolution(config, fitness_fn)
        algo.run()
        return

    print("Running evolution with live tracking...")
    algo = MonitoredCartPoleEvolution(config, fitness_fn, visualizer)
    final_population = algo.run()

    final_fitness = np.array([fitness_fn(individual) for individual in final_population])
    visualizer.record_step(final_population, final_fitness)

    best_idx = int(np.argmax(final_fitness))
    best_individual = final_population[best_idx]
    best_fitness = float(final_fitness[best_idx])

    print(f"\nBest training fitness (avg reward over 1 episode): {best_fitness:.2f}")
    print("Evaluating best controller over 10 additional episodes...")

    test_rewards = [evaluate_cartpole(best_individual, num_episodes=1) for _ in range(10)]
    test_reward = float(np.mean(test_rewards))
    print(f"Average test reward: {test_reward:.2f}")

    if test_reward >= 475:
        print("✓ Task solved! (reward >= 475)")
    else:
        print("Task not yet solved. Try more steps or larger population.")

    print("\nRendering animation (also saved to 'cartpole_evolution.gif')...")
    visualizer.show(interval=200, save_path="cartpole_evolution.gif")


if __name__ == "__main__":
    run_cartpole_live()
