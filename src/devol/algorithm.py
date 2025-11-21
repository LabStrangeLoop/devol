"""Main Diffusion Evolution algorithm."""

from collections.abc import Callable
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from devol.config import DiffusionConfig
from devol.distance import create_distance_computer
from devol.evolution import compute_epsilon_hat, estimate_x0, evolution_step
from devol.fitness import (
    DirectMapper,
    ExponentialMapper,
    create_fitness_mapper,
    create_fitness_normalizer,
)
from devol.schedules import create_alpha_schedule, create_sigma_schedule


class DiffusionEvolution:
    def __init__(self, config: DiffusionConfig, fitness_fn: Callable[[NDArray], float]) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.rng = np.random.default_rng(config.seed)

        self.alpha = create_alpha_schedule(config.schedule.type.value, config.num_steps, config.schedule.epsilon)
        self.sigma = create_sigma_schedule(self.alpha, config.sigma_m)

        self.distance_computer = create_distance_computer(
            config.distance.type.value,
            config.param_dim,
            config.distance.latent_dim,
            config.seed,
        )

        self.fitness_normalizer = create_fitness_normalizer(config.fitness.normalize)
        self.fitness_mapper = create_fitness_mapper(
            config.fitness.mapping,
            config.fitness.temperature,
        )

        debug_flag = os.environ.get("DEVOL_DEBUG_FITNESS", "1").lower()
        self._visualize_mappings = debug_flag not in {"0", "false", "off"}
        self._debug_output_dir = Path(os.environ.get("DEVOL_FITNESS_DEBUG_DIR", "fitness_landscapes"))
        self._visualization_bootstrapped = False
        self._visualization_setup_error_reported = False
        self._matplotlib_warning_issued = False
        self._direct_mapper = DirectMapper()
        self._exponential_mapper = ExponentialMapper(config.fitness.temperature)

        self.population: NDArray | None = None

    # TODO: Is this init optimal? do we want to abstract it?
    # Make it a docstring
    # Explain how the noising op is shifting the original pdf to a ~N(0, 1)
    def initialize_population(self) -> NDArray:  # TODO: maybe make it of type Population
        self.population = self.rng.standard_normal((self.config.population_size, self.config.param_dim))
        return self.population

    def evaluate_fitness(self, population: NDArray) -> NDArray:
        return np.array([self.fitness_fn(ind) for ind in population])

    def _prepare_visualization_output(self) -> bool:
        if self._visualization_bootstrapped:
            return True

        try:
            self._debug_output_dir.mkdir(parents=True, exist_ok=True)
            for existing in self._debug_output_dir.glob("*.png"):
                existing.unlink()
        except OSError as exc:
            if not self._visualization_setup_error_reported:
                print(f"Could not prepare fitness visualization directory '{self._debug_output_dir}': {exc}")
                self._visualization_setup_error_reported = True
            self._visualize_mappings = False
            return False

        print(f"Saving fitness mapping plots to '{self._debug_output_dir}'.")
        self._visualization_bootstrapped = True
        return True

    def _maybe_save_fitness_landscape(
        self,
        timestamp: int,
        fitness: NDArray,
        direct_weights: NDArray,
        exponential_weights: NDArray,
    ) -> None:
        if not self._visualize_mappings:
            return

        if not self._prepare_visualization_output():
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            if not self._matplotlib_warning_issued:
                print("Matplotlib not available; skipping fitness mapping visualization.")
                self._matplotlib_warning_issued = True
            self._visualize_mappings = False
            return

        step_index = self.config.num_steps - timestamp + 1
        order = np.argsort(fitness)[::-1]
        ranks = np.arange(1, len(fitness) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        axes[0].plot(ranks, direct_weights[order], label="Direct weights", color="tab:blue")
        axes[0].set_title("Direct mapping")
        axes[0].set_xlabel("Ranked individual")
        axes[0].set_ylabel("Raw value")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            ranks,
            exponential_weights[order],
            label="Exponential weights",
            color="tab:orange",
        )
        axes[1].plot(
            ranks,
            fitness[order],
            label="Raw fitness (for reference)",
            color="tab:gray",
            linestyle="--",
            alpha=0.7,
        )
        axes[1].set_title("Exponential mapping")
        axes[1].set_xlabel("Ranked individual")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(
            f"Fitness landscape comparison - Step {step_index}/{self.config.num_steps}",
            fontsize=12,
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        filename = self._debug_output_dir / f"fitness_landscape_step_{step_index:03d}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def step(self, timestamp: int, population: NDArray) -> NDArray:
        fitness = self.evaluate_fitness(population)
        normalized_fitness = self.fitness_normalizer(fitness)

        if self._visualize_mappings:
            direct_weights = self._direct_mapper(normalized_fitness)
            exponential_weights = self._exponential_mapper(normalized_fitness)
            self._maybe_save_fitness_landscape(timestamp, normalized_fitness, direct_weights, exponential_weights)

        fitness_weights = self.fitness_mapper(normalized_fitness)

        alpha_t = self.alpha[timestamp]
        alpha_t_minus_1 = self.alpha[timestamp - 1]
        sigma_t = self.sigma[timestamp]

        new_population = np.zeros_like(population)

        for i in range(len(population)):
            x_t = population[i]
            x_hat_0 = estimate_x0(x_t, population, fitness_weights, alpha_t, self.distance_computer)
            epsilon_hat = compute_epsilon_hat(x_t, x_hat_0, alpha_t)
            new_population[i] = evolution_step(x_t, x_hat_0, epsilon_hat, alpha_t, alpha_t_minus_1, sigma_t, self.rng)

        return new_population

    def run(self) -> NDArray:
        population = self.initialize_population()

        for timestamp in range(self.config.num_steps, 0, -1):
            population = self.step(timestamp, population)

        self.population = population
        return population

    def get_best_individual(self) -> tuple[NDArray, float]:
        if self.population is None:
            raise ValueError("Algorithm has not been run yet")
        fitness = self.evaluate_fitness(self.population)
        best_idx = np.argmax(fitness)
        return self.population[best_idx], fitness[best_idx]
