"""Main Diffusion Evolution algorithm."""

from typing import Callable

import numpy as np

from devol.config import DiffusionConfig
from devol.distance import create_distance_computer
from devol.evolution import compute_epsilon_hat, estimate_x0, evolution_step
from devol.exceptions import EvolutionError, FitnessComputationError
from devol.fitness import create_fitness_mapper, preprocess_fitness
from devol.schedules import create_alpha_schedule, create_sigma_schedule
from devol.types import FloatArray


class DiffusionEvolution:
    def __init__(self, config: DiffusionConfig, fitness_fn: Callable[[FloatArray], float]) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.rng = np.random.default_rng(config.seed)

        self.alpha = create_alpha_schedule(
            config.schedule.type, config.num_steps, config.schedule.epsilon
        )
        self.sigma = create_sigma_schedule(self.alpha, config.sigma_m)

        self.distance_computer = create_distance_computer(
            config.distance.type,
            config.param_dim,
            config.distance.latent_dim,
            config.seed,
        )

        self.fitness_mapper, self.normalize, self.shift_negative = create_fitness_mapper(
            config.fitness.mapping,
            config.fitness.temperature,
            config.fitness.normalize,
            config.fitness.shift_negative,
        )

        self.population: FloatArray | None = None
        self.fitness_history: list[FloatArray] = []

    def initialize_population(self) -> FloatArray:
        self.population = self.rng.standard_normal(
            (self.config.population_size, self.config.param_dim)
        )
        return self.population

    def evaluate_fitness(self, population: FloatArray) -> FloatArray:
        try:
            raw_fitness = np.array([self.fitness_fn(individual) for individual in population])
        except Exception as exc:  # pragma: no cover - user fitness failures  # noqa: BLE001
            msg = "User-provided fitness function raised an exception."
            raise FitnessComputationError(msg) from exc
        return preprocess_fitness(raw_fitness, self.shift_negative, self.normalize)

    def step(self, t: int, population: FloatArray) -> FloatArray:
        fitness = self.evaluate_fitness(population)
        self.fitness_history.append(fitness)
        fitness_weights = self.fitness_mapper(fitness)

        alpha_t = self.alpha[t]
        alpha_t_minus_1 = self.alpha[t - 1]
        sigma_t = self.sigma[t]

        new_population = np.zeros_like(population, dtype=float)

        for idx, x_t in enumerate(population):
            x_hat_0 = estimate_x0(x_t, population, fitness_weights, alpha_t, self.distance_computer)
            epsilon_hat = compute_epsilon_hat(x_t, x_hat_0, alpha_t)
            new_population[idx] = evolution_step(
                x_t, x_hat_0, epsilon_hat, alpha_t, alpha_t_minus_1, sigma_t, self.rng
            )

        return new_population

    def run(self) -> FloatArray:
        try:
            population = self.initialize_population()

            for t in range(self.config.num_steps, 0, -1):
                population = self.step(t, population)
        except FitnessComputationError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected runtime  # noqa: BLE001
            msg = "Diffusion evolution failed to complete."
            raise EvolutionError(msg) from exc

        self.population = population
        return population

    def get_best_individual(self) -> tuple[FloatArray, float]:
        if self.population is None:
            raise EvolutionError("Algorithm has not been run yet.")
        fitness = self.evaluate_fitness(self.population)
        best_idx = np.argmax(fitness)
        return self.population[best_idx], fitness[best_idx]
