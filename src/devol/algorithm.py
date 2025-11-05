"""Main Diffusion Evolution algorithm."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from devol.config import DiffusionConfig
from devol.distance import create_distance_computer
from devol.evolution import compute_epsilon_hat, estimate_x0, evolution_step
from devol.fitness import create_fitness_mapper
from devol.schedules import create_alpha_schedule, create_sigma_schedule


class DiffusionEvolution:
    def __init__(self, config: DiffusionConfig, fitness_fn: Callable[[NDArray], float]) -> None:
        self.config = config
        self.fitness_fn = fitness_fn
        self.rng = np.random.default_rng(config.seed)

        self.alpha = create_alpha_schedule(
            config.schedule.type.value, config.num_steps, config.schedule.epsilon
        )
        self.sigma = create_sigma_schedule(self.alpha, config.sigma_m)

        self.distance_computer = create_distance_computer(
            config.distance.type.value,
            config.param_dim,
            config.distance.latent_dim,
            config.seed,
        )

        self.fitness_mapper = create_fitness_mapper(
            config.fitness.mapping.value,
            config.fitness.temperature,
        )

        self.population: NDArray | None = None

    # TODO: Is this init optimal? do we want to abstract it?
    # Make it a docstring
    # Explain how the noising op is shifting the original pdf to a ~N(0, 1)
    def initialize_population(self) -> NDArray:  # TODO: maybe make it of type Population
        self.population = self.rng.standard_normal(
            (self.config.population_size, self.config.param_dim)
        )
        return self.population

    def evaluate_fitness(self, population: NDArray) -> NDArray:
        return np.array([self.fitness_fn(ind) for ind in population])

    def step(self, timestamp: int, population: NDArray) -> NDArray:
        fitness = self.evaluate_fitness(population)
        fitness_weights = self.fitness_mapper(fitness)

        alpha_t = self.alpha[timestamp]
        alpha_t_minus_1 = self.alpha[timestamp - 1]
        sigma_t = self.sigma[timestamp]

        new_population = np.zeros_like(population)

        for i in range(len(population)):
            x_t = population[i]
            x_hat_0 = estimate_x0(x_t, population, fitness_weights, alpha_t, self.distance_computer)
            epsilon_hat = compute_epsilon_hat(x_t, x_hat_0, alpha_t)
            new_population[i] = evolution_step(
                x_t, x_hat_0, epsilon_hat, alpha_t, alpha_t_minus_1, sigma_t, self.rng
            )

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
