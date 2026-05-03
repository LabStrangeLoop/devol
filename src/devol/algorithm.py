"""Main Diffusion Evolution algorithm."""

from collections.abc import Callable

import numpy as np

from devol.config import DiffusionConfig
from devol.distance import FloatArray, create_distance_computer
from devol.evolution import compute_epsilon_hat, estimate_x0, evolution_step
from devol.fitness import create_fitness_mapper, create_fitness_normalizer
from devol.schedules import create_alpha_schedule, create_sigma_schedule


class DiffusionEvolution:
    def __init__(self, config: DiffusionConfig, fitness_fn: Callable[[FloatArray], float]) -> None:
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

        self.population: FloatArray | None = None

    def initialize_population(self) -> FloatArray:
        self.population = self.rng.standard_normal((self.config.population_size, self.config.param_dim))
        return self.population

    def evaluate_fitness(self, population: FloatArray) -> FloatArray:
        return np.array([self.fitness_fn(ind) for ind in population])

    def step(self, timestamp: int, population: FloatArray) -> FloatArray:
        fitness = self.evaluate_fitness(population)
        normalized_fitness = self.fitness_normalizer(fitness)

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

    def run(self, initial_population: FloatArray | None = None) -> FloatArray:
        population = initial_population
        if population is None:
            population = self.initialize_population()

        for timestamp in range(self.config.num_steps, 0, -1):
            population = self.step(timestamp, population)

        self.population = population
        return population

    def get_best_individual(self) -> tuple[FloatArray, float]:
        if self.population is None:
            raise ValueError("Algorithm has not been run yet")
        fitness = self.evaluate_fitness(self.population)
        best_idx = int(np.argmax(fitness))
        return self.population[best_idx], float(fitness[best_idx])
