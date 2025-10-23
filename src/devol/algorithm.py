"""Main Diffusion Evolution algorithm."""

from numpy.typing import NDArray
from typing import Callable
import numpy as np

from devol.config import DiffusionConfig
from devol.distance import create_distance_computer
from devol.evolution import compute_epsilon_hat, estimate_x0, evolution_step
from devol.fitness import create_fitness_mapper, preprocess_fitness
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

        self.fitness_mapper, self.normalize, self.shift_negative = create_fitness_mapper(
            config.fitness.mapping.value,
            config.fitness.temperature,
            config.fitness.normalize,
            config.fitness.shift_negative,
        )

        self.population: NDArray | None = None
        self.fitness_history: list[NDArray] = []

    def initialize_population(self) -> NDArray:
        self.population = self.rng.standard_normal(
            (self.config.population_size, self.config.param_dim)
        )
        return self.population

    def evaluate_fitness(self, population: NDArray) -> NDArray:
        fitness = np.array([self.fitness_fn(ind) for ind in population])
        return preprocess_fitness(fitness, self.shift_negative, self.normalize)

    def step(self, t: int, population: NDArray) -> NDArray:
        fitness = self.evaluate_fitness(population)
        self.fitness_history.append(fitness)
        fitness_weights = self.fitness_mapper(fitness)

        alpha_t = self.alpha[t]
        alpha_t_minus_1 = self.alpha[t - 1]
        sigma_t = self.sigma[t]

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

        for t in range(self.config.num_steps, 0, -1):
            population = self.step(t, population)

        self.population = population
        return population

    def get_best_individual(self) -> tuple[NDArray, float]:
        if self.population is None:
            raise ValueError("Algorithm has not been run yet")
        fitness = self.evaluate_fitness(self.population)
        best_idx = np.argmax(fitness)
        return self.population[best_idx], fitness[best_idx]
