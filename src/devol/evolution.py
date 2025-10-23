"""Core evolution step logic."""

import numpy as np
from numpy.typing import NDArray

from devol.distance import DistanceComputer
from devol.fitness import FitnessMapper


def compute_gaussian_weights(
    distances_squared: NDArray, alpha_t: float, epsilon: float = 1e-10
) -> NDArray:
    variance = 1 - alpha_t
    if variance < epsilon:
        return np.ones_like(distances_squared)
    exp_term = -distances_squared / (2 * variance)
    return np.exp(np.clip(exp_term, -50, 50))


def estimate_x0(
    x_t: NDArray,
    population: NDArray,
    fitness_weights: NDArray,
    alpha_t: float,
    distance_computer: DistanceComputer,
) -> NDArray:
    expected_positions = np.sqrt(alpha_t) * population
    distances_squared = distance_computer.compute_distances(x_t, expected_positions)
    gaussian_weights = compute_gaussian_weights(distances_squared, alpha_t)
    combined_weights = fitness_weights * gaussian_weights

    z = np.sum(combined_weights)
    if z < 1e-10:
        combined_weights = np.ones_like(combined_weights) / len(combined_weights)
        z = 1.0

    normalized_weights = combined_weights / z
    x_hat_0 = np.sum(normalized_weights[:, np.newaxis] * population, axis=0)
    return x_hat_0


def compute_epsilon_hat(x_t: NDArray, x_hat_0: NDArray, alpha_t: float) -> NDArray:
    numerator = x_t - np.sqrt(alpha_t) * x_hat_0
    denominator = np.sqrt(1 - alpha_t)
    return numerator / denominator if denominator > 1e-10 else np.zeros_like(x_t)


def evolution_step(
    x_t: NDArray,
    x_hat_0: NDArray,
    epsilon_hat: NDArray,
    alpha_t: float,
    alpha_t_minus_1: float,
    sigma_t: float,
    rng: np.random.Generator,
) -> NDArray:
    sqrt_alpha_prev = np.sqrt(alpha_t_minus_1)
    direction_term = np.sqrt(max(0, 1 - alpha_t_minus_1 - sigma_t**2))
    mutation = sigma_t * rng.standard_normal(x_t.shape)
    return sqrt_alpha_prev * x_hat_0 + direction_term * epsilon_hat + mutation
