"""Core evolution step logic."""

import numpy as np

from devol.distance import DistanceComputer, FloatArray


def compute_gaussian_weights(distances_squared: FloatArray, alpha_t: float, epsilon: float = 1e-10) -> FloatArray:
    variance = 1 - alpha_t
    if variance < epsilon:
        return np.ones_like(distances_squared)
    exp_term = -distances_squared / (2 * variance)
    result: FloatArray = np.exp(np.clip(exp_term, -50, 50))
    return result


def estimate_x0(
    x_t: FloatArray,
    population: FloatArray,
    fitness_weights: FloatArray,
    alpha_t: float,
    distance_computer: DistanceComputer,
) -> FloatArray:
    """Estimate the clean data point x₀ from noisy observation xₜ using Bayesian inference.

    Implements Equations 8 and 9 from "Diffusion Models are Evolutionary Algorithms"
    (https://arxiv.org/abs/2410.02543):

    Equation 8: x̂₀(xₜ, α, t) = (1/Z) Σ[x∈Xₜ] g[f(x)] 𝒩(xₜ; √(αₜ)x, 1-αₜ) x
    Equation 9: Z = p(xₜ) = Σ[x∈Xₜ] g[f(x)] 𝒩(xₜ; √(αₜ)x, 1-αₜ)

    This function combines two weighting mechanisms:
    1. Fitness weights g[f(x)]: Prioritize high-fitness individuals from the population
    2. Gaussian weights 𝒩(xₜ; √(αₜ)x, 1-αₜ): Weight by proximity to the noisy observation

    The Gaussian term makes each population member sensitive only to local neighbors,
    while the fitness term guides selection toward better solutions. The combination
    creates a fitness-weighted, proximity-aware estimate of the original data point.

    Args:
        x_t: Current noisy observation at diffusion step t
        population: Array of candidate solutions from the evolutionary population
        fitness_weights: Pre-computed fitness weights g[f(x)] for each population member
        alpha_t: Noise schedule parameter αₜ controlling diffusion variance
        distance_computer: Object to compute distances between points

    Returns:
        x̂₀: Estimated clean data point (weighted average of population)
    """
    expected_positions = np.sqrt(alpha_t) * population
    distances_squared = distance_computer.compute_distances(x_t, expected_positions)
    gaussian_weights = compute_gaussian_weights(distances_squared, alpha_t)
    combined_weights = fitness_weights * gaussian_weights

    z = float(np.sum(combined_weights))
    if z < 1e-10:
        combined_weights = np.ones_like(combined_weights) / len(combined_weights)
        z = 1.0

    normalized_weights = combined_weights / z
    x_hat_0: FloatArray = np.sum(normalized_weights[:, np.newaxis] * population, axis=0)
    return x_hat_0


def compute_epsilon_hat(x_t: FloatArray, x_hat_0: FloatArray, alpha_t: float) -> FloatArray:
    """
    This is implementing equation 10 from section 3
    """
    numerator = x_t - np.sqrt(alpha_t) * x_hat_0
    denominator = np.sqrt(1 - alpha_t)
    result: FloatArray = numerator / denominator if denominator > 1e-10 else np.zeros_like(x_t)
    return result


def evolution_step(
    x_t: FloatArray,
    x_hat_0: FloatArray,
    epsilon_hat: FloatArray,
    alpha_t: float,
    alpha_t_minus_1: float,
    sigma_t: float,
    rng: np.random.Generator,
) -> FloatArray:
    sqrt_alpha_prev = np.sqrt(alpha_t_minus_1)
    direction_term = np.sqrt(max(0, 1 - alpha_t_minus_1 - sigma_t**2))
    mutation = sigma_t * rng.standard_normal(x_t.shape)
    result: FloatArray = sqrt_alpha_prev * x_hat_0 + direction_term * epsilon_hat + mutation
    return result
