"""Tests for low-level evolution utilities."""

import numpy as np

from devol.evolution import compute_epsilon_hat, compute_gaussian_weights, evolution_step
from devol.schedules import create_alpha_schedule, create_sigma_schedule


def test_gaussian_weights_return_uniform_when_variance_small() -> None:
    distances = np.array([0.0, 1.0, 2.0])
    weights = compute_gaussian_weights(distances, alpha_t=0.9999)
    assert np.allclose(weights, np.ones_like(distances))


def test_epsilon_hat_returns_zero_when_denom_zero() -> None:
    x_t = np.array([1.0, -1.0])
    x_hat = np.array([1.0, -1.0])
    epsilon = compute_epsilon_hat(x_t, x_hat, alpha_t=1.0)
    assert np.allclose(epsilon, np.zeros_like(x_t))


def test_evolution_step_respects_output_shape() -> None:
    rng = np.random.default_rng(0)
    x_t = np.array([0.2, -0.1, 0.5])
    x_hat_0 = np.zeros_like(x_t)
    epsilon_hat = np.ones_like(x_t) * 0.1
    result = evolution_step(x_t, x_hat_0, epsilon_hat, alpha_t=0.9, alpha_t_minus_1=0.91, sigma_t=0.05, rng=rng)
    assert result.shape == x_t.shape


def test_sigma_schedule_is_non_negative() -> None:
    alpha = create_alpha_schedule("cosine", total_steps=5, epsilon=1e-4)
    sigma = create_sigma_schedule(alpha, sigma_m=0.5)
    assert np.all(sigma >= 0)
