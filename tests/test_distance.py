"""Tests for distance computation strategies."""

import numpy as np
import pytest

from devol.distance import CosineDistance, EuclideanDistance, LatentDistance


def test_euclidean_distance():
    computer = EuclideanDistance()
    x_i = np.array([0.0, 0.0])
    population = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    distances = computer.compute_distances(x_i, population)
    expected = np.array([1.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(distances, expected)


def test_latent_distance():
    computer = LatentDistance(param_dim=4, latent_dim=2, seed=42)
    x_i = np.array([1.0, 0.0, 0.0, 0.0])
    population = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    distances = computer.compute_distances(x_i, population)
    assert distances.shape == (2,)
    assert distances[0] < distances[1]


def test_cosine_distance():
    computer = CosineDistance()
    x_i = np.array([1.0, 0.0])
    population = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    distances = computer.compute_distances(x_i, population)
    expected = np.array([0.0, 2.0, 1.0])
    np.testing.assert_array_almost_equal(distances, expected)


def test_cosine_distance_zero_vector():
    computer = CosineDistance()
    x_i = np.array([0.0, 0.0])
    population = np.array([[1.0, 0.0]])
    distances = computer.compute_distances(x_i, population)
    assert distances[0] == 1.0
